"""
LLM Zero-shot分类器模块

使用大语言模型进行零样本精神疾病辅助诊断
使用<think>...</think><box>...</box>格式进行诊断输出
Prompt模板从prompts文件夹统一加载
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入LLM客户端
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import create_llm_client, extract_reasoning_content

try:
    from .config import (
        LLM_CONFIG, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS,
        TWELVE_CLASS_DETAILED_LABELS, VALID_ICD_SUBCODES
    )
    from .data_utils import load_and_process_data, prepare_classification_dataset
    from .metrics import (
        calculate_singlelabel_metrics, calculate_multilabel_metrics,
        format_metrics_for_print
    )
    from .prompts import PromptLoader
except ImportError:
    from config import (
        LLM_CONFIG, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS,
        TWELVE_CLASS_DETAILED_LABELS, VALID_ICD_SUBCODES
    )
    from data_utils import load_and_process_data, prepare_classification_dataset
    from metrics import (
        calculate_singlelabel_metrics, calculate_multilabel_metrics,
        format_metrics_for_print
    )
    from prompts import PromptLoader


# ============================================================
# 中文到英文标签的映射
# ============================================================

CHINESE_TO_ENGLISH = {
    # 2分类
    "抑郁": "Depression",
    "焦虑": "Anxiety",
    # 4分类
    "mix": "Mixed",
    "Mix": "Mixed",
    "mixed": "Mixed",
    "Mixed": "Mixed",
    "混合": "Mixed",
    "others": "Others",
    "Others": "Others",
    "其他": "Others",
    # ICD代码标准化
    "F20": "F20",
    "F31": "F31",
    "F32": "F32",
    "F39": "F39",
    "F41": "F41",
    "F42": "F42",
    "F43": "F43",
    "F45": "F45",
    "F51": "F51",
    "F98": "F98",
    "Z71": "Z71",
}


# ============================================================
# LLM分类器
# ============================================================

class LLMZeroshotClassifier:
    """
    LLM零样本分类器
    
    使用大语言模型进行零样本诊断分类
    使用<think>...</think><box>...</box>格式进行诊断输出
    Prompt模板从prompts文件夹统一加载
    """
    
    def __init__(
        self,
        classification_type: str = "12class",
        model_name: str = None,
        temperature: float = None,
        max_tokens: int = None,
        max_workers: int = 16,
        prompts_dir: str = None,
        api_key: str = None
    ):
        """
        初始化分类器
        
        Args:
            classification_type: 分类类型 ("2class", "4class", "12class")
            model_name: LLM模型名称
            temperature: 生成温度
            max_tokens: 最大生成token数
            max_workers: 并行处理的最大工作线程数
            prompts_dir: prompt模板目录路径
            api_key: API密钥
        """
        self.classification_type = classification_type
        self.model_name = model_name or LLM_CONFIG['model_name']
        self.temperature = temperature if temperature is not None else LLM_CONFIG['temperature']
        self.max_tokens = max_tokens or LLM_CONFIG['max_tokens']
        self.max_workers = max_workers
        
        # 初始化Prompt加载器
        if prompts_dir is None:
            prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        self.prompt_loader = PromptLoader(prompts_dir)
        
        # 初始化LLM客户端
        self.client = create_llm_client(self.model_name, api_key="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")
        
        # 标签设置
        if classification_type == "2class":
            self.labels = TWO_CLASS_LABELS
            self.is_multilabel = False
        elif classification_type == "4class":
            self.labels = FOUR_CLASS_LABELS
            self.is_multilabel = False
        elif classification_type == "12class_detailed":
            # 使用ICD-10小类标签
            self.labels = TWELVE_CLASS_DETAILED_LABELS
            self.is_multilabel = True
        else:  # 12class（大类）
            self.labels = TWELVE_CLASS_LABELS
            self.is_multilabel = True
        
        # 加载prompt模板
        self._load_prompts()
    
    def _load_prompts(self):
        """加载所有需要的prompt模板"""
        # 根据分类类型加载对应的system和user prompt
        # 12class_detailed使用与12class相同的prompt模板（prompt已包含小类说明）
        if self.classification_type == "12class_detailed":
            type_suffix = "12class"
        else:
            type_suffix = self.classification_type  # "2class", "4class", "12class"
        
        self.system_prompt = self.prompt_loader.load(f"system_{type_suffix}")
        self.user_template = self.prompt_loader.load(f"user_{type_suffix}")
    
    def _build_messages(self, conversation: str) -> List[Dict[str, str]]:
        """构建消息列表"""
        # 截断过长的对话
        conversation = conversation[:12000]
        
        user_content = self.user_template.format(conversation=conversation)
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def _parse_response(self, response_text: str) -> Tuple[str, Any]:
        """
        解析LLM响应，提取<think>和<box>内容
        
        Args:
            response_text: LLM原始响应文本
            
        Returns:
            Tuple[thinking, prediction]: 思考过程和预测结果
        """
        # 提取<think>内容
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        thinking = think_match.group(1).strip() if think_match else ""
        
        # 提取<box>内容
        box_match = re.search(r'<box>(.*?)</box>', response_text, re.DOTALL)
        
        if not box_match:
            # 尝试备用模式：直接在响应末尾查找诊断关键词
            return thinking, self._fallback_parse(response_text)
        
        box_content = box_match.group(1).strip()
        
        # 解析box内容
        return thinking, self._parse_box_content(box_content)
    
    def _parse_prediction_only(self, response_text: str) -> Any:
        """
        仅解析预测结果（<box>内容），不处理thinking
        
        Args:
            response_text: LLM原始响应文本
            
        Returns:
            预测结果
        """
        # 提取<box>内容
        box_match = re.search(r'<box>(.*?)</box>', response_text, re.DOTALL)
        
        if not box_match:
            # 尝试备用模式：直接在响应末尾查找诊断关键词
            return self._fallback_parse(response_text)
        
        box_content = box_match.group(1).strip()
        
        # 解析box内容
        return self._parse_box_content(box_content)
    
    def _parse_box_content(self, box_content: str) -> Any:
        """
        解析<box>标签内的内容
        
        Args:
            box_content: box标签内的文本
            
        Returns:
            预测标签（单标签为字符串，多标签为列表）
        """
        if self.classification_type == "2class":
            return self._parse_2class(box_content)
        elif self.classification_type == "4class":
            return self._parse_4class(box_content)
        elif self.classification_type == "12class_detailed":
            return self._parse_12class_detailed(box_content)
        else:  # 12class（大类）
            return self._parse_12class(box_content)
    
    def _parse_2class(self, content: str) -> str:
        """解析2分类结果"""
        content = content.strip()
        
        # 直接匹配关键词
        if "抑郁" in content or "Depression" in content.lower():
            return "Depression"
        elif "焦虑" in content or "Anxiety" in content.lower():
            return "Anxiety"
        
        # 默认返回
        return "Depression"  # 2分类时默认返回Depression
    
    def _parse_4class(self, content: str) -> str:
        """解析4分类结果"""
        content = content.strip().lower()
        
        # 按优先级匹配（mix和others优先，因为可能同时包含抑郁/焦虑关键词）
        if "mix" in content or "混合" in content:
            return "Mixed"
        elif "other" in content or "其他" in content:
            return "Others"
        elif "抑郁" in content or "depression" in content:
            return "Depression"
        elif "焦虑" in content or "anxiety" in content:
            return "Anxiety"
        
        return "Others"
    
    def _parse_12class(self, content: str) -> List[str]:
        """
        解析12分类结果（多标签，大类）
        
        支持格式：
        - F32.1;F41.0 -> ["F32", "F41"]
        - F32;F41 -> ["F32", "F41"]
        - F32.1 -> ["F32"]
        """
        # 提取所有ICD代码（大类）
        # 匹配 Fxx 或 Z71 格式
        codes = re.findall(r'(F\d{2}|Z71)', content.upper())
        
        # 去重并保持顺序
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen and code in self.labels:
                seen.add(code)
                unique_codes.append(code)
        
        # 如果没有找到有效代码，返回Others
        if not unique_codes:
            return ["Others"]
        
        return unique_codes
    
    def _parse_12class_detailed(self, content: str) -> List[str]:
        """
        解析12分类结果（多标签，小类）
        
        支持格式：
        - F32.1;F41.0 -> ["F32.1", "F41.0"]
        - F32.10;F41.00 -> ["F32.1", "F41.0"]
        - F32;F41 -> ["F32.9", "F41.9"]（回退到未特指）
        - F39 -> ["F39"]（F39没有小类）
        - Z71.9 -> ["Z71.9"]
        """
        content = content.upper()
        
        extracted_codes = []
        
        # 首先尝试匹配带小数点的代码（如 F32.1, F32.10, F32.100）
        # 匹配格式: Fxx.x 或 Fxx.xx 或 Fxx.xxx
        detailed_matches = re.findall(r'(F\d{2})\.(\d+)', content)
        for major, minor in detailed_matches:
            # 取小数点后第一位作为小类代码
            subcode = minor[0] if minor else "9"
            code_normalized = f"{major}.{subcode}"
            
            if code_normalized not in extracted_codes:
                if code_normalized in VALID_ICD_SUBCODES:
                    extracted_codes.append(code_normalized)
                else:
                    # 回退到未特指代码
                    unspecified = f"{major}.9"
                    if unspecified in VALID_ICD_SUBCODES and unspecified not in extracted_codes:
                        extracted_codes.append(unspecified)
        
        # 匹配Z71.x格式
        z71_matches = re.findall(r'Z71\.?(\d)?', content)
        for match in z71_matches:
            subcode = match if match else "9"
            code_normalized = f"Z71.{subcode}"
            if code_normalized not in extracted_codes:
                if code_normalized in VALID_ICD_SUBCODES:
                    extracted_codes.append(code_normalized)
                elif "Z71.9" not in extracted_codes:
                    extracted_codes.append("Z71.9")
        
        # 匹配F39（特殊处理，没有小数点形式）
        if re.search(r'F39(?!\d)', content):
            if "F39" not in extracted_codes and "F39" in VALID_ICD_SUBCODES:
                extracted_codes.append("F39")
        
        # 匹配只有大类代码的情况（如 F32, F41）
        # 这些可能是没有小数点的大类代码，需要回退到.9
        major_only = re.findall(r'(?<!\d)(F\d{2})(?!\.\d)', content)
        for major in major_only:
            # 检查是否已经有该大类的小类代码
            has_subcode = any(c.startswith(major + ".") for c in extracted_codes)
            if not has_subcode:
                unspecified = f"{major}.9"
                if unspecified in VALID_ICD_SUBCODES and unspecified not in extracted_codes:
                    extracted_codes.append(unspecified)
        
        # 如果没有找到有效代码，返回Others
        if not extracted_codes:
            return ["Others"]
        
        return extracted_codes
    
    def _fallback_parse(self, response_text: str) -> Any:
        """
        备用解析方法，当没有找到<box>标签时使用
        
        Args:
            response_text: 完整响应文本
            
        Returns:
            预测标签
        """
        if self.classification_type == "2class":
            # 检查最后出现的是抑郁还是焦虑
            dep_pos = max(response_text.rfind("抑郁"), response_text.lower().rfind("depression"))
            anx_pos = max(response_text.rfind("焦虑"), response_text.lower().rfind("anxiety"))
            
            if dep_pos > anx_pos:
                return "Depression"
            elif anx_pos > dep_pos:
                return "Anxiety"
            return "Depression"
        
        elif self.classification_type == "4class":
            # 检查最后出现的关键词
            keywords = {
                "Depression": max(response_text.rfind("抑郁"), response_text.lower().rfind("depression")),
                "Anxiety": max(response_text.rfind("焦虑"), response_text.lower().rfind("anxiety")),
                "Mixed": max(response_text.rfind("mix"), response_text.rfind("混合")),
                "Others": max(response_text.rfind("other"), response_text.rfind("其他"))
            }
            return max(keywords, key=keywords.get) if max(keywords.values()) >= 0 else "Others"
        
        elif self.classification_type == "12class_detailed":
            return self._parse_12class_detailed(response_text)
        
        else:  # 12class（大类）
            return self._parse_12class(response_text)
    
    def _classify_single(self, text: str) -> Tuple[Any, str, str]:
        """
        对单个文本进行分类
        
        Returns:
            Tuple[prediction, thinking, raw_response]: 预测结果、思考过程、原始响应
        """
        messages = self._build_messages(text)
        
        try:
            # 调用LLM（不使用JSON格式，使用自然语言输出，返回原始响应对象）
            result = self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                use_json_format=False,
                return_raw_response=True
            )
            
            # 解包返回值（包含原始响应对象）
            response, tokens, raw_response_obj = result
            
            # 获取响应文本
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # 优先从API响应对象中提取reasoning（适用于reasoning模型如DeepSeek R1）
            thinking = extract_reasoning_content(raw_response_obj)
            
            # 如果API响应中没有reasoning字段，再从响应文本中提取<think>标签内容
            if not thinking:
                think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
                thinking = think_match.group(1).strip() if think_match else ""
            
            # 解析预测结果（只提取<box>内容）
            prediction = self._parse_prediction_only(response_text)
            
            return prediction, thinking, response_text
            
        except Exception as e:
            print(f"LLM调用失败: {e}")
            default = ["Others"] if self.is_multilabel else "Others"
            return default, "", str(e)
    
    def predict(self, texts: List[str], show_progress: bool = True) -> List[Any]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度
            
        Returns:
            预测标签列表
        """
        predictions = [None] * len(texts)
        
        print(f"正在使用LLM进行{self.classification_type}分类预测...")
        print(f"模型: {self.model_name}, 样本数: {len(texts)}")
        print(f"输出格式: <think>思考过程</think><box>诊断结果</box>")
        
        start_time = time.time()
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._classify_single, text): idx
                for idx, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    pred, thinking, raw = future.result()
                    predictions[idx] = pred
                except Exception as e:
                    print(f"样本 {idx} 处理失败: {e}")
                    predictions[idx] = ["Others"] if self.is_multilabel else "Others"
                
                completed += 1
                if show_progress and completed % 10 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed) * (len(texts) - completed)
                    print(f"进度: {completed}/{len(texts)} ({100*completed/len(texts):.1f}%), "
                          f"已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
        
        elapsed = time.time() - start_time
        print(f"预测完成，总耗时: {elapsed:.2f}秒")
        
        return predictions
    
    def predict_with_details(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        批量预测并返回详细结果（包含思考过程和原始响应）
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度
            
        Returns:
            包含完整预测信息的字典列表
        """
        results = [None] * len(texts)
        
        print(f"正在使用LLM进行{self.classification_type}分类预测（详细模式）...")
        print(f"模型: {self.model_name}, 样本数: {len(texts)}")
        
        start_time = time.time()
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._classify_single, text): idx
                for idx, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    pred, thinking, raw = future.result()
                    results[idx] = {
                        'prediction': pred,
                        'thinking': thinking,
                        'raw_response': raw
                    }
                except Exception as e:
                    results[idx] = {
                        'prediction': ["Others"] if self.is_multilabel else "Others",
                        'thinking': "",
                        'raw_response': "",
                        'error': str(e)
                    }
                
                completed += 1
                if show_progress and completed % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"进度: {completed}/{len(texts)} ({100*completed/len(texts):.1f}%)")
        
        elapsed = time.time() - start_time
        print(f"预测完成，总耗时: {elapsed:.2f}秒")
        
        return results
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[Any],
        show_progress: bool = True,
        return_details: bool = False
    ) -> Dict[str, Any]:
        """
        评估分类器
        
        Args:
            texts: 文本列表
            labels: 真实标签列表
            show_progress: 是否显示进度
            return_details: 是否返回详细的推理结果
            
        Returns:
            评估指标字典，如果return_details=True，还包含详细预测结果
        """
        if return_details:
            # 使用详细预测模式
            detailed_results = self.predict_with_details(texts, show_progress)
            predictions = [r['prediction'] for r in detailed_results]
        else:
            predictions = self.predict(texts, show_progress)
            detailed_results = None
        
        if self.is_multilabel:
            metrics = calculate_multilabel_metrics(labels, predictions, self.labels)
        else:
            metrics = calculate_singlelabel_metrics(labels, predictions, self.labels)
        
        result = {'metrics': metrics}
        
        if return_details:
            # 组装详细结果，包含输入文本、真实标签和预测详情
            sample_details = []
            for i, (text, label, detail) in enumerate(zip(texts, labels, detailed_results)):
                sample_details.append({
                    'index': i,
                    'text': text[:500] + '...' if len(text) > 500 else text,  # 截断过长文本
                    'true_label': label,
                    'prediction': detail['prediction'],
                    'thinking': detail.get('thinking', ''),
                    'raw_response': detail.get('raw_response', ''),
                    'error': detail.get('error', None)
                })
            result['sample_details'] = sample_details
        
        return result
    
    def get_loaded_prompts(self) -> Dict[str, str]:
        """
        获取已加载的prompt模板（用于调试）
        
        Returns:
            prompt名称到内容的映射
        """
        return {
            'system_prompt': self.system_prompt,
            'user_template': self.user_template
        }


def train_and_evaluate_llm(
    test_data: List[Dict],
    classification_type: str = "12class",
    model_name: str = None,
    save_details: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    评估LLM零样本分类器（不需要训练）
    
    Args:
        test_data: 测试数据
        classification_type: 分类类型
        model_name: 模型名称
        save_details: 是否保存详细的推理结果
        
    Returns:
        评估结果字典，包含metrics和可选的sample_details
    """
    print(f"\n{'='*60}")
    print(f"LLM Zero-shot {classification_type}分类评估")
    print("="*60)
    
    # 准备数据
    test_texts, test_labels = prepare_classification_dataset(
        test_data, classification_type
    )
    
    print(f"测试集大小: {len(test_texts)}")
    
    # 创建分类器
    clf = LLMZeroshotClassifier(
        classification_type=classification_type,
        model_name=model_name,
        **kwargs
    )
    
    # 评估（根据save_details决定是否返回详细结果）
    eval_result = clf.evaluate(test_texts, test_labels, return_details=save_details)
    
    # 打印结果
    task_name = f"LLM Zero-shot {classification_type}分类 ({clf.model_name})"
    clf_type = "multi" if classification_type == "12class" else "single"
    print(format_metrics_for_print(eval_result['metrics'], task_name, clf_type))
    
    result = {
        'classification_type': classification_type,
        'model_name': clf.model_name,
        'method': 'LLM-Zeroshot',
        'metrics': eval_result['metrics']
    }
    
    # 如果有详细结果，也加入返回值
    if save_details and 'sample_details' in eval_result:
        result['sample_details'] = eval_result['sample_details']
    
    return result


def run_llm_benchmark(
    test_file: str,
    model_name: str = None,
    classification_types: List[str] = None,
    save_details: bool = True
) -> List[Dict[str, Any]]:
    """
    运行完整的LLM benchmark
    
    Args:
        test_file: 测试数据文件路径
        model_name: 模型名称
        classification_types: 要测试的分类类型列表
            支持: "2class", "4class", "12class", "12class_detailed"
        save_details: 是否保存详细的推理结果（包含thinking和raw_response）
        
    Returns:
        所有评估结果列表，每个结果包含metrics和可选的sample_details
    """
    if classification_types is None:
        classification_types = ["2class", "4class", "12class", "12class_detailed"]
    
    # 加载数据
    print("正在加载数据...")
    test_data = load_and_process_data(test_file)
    
    results = []
    
    for class_type in classification_types:
        try:
            result = train_and_evaluate_llm(
                test_data,
                classification_type=class_type,
                model_name=model_name,
                save_details=save_details
            )
            results.append(result)
        except Exception as e:
            print(f"评估 {class_type} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


if __name__ == "__main__":
    from config import TEST_DATA_100_FILE
    
    # 测试prompt加载
    print("测试Prompt加载...")
    for ct in ["2class", "4class", "12class", "12class_detailed"]:
        clf = LLMZeroshotClassifier(classification_type=ct)
        prompts = clf.get_loaded_prompts()
        print(f"\n{ct} 分类:")
        print(f"  系统提示: {prompts['system_prompt'][:80]}...")
        print(f"  用户模板: {prompts['user_template'][:80]}...")
        print(f"  标签数量: {len(clf.labels)}")
    print("\nPrompt加载成功!")
    
    # 测试解析逻辑
    print("\n测试解析逻辑...")
    
    # 2分类测试
    clf_2 = LLMZeroshotClassifier(classification_type="2class")
    test_response_2 = "<think>患者主诉情绪低落，兴趣减退，睡眠障碍，符合抑郁症诊断标准。</think><box>抑郁</box>"
    thinking, pred = clf_2._parse_response(test_response_2)
    print(f"2分类测试: 输入='{test_response_2[:50]}...', 预测='{pred}'")
    
    # 4分类测试
    clf_4 = LLMZeroshotClassifier(classification_type="4class")
    test_response_4 = "<think>患者同时有抑郁和焦虑症状。</think><box>mix</box>"
    thinking, pred = clf_4._parse_response(test_response_4)
    print(f"4分类测试: 输入='{test_response_4[:50]}...', 预测='{pred}'")
    
    # 12分类测试（大类）
    clf_12 = LLMZeroshotClassifier(classification_type="12class")
    test_response_12 = "<think>根据症状分析，患者符合抑郁发作和焦虑障碍。</think><box>F32.1;F41.0</box>"
    thinking, pred = clf_12._parse_response(test_response_12)
    print(f"12分类(大类)测试: 输入='{test_response_12[:50]}...', 预测={pred}")
    
    # 12分类测试（小类）
    clf_12d = LLMZeroshotClassifier(classification_type="12class_detailed")
    test_response_12d = "<think>根据症状分析，患者符合中度抑郁发作和惊恐障碍。</think><box>F32.1;F41.0</box>"
    thinking, pred = clf_12d._parse_response(test_response_12d)
    print(f"12分类(小类)测试: 输入='{test_response_12d[:50]}...', 预测={pred}")
    
    # 测试更多小类解析格式
    print("\n测试小类解析格式...")
    test_cases = [
        "F32.100;F41.000",  # 三位小数格式
        "F32.1;F41.0",      # 一位小数格式
        "F32;F41",          # 只有大类，应回退到.9
        "F39",              # 特殊处理的F39
        "Z71.9",            # Z71格式
        "F32.1;F32.2",      # 同一大类多个小类
    ]
    for case in test_cases:
        pred = clf_12d._parse_12class_detailed(case)
        print(f"  '{case}' -> {pred}")
    
    print("\n解析测试完成!")
    
    # 运行简单benchmark（只测试12class_detailed作为快速验证）
    print("\n" + "="*60)
    print("运行LLM Benchmark（12class_detailed快速验证）")
    print("="*60)
    
    results = run_llm_benchmark(
        TEST_DATA_100_FILE,
        model_name="qwen3-32b:9041",
        classification_types=["12class_detailed"],
        save_details=True  # 保存详细的推理结果
    )
    
    # 保存结果（包含详细推理结果）
    model_short_name = "qwen3-32b"
    output_file = os.path.join(OUTPUT_DIR, f"llm_zeroshot_{model_short_name}_12class_detailed_results.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")
