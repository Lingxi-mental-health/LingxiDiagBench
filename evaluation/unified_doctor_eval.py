#!/usr/bin/env python3
"""
统一的 Doctor Agent 评估程序（LLM-as-Judge）

评估指标（5个维度）：
1. 临床准确性与能力 (Clinical_Accuracy_Competence) - 治疗知识运用、干预匹配、临床推理
2. 伦理与专业行为 (Ethical_Professional_Conduct) - 专业边界、文化敏感性、尊重包容
3. 评估与回应 (Assessment_Response) - 理解成员输入、共情调频、优先级判断
4. 治疗关系与联盟 (Therapeutic_Relationship_Alliance) - 协作关系、自主权支持、平衡视角
5. AI沟通质量 (AI_Communication_Quality) - 自然流畅、避免LLM常见问题（机械、重复、模板化）

每个维度采用1-6分量表评分。

支持两种评估模式：
1. 原始对话模式 (--use-original-dialogue): 评估数据集中真实的对话记录
2. Doctor Agent 模式: 使用 Doctor Agent 生成对话并评估

使用方法:
    # 模式1: 使用原始数据集中的对话（评估真实对话数据）
    python unified_doctor_eval.py \\
        --data-file /path/to/conversations.json \\
        --use-original-dialogue \\
        --eval-models "gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041" \\
        --max-workers 8 \\
        --limit 10

    # 模式2: 使用 Doctor Agent 生成对话（评估 Doctor Agent）
    python unified_doctor_eval.py \\
        --data-file /path/to/cases.json \\
        --doctor-model "qwen3-30b@10.119.28.185:9041" \\
        --doctor-version v2 \\
        --patient-model "qwen3-30b@10.119.28.185:9041" \\
        --patient-version v3 \\
        --eval-models "gemma-3-27b-it@10.119.28.185:9051" \\
        --max-workers 8 \\
        --max-turns 20 \\
        --limit 10

    # 模式3: 只生成对话，不进行评估（--generate-only）
    python unified_doctor_eval.py \\
        --data-file /path/to/cases.json \\
        --doctor-model "qwen3-30b@10.119.28.185:9041" \\
        --doctor-version v2 \\
        --patient-model "qwen3-30b@10.119.28.185:9041" \\
        --patient-version v3 \\
        --generate-only \\
        --max-workers 8 \\
        --max-turns 20 \\
        --limit 10
    # 输出: dialogues_v2_qwen3-30b_xxx_20250102_120000.json

    # 模式4: 只评估，从文件加载对话（--eval-only）
    python unified_doctor_eval.py \\
        --eval-only \\
        --dialogue-file /path/to/dialogues_v2_xxx.json \\
        --eval-models "gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041" \\
        --max-workers 8
        
    # 支持的 Doctor 版本 (动态加载，可扩展):
    # - v1: 基础版本 (src/doctor/doctor_v1.py)
    # - v2: 阶段式诊断树版本 (src/doctor/doctor_v2.py)
    # - v3: 增强版本 (src/doctor/doctor_v3.py)
    # - 以及任意新版本: 只需创建 src/doctor/doctor_{version}.py 并定义 Doctor 类
"""

import argparse
import json
import math
import os
import sys
import re
import concurrent.futures
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from evaluation.llm_client import (
    create_llm_client,
    PromptLoader,
)

# 导入诊断分类相关模块
from evaluation.static.config import (
    TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS,
    VALID_ICD_CODES, VALID_ICD_SUBCODES
)
from evaluation.static.metrics import (
    calculate_singlelabel_metrics, calculate_multilabel_metrics,
    format_metrics_for_print
)
from evaluation.static.data_utils import (
    extract_f_codes_from_diagnosis_code, extract_detailed_codes,
    classify_2class, classify_4class
)
from evaluation.static.prompts import PromptLoader as ClassificationPromptLoader


# ============================================================
# 诊断验证器（用于动态诊断正确率评估）
# ============================================================

class DiagnosisVerifier:
    """
    诊断验证器
    
    使用LLM对对话结果进行2分类、4分类、12分类诊断，
    并与真实标签进行比较计算正确率
    """
    
    # 分类类型配置
    CLASSIFICATION_TYPES = ["2class", "4class", "12class"]
    
    def __init__(
        self,
        verifier_model: str,
        prompts_dir: str = None,
        api_key: str = None,
        max_workers: int = 16,
    ):
        """
        初始化诊断验证器
        
        Args:
            verifier_model: 验证模型名称
            prompts_dir: prompts 目录路径
            api_key: API Key（仅用于 OpenRouter）
            max_workers: 最大并行工作线程数
        """
        self.client = create_llm_client(verifier_model, api_key=api_key)
        
        if prompts_dir is None:
            prompts_dir = PROJECT_ROOT / "evaluation" / "static" / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self.verifier_model = verifier_model
        self.max_workers = max_workers
        self._prompt_cache = {}
    
    def _load_prompts(self, classification_type: str) -> Tuple[str, str]:
        """
        加载分类prompts
        
        Args:
            classification_type: 分类类型 ("2class", "4class", "12class")
            
        Returns:
            Tuple[system_prompt, user_template]
        """
        cache_key = f"{classification_type}_prompts"
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]
        
        # 加载system prompt
        system_file = self.prompts_dir / f"system_{classification_type}.txt"
        if not system_file.exists():
            raise FileNotFoundError(f"System prompt文件不存在: {system_file}")
        
        with open(system_file, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        # 加载user template
        user_file = self.prompts_dir / f"user_{classification_type}.txt"
        if not user_file.exists():
            raise FileNotFoundError(f"User template文件不存在: {user_file}")
        
        with open(user_file, 'r', encoding='utf-8') as f:
            user_template = f.read()
        
        self._prompt_cache[cache_key] = (system_prompt, user_template)
        return system_prompt, user_template
    
    def _parse_box_content(self, response_text: str, classification_type: str) -> Any:
        """
        从响应中提取<box>标签内容并解析
        
        Args:
            response_text: LLM响应文本
            classification_type: 分类类型
            
        Returns:
            解析后的预测结果
        """
        # 提取<box>内容
        box_match = re.search(r'<box>(.*?)</box>', response_text, re.DOTALL)
        
        if not box_match:
            # 尝试备用解析
            return self._fallback_parse(response_text, classification_type)
        
        box_content = box_match.group(1).strip()
        
        if classification_type == "2class":
            return self._parse_2class(box_content)
        elif classification_type == "4class":
            return self._parse_4class(box_content)
        else:  # 12class
            return self._parse_12class(box_content)
    
    def _parse_2class(self, content: str) -> str:
        """解析2分类结果"""
        content = content.strip().lower()
        
        if "抑郁" in content or "depression" in content:
            return "Depression"
        elif "焦虑" in content or "anxiety" in content:
            return "Anxiety"
        
        return "Depression"  # 默认返回
    
    def _parse_4class(self, content: str) -> str:
        """解析4分类结果"""
        content = content.strip().lower()
        
        # 按优先级匹配
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
        """解析12分类结果（多标签）"""
        # 提取所有ICD代码（大类）
        codes = re.findall(r'(F\d{2}|Z71)', content.upper())
        
        # 去重并保持顺序
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen and code in TWELVE_CLASS_LABELS:
                seen.add(code)
                unique_codes.append(code)
        
        if not unique_codes:
            return ["Others"]
        
        return unique_codes
    
    def _fallback_parse(self, response_text: str, classification_type: str) -> Any:
        """备用解析方法"""
        if classification_type == "2class":
            dep_pos = max(response_text.rfind("抑郁"), response_text.lower().rfind("depression"))
            anx_pos = max(response_text.rfind("焦虑"), response_text.lower().rfind("anxiety"))
            
            if dep_pos > anx_pos:
                return "Depression"
            elif anx_pos > dep_pos:
                return "Anxiety"
            return "Depression"
        
        elif classification_type == "4class":
            return self._parse_4class(response_text)
        
        else:  # 12class
            return self._parse_12class(response_text)
    
    def verify_single(
        self,
        dialogue_history: str,
        classification_type: str,
    ) -> Dict[str, Any]:
        """
        对单个对话进行诊断验证
        
        Args:
            dialogue_history: 对话历史
            classification_type: 分类类型
            
        Returns:
            Dict 包含:
                - prediction: 预测结果
                - content: 模型输出的 content 内容
                - reasoning: 模型输出的 reasoning 内容
                - prompt: 发送给模型的完整 prompt
                - tokens: token 使用量
        """
        from evaluation.llm_client import extract_reasoning_content
        
        try:
            system_prompt, user_template = self._load_prompts(classification_type)
            
            # 截断过长的对话
            dialogue_truncated = dialogue_history[:12000]
            
            user_content = user_template.format(conversation=dialogue_truncated)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # 保存完整的 prompt 用于 debug
            full_prompt = f"[System]\n{system_prompt}\n\n[User]\n{user_content}"
            
            response, tokens = self.client.chat_completion(
                messages,
                temperature=0.6,
                max_tokens=4096,
                use_json_format=False,  # 使用自由文本格式，输出 <box>...</box>
            )
            
            # 获取响应文本（content）
            content = ""
            if hasattr(response, 'content'):
                content = response.content or ""
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            
            # 获取 reasoning 内容（支持 reasoning 模型）
            reasoning = ""
            try:
                reasoning = extract_reasoning_content(response) or ""
            except Exception:
                pass
            
            # 如果 content 为空但 reasoning 有内容（\no_think 模式），使用 reasoning 作为 content
            if not content and reasoning:
                content = reasoning
                reasoning = ""
            
            # 解析预测结果
            prediction = self._parse_box_content(content, classification_type)
            
            return {
                'prediction': prediction,
                'content': content,
                'reasoning': reasoning,
                'prompt': full_prompt,
                'tokens': tokens,
            }
            
        except Exception as e:
            print(f"[DiagnosisVerifier] 验证失败: {e}")
            default = ["Others"] if classification_type == "12class" else "Others"
            return {
                'prediction': default,
                'content': "",
                'reasoning': "",
                'prompt': "",
                'tokens': (0, 0),
                'error': str(e),
            }
    
    def verify_batch(
        self,
        dialogues: List[str],
        classification_type: str,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量验证对话诊断
        
        Args:
            dialogues: 对话历史列表
            classification_type: 分类类型
            show_progress: 是否显示进度
            
        Returns:
            预测结果列表
        """
        results = [None] * len(dialogues)
        
        print(f"[DiagnosisVerifier] 正在进行{classification_type}分类验证...")
        print(f"[DiagnosisVerifier] 模型: {self.verifier_model}, 样本数: {len(dialogues)}")
        
        lock = threading.Lock()
        progress = {"count": 0}
        
        def verify_single_task(idx: int, dialogue: str) -> Tuple[int, Dict]:
            result = self.verify_single(dialogue, classification_type)
            
            with lock:
                progress["count"] += 1
                if show_progress and (progress["count"] % 10 == 0 or progress["count"] == len(dialogues)):
                    print(f"  [{classification_type}] 验证进度: {progress['count']}/{len(dialogues)}")
            
            return idx, result
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(verify_single_task, idx, dialogue)
                for idx, dialogue in enumerate(dialogues)
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    print(f"[DiagnosisVerifier] 任务失败: {e}")
        
        print(f"[DiagnosisVerifier] {classification_type}分类验证完成!")
        return results
    
    def calculate_metrics(
        self,
        predictions: List[Any],
        true_labels: List[Any],
        classification_type: str,
    ) -> Dict[str, Any]:
        """
        计算分类指标
        
        Args:
            predictions: 预测标签列表
            true_labels: 真实标签列表
            classification_type: 分类类型
            
        Returns:
            指标字典
        """
        if classification_type == "2class":
            return calculate_singlelabel_metrics(true_labels, predictions, TWO_CLASS_LABELS)
        elif classification_type == "4class":
            return calculate_singlelabel_metrics(true_labels, predictions, FOUR_CLASS_LABELS)
        else:  # 12class
            return calculate_multilabel_metrics(true_labels, predictions, TWELVE_CLASS_LABELS)


def get_true_labels_for_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    从样本中提取真实标签
    
    Args:
        sample: 样本数据
        
    Returns:
        包含各分类任务真实标签的字典
    """
    diagnosis_code = sample.get("DiagnosisCode", "")
    
    # 提取12分类标签（大类）
    icd_labels = sample.get("icd_clf_label")
    if not icd_labels:
        icd_labels = extract_f_codes_from_diagnosis_code(diagnosis_code)
    
    # 提取详细信息用于2分类和4分类
    detailed_info = extract_detailed_codes(diagnosis_code)
    
    # 2分类标签
    two_class_label = sample.get("two_class_label")
    if two_class_label is None:
        two_class_label = classify_2class(detailed_info)
    
    # 4分类标签
    four_class_label = sample.get("four_class_label")
    if not four_class_label:
        four_class_label = classify_4class(detailed_info)
    
    return {
        "2class": two_class_label,
        "4class": four_class_label,
        "12class": icd_labels,
    }


# ============================================================
# Pydantic 模型定义
# ============================================================

class DoctorMetricScore(BaseModel):
    """单个指标的评分"""
    score: int = Field(..., ge=1, le=6, description="评分 1-6")
    reasoning: str = Field(..., description="简要分析")


class DoctorEvalMetrics(BaseModel):
    """Doctor Agent 评估的5个维度"""
    Clinical_Accuracy_Competence: DoctorMetricScore = Field(..., description="临床准确性与能力")
    Ethical_Professional_Conduct: DoctorMetricScore = Field(..., description="伦理与专业行为")
    Assessment_Response: DoctorMetricScore = Field(..., description="评估与回应")
    Therapeutic_Relationship_Alliance: DoctorMetricScore = Field(..., description="治疗关系与联盟")
    AI_Communication_Quality: DoctorMetricScore = Field(..., description="AI沟通质量")


class DoctorEvaluation(BaseModel):
    """Doctor Agent 评估结果"""
    metrics: DoctorEvalMetrics = Field(..., description="5个维度的评分")

# ============================================================
# Doctor Agent 导入 (延迟加载，带线程锁)
# ============================================================

_doctor_classes_cache: Dict[str, Any] = {}
_doctor_classes_lock = threading.Lock()

def _load_doctor_class(version: str):
    """
    按需加载指定版本的 Doctor 类（线程安全）
    
    支持动态加载任意版本的 Doctor 类，无需硬编码每个版本。
    只需在 src/doctor/ 目录下创建对应的 doctor_{version}.py 文件，
    并在其中定义 Doctor 类即可。
    
    Args:
        version: 医生版本，如 "v1", "v2", "v3", "base" 等
    
    Returns:
        Doctor 类，如果加载失败则返回 None
    
    Examples:
        - version="v1" -> 从 src.doctor.doctor_v1 导入 Doctor
        - version="v2" -> 从 src.doctor.doctor_v2 导入 Doctor  
        - version="v3" -> 从 src.doctor.doctor_v3 导入 Doctor
        - version="base" -> 从 src.doctor.doctor_base 导入 DoctorBase
    """
    import importlib
    
    global _doctor_classes_cache
    
    with _doctor_classes_lock:
        # 只有当缓存值不为 None 时才直接返回，否则尝试重新加载
        if version in _doctor_classes_cache and _doctor_classes_cache[version] is not None:
            return _doctor_classes_cache[version]
        
        doctor_cls = None
        try:
            # 特殊处理 base 版本
            if version == "base":
                module_name = "src.doctor.doctor_base"
                class_name = "DoctorBase"
            else:
                # 动态构建模块名：v1 -> doctor_v1, v2 -> doctor_v2, v3 -> doctor_v3, etc.
                module_name = f"src.doctor.doctor_{version}"
                class_name = "Doctor"
            
            # 动态导入模块
            module = importlib.import_module(module_name)
            doctor_cls = getattr(module, class_name)
            
            _doctor_classes_cache[version] = doctor_cls
            print(f"[DoctorLoader] 成功加载 Doctor{version.upper()} (from {module_name})")
            return doctor_cls
            
        except ModuleNotFoundError:
            print(f"[WARNING] 找不到 Doctor 版本模块: {module_name}")
            print(f"  请确保 src/doctor/doctor_{version}.py 文件存在")
            # 不缓存失败结果，下次会尝试重新加载
            return None
        except AttributeError:
            print(f"[WARNING] 模块 {module_name} 中找不到 {class_name} 类")
            return None
        except Exception as e:
            print(f"[WARNING] 无法加载 Doctor{version.upper()}: {e}")
            return None


def get_doctor_class(version: str):
    """获取指定版本的 Doctor 类"""
    return _load_doctor_class(version)


# Patient Agent 导入
_patient_classes_cache: Dict[str, Any] = {}
_patient_classes_lock = threading.Lock()

def _load_patient_class(version: str):
    """按需加载指定版本的 Patient 类（线程安全）"""
    global _patient_classes_cache
    
    with _patient_classes_lock:
        if version in _patient_classes_cache:
            return _patient_classes_cache[version]
        
        patient_cls = None
        try:
            if version == "v1":
                from src.patient.patient_v1 import Patient as _PatientV1
                patient_cls = _PatientV1
            elif version == "cot":
                from src.patient.patient_cot import Patient as _PatientCoT
                patient_cls = _PatientCoT
            elif version == "mdd5k":
                from src.patient.patient_mdd5k import Patient as _PatientMDD5K
                patient_cls = _PatientMDD5K
            elif version == "v3":
                from src.patient.patient_v3 import Patient as _PatientV3
                patient_cls = _PatientV3
            else:
                print(f"[WARNING] 不支持的 Patient 版本: {version}")
                return None
            
            _patient_classes_cache[version] = patient_cls
            print(f"[PatientLoader] 成功加载 Patient{version.upper()}")
            return patient_cls
            
        except Exception as e:
            print(f"[WARNING] 无法加载 Patient{version.upper()}: {e}")
            _patient_classes_cache[version] = None
            return None


def get_patient_class(version: str):
    """获取指定版本的 Patient 类"""
    return _load_patient_class(version)


# 已知的 Doctor 版本（仅供参考，实际支持动态加载任意版本）
DOCTOR_VERSION_CHOICES = ["base", "v1", "v2", "v3"]
PATIENT_VERSION_CHOICES = ["v1", "cot", "mdd5k", "v3"]


# ============================================================
# 数据结构定义
# ============================================================

@dataclass
class DoctorEvaluationResult:
    """单个对话的评估结果"""
    sample_id: str
    total_turns: int
    # 5个维度的评估
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    average_score: Optional[float] = None
    # Token 使用量
    tokens: Tuple[int, int] = (0, 0)
    # 对话内容（用于调试）
    dialogue_history: str = ""
    patient_info: str = ""
    # 错误信息
    error: Optional[str] = None
    # 诊断验证结果（Dynamic评估）
    diagnosis_verification: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchEvaluationResult:
    """批量评估结果"""
    total_samples: int
    evaluated_samples: int
    results: List[DoctorEvaluationResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 工具函数
# ============================================================

def calc_average(scores: List[Optional[float]]) -> float:
    """计算平均值"""
    valid = [s for s in scores if isinstance(s, (int, float))]
    return sum(valid) / len(valid) if valid else 0.0


def calc_std(scores: List[Optional[float]]) -> float:
    """计算标准差"""
    valid = [float(s) for s in scores if isinstance(s, (int, float))]
    if len(valid) < 2:
        return 0.0
    mean = sum(valid) / len(valid)
    variance = sum((x - mean) ** 2 for x in valid) / len(valid)
    return math.sqrt(variance)


def get_icd_category(sample: Dict[str, Any]) -> str:
    """
    从样本中提取ICD大类（12分类）
    
    按照 doctor_eval_multilabel.py 中定义的12个类别进行分类：
    ["F20", "F31", "F32", "F39", "F41", "F42", "F43", "F45", "F51", "F98", "Z71", "Others"]
    """
    # 定义有效的12大类
    VALID_ICD_CATEGORIES = ["F20", "F31", "F32", "F39", "F41", "F42", "F43", "F45", "F51", "F98", "Z71"]
    
    def extract_category_from_code(code: str) -> str:
        """从诊断代码中提取类别"""
        if not code:
            return "Others"
        
        code = code.strip().upper()
        
        # 检查 Z71
        if "Z71" in code:
            return "Z71"
        
        # 使用正则表达式提取 F 开头的代码（如 F32.1 -> F32）
        match = re.search(r"F(\d{2})", code)
        if match:
            f_code = f"F{match.group(1)}"
            if f_code in VALID_ICD_CATEGORIES:
                return f_code
        
        return "Others"
    
    # 优先使用 icd_clf_label 字段
    icd_labels = sample.get("icd_clf_label", [])
    if icd_labels:
        if isinstance(icd_labels, list) and len(icd_labels) > 0:
            label = icd_labels[0]
            if label in VALID_ICD_CATEGORIES:
                return label
            return "Others"
        elif isinstance(icd_labels, str):
            if icd_labels in VALID_ICD_CATEGORIES:
                return icd_labels
            return "Others"
    
    # 回退到 DiagnosisCode
    diag_code = sample.get("DiagnosisCode", "")
    return extract_category_from_code(diag_code)


def stratified_sample_by_icd(
    samples: List[Dict[str, Any]],
    limit: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    按ICD 12大类进行均匀分层采样
    
    12大类定义（与 doctor_eval_multilabel.py 一致）：
    ["F20", "F31", "F32", "F39", "F41", "F42", "F43", "F45", "F51", "F98", "Z71", "Others"]
    
    Args:
        samples: 原始样本列表
        limit: 目标样本数量
        seed: 随机种子
    
    Returns:
        均匀采样后的样本列表
    """
    import random
    random.seed(seed)
    
    # 按ICD 12大类分组
    icd_groups: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        icd_category = get_icd_category(sample)
        if icd_category not in icd_groups:
            icd_groups[icd_category] = []
        icd_groups[icd_category].append(sample)
    
    num_categories = len(icd_groups)
    if num_categories == 0:
        return samples[:limit]
    
    # 计算每个类别应采样的数量
    base_per_category = limit // num_categories
    remainder = limit % num_categories
    
    # 打印采样信息
    print(f"\n[分层采样] 按 ICD 12大类进行均匀采样（实际存在 {num_categories} 种类别）")
    print(f"[分层采样] 12大类: F20, F31, F32, F39, F41, F42, F43, F45, F51, F98, Z71, Others")
    print(f"[分层采样] 目标样本数: {limit}, 每类基础样本数: {base_per_category}")
    
    sampled_samples = []
    category_counts = {}
    
    # 对每个类别进行采样
    sorted_categories = sorted(icd_groups.keys())
    for idx, category in enumerate(sorted_categories):
        category_samples = icd_groups[category]
        # 前 remainder 个类别多采样一个
        target_count = base_per_category + (1 if idx < remainder else 0)
        actual_count = min(target_count, len(category_samples))
        
        if actual_count > 0:
            selected = random.sample(category_samples, actual_count)
            sampled_samples.extend(selected)
            category_counts[category] = actual_count
    
    # 如果因为某些类别样本不足导致总数不够，从其他类别补充
    if len(sampled_samples) < limit:
        remaining_needed = limit - len(sampled_samples)
        # 收集未被采样的样本
        sampled_ids = {id(s) for s in sampled_samples}
        remaining_samples = [s for s in samples if id(s) not in sampled_ids]
        
        if remaining_samples:
            additional = random.sample(
                remaining_samples, 
                min(remaining_needed, len(remaining_samples))
            )
            for s in additional:
                category = get_icd_category(s)
                category_counts[category] = category_counts.get(category, 0) + 1
            sampled_samples.extend(additional)
    
    # 打印各类别采样结果
    print(f"[分层采样] 各类别采样数量:")
    for category in sorted_categories:
        original_count = len(icd_groups[category])
        sampled_count = category_counts.get(category, 0)
        print(f"  {category}: {sampled_count}/{original_count}")
    print(f"[分层采样] 总采样数: {len(sampled_samples)}")
    
    # 打乱顺序以避免类别聚集
    random.shuffle(sampled_samples)
    
    return sampled_samples


def sanitize_model_name(model: str) -> str:
    """将模型名转换为适合作为文件名的安全字符串"""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip())
    return safe or "model"


def strip_reasoning_from_text(text: str) -> str:
    """
    去除文本中的 reasoning 标记，包括:
    - <think>...</think>
    - <reasoning>...</reasoning>
    - 其他常见的 reasoning 标记
    """
    if not text:
        return text
    
    # 去除 <think>...</think> 标记（包括空标签）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # 去除 <reasoning>...</reasoning> 标记
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL)
    
    # 去除开头的 "xxx" 或类似的无意义占位符（有时模型会输出）
    text = re.sub(r"^\s*xxx\s*", "", text)
    
    # 清理多余的空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def remove_doctor_diagnosis_from_dialogue(dialogue_history: List[str]) -> List[str]:
    """
    移除对话历史中最后一轮医生的诊断内容
    
    为了防止 doctor_xx.py 中生成的诊断结果被包含在评估用的对话历史中，
    需要检查对话历史的最后一条，如果是以"医生:"或"医生："开头且包含诊断相关内容，
    则将其移除。
    
    Args:
        dialogue_history: 对话历史列表
        
    Returns:
        清理后的对话历史列表
    """
    if not dialogue_history:
        return dialogue_history
    
    # 诊断相关的关键词
    diagnosis_keywords = [
        "诊断结束", "诊断结果", "诊断为", "诊断是",
        "ICD-10", "ICD10", "F32", "F31", "F41", "F20", "F39", "F42", "F43", "F45", "F51", "F98", "Z71",
        "<box>", "</box>",
        "最终诊断", "初步诊断", "临床诊断",
    ]
    
    # 检查最后一条是否是医生的回复
    last_entry = dialogue_history[-1].strip()
    is_doctor_entry = last_entry.startswith("医生:") or last_entry.startswith("医生：")
    
    if is_doctor_entry:
        # 检查是否包含诊断相关内容
        contains_diagnosis = any(keyword in last_entry for keyword in diagnosis_keywords)
        
        if contains_diagnosis:
            print(f"[DialogueClean] 检测到最后一轮是医生的诊断内容，已移除")
            return dialogue_history[:-1]
    
    return dialogue_history


def clean_dialogue_string(dialogue_str: str) -> str:
    """
    清理字符串格式的对话历史，移除最后一轮医生的诊断内容
    
    Args:
        dialogue_str: 对话历史字符串（每轮用换行符分隔）
        
    Returns:
        清理后的对话历史字符串
    """
    if not dialogue_str:
        return dialogue_str
    
    # 按换行符分割，但需要处理多行的情况
    lines = dialogue_str.strip().split('\n')
    
    # 重新组织为对话轮次（医生/患者）
    dialogue_entries = []
    current_entry = []
    
    for line in lines:
        line_stripped = line.strip()
        # 检查是否是新的对话轮次开始
        if line_stripped.startswith("医生:") or line_stripped.startswith("医生：") or \
           line_stripped.startswith("患者:") or line_stripped.startswith("患者："):
            if current_entry:
                dialogue_entries.append('\n'.join(current_entry))
            current_entry = [line]
        else:
            # 继续当前轮次
            current_entry.append(line)
    
    # 添加最后一个条目
    if current_entry:
        dialogue_entries.append('\n'.join(current_entry))
    
    # 使用列表版本的清理函数
    cleaned_entries = remove_doctor_diagnosis_from_dialogue(dialogue_entries)
    
    return '\n'.join(cleaned_entries)


def extract_patient_info(sample: Dict[str, Any]) -> str:
    """
    从样本中提取患者基本信息（仅年龄、性别、真实诊断）
    
    用于 Doctor Agent 评估，只提供基本信息和真实诊断作为评估标准，
    不泄露详细病史信息，以便评估医生是否能通过问诊收集到正确的信息。
    """
    parts = []
    
    # 只提取年龄、性别和诊断结果
    age = sample.get("Age") or sample.get("age")
    gender = sample.get("Gender") or sample.get("gender")
    
    if age is not None:
        parts.append(f"年龄: {age}")
    if gender:
        parts.append(f"性别: {gender}")
    
    # 真实诊断结果（作为评估标准）
    diag = sample.get("Diagnosis") or sample.get("诊断", "")
    if diag:
        parts.append(f"真实诊断: {diag}")
    
    return " | ".join(str(p).strip() for p in parts if str(p).strip())


def extract_dialogue_from_sample(sample: Dict[str, Any]) -> str:
    """从样本中提取对话历史"""
    # 尝试从不同字段获取对话
    dialogue = sample.get("dialogue_history", "")
    if dialogue:
        return dialogue
    
    cleaned_text = sample.get("cleaned_text", "")
    if cleaned_text:
        return cleaned_text
    
    # 尝试从 conversation 字段获取
    conversation = sample.get("conversation", [])
    if conversation:
        lines = []
        for turn in conversation:
            if isinstance(turn, dict):
                doctor = turn.get("doctor", "")
                patient = turn.get("patient", "")
                if doctor:
                    lines.append(f"医生: {doctor}")
                if patient:
                    lines.append(f"患者: {patient}")
            elif isinstance(turn, str):
                lines.append(turn)
        return "\n".join(lines)
    
    return ""


def format_dialogue_for_eval(dialogue: str) -> str:
    """格式化对话用于评估"""
    if not dialogue:
        return "（无对话记录）"
    return dialogue.strip()


# ============================================================
# 对话生成函数
# ============================================================

def generate_conversation(
    sample: Dict[str, Any],
    doctor_model: str,
    doctor_version: str,
    patient_model: str,
    patient_version: str,
    max_turns: int = 20,
) -> Tuple[str, int]:
    """
    使用 Doctor Agent 和 Patient Agent 生成对话
    
    Args:
        sample: 病例样本
        doctor_model: 医生模型
        doctor_version: 医生版本
        patient_model: 患者模型
        patient_version: 患者版本
        max_turns: 最大对话轮次
    
    Returns:
        Tuple[str, int]: (对话历史, 对话轮次数)
    """
    try:
        # 获取 Doctor 和 Patient 类
        doctor_cls = get_doctor_class(doctor_version)
        patient_cls = get_patient_class(patient_version)
        
        if not doctor_cls or not patient_cls:
            print(f"[ConvGen] 无法加载 Doctor 或 Patient 类")
            return "", 0
        
        # 配置路径
        doctor_prompt_path = str(PROJECT_ROOT / "prompts" / "doctor" / "doctor_persona.json")
        diagtree_path = str(PROJECT_ROOT / "prompts" / "diagtree")
        
        # 初始化 Doctor 和 Patient
        doctor = doctor_cls(
            patient_template=sample,
            doctor_prompt_path=doctor_prompt_path,
            diagtree_path=diagtree_path,
            model_path=doctor_model,
            use_api=True,
        )
        
        patient = patient_cls(
            patient_template=sample,
            model_path=patient_model,
            use_api=True,
        )
        
        # 生成对话
        dialogue_history = []
        patient_response = ""
        
        # 保护参数
        MAX_PATIENT_RESPONSE_LEN = 300  # 患者回复最大字符数
        MAX_DIALOGUE_CHARS = 12000       # 对话历史最大字符数（约4000 tokens）
        MAX_RETRIES_PER_TURN = 2         # 每轮最大重试次数
        
        for turn_idx in range(max_turns):
            try:
                # 检查对话历史长度，如果过长则提前结束
                current_dialogue_len = sum(len(h) for h in dialogue_history)
                if current_dialogue_len > MAX_DIALOGUE_CHARS:
                    print(f"[ConvGen] 对话历史过长 ({current_dialogue_len} chars)，提前结束")
                    break
                
                # 医生回复
                doctor_response, topic, _ = doctor.doctor_response_gen(
                    patient_response,
                    dialogue_history,
                )
                
                if not doctor_response:
                    break
                
                print("doctor_response: ", doctor_response)
                
                # 先去除 reasoning 标记（<think>...</think>）
                doctor_response = strip_reasoning_from_text(doctor_response)
                
                print("doctor_response after strip reasoning: ", doctor_response)
                
                # 检查是否诊断结束
                if "诊断结束" in doctor_response or "诊断结果" in doctor_response:
                    dialogue_history.append(f"医生: {doctor_response}")
                    break
                
                dialogue_history.append(f"医生: {doctor_response}")
                
                # 患者回复（带重试机制）
                dialogue_str = "\n".join(dialogue_history)
                
                print("dialogue_str: ", dialogue_str)
                
                # 截断过长的对话历史以避免 token 超限
                if len(dialogue_str) > MAX_DIALOGUE_CHARS:
                    # 保留最近的对话
                    truncated_history = dialogue_history[-20:]  # 保留最近20轮
                    dialogue_str = "\n".join(truncated_history)
                    print(f"[ConvGen] 截断对话历史到 {len(truncated_history)} 轮")
                
                patient_response_result = None
                for retry in range(MAX_RETRIES_PER_TURN):
                    try:
                        patient_response_result = patient.patient_response_gen(
                            current_topic=topic,
                            dialogue_history=dialogue_str,
                            current_doctor_question=doctor_response,
                        )
                        break
                    except Exception as patient_error:
                        if "maximum context length" in str(patient_error) or "16384 tokens" in str(patient_error):
                            # Token 超限，进一步截断对话历史
                            truncated_history = dialogue_history[-10:]  # 保留最近10轮
                            dialogue_str = "\n".join(truncated_history)
                            print(f"[ConvGen] Token超限，截断到 {len(truncated_history)} 轮，重试 {retry+1}/{MAX_RETRIES_PER_TURN}")
                            if retry == MAX_RETRIES_PER_TURN - 1:
                                print(f"[ConvGen] 重试失败，结束对话生成")
                                break
                        else:
                            raise patient_error
                
                print("patient_response_result: ", patient_response_result)
                
                if patient_response_result is None:
                    # 无法获取患者回复，结束对话
                    break
                
                if isinstance(patient_response_result, tuple):
                    patient_response = patient_response_result[0]
                else:
                    patient_response = patient_response_result
                
                patient_response = strip_reasoning_from_text(patient_response)
                print("patient_response after strip reasoning: ", patient_response)
                
                dialogue_history.append(f"患者: {patient_response}")
                    
            except Exception as turn_error:
                print(f"[ConvGen] 第 {turn_idx+1} 轮生成失败: {turn_error}")
                # 继续尝试下一轮，或者结束
                if "maximum context length" in str(turn_error):
                    print(f"[ConvGen] Token超限，结束对话生成")
                    break
                continue
        
        # 清理对话历史：移除最后一轮医生的诊断内容（如果存在）
        dialogue_history = remove_doctor_diagnosis_from_dialogue(dialogue_history)
        
        return "\n".join(dialogue_history), len(dialogue_history) // 2
        
    except Exception as e:
        print(f"[ConvGen] 生成对话失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果已经生成了部分对话，返回已生成的内容
        if dialogue_history:
            print(f"[ConvGen] 返回已生成的 {len(dialogue_history)} 轮对话")
            # 清理对话历史：移除最后一轮医生的诊断内容（如果存在）
            dialogue_history = remove_doctor_diagnosis_from_dialogue(dialogue_history)
            return "\n".join(dialogue_history), len(dialogue_history) // 2
        return "", 0


# ============================================================
# 评估器类
# ============================================================

class UnifiedDoctorEvaluator:
    """统一的 Doctor Agent 评估器"""
    
    # 评估的5个维度
    EVAL_METRICS = [
        "Clinical_Accuracy_Competence",
        "Ethical_Professional_Conduct",
        "Assessment_Response",
        "Therapeutic_Relationship_Alliance",
        "AI_Communication_Quality",
    ]
    
    # 中文名称映射
    METRIC_NAMES_CN = {
        "Clinical_Accuracy_Competence": "临床准确性与能力",
        "Ethical_Professional_Conduct": "伦理与专业行为",
        "Assessment_Response": "评估与回应",
        "Therapeutic_Relationship_Alliance": "治疗关系与联盟",
        "AI_Communication_Quality": "AI沟通质量",
    }
    
    def __init__(
        self,
        eval_model: str,
        prompts_dir: str = None,
        api_key: str = None,
    ):
        """
        初始化评估器
        
        Args:
            eval_model: 评估模型名称
            prompts_dir: prompts 目录路径
            api_key: API Key（仅用于 OpenRouter）
        """
        self.client = create_llm_client(eval_model, api_key=api_key)
        if prompts_dir is None:
            prompts_dir = PROJECT_ROOT / "prompts" / "evaluation"
        self.prompts_dir = Path(prompts_dir)
        self.eval_model = eval_model
        self._prompt_cache = {}
    
    def _load_prompt(self) -> str:
        """加载评估 prompt"""
        if "doctor_eval" in self._prompt_cache:
            return self._prompt_cache["doctor_eval"]
        
        prompt_file = self.prompts_dir / "doctor_eval_prompt.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._prompt_cache["doctor_eval"] = content
        return content
    
    def _is_model_support_structured_output(self) -> bool:
        """
        判断当前评估模型是否支持 structured output (json_schema)
        
        不支持 structured output 的模型列表:
        - gpt-oss-* 系列 (开源 GPT 模型)
        - 大部分 vLLM 部署的开源模型（除非特别配置）
        """
        model_lower = self.eval_model.lower()
        # 不支持 structured output 的模型模式
        unsupported_patterns = [
            "gpt-oss",
            "llama",
            "mistral",
            "yi-",
            "baichuan",
            "chatglm",
            "internlm",
            "deepseek",
        ]
        for pattern in unsupported_patterns:
            if pattern in model_lower:
                return False
        return True

    def evaluate_dialogue(
        self,
        patient_info: str,
        dialogue_history: str,
    ) -> Tuple[Dict[str, Any], str, Optional[float], Tuple[int, int]]:
        """
        评估一段完整的问诊对话
        
        Args:
            patient_info: 患者信息
            dialogue_history: 问诊对话历史
        
        Returns:
            Tuple: (metrics_dict, average_score, tokens)
        """
        prompt_template = self._load_prompt()
        prompt = prompt_template.format(
            patient_info=patient_info,
            dialogue_history=dialogue_history,
        )
        
        messages = [
            {"role": "system", "content": "你是一位资深的精神科医学教育专家。请严格按照 JSON 格式输出评估结果。"},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # 判断模型是否支持 structured output
            use_structured_output = self._is_model_support_structured_output()
            
            if use_structured_output:
                # 支持 structured output 的模型（如 gemma, qwen3-30b 等）
                response, tokens = self.client.chat_completion(
                    messages,
                    temperature=0.1,
                    max_tokens=4096,
                    response_model=DoctorEvaluation,
                )
            else:
                # 不支持 structured output 的模型（如 gpt-oss-20b）
                # 使用简单的 json_object 格式
                response, tokens = self.client.chat_completion(
                    messages,
                    temperature=0.7,
                    max_tokens=4096,
                    use_json_format=True,  # 使用简单的 json_object 格式
                )
            
            # 解析响应
            if isinstance(response, DoctorEvaluation):
                metrics = {}
                scores = []
                for metric_name in self.EVAL_METRICS:
                    metric_obj = getattr(response.metrics, metric_name, None)
                    if metric_obj:
                        metrics[metric_name] = {
                            "score": metric_obj.score,
                            "reasoning": metric_obj.reasoning,
                        }
                        scores.append(metric_obj.score)
                
                # 自动计算平均分
                average_score = sum(scores) / len(scores) if scores else None
                
                return (
                    metrics,
                    average_score,
                    tokens,
                )
            else:
                # 尝试解析字符串响应
                return self._parse_string_response(response, tokens)
                
        except Exception as e:
            print(f"警告：Doctor 评估失败: {e}")
            return {}, str(e), None, (0, 0)
    
    def _parse_string_response(
        self,
        response: str,
        tokens: Tuple[int, int],
    ) -> Tuple[Dict[str, Any], str, Optional[float], Tuple[int, int]]:
        """解析字符串格式的响应"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not json_match:
                json_match = re.search(r"(\{.*\})", response, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(1))
                metrics = {}
                scores = []
                for metric_name in self.EVAL_METRICS:
                    if "metrics" in data and metric_name in data["metrics"]:
                        metric_data = data["metrics"][metric_name]
                        score = metric_data.get("score")
                        metrics[metric_name] = {
                            "score": score,
                            "reasoning": metric_data.get("reasoning", ""),
                        }
                        if score is not None:
                            scores.append(score)
                
                # 自动计算平均分
                average_score = sum(scores) / len(scores) if scores else None
                
                return (
                    metrics,
                    average_score,
                    tokens,
                )
        except Exception:
            pass
        
        return {},  None, tokens
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        use_original_dialogue: bool = True,
        doctor_model: Optional[str] = None,
        doctor_version: str = "v2",
        patient_model: Optional[str] = None,
        patient_version: str = "v3",
        max_turns: int = 20,
    ) -> DoctorEvaluationResult:
        """
        评估单个样本
        
        Args:
            sample: 样本数据
            use_original_dialogue: 是否使用原始对话
            doctor_model: Doctor Agent 模型
            doctor_version: Doctor Agent 版本
            patient_model: Patient Agent 模型
            patient_version: Patient Agent 版本
            max_turns: 最大对话轮次
        """
        sample_id = str(sample.get("patient_id", sample.get("id", "unknown")))
        
        # 获取患者信息
        patient_info = extract_patient_info(sample)
        
        # 获取对话
        if use_original_dialogue:
            dialogue_history = extract_dialogue_from_sample(sample)
            total_turns = dialogue_history.count("医生:") + dialogue_history.count("医生：")
        else:
            if not doctor_model or not patient_model:
                return DoctorEvaluationResult(
                    sample_id=sample_id,
                    total_turns=0,
                    error="未指定 doctor_model 或 patient_model",
                )
            
            dialogue_history, total_turns = generate_conversation(
                sample=sample,
                doctor_model=doctor_model,
                doctor_version=doctor_version,
                patient_model=patient_model,
                patient_version=patient_version,
                max_turns=max_turns,
            )
        
        if not dialogue_history:
            return DoctorEvaluationResult(
                sample_id=sample_id,
                total_turns=0,
                error="无有效对话",
            )
        
        # 评估对话
        metrics, avg_score, tokens = self.evaluate_dialogue(
            patient_info=patient_info,
            dialogue_history=format_dialogue_for_eval(dialogue_history),
        )
        
        return DoctorEvaluationResult(
            sample_id=sample_id,
            total_turns=total_turns,
            metrics=metrics,
            average_score=avg_score,
            tokens=tokens,
            dialogue_history=dialogue_history,
            patient_info=patient_info,
        )


# ============================================================
# 批量评估函数
# ============================================================

@dataclass
class EvalTask:
    """评估任务数据结构"""
    sample_id: str
    sample_idx: int
    patient_info: str
    dialogue_history: str
    total_turns: int
    sample: Dict[str, Any]
    # 真实标签（用于诊断验证）
    true_labels: Dict[str, Any] = field(default_factory=dict)


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """加载数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if "data" in data:
            return data["data"]
        elif "samples" in data:
            return data["samples"]
        elif "cases" in data:
            return data["cases"]
        elif "conversations" in data:
            return data["conversations"]
    
    return []


def prepare_eval_tasks(
    samples: List[Dict[str, Any]],
    use_original_dialogue: bool = True,
    doctor_model: Optional[str] = None,
    doctor_version: str = "v2",
    patient_model: Optional[str] = None,
    patient_version: str = "v3",
    max_turns: int = 20,
    max_workers: int = 8,
) -> List[EvalTask]:
    """
    准备所有评估任务，包括并行生成对话（如果需要）
    """
    print(f"\n[阶段1] 准备评估任务...")
    
    eval_tasks = []
    
    if use_original_dialogue:
        # 使用原始对话
        for sample_idx, sample in enumerate(samples):
            sample_id = str(sample.get("patient_id", sample.get("id", f"sample_{sample_idx}")))
            patient_info = extract_patient_info(sample)
            dialogue_history = extract_dialogue_from_sample(sample)
            true_labels = get_true_labels_for_sample(sample)
            
            if dialogue_history:
                # 清理对话历史：移除最后一轮医生的诊断内容（如果存在）
                dialogue_history = clean_dialogue_string(dialogue_history)
                
                total_turns = dialogue_history.count("医生:") + dialogue_history.count("医生：")
                eval_tasks.append(EvalTask(
                    sample_id=sample_id,
                    sample_idx=sample_idx,
                    patient_info=patient_info,
                    dialogue_history=dialogue_history,
                    total_turns=total_turns,
                    sample=sample,
                    true_labels=true_labels,
                ))
        
        print(f"[阶段1] 使用原始对话，共 {len(eval_tasks)} 个任务")
        return eval_tasks
    
    # 需要生成对话
    print(f"[阶段1] 需要生成 {len(samples)} 个对话")
    print(f"[阶段1] Doctor: {doctor_model} ({doctor_version})")
    print(f"[阶段1] Patient: {patient_model} ({patient_version})")
    print(f"[阶段1] 最大轮次: {max_turns}")
    print(f"[阶段1] 并行线程数: {max_workers}")
    
    # 预加载类
    get_doctor_class(doctor_version)
    get_patient_class(patient_version)
    
    lock = threading.Lock()
    progress = {"count": 0}
    
    def generate_task(sample_idx: int, sample: Dict[str, Any]) -> Optional[EvalTask]:
        """生成单个对话任务"""
        try:
            sample_id = str(sample.get("patient_id", sample.get("id", f"sample_{sample_idx}")))
            patient_info = extract_patient_info(sample)
            true_labels = get_true_labels_for_sample(sample)
            
            dialogue_history, total_turns = generate_conversation(
                sample=sample,
                doctor_model=doctor_model,
                doctor_version=doctor_version,
                patient_model=patient_model,
                patient_version=patient_version,
                max_turns=max_turns,
            )
            
            with lock:
                progress["count"] += 1
                if progress["count"] % 5 == 0 or progress["count"] == len(samples):
                    print(f"[阶段1] 生成进度: {progress['count']}/{len(samples)}")
            
            if dialogue_history:
                return EvalTask(
                    sample_id=sample_id,
                    sample_idx=sample_idx,
                    patient_info=patient_info,
                    dialogue_history=dialogue_history,
                    total_turns=total_turns,
                    sample=sample,
                    true_labels=true_labels,
                )
            return None
            
        except Exception as e:
            print(f"[阶段1] 生成对话失败: {e}")
            return None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_task, idx, sample) 
            for idx, sample in enumerate(samples)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    eval_tasks.append(result)
            except Exception as e:
                print(f"[阶段1] 任务执行失败: {e}")
    
    print(f"[阶段1] 完成! 成功生成 {len(eval_tasks)} 个对话")
    return eval_tasks


def evaluate_tasks_parallel(
    eval_tasks: List[EvalTask],
    eval_models: List[str],
    max_workers: int = 8,
    api_key: Optional[str] = None,
) -> Dict[str, Dict[str, DoctorEvaluationResult]]:
    """
    并行评估所有任务
    
    Returns:
        模型名称 -> (sample_id -> 评估结果) 的映射
    """
    print(f"\n[阶段2] 开始并行评估...")
    print(f"[阶段2] 评估任务数: {len(eval_tasks)}")
    print(f"[阶段2] 评估模型: {', '.join(eval_models)}")
    print(f"[阶段2] 每模型并行线程数: {max_workers}")
    
    # 初始化所有评估器
    evaluators = {}
    for model_name in eval_models:
        evaluators[model_name] = UnifiedDoctorEvaluator(
            eval_model=model_name,
            api_key=api_key,
        )
    
    # 结果存储
    all_model_results: Dict[str, Dict[str, DoctorEvaluationResult]] = {
        model: {} for model in eval_models
    }
    
    def evaluate_single_model(model_name: str) -> Dict[str, DoctorEvaluationResult]:
        """评估单个模型"""
        evaluator = evaluators[model_name]
        model_results: Dict[str, DoctorEvaluationResult] = {}
        
        lock = threading.Lock()
        progress = {"count": 0}
        
        def evaluate_single_task(task: EvalTask) -> Tuple[str, DoctorEvaluationResult]:
            """评估单个任务"""
            metrics, avg_score, tokens = evaluator.evaluate_dialogue(
                patient_info=task.patient_info,
                dialogue_history=format_dialogue_for_eval(task.dialogue_history),
            )
            
            result = DoctorEvaluationResult(
                sample_id=task.sample_id,
                total_turns=task.total_turns,
                metrics=metrics,
                average_score=avg_score,
                tokens=tokens,
                dialogue_history=task.dialogue_history,
                patient_info=task.patient_info,
            )
            
            with lock:
                progress["count"] += 1
                if progress["count"] % 10 == 0 or progress["count"] == len(eval_tasks):
                    print(f"  [{model_name}] 评估进度: {progress['count']}/{len(eval_tasks)}")
            
            return task.sample_id, result
        
        print(f"\n  [{model_name}] 开始评估 {len(eval_tasks)} 个任务...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single_task, task) for task in eval_tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    sample_id, result = future.result()
                    with lock:
                        model_results[sample_id] = result
                except Exception as e:
                    print(f"  [{model_name}] 评估失败: {e}")
        
        print(f"  [{model_name}] 完成! 评估 {len(model_results)} 个样本")
        return model_results
    
    # 并行评估所有模型
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(eval_models)) as executor:
        model_futures = {
            executor.submit(evaluate_single_model, model): model 
            for model in eval_models
        }
        
        for future in concurrent.futures.as_completed(model_futures):
            model_name = model_futures[future]
            try:
                all_model_results[model_name] = future.result()
            except Exception as e:
                print(f"[阶段2] 模型 {model_name} 评估失败: {e}")
    
    print(f"\n[阶段2] 所有模型评估完成!")
    return all_model_results


def run_diagnosis_verification(
    eval_tasks: List[EvalTask],
    verifier_model: str,
    max_workers: int = 16,
    api_key: Optional[str] = None,
    output_dir: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    运行诊断验证（Dynamic评估）
    
    Args:
        eval_tasks: 评估任务列表
        verifier_model: 验证模型
        max_workers: 最大并行工作线程数
        api_key: API Key
        output_dir: 输出目录（用于保存诊断详情文件）
        timestamp: 时间戳（用于文件名）
        
    Returns:
        分类类型 -> 指标结果的映射
    """
    print(f"\n[阶段3] 开始诊断验证（Dynamic评估）...")
    print(f"[阶段3] 验证模型: {verifier_model}")
    print(f"[阶段3] 任务数: {len(eval_tasks)}")
    
    verifier = DiagnosisVerifier(
        verifier_model=verifier_model,
        max_workers=max_workers,
        api_key=api_key,
    )
    
    # 准备对话和标签
    dialogues = [task.dialogue_history for task in eval_tasks]
    
    verification_results = {}
    
    for classification_type in DiagnosisVerifier.CLASSIFICATION_TYPES:
        print(f"\n[阶段3] 正在进行 {classification_type} 分类验证...")
        
        # 获取该分类任务的真实标签
        true_labels = []
        valid_indices = []
        
        for idx, task in enumerate(eval_tasks):
            label = task.true_labels.get(classification_type)
            if label is not None:  # 2分类可能有None值
                true_labels.append(label)
                valid_indices.append(idx)
        
        if not true_labels:
            print(f"  [警告] {classification_type} 没有有效的真实标签，跳过")
            continue
        
        # 只验证有真实标签的对话
        valid_dialogues = [dialogues[i] for i in valid_indices]
        
        # 批量验证
        batch_results = verifier.verify_batch(
            dialogues=valid_dialogues,
            classification_type=classification_type,
            show_progress=True,
        )
        
        # 提取预测结果
        predictions = [r['prediction'] for r in batch_results]
        
        # 计算指标
        metrics = verifier.calculate_metrics(
            predictions=predictions,
            true_labels=true_labels,
            classification_type=classification_type,
        )
        
        verification_results[classification_type] = {
            'metrics': metrics,
            'predictions': predictions,
            'true_labels': true_labels,
            'valid_sample_count': len(true_labels),
            'raw_results': batch_results,
        }
        
        # 打印结果
        print(f"\n[阶段3] {classification_type} 分类验证结果:")
        print(f"  样本数: {len(true_labels)}")
        print(f"  准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        if 'macro_f1' in metrics:
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        if 'weighted_f1' in metrics:
            print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        if 'top1_accuracy' in metrics:
            print(f"  Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        if 'top3_accuracy' in metrics:
            print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        
        # 保存诊断详情到中间文件（用于 debug）
        if output_dir:
            _save_diagnosis_details(
                output_dir=output_dir,
                classification_type=classification_type,
                eval_tasks=[eval_tasks[i] for i in valid_indices],
                batch_results=batch_results,
                true_labels=true_labels,
                predictions=predictions,
                verifier_model=verifier_model,
                timestamp=timestamp,
            )
    
    print(f"\n[阶段3] 诊断验证完成!")
    return verification_results


def _save_diagnosis_details(
    output_dir: Path,
    classification_type: str,
    eval_tasks: List[EvalTask],
    batch_results: List[Dict[str, Any]],
    true_labels: List[Any],
    predictions: List[Any],
    verifier_model: str,
    timestamp: Optional[str] = None,
) -> None:
    """
    保存每个样本的诊断详情到中间文件（用于 debug）
    
    保存内容包括：
    - sample_id: 样本ID
    - ground_truth: 真实标签
    - prediction: 模型预测结果
    - is_correct: 是否正确
    - content: 模型输出的 content 内容（包含 <box> 标签）
    - reasoning: 模型输出的 reasoning 内容（<think> 标签内容）
    - prompt: 发送给模型的完整 prompt
    - dialogue_history: 完整的对话历史
    
    Args:
        output_dir: 输出目录
        classification_type: 分类类型
        eval_tasks: 评估任务列表
        batch_results: 批量验证结果
        true_labels: 真实标签列表
        predictions: 预测结果列表
        verifier_model: 验证模型名称
        timestamp: 时间戳
    """
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    verifier_safe_name = sanitize_model_name(verifier_model)
    output_file = output_dir / f"diagnosis_details_{classification_type}_{verifier_safe_name}_{timestamp}.json"
    
    details = []
    for idx, (task, result, true_label, pred) in enumerate(zip(eval_tasks, batch_results, true_labels, predictions)):
        # 判断是否预测正确
        if classification_type == "12class":
            # 12分类是多标签，检查是否有交集
            pred_set = set(pred) if isinstance(pred, list) else {pred}
            true_set = set(true_label) if isinstance(true_label, list) else {true_label}
            is_correct = bool(pred_set & true_set)
        else:
            is_correct = pred == true_label
        
        detail = {
            "idx": idx,
            "sample_id": task.sample_id,
            "ground_truth": true_label,
            "prediction": pred,
            "is_correct": is_correct,
            "content": result.get("content", ""),  # 模型输出的 content（包含诊断结果）
            "reasoning": result.get("reasoning", ""),  # 模型输出的 reasoning
            "prompt": result.get("prompt", ""),  # 发送给模型的完整 prompt
            "dialogue_history": task.dialogue_history,  # 完整的对话历史
        }
        
        # 如果有错误信息也保存
        if result.get("error"):
            detail["error"] = result.get("error")
        
        details.append(detail)
    
    # 统计信息
    correct_count = sum(1 for d in details if d["is_correct"])
    summary = {
        "classification_type": classification_type,
        "verifier_model": verifier_model,
        "total_samples": len(details),
        "correct_count": correct_count,
        "accuracy": correct_count / len(details) if details else 0,
        "timestamp": timestamp,
    }
    
    output_data = {
        "summary": summary,
        "details": details,
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  [Debug] 诊断详情已保存到: {output_file}")
    except Exception as e:
        print(f"  [警告] 保存诊断详情失败: {e}")


def compute_statistics(
    all_results: Dict[str, DoctorEvaluationResult],
) -> Dict[str, Any]:
    """计算统计信息"""
    metric_scores = {
        "Clinical_Accuracy_Competence": [],
        "Ethical_Professional_Conduct": [],
        "Assessment_Response": [],
        "Therapeutic_Relationship_Alliance": [],
        "AI_Communication_Quality": [],
    }
    
    total_tokens = [0, 0]
    average_scores = []
    
    for sample_id, result in all_results.items():
        total_tokens[0] += result.tokens[0]
        total_tokens[1] += result.tokens[1]
        
        if result.average_score is not None:
            average_scores.append(result.average_score)
        
        for metric_name in metric_scores:
            if metric_name in result.metrics:
                score = result.metrics[metric_name].get("score")
                if score is not None:
                    metric_scores[metric_name].append(score)
    
    # 计算统计
    stats = {
        "total_samples": len(all_results),
        "metrics": {},
        "overall_average": {
            "average": calc_average(average_scores),
            "std": calc_std(average_scores),
        },
        "total_prompt_tokens": total_tokens[0],
        "total_completion_tokens": total_tokens[1],
    }
    
    for metric_name, scores in metric_scores.items():
        stats["metrics"][metric_name] = {
            "count": len(scores),
            "average": calc_average(scores),
            "std": calc_std(scores),
        }
    
    return stats


def print_summary(stats: Dict[str, Any], model_name: str, data_file: str):
    """打印评估摘要"""
    print("\n" + "=" * 80)
    print("Doctor Agent 评估结果摘要")
    print("=" * 80)
    print(f"数据文件: {data_file}")
    print(f"评估模型: {model_name}")
    print(f"样本总数: {stats['total_samples']}")
    print()
    
    print("【各维度评估结果】(1-6分)")
    print("-" * 60)
    
    metric_names_cn = UnifiedDoctorEvaluator.METRIC_NAMES_CN
    
    for metric, display_name in metric_names_cn.items():
        if metric in stats["metrics"]:
            data = stats["metrics"][metric]
            print(f"  {display_name}: {data['average']:.2f} ± {data['std']:.2f} (样本数: {data['count']})")
    
    print()
    print("【整体评分】")
    print("-" * 60)
    overall = stats["overall_average"]
    print(f"  平均分: {overall['average']:.2f} ± {overall['std']:.2f}")
    
    print()
    print("【Token 使用量】")
    print(f"  Prompt tokens: {stats['total_prompt_tokens']}")
    print(f"  Completion tokens: {stats['total_completion_tokens']}")
    print("=" * 80)


def result_to_dict(result: DoctorEvaluationResult) -> Dict[str, Any]:
    """将评估结果转换为字典"""
    return {
        "sample_id": result.sample_id,
        "total_turns": result.total_turns,
        "metrics": result.metrics,
        "average_score": result.average_score,
        "tokens": result.tokens,
        "dialogue_history": result.dialogue_history,
        "patient_info": result.patient_info,
        "error": result.error,
    }


def save_generated_dialogues(
    output_dir: Path,
    eval_tasks: List[EvalTask],
    data_file: Path,
    timestamp: str,
    doctor_model: str,
    doctor_version: str,
    patient_model: str,
    patient_version: str,
    max_turns: int,
) -> Path:
    """
    保存生成的对话到文件
    
    Args:
        output_dir: 输出目录
        eval_tasks: 评估任务列表（包含生成的对话）
        data_file: 原始数据文件路径
        timestamp: 时间戳
        doctor_model: Doctor Agent 模型
        doctor_version: Doctor Agent 版本
        patient_model: Patient Agent 模型
        patient_version: Patient Agent 版本
        max_turns: 最大对话轮次
    
    Returns:
        保存的文件路径
    """
    doctor_safe_name = sanitize_model_name(doctor_model)
    patient_safe_name = sanitize_model_name(patient_model)
    
    output_file = output_dir / f"dialogues_{doctor_version}_{doctor_safe_name}_{data_file.stem}_{timestamp}.json"
    
    dialogues_data = {
        "metadata": {
            "data_file": str(data_file),
            "doctor_model": doctor_model,
            "doctor_version": doctor_version,
            "patient_model": patient_model,
            "patient_version": patient_version,
            "max_turns": max_turns,
            "timestamp": timestamp,
            "total_dialogues": len(eval_tasks),
        },
        "dialogues": [
            {
                "sample_id": task.sample_id,
                "sample_idx": task.sample_idx,
                "patient_info": task.patient_info,
                "dialogue_history": task.dialogue_history,
                "total_turns": task.total_turns,
                "sample": task.sample,
            }
            for task in eval_tasks
        ],
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dialogues_data, f, ensure_ascii=False, indent=2)
    
    print(f"[对话生成] 对话已保存到: {output_file}")
    return output_file


def load_generated_dialogues(dialogue_file: Path) -> Tuple[List[EvalTask], Dict[str, Any]]:
    """
    从文件加载生成的对话
    
    支持两种格式的文件：
    1. dialogues 格式（--generate-only 生成的文件）
    2. results 格式（完整评估结果文件）
    
    Args:
        dialogue_file: 对话文件路径
    
    Returns:
        Tuple: (评估任务列表, 元数据)
    """
    print(f"[对话加载] 从文件加载对话: {dialogue_file}")
    
    with open(dialogue_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    
    # 支持两种格式：dialogues（generate-only 生成）和 results（完整评估结果）
    dialogues = data.get("dialogues", [])
    if not dialogues:
        dialogues = data.get("results", [])
        if dialogues:
            print(f"[对话加载] 检测到 'results' 格式的评估结果文件")
    
    # 尝试加载原始数据文件以获取 true_labels
    original_samples = {}
    original_data_file = metadata.get("data_file")
    if original_data_file and Path(original_data_file).exists():
        try:
            print(f"[对话加载] 尝试加载原始数据文件: {original_data_file}")
            original_data = load_data(original_data_file)
            for sample in original_data:
                sample_id = str(sample.get("patient_id", sample.get("id", "")))
                if sample_id:
                    original_samples[sample_id] = sample
            print(f"[对话加载] 成功加载 {len(original_samples)} 个原始样本")
        except Exception as e:
            print(f"[对话加载] 加载原始数据文件失败: {e}")
    
    eval_tasks = []
    for idx, d in enumerate(dialogues):
        dialogue_history = d.get("dialogue_history", "")
        
        # 清理对话历史：移除最后一轮医生的诊断内容（如果存在）
        if dialogue_history:
            dialogue_history = clean_dialogue_string(dialogue_history)
        
        # 获取 sample 字段（用于提取真实标签）
        sample = d.get("sample", {})
        sample_id = d.get("sample_id", f"sample_{idx}")
        
        # 如果 sample 为空，尝试从原始数据中获取
        if not sample and sample_id in original_samples:
            sample = original_samples[sample_id]
        
        # 提取真实标签
        true_labels = {}
        if sample:
            try:
                true_labels = get_true_labels_for_sample(sample)
            except Exception:
                pass
        
        task = EvalTask(
            sample_id=sample_id,
            sample_idx=d.get("sample_idx", idx),
            patient_info=d.get("patient_info", ""),
            dialogue_history=dialogue_history,
            total_turns=d.get("total_turns", 0),
            sample=sample,
            true_labels=true_labels,
        )
        eval_tasks.append(task)
    
    print(f"[对话加载] 成功加载 {len(eval_tasks)} 个对话")
    print(f"[对话加载] 元数据: Doctor={metadata.get('doctor_model')} ({metadata.get('doctor_version')})")
    
    return eval_tasks, metadata


def save_results(
    output_dir: Path,
    model_name: str,
    all_results: Dict[str, DoctorEvaluationResult],
    stats: Dict[str, Any],
    data_file: Path,
    timestamp: str,
    use_original_dialogue: bool = True,
    doctor_model: Optional[str] = None,
    doctor_version: str = "v2",
    patient_model: Optional[str] = None,
    patient_version: str = "v3",
):
    """保存评估结果"""
    model_safe_name = sanitize_model_name(model_name)
    
    # 根据模式生成不同的文件名前缀
    if use_original_dialogue:
        prefix = "doctor_eval_original"
    else:
        doctor_safe_name = sanitize_model_name(doctor_model or "unknown")
        prefix = f"doctor_eval_{doctor_version}_{doctor_safe_name}"
    
    # 保存详细结果
    output_file = output_dir / f"{prefix}_{data_file.stem}_{model_safe_name}_{timestamp}.json"
    output_data = {
        "metadata": {
            "data_file": str(data_file),
            "eval_model": model_name,
            "timestamp": timestamp,
            "use_original_dialogue": use_original_dialogue,
            "doctor_model": doctor_model,
            "doctor_version": doctor_version,
            "patient_model": patient_model,
            "patient_version": patient_version,
        },
        "statistics": stats,
        "results": [result_to_dict(r) for r in all_results.values()],
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"[{model_name}] 评估结果已保存到: {output_file}")
    
    return output_file


# ============================================================
# 聚合结果函数
# ============================================================

def aggregate_doctor_results(
    model_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """聚合多个模型的 Doctor 评估结果"""
    if not model_results:
        return {}
    
    # 收集所有模型的统计信息
    model_stats = {}
    for model_name, results in model_results.items():
        if "statistics" in results:
            model_stats[model_name] = results["statistics"]
    
    # 聚合各维度
    aggregated = {
        "models": list(model_results.keys()),
        "aggregated_stats": {
            "metrics": {},
            "overall_average": {},
        },
        "per_model_stats": model_stats,
    }
    
    metrics = [
        "Clinical_Accuracy_Competence",
        "Ethical_Professional_Conduct",
        "Assessment_Response",
        "Therapeutic_Relationship_Alliance",
        "AI_Communication_Quality",
    ]
    
    for metric_name in metrics:
        values = []
        per_model = {}
        for model_name, stats in model_stats.items():
            if "metrics" in stats and metric_name in stats["metrics"]:
                avg = stats["metrics"][metric_name].get("average")
                if avg is not None:
                    values.append(avg)
                    per_model[model_name] = avg
        
        aggregated["aggregated_stats"]["metrics"][metric_name] = {
            "average": calc_average(values),
            "std": calc_std(values),
            "per_model": per_model,
        }
    
    # 聚合整体评分
    overall_values = []
    for stats in model_stats.values():
        if "overall_average" in stats:
            avg = stats["overall_average"].get("average")
            if avg is not None:
                overall_values.append(avg)
    
    aggregated["aggregated_stats"]["overall_average"] = {
        "average": calc_average(overall_values),
        "std": calc_std(overall_values),
    }
    
    return aggregated


def print_aggregated_summary(aggregated: Dict[str, Any]):
    """打印聚合结果摘要"""
    print("\n" + "=" * 80)
    print("多模型融合 Doctor Agent 评估结果摘要")
    print("=" * 80)
    print(f"评估模型: {', '.join(aggregated.get('models', []))}")
    print()
    
    agg_stats = aggregated.get("aggregated_stats", {})
    
    print("【各维度评估结果】(多模型平均, 1-6分)")
    print("-" * 60)
    
    metric_names_cn = UnifiedDoctorEvaluator.METRIC_NAMES_CN
    
    for metric, display_name in metric_names_cn.items():
        if "metrics" in agg_stats and metric in agg_stats["metrics"]:
            data = agg_stats["metrics"][metric]
            print(f"  {display_name}: {data['average']:.2f} ± {data['std']:.2f}")
            for model, score in data.get("per_model", {}).items():
                print(f"    - {model}: {score:.2f}")
    
    print()
    print("【整体评分】")
    print("-" * 60)
    overall = agg_stats.get("overall_average", {})
    print(f"  平均分: {overall.get('average', 0):.2f} ± {overall.get('std', 0):.2f}")
    print("=" * 80)


def save_aggregated_results(aggregated: Dict[str, Any], output_file: Path):
    """保存聚合结果（JSON格式）"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"聚合结果已保存到: {output_file}")


def save_aggregated_summary_txt(
    aggregated: Dict[str, Any],
    output_file: Path,
    data_file: str = "",
    use_original_dialogue: bool = True,
    doctor_model: Optional[str] = None,
    doctor_version: str = "v2",
):
    """
    保存聚合结果的文本摘要
    
    Args:
        aggregated: 聚合后的评估结果
        output_file: 输出文件路径
        data_file: 数据文件路径
        use_original_dialogue: 是否使用原始对话
        doctor_model: Doctor Agent 模型
        doctor_version: Doctor Agent 版本
    """
    lines = []
    
    # 标题
    lines.append("=" * 70)
    lines.append("多模型融合 (Average Vote) Doctor Agent 评估结果")
    lines.append("=" * 70)
    
    # 模型信息
    models = aggregated.get('models', [])
    lines.append(f"评估模型: {', '.join(models)}")
    
    # 数据来源信息
    if data_file:
        lines.append(f"数据文件: {data_file}")
    
    if use_original_dialogue:
        lines.append("对话来源: 原始数据集对话")
    else:
        lines.append(f"对话来源: Doctor Agent 生成")
        lines.append(f"  - Doctor 模型: {doctor_model or 'Unknown'}")
        lines.append(f"  - Doctor 版本: {doctor_version}")
    
    # 计算评估样本数
    total_samples = 0
    for stats in aggregated.get("per_model_stats", {}).values():
        total_samples = max(total_samples, stats.get("total_samples", 0))
    lines.append(f"评估样本数: {total_samples}")
    lines.append("")
    
    agg_stats = aggregated.get("aggregated_stats", {})
    
    # 各维度评估结果
    lines.append("【Doctor Agent 问诊能力评估】(多模型平均, 1-6分)")
    lines.append("-" * 50)
    
    metric_names_cn = {
        "Clinical_Accuracy_Competence": "临床准确性与能力",
        "Ethical_Professional_Conduct": "伦理与专业行为",
        "Assessment_Response": "评估与回应",
        "Therapeutic_Relationship_Alliance": "治疗关系与联盟",
        "AI_Communication_Quality": "AI沟通质量",
    }
    
    metrics_data = agg_stats.get("metrics", {})
    for metric, display_name in metric_names_cn.items():
        if metric in metrics_data:
            data = metrics_data[metric]
            # 获取样本数量
            count = 0
            for stats in aggregated.get("per_model_stats", {}).values():
                if "metrics" in stats and metric in stats["metrics"]:
                    count = stats["metrics"][metric].get("count", 0)
                    break
            lines.append(f"  {display_name}: {data['average']:.2f} ± {data['std']:.2f} (n={count})")
    
    lines.append("")
    
    # 整体评分
    lines.append("【整体评分】(多模型平均)")
    lines.append("-" * 50)
    overall = agg_stats.get("overall_average", {})
    lines.append(f"  平均分: {overall.get('average', 0):.2f} ± {overall.get('std', 0):.2f} / 6")
    
    lines.append("=" * 70)
    lines.append("")
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"聚合摘要已保存到: {output_file}")


# ============================================================
# Excel 结果保存功能
# ============================================================

def save_results_to_excel(
    output_path: Path,
    doctor_version: str,
    model_name: str,
    llm_judge_stats: Dict[str, Any],
    verification_results: Optional[Dict[str, Dict[str, Any]]] = None,
    append_mode: bool = True,
    verifier_model: Optional[str] = None,
) -> None:
    """
    保存评估结果到Excel文件
    
    表格结构:
    | Doctor Agent | Model | LLM-as-Judge (5维度) | Dynamic (2class/4class/12class) | Verifier Model |
    
    Args:
        output_path: 输出Excel文件路径
        doctor_version: Doctor版本
        model_name: Doctor模型名称
        llm_judge_stats: LLM-as-Judge评估统计
        verification_results: 诊断验证结果
        append_mode: 是否追加模式（True则追加行，False则覆盖）
        verifier_model: 诊断验证使用的模型名称
    """
    try:
        import pandas as pd
    except ImportError:
        print("[警告] pandas未安装，无法保存Excel文件。请运行: pip install pandas")
        return
    
    try:
        import openpyxl
    except ImportError:
        print("[警告] openpyxl未安装，无法保存Excel文件。请运行: pip install openpyxl")
        return
    
    print(f"[Excel] 准备保存结果到: {output_path}")
    
    # 构建行数据
    row_data = {
        'Doctor Agent': f'Doctor {doctor_version.upper()}',
        'Model': model_name,
    }
    
    # LLM-as-Judge 5个维度
    llm_metrics = llm_judge_stats.get('metrics', {})
    for metric_name in ['Clinical_Accuracy_Competence', 'Ethical_Professional_Conduct',
                        'Assessment_Response', 'Therapeutic_Relationship_Alliance',
                        'AI_Communication_Quality']:
        if metric_name in llm_metrics:
            row_data[metric_name] = llm_metrics[metric_name].get('average', 0)
        else:
            row_data[metric_name] = 0
    
    # Dynamic评估 - 2class
    if verification_results and '2class' in verification_results:
        metrics_2class = verification_results['2class'].get('metrics', {})
        row_data['2class_Acc'] = metrics_2class.get('accuracy', 0)
        row_data['2class_F1_macro'] = metrics_2class.get('macro_f1', 0)
        row_data['2class_F1_weighted'] = metrics_2class.get('weighted_f1', 0)
    else:
        row_data['2class_Acc'] = None
        row_data['2class_F1_macro'] = None
        row_data['2class_F1_weighted'] = None
    
    # Dynamic评估 - 4class
    if verification_results and '4class' in verification_results:
        metrics_4class = verification_results['4class'].get('metrics', {})
        row_data['4class_Acc'] = metrics_4class.get('accuracy', 0)
        row_data['4class_F1_macro'] = metrics_4class.get('macro_f1', 0)
        row_data['4class_F1_weighted'] = metrics_4class.get('weighted_f1', 0)
    else:
        row_data['4class_Acc'] = None
        row_data['4class_F1_macro'] = None
        row_data['4class_F1_weighted'] = None
    
    # Dynamic评估 - 12class
    if verification_results and '12class' in verification_results:
        metrics_12class = verification_results['12class'].get('metrics', {})
        row_data['12class_Acc'] = metrics_12class.get('accuracy', 0)
        row_data['12class_Top1_Acc'] = metrics_12class.get('top1_accuracy', 0)
        row_data['12class_Top3_Acc'] = metrics_12class.get('top3_accuracy', 0)
        row_data['12class_F1_macro'] = metrics_12class.get('macro_f1', 0)
        row_data['12class_F1_weighted'] = metrics_12class.get('weighted_f1', 0)
    else:
        row_data['12class_Acc'] = None
        row_data['12class_Top1_Acc'] = None
        row_data['12class_Top3_Acc'] = None
        row_data['12class_F1_macro'] = None
        row_data['12class_F1_weighted'] = None
    
    # 添加 Verifier Model 列
    row_data['Verifier_Model'] = verifier_model if verifier_model else ""
    
    # 转换为DataFrame
    new_row = pd.DataFrame([row_data])
    
    # 判断是追加还是覆盖
    if append_mode and output_path.exists():
        try:
            existing_df = pd.read_excel(output_path)
            df = pd.concat([existing_df, new_row], ignore_index=True)
        except Exception as e:
            print(f"[警告] 读取现有Excel文件失败: {e}，将创建新文件")
            df = new_row
    else:
        df = new_row
    
    # 确保输出目录存在
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 将数值列保留3位小数
    numeric_columns = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
    for col in numeric_columns:
        df[col] = df[col].apply(lambda x: round(x, 3) if pd.notna(x) else x)
    
    # 保存Excel
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"[Excel] 结果已保存到: {output_path}")
    except Exception as e:
        print(f"[警告] 保存Excel失败: {e}")
        import traceback
        traceback.print_exc()


def save_results_to_json_summary(
    output_path: Path,
    doctor_version: str,
    model_name: str,
    llm_judge_stats: Dict[str, Any],
    verification_results: Optional[Dict[str, Dict[str, Any]]] = None,
    append_mode: bool = True,
) -> None:
    """
    保存评估结果汇总到JSON文件
    
    Args:
        output_path: 输出JSON文件路径
        doctor_version: Doctor版本
        model_name: Doctor模型名称
        llm_judge_stats: LLM-as-Judge评估统计
        verification_results: 诊断验证结果
        append_mode: 是否追加模式
    """
    from datetime import datetime
    
    # 构建结果数据
    result_entry = {
        'timestamp': datetime.now().isoformat(),
        'doctor_version': doctor_version,
        'model_name': model_name,
        'llm_judge': {
            'metrics': {},
            'overall_average': llm_judge_stats.get('overall_average', {})
        },
        'dynamic_verification': {},
    }
    
    # 添加LLM-as-Judge各维度
    for metric_name in ['Clinical_Accuracy_Competence', 'Ethical_Professional_Conduct',
                        'Assessment_Response', 'Therapeutic_Relationship_Alliance',
                        'AI_Communication_Quality']:
        if metric_name in llm_judge_stats.get('metrics', {}):
            result_entry['llm_judge']['metrics'][metric_name] = llm_judge_stats['metrics'][metric_name]
    
    # 添加Dynamic验证结果
    if verification_results:
        for cls_type in ['2class', '4class', '12class']:
            if cls_type in verification_results:
                metrics = verification_results[cls_type].get('metrics', {})
                # 移除class_metrics以减小文件大小
                metrics_clean = {k: v for k, v in metrics.items() if k != 'class_metrics'}
                result_entry['dynamic_verification'][cls_type] = metrics_clean
    
    # 读取或创建结果列表
    if append_mode and output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results_list = json.load(f)
            if not isinstance(results_list, list):
                results_list = [results_list]
        except Exception as e:
            print(f"[警告] 读取现有JSON文件失败: {e}，将创建新文件")
            results_list = []
    else:
        results_list = []
    
    results_list.append(result_entry)
    
    # 保存JSON
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        print(f"[JSON] 汇总结果已保存到: {output_path}")
    except Exception as e:
        print(f"[警告] 保存JSON失败: {e}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="统一的 Doctor Agent 评估程序（LLM-as-Judge）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 模式1: 使用原始数据集中的对话（评估真实对话数据）
  python unified_doctor_eval.py \\
      --data-file /path/to/conversations.json \\
      --use-original-dialogue \\
      --eval-models "gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041"

  # 模式2: 使用 Doctor Agent 生成对话（评估 Doctor Agent）
  python unified_doctor_eval.py \\
      --data-file /path/to/cases.json \\
      --doctor-model "qwen3-30b@10.119.28.185:9041" \\
      --doctor-version v2 \\
      --patient-model "qwen3-30b@10.119.28.185:9041" \\
      --patient-version v3 \\
      --eval-models "gemma-3-27b-it@10.119.28.185:9051"

  # 模式3: 只生成对话，不进行评估
  python unified_doctor_eval.py \\
      --data-file /path/to/cases.json \\
      --doctor-model "qwen3-30b@10.119.28.185:9041" \\
      --patient-model "qwen3-30b@10.119.28.185:9041" \\
      --generate-only

  # 模式4: 只评估，从文件加载对话
  python unified_doctor_eval.py \\
      --eval-only \\
      --dialogue-file /path/to/dialogues_v2_xxx.json \\
      --eval-models "gemma-3-27b-it@10.119.28.185:9051"
""",
    )
    
    parser.add_argument(
        "--data-file",
        type=str,
        required=False,
        default=None,
        help="数据文件路径（JSON格式）。使用 --eval-only + --dialogue-file 时可不指定",
    )
    
    parser.add_argument(
        "--eval-models",
        type=str,
        default="gemma-3-27b-it@10.119.28.185:9051,qwen3-30b@10.119.28.185:9041",
        help="评估模型列表，逗号分隔",
    )
    
    parser.add_argument(
        "--doctor-model",
        type=str,
        default=None,
        help="Doctor Agent 使用的模型（当不使用 --use-original-dialogue 时必须指定）",
    )
    
    parser.add_argument(
        "--doctor-version",
        type=str,
        default="v2",
        help="Doctor Agent 版本（如 v1, v2, v3, base 等，默认: v2）。支持动态加载任意版本，只需确保 src/doctor/doctor_{version}.py 存在",
    )
    
    parser.add_argument(
        "--patient-model",
        type=str,
        default=None,
        help="Patient Agent 使用的模型（当不使用 --use-original-dialogue 时必须指定）",
    )
    
    parser.add_argument(
        "--patient-version",
        type=str,
        choices=PATIENT_VERSION_CHOICES,
        default="v3",
        help="Patient Agent 版本（默认: v3，可选: v1, cot, mdd5k, v3）",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API Key（仅用于 OpenRouter 模型）",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="输出目录（默认: evaluation_results）",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评估的样本数量",
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="生成对话的最大轮次（默认: 20）",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行线程数（默认: 8）",
    )
    
    parser.add_argument(
        "--use-original-dialogue",
        action="store_true",
        help="使用数据集中原始的对话记录（不指定则使用 Doctor Agent 生成）",
    )
    
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="只生成对话，不进行评估。生成的对话将保存到文件，可后续使用 --eval-only 进行评估",
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="只进行评估，跳过对话生成。需要配合 --dialogue-file 指定之前保存的对话文件",
    )
    
    parser.add_argument(
        "--dialogue-file",
        type=str,
        default=None,
        help="对话文件路径（与 --eval-only 配合使用，指定之前 --generate-only 保存的对话文件）",
    )
    
    parser.add_argument(
        "--verifier-model",
        type=str,
        default=None,
        help="诊断验证模型（用于Dynamic评估），默认使用 doctor-model",
    )
    
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="跳过诊断验证（Dynamic评估），只进行LLM-as-Judge评估",
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只进行诊断验证，跳过LLM-as-Judge评估。需要配合 --dialogue-file 和 --verifier-model 使用",
    )
    
    parser.add_argument(
        "--output-excel",
        type=str,
        default=None,
        help="输出Excel文件路径（用于汇总多次评估结果）",
    )
    
    args = parser.parse_args()
    
    # 检查参数一致性
    use_original_dialogue = args.use_original_dialogue
    doctor_model = args.doctor_model
    doctor_version = args.doctor_version
    patient_model = args.patient_model
    patient_version = args.patient_version
    generate_only = args.generate_only
    eval_only = args.eval_only
    verify_only = args.verify_only
    dialogue_file = args.dialogue_file
    verifier_model = args.verifier_model
    skip_verification = args.skip_verification
    output_excel = args.output_excel
    
    # 检查互斥参数
    if generate_only and eval_only:
        print("错误: --generate-only 和 --eval-only 不能同时使用")
        sys.exit(1)
    
    if verify_only and generate_only:
        print("错误: --verify-only 和 --generate-only 不能同时使用")
        sys.exit(1)
    
    if verify_only and not dialogue_file:
        print("错误: 使用 --verify-only 时，必须通过 --dialogue-file 指定对话文件")
        sys.exit(1)
    
    if verify_only and not verifier_model:
        print("错误: 使用 --verify-only 时，必须通过 --verifier-model 指定验证模型")
        sys.exit(1)
    
    if verify_only and skip_verification:
        print("错误: --verify-only 和 --skip-verification 不能同时使用")
        sys.exit(1)
    
    if generate_only and use_original_dialogue:
        print("错误: --generate-only 和 --use-original-dialogue 不能同时使用（原始对话无需生成）")
        sys.exit(1)
    
    if eval_only and not dialogue_file and not use_original_dialogue:
        print("错误: 使用 --eval-only 时，必须通过 --dialogue-file 指定对话文件，或使用 --use-original-dialogue")
        sys.exit(1)
    
    if dialogue_file and not eval_only and not verify_only:
        print("警告: --dialogue-file 只在 --eval-only 或 --verify-only 模式下生效")
    
    if not use_original_dialogue and not eval_only and not verify_only:
        if not doctor_model:
            print("错误: 当不使用 --use-original-dialogue 或 --eval-only 或 --verify-only 时，必须通过 --doctor-model 指定 Doctor Agent 模型")
            sys.exit(1)
        if not patient_model:
            print("错误: 当不使用 --use-original-dialogue 或 --eval-only 或 --verify-only 时，必须通过 --patient-model 指定 Patient Agent 模型")
            sys.exit(1)
    
    # 检查 --data-file 是否需要
    if not args.data_file and not ((eval_only or verify_only) and dialogue_file):
        print("错误: 必须通过 --data-file 指定数据文件（除非使用 --eval-only/--verify-only + --dialogue-file）")
        sys.exit(1)
    
    # 解析模型列表
    model_list = [m.strip() for m in args.eval_models.split(",") if m.strip()]
    
    # 解析数据文件路径
    data_file = None
    if args.data_file:
        data_file = Path(args.data_file)
        if not data_file.is_absolute():
            data_file = PROJECT_ROOT / data_file
        
        if not data_file.exists():
            print(f"错误: 数据文件不存在: {data_file}")
            sys.exit(1)
    
    # 解析输出目录
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据（如果不是 eval_only + dialogue_file 模式）
    samples = []
    if not ((eval_only or verify_only) and dialogue_file):
        print(f"加载数据: {data_file}")
        samples = load_data(str(data_file))
        print(f"加载了 {len(samples)} 个样本")
        
        # 限制样本数量（按ICD大类均匀分层采样）
        if args.limit and args.limit < len(samples):
            samples = stratified_sample_by_icd(samples, args.limit, seed=42)
    else:
        print("跳过数据加载（将从对话文件加载）")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 打印配置信息
    print("\n" + "=" * 60)
    print("Doctor Agent 评估配置")
    print("=" * 60)
    print(f"数据文件: {data_file}")
    
    # 显示运行模式
    if generate_only:
        print("运行模式: 仅生成对话 (--generate-only)")
    elif verify_only:
        print("运行模式: 仅诊断验证 (--verify-only)")
        print(f"对话文件: {dialogue_file}")
        print(f"验证模型: {verifier_model}")
    elif eval_only:
        print("运行模式: 仅评估 (--eval-only)")
        if dialogue_file:
            print(f"对话文件: {dialogue_file}")
    else:
        print("运行模式: 完整流程（生成+评估）")
    
    if not generate_only and not verify_only:
        print(f"评估模型: {', '.join(model_list)}")
    
    if use_original_dialogue:
        print("对话来源: 使用数据集中的原始对话")
    elif (eval_only or verify_only) and dialogue_file:
        print("对话来源: 从文件加载")
    else:
        print(f"对话来源: Doctor Agent 生成")
        print(f"  - Doctor 模型: {doctor_model}")
        print(f"  - Doctor 版本: {doctor_version}")
        print(f"  - Patient 模型: {patient_model}")
        print(f"  - Patient 版本: {patient_version}")
        print(f"  - 最大轮次: {args.max_turns}")
    print("=" * 60)
    
    # ================================================================
    # 根据模式执行不同的流程
    # ================================================================
    
    eval_tasks = None
    dialogue_metadata = None
    
    if (eval_only or verify_only) and dialogue_file:
        # 模式：从文件加载对话（eval-only 或 verify-only）
        dialogue_path = Path(dialogue_file)
        if not dialogue_path.is_absolute():
            dialogue_path = PROJECT_ROOT / dialogue_path
        
        if not dialogue_path.exists():
            print(f"错误: 对话文件不存在: {dialogue_path}")
            sys.exit(1)
        
        eval_tasks, dialogue_metadata = load_generated_dialogues(dialogue_path)
        
        # 从对话文件元数据获取 doctor 信息（用于保存结果时标识）
        if dialogue_metadata:
            doctor_model = dialogue_metadata.get("doctor_model", doctor_model)
            doctor_version = dialogue_metadata.get("doctor_version", doctor_version)
            patient_model = dialogue_metadata.get("patient_model", patient_model)
            patient_version = dialogue_metadata.get("patient_version", patient_version)
            # 如果 data_file 为 None，从元数据获取原始数据文件路径
            if data_file is None:
                orig_data_file = dialogue_metadata.get("data_file")
                if orig_data_file:
                    data_file = Path(orig_data_file)
                else:
                    # 使用对话文件名作为标识
                    data_file = dialogue_path
        
    else:
        # 模式：需要生成或提取对话
        # 阶段1：准备评估任务
        eval_tasks = prepare_eval_tasks(
            samples=samples,
            use_original_dialogue=use_original_dialogue,
            doctor_model=doctor_model,
            doctor_version=doctor_version,
            patient_model=patient_model,
            patient_version=patient_version,
            max_turns=args.max_turns,
            max_workers=args.max_workers,
        )
    
    if not eval_tasks:
        print("错误: 没有可评估的任务")
        sys.exit(1)
    
    # 如果是 generate_only 模式，保存对话并退出
    if generate_only:
        dialogue_output_file = save_generated_dialogues(
            output_dir=output_dir,
            eval_tasks=eval_tasks,
            data_file=data_file,
            timestamp=timestamp,
            doctor_model=doctor_model,
            doctor_version=doctor_version,
            patient_model=patient_model,
            patient_version=patient_version,
            max_turns=args.max_turns,
        )
        print(f"\n{'='*60}")
        print("对话生成完成!")
        print(f"对话已保存到: {dialogue_output_file}")
        print(f"可使用以下命令进行评估:")
        print(f"  python {Path(__file__).name} \\")
        print(f"      --eval-only \\")
        print(f"      --dialogue-file {dialogue_output_file} \\")
        print(f"      --eval-models \"{args.eval_models}\"")
        print(f"{'='*60}")
        sys.exit(0)
    
    # 阶段2：并行评估所有任务（多模型并行）
    # 如果是 verify_only 模式，跳过 LLM-as-Judge 评估
    all_model_results = {}
    if not verify_only:
        all_model_results = evaluate_tasks_parallel(
            eval_tasks=eval_tasks,
            eval_models=model_list,
            max_workers=args.max_workers,
            api_key=args.api_key,
        )
    else:
        print("[信息] 跳过 LLM-as-Judge 评估（--verify-only）")
    
    # 阶段3：诊断验证（Dynamic评估）
    verification_results = None
    actual_verifier_model = None
    if verify_only or not skip_verification:
        # 确定验证模型
        actual_verifier_model = verifier_model or doctor_model
        if actual_verifier_model:
            verification_results = run_diagnosis_verification(
                eval_tasks=eval_tasks,
                verifier_model=actual_verifier_model,
                max_workers=args.max_workers,
                api_key=args.api_key,
                output_dir=output_dir,
                timestamp=timestamp,
            )
        else:
            print("[警告] 未指定验证模型，跳过诊断验证")
    else:
        print("[信息] 跳过诊断验证（--skip-verification）")
    
    # 计算统计并保存结果
    all_model_stats = {}
    saved_files = []
    
    for model_name in model_list:
        model_results = all_model_results.get(model_name, {})
        
        if not model_results:
            print(f"[警告] 模型 {model_name} 没有评估结果")
            continue
        
        # 计算统计信息
        stats = compute_statistics(model_results)
        all_model_stats[model_name] = stats
        
        # 打印摘要
        print_summary(stats, model_name, str(data_file))
        
        # 保存结果
        output_file = save_results(
            output_dir, model_name, model_results, stats, data_file, timestamp,
            use_original_dialogue=use_original_dialogue,
            doctor_model=doctor_model,
            doctor_version=doctor_version,
            patient_model=patient_model,
            patient_version=patient_version,
        )
        saved_files.append(output_file)
    
    print(f"\n{'='*60}")
    print("所有模型评估完成!")
    print(f"结果已保存到: {output_dir}")
    print(f"{'='*60}")
    
    # 自动聚合多模型评估结果
    if len(saved_files) > 1:
        print(f"\n正在自动聚合多模型评估结果...")
        
        try:
            model_results_data = {}
            for file_path in saved_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                model_name = data.get("metadata", {}).get("eval_model", file_path.stem)
                model_results_data[model_name] = data
            
            if model_results_data:
                aggregated = aggregate_doctor_results(model_results_data)
                print_aggregated_summary(aggregated)
                
                # 保存聚合结果（JSON格式）
                aggregated_file = output_dir / f"doctor_aggregated_results_{timestamp}.json"
                save_aggregated_results(aggregated, aggregated_file)
                
                # 保存聚合摘要（TXT格式）
                summary_file = output_dir / f"doctor_aggregated_summary_{timestamp}.txt"
                save_aggregated_summary_txt(
                    aggregated,
                    summary_file,
                    data_file=str(data_file),
                    use_original_dialogue=use_original_dialogue,
                    doctor_model=doctor_model,
                    doctor_version=doctor_version,
                )
        except Exception as e:
            print(f"警告: 自动聚合失败: {e}")
    
    # 保存到Excel和JSON汇总文件
    # verify_only 模式下创建空的统计数据
    if verify_only:
        first_model_stats = {"metrics": {}, "overall_average": {}}
        doctor_model_short = sanitize_model_name(doctor_model or "unknown")
    elif all_model_stats:
        # 取第一个eval模型的统计结果作为LLM-as-Judge结果
        first_model_stats = list(all_model_stats.values())[0]
        doctor_model_short = sanitize_model_name(doctor_model or "unknown")
    else:
        first_model_stats = None
        doctor_model_short = None
    
    if first_model_stats is not None:
        
        # 确定Excel输出路径
        if output_excel:
            excel_path = Path(output_excel)
        else:
            excel_path = output_dir / "evaluation_summary.xlsx"
        
        # 确定JSON汇总输出路径
        json_summary_path = output_dir / "evaluation_summary.json"
        
        # 保存Excel
        save_results_to_excel(
            output_path=excel_path,
            doctor_version=doctor_version,
            model_name=doctor_model_short,
            llm_judge_stats=first_model_stats,
            verification_results=verification_results,
            append_mode=True,
            verifier_model=sanitize_model_name(actual_verifier_model) if actual_verifier_model else None,
        )
        
        # 保存JSON汇总
        save_results_to_json_summary(
            output_path=json_summary_path,
            doctor_version=doctor_version,
            model_name=doctor_model_short,
            llm_judge_stats=first_model_stats,
            verification_results=verification_results,
            append_mode=True,
        )
    
    print(f"\n{'='*60}")
    print("评估流程全部完成!")
    if verification_results:
        print("包含: LLM-as-Judge评估 + Dynamic诊断验证")
    else:
        print("包含: LLM-as-Judge评估")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

