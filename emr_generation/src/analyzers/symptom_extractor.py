"""
症状提取器 - 使用LLM从病例中提取251个标准症状
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..utils.llm_client import LLMClient
from ..config import Config
from .distribution_analyzer import parse_diagnosis_codes, get_primary_diagnosis_code


class SymptomExtractor:
    """症状提取器 - 使用LLM判断251个症状是否出现"""
    
    def __init__(
        self,
        symptoms_file: Path = None,
        llm_client: LLMClient = None,
    ):
        """
        初始化症状提取器
        
        Args:
            symptoms_file: 症状定义Excel文件路径
            llm_client: LLM客户端
        """
        self.symptoms_file = symptoms_file or Config.SYMPTOMS_FILE
        self.llm_client = llm_client
        
        # 加载症状定义
        self.symptoms_df = self._load_symptoms()
        self.symptom_names = self.symptoms_df['症状名称'].tolist()
        
        # 存储提取结果
        # {诊断编码: {症状名称: 出现次数}}
        self.diagnosis_symptoms: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # {诊断编码: 总样本数}
        self.diagnosis_counts: Dict[str, int] = defaultdict(int)
        
        # 构建system prompt
        self.system_prompt = self._build_system_prompt()
    
    def _load_symptoms(self) -> pd.DataFrame:
        """加载症状定义"""
        df = pd.read_excel(self.symptoms_file)
        print(f"加载了 {len(df)} 个症状定义")
        return df
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        # 构建症状列表和说明
        symptoms_text = ""
        for idx, row in self.symptoms_df.iterrows():
            name = row['症状名称']
            desc = row['解释'] if pd.notna(row['解释']) else ""
            symptoms_text += f"{idx+1}. {name}: {desc}\n"
        
        system_prompt = f"""你是一位专业的精神科医生，擅长从病历和医患对话中识别精神症状。

你的任务是判断以下251个精神症状在给定的病历文本中是否出现。

## 症状列表及说明：
{symptoms_text}

## 输出格式要求：
请直接输出出现的症状名称，用逗号分隔。如果没有发现任何症状，输出"无"。
只输出症状名称，不要输出解释或其他内容。

## 示例输出：
失眠,情绪低落,焦虑,注意力不集中"""
        
        return system_prompt
    
    def _build_user_prompt(self, present_illness: str, dialogue: str = None) -> str:
        """构建用户提示词"""
        prompt = f"""请分析以下病历内容，判断其中出现了哪些症状。

## 现病史：
{present_illness}
"""
        if dialogue:
            # 对话可能很长，只取前2000字
            dialogue_truncated = dialogue[:2000] if len(dialogue) > 2000 else dialogue
            prompt += f"""
## 医患对话：
{dialogue_truncated}
"""
        
        prompt += """
请列出在上述内容中出现的症状名称（从251个标准症状中选择），用逗号分隔。"""
        
        return prompt
    
    def extract_symptoms_from_record(
        self, 
        record: Dict[str, Any]
    ) -> List[str]:
        """
        从单条记录中提取症状
        
        Args:
            record: 病例记录
            
        Returns:
            出现的症状列表
        """
        present_illness = record.get("PresentIllnessHistory", "")
        dialogue = record.get("cleaned_text", "")
        
        if not present_illness and not dialogue:
            return []
        
        user_prompt = self._build_user_prompt(present_illness, dialogue)
        
        try:
            response = self.llm_client.generate_text(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,  # 低温度保证一致性
                max_tokens=1000,
            )
            
            if not response:
                return []
            
            # 解析响应
            symptoms = self._parse_response(response)
            return symptoms
            
        except Exception as e:
            print(f"症状提取失败: {e}")
            return []
    
    def _parse_response(self, response: str) -> List[str]:
        """解析LLM响应，提取症状列表"""
        response = response.strip()
        
        if response == "无" or not response:
            return []
        
        # 按逗号或顿号分割
        symptoms = []
        for sep in [',', '，', '、']:
            if sep in response:
                parts = response.split(sep)
                symptoms = [s.strip() for s in parts if s.strip()]
                break
        
        if not symptoms:
            symptoms = [response.strip()]
        
        # 验证症状是否在标准列表中
        valid_symptoms = []
        for s in symptoms:
            if s in self.symptom_names:
                valid_symptoms.append(s)
            else:
                # 尝试模糊匹配
                for name in self.symptom_names:
                    if name in s or s in name:
                        valid_symptoms.append(name)
                        break
        
        return valid_symptoms
    
    def analyze_records(
        self,
        records: List[Dict[str, Any]],
        num_workers: int = 4,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        分析所有记录，提取症状分布
        
        Args:
            records: 病例记录列表
            num_workers: 并行工作线程数
            progress_callback: 进度回调函数
            
        Returns:
            分析结果
        """
        total = len(records)
        processed = 0
        lock = threading.Lock()
        
        def process_record(record):
            nonlocal processed
            
            # 获取诊断编码
            diagnosis_code = record.get("DiagnosisCode", "")
            primary_diag = get_primary_diagnosis_code(diagnosis_code)
            
            if not primary_diag:
                return None
            
            # 提取症状
            symptoms = self.extract_symptoms_from_record(record)
            
            # 更新统计
            with lock:
                self.diagnosis_counts[primary_diag] += 1
                for symptom in symptoms:
                    self.diagnosis_symptoms[primary_diag][symptom] += 1
                
                processed += 1
                if progress_callback:
                    progress_callback(processed, total)
            
            return {
                "diagnosis_code": primary_diag,
                "symptoms": symptoms,
            }
        
        # 并行处理
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_record, record): record for record in records}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        return self._build_summary()
    
    def _build_summary(self) -> Dict[str, Any]:
        """构建分析摘要"""
        summary = {
            "total_symptoms": len(self.symptom_names),
            "symptom_names": self.symptom_names,
            "diagnosis_symptom_occurrence": {},
        }
        
        # 计算每个诊断下的症状出现率
        for diag_code, symptom_counts in self.diagnosis_symptoms.items():
            total_samples = self.diagnosis_counts[diag_code]
            if total_samples == 0:
                continue
            
            occurrence = {}
            for symptom, count in symptom_counts.items():
                occurrence[symptom] = {
                    "count": count,
                    "rate": count / total_samples,
                }
            
            # 按出现率排序
            sorted_occurrence = dict(sorted(
                occurrence.items(),
                key=lambda x: x[1]["rate"],
                reverse=True
            ))
            
            summary["diagnosis_symptom_occurrence"][diag_code] = {
                "total_samples": total_samples,
                "symptoms": sorted_occurrence,
            }
        
        return summary
    
    def save_results(self, filepath: Path = None):
        """保存提取结果"""
        filepath = filepath or Config.MAPPING_DIR / "symptom_occurrence.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self._build_summary()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"症状提取结果已保存到: {filepath}")
        return summary


def extract_symptoms_from_data(
    data_file: Path = None,
    output_file: Path = None,
    llm_host: str = None,
    llm_port: int = None,
    llm_model: str = None,
    num_workers: int = 4,
    max_records: int = None,
):
    """
    从数据文件中提取症状
    
    Args:
        data_file: 数据文件路径
        output_file: 输出文件路径
        llm_host: LLM服务地址
        llm_port: LLM服务端口
        llm_model: LLM模型名称
        num_workers: 并行工作线程数
        max_records: 最大处理记录数（用于测试）
    """
    import json
    
    # 加载数据
    data_file = data_file or Config.DEFAULT_DATA_FILE
    print(f"加载数据: {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    if max_records:
        records = records[:max_records]
    
    print(f"共 {len(records)} 条记录")
    
    # 初始化LLM客户端
    llm_client = LLMClient(
        host=llm_host,
        port=llm_port,
        model=llm_model,
    )
    
    # 初始化提取器
    extractor = SymptomExtractor(llm_client=llm_client)
    
    # 提取症状
    def progress_callback(current, total):
        if current % 10 == 0 or current == total:
            print(f"进度: {current}/{total} ({current/total*100:.1f}%)")
    
    print("\n开始提取症状...")
    summary = extractor.analyze_records(
        records,
        num_workers=num_workers,
        progress_callback=progress_callback,
    )
    
    # 保存结果
    extractor.save_results(output_file)
    
    # 打印摘要
    print("\n=== 提取结果摘要 ===")
    for diag_code, data in summary["diagnosis_symptom_occurrence"].items():
        print(f"\n[{diag_code}] 样本数: {data['total_samples']}")
        top_symptoms = list(data["symptoms"].items())[:5]
        for symptom, info in top_symptoms:
            print(f"  {symptom}: {info['rate']:.1%} ({info['count']})")
    
    return summary
