"""
关键词分析器 - 分析主诉和现病史中的关键词分布
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import Counter, defaultdict
import random

from ..utils.tokenizer import ChineseTokenizer
from ..config import Config


def parse_diagnosis_codes(diagnosis_code: str) -> List[str]:
    """
    解析诊断编码为大类列表
    
    例如:
    - "F28.x02,F39.x00,F42.000x011" -> ["F28", "F39", "F42.0"]
    - "F32.100" -> ["F32.1"]
    - "F41.200x002" -> ["F41.2"]
    """
    if not diagnosis_code:
        return []
    
    codes = []
    raw_codes = [c.strip() for c in diagnosis_code.split(",")]
    
    for raw_code in raw_codes:
        if not raw_code:
            continue
        
        match = re.match(r'^([A-Z]\d+)(?:\.(\d))?', raw_code)
        
        if match:
            main_code = match.group(1)
            sub_code = match.group(2)
            
            if sub_code:
                parsed = f"{main_code}.{sub_code}"
            else:
                parsed = main_code
            
            if parsed not in codes:
                codes.append(parsed)
    
    return codes


class KeywordAnalyzer:
    """关键词分析器 - 分析文本中的关键词分布"""
    
    def __init__(self):
        self.tokenizer = ChineseTokenizer()
        
        # 主诉关键词分布：{诊断: {症状: Counter(关键词)}}
        self.chief_complaint_keywords: Dict[str, Dict[str, Counter]] = \
            defaultdict(lambda: defaultdict(Counter))
        
        # 现病史关键词分布：{诊断: Counter(关键词)}
        self.present_illness_keywords: Dict[str, Counter] = defaultdict(Counter)
        
        # 症状-关键词共现（按诊断编码分类）：{诊断: {症状: Counter(关键词)}}
        self.symptom_keyword_cooccurrence: Dict[str, Dict[str, Counter]] = \
            defaultdict(lambda: defaultdict(Counter))
        
        # 诱因关键词（按诊断编码分类）：{诊断: Counter(诱因)}
        self.trigger_keywords: Dict[str, Counter] = defaultdict(Counter)
        
        # 时间表达式模板（按诊断编码分类）：{诊断: Counter(时间表达式)}
        self.time_templates: Dict[str, Counter] = defaultdict(Counter)
        
        # 现病史句式模板
        self.sentence_templates: List[str] = []
    
    def analyze_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析所有记录
        
        Args:
            records: 原始记录列表
            
        Returns:
            分析结果
        """
        for record in records:
            self._analyze_chief_complaint(record)
            self._analyze_present_illness(record)
        
        return self._build_summary()
    
    def _analyze_chief_complaint(self, record: Dict[str, Any]):
        """分析主诉"""
        chief_complaint = record.get("ChiefComplaint", "")
        diagnosis_code = record.get("DiagnosisCode", "")
        diagnosis_categories = parse_diagnosis_codes(diagnosis_code)
        
        if not chief_complaint:
            return
        
        # 分词
        words = self.tokenizer.tokenize(chief_complaint)
        
        # 提取症状关键词
        symptoms = self.tokenizer.extract_symptom_keywords(chief_complaint)
        
        # 记录症状-关键词共现（按诊断编码分类）
        for diag_cat in diagnosis_categories:
            for symptom in symptoms:
                for word in words:
                    if word != symptom:
                        self.symptom_keyword_cooccurrence[diag_cat][symptom][word] += 1
        
        # 记录诊断编码大类-关键词
        for diag_cat in diagnosis_categories:
            for word in words:
                self.chief_complaint_keywords[diag_cat]["__all__"][word] += 1
        
        # 提取时间表达式（按诊断编码分类）
        times = self.tokenizer.extract_time_expressions(chief_complaint)
        for diag_cat in diagnosis_categories:
            for t in times:
                self.time_templates[diag_cat][t] += 1
    
    def _analyze_present_illness(self, record: Dict[str, Any]):
        """分析现病史"""
        present_illness = record.get("PresentIllnessHistory", "")
        diagnosis_code = record.get("DiagnosisCode", "")
        diagnosis_categories = parse_diagnosis_codes(diagnosis_code)
        
        if not present_illness:
            return
        
        # 分词
        words = self.tokenizer.tokenize(present_illness)
        
        # 记录诊断编码大类-关键词
        for diag_cat in diagnosis_categories:
            for word in words:
                self.present_illness_keywords[diag_cat][word] += 1
        
        # 提取诱因（按诊断编码分类）
        trigger_patterns = [
            "学习压力", "工作压力", "家庭问题", "感情问题", "人际关系",
            "情感问题", "经济压力", "失业", "分手", "离婚", "丧亲",
            "疫情", "封控", "父母不在身边", "被责怪", "被批评",
            "无明显诱因",
        ]
        for trigger in trigger_patterns:
            if trigger in present_illness:
                for diag_cat in diagnosis_categories:
                    self.trigger_keywords[diag_cat][trigger] += 1
        
        # 提取时间表达式（按诊断编码分类）
        times = self.tokenizer.extract_time_expressions(present_illness)
        for diag_cat in diagnosis_categories:
            for t in times:
                self.time_templates[diag_cat][t] += 1
        
        # 提取句式模板（简化版）
        self._extract_sentence_templates(present_illness)
    
    def _extract_sentence_templates(self, text: str):
        """提取句式模板"""
        # 常见句式模式
        patterns = [
            "患者{时间}无明显诱因下出现{症状}",
            "患者{时间}因{诱因}出现{症状}",
            "近{时间}来{症状}",
            "曾在{医院}就诊",
            "诊断{诊断}",
            "服用{药物}",
            "效果{效果}",
            "故来就诊",
            "为进一步治疗来我院门诊",
        ]
        
        for pattern in patterns:
            if any(keyword in text for keyword in pattern.replace("{", " ").replace("}", " ").split()):
                if pattern not in self.sentence_templates:
                    self.sentence_templates.append(pattern)
    
    def _build_summary(self) -> Dict[str, Any]:
        """构建分析摘要 - 只输出概率分布"""
        summary = {
            "chief_complaint_keywords": {},
            "present_illness_keywords": {},
            "symptom_keyword_cooccurrence": {},
            "trigger_keywords": {},
            "time_templates": {},
            "sentence_templates": self.sentence_templates,
        }
        
        # 主诉关键词（按诊断编码大类，只保留概率）
        for diagnosis, symptom_keywords in self.chief_complaint_keywords.items():
            summary["chief_complaint_keywords"][diagnosis] = {}
            for symptom, counter in symptom_keywords.items():
                total = sum(counter.values())
                if total > 0:
                    # 只保留 top 50，只输出概率
                    top_keywords = counter.most_common(50)
                    summary["chief_complaint_keywords"][diagnosis][symptom] = {
                        k: v / total for k, v in top_keywords
                    }
        
        # 现病史关键词（按诊断编码大类，只保留概率）
        for diagnosis, counter in self.present_illness_keywords.items():
            total = sum(counter.values())
            if total > 0:
                top_keywords = counter.most_common(100)
                summary["present_illness_keywords"][diagnosis] = {
                    k: v / total for k, v in top_keywords
                }
        
        # 症状-关键词共现（按诊断编码分类，只保留概率）
        for diagnosis, symptom_counters in self.symptom_keyword_cooccurrence.items():
            summary["symptom_keyword_cooccurrence"][diagnosis] = {}
            for symptom, counter in symptom_counters.items():
                total = sum(counter.values())
                if total > 0:
                    top_keywords = counter.most_common(30)
                    summary["symptom_keyword_cooccurrence"][diagnosis][symptom] = {
                        k: v / total for k, v in top_keywords
                    }
        
        # 诱因关键词（按诊断编码分类，只保留概率）
        for diagnosis, counter in self.trigger_keywords.items():
            total = sum(counter.values())
            if total > 0:
                summary["trigger_keywords"][diagnosis] = {
                    k: v / total for k, v in counter.items()
                }
        
        # 时间模板（按诊断编码分类，只保留概率）
        for diagnosis, counter in self.time_templates.items():
            total = sum(counter.values())
            if total > 0:
                # 保留 top 50 时间模板
                top_templates = counter.most_common(50)
                summary["time_templates"][diagnosis] = {
                    k: v / total for k, v in top_templates
                }
        
        return summary
    
    def save_mapping(self, filepath: Path = None):
        """保存关键词映射到文件"""
        filepath = filepath or Config.KEYWORD_MAPPING_FILE
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self._build_summary()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"关键词映射已保存到: {filepath}")
        return summary
    
    @classmethod
    def load_mapping(cls, filepath: Path = None) -> Dict[str, Any]:
        """从文件加载关键词映射"""
        filepath = filepath or Config.KEYWORD_MAPPING_FILE
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


class KeywordSampler:
    """关键词采样器"""
    
    def __init__(self, mapping: Dict[str, Any] = None, mapping_file: Path = None):
        """
        初始化采样器
        
        Args:
            mapping: 关键词映射字典
            mapping_file: 关键词映射文件路径
        """
        if mapping:
            self.mapping = mapping
        elif mapping_file:
            self.mapping = KeywordAnalyzer.load_mapping(mapping_file)
        else:
            self.mapping = KeywordAnalyzer.load_mapping()
    
    def sample_keywords_for_symptom(
        self,
        symptom: str,
        diagnosis_code: str = None,
        n: int = 5
    ) -> List[str]:
        """
        为特定症状采样相关关键词
        
        Args:
            symptom: 症状名称
            diagnosis_code: 诊断编码大类（如 F32.9）
            n: 采样数量
            
        Returns:
            关键词列表
        """
        cooccurrence = self.mapping.get("symptom_keyword_cooccurrence", {})
        
        # 优先使用诊断特定的共现分布
        if diagnosis_code and diagnosis_code in cooccurrence:
            if symptom in cooccurrence[diagnosis_code]:
                probs = cooccurrence[diagnosis_code][symptom]
                return self._sample_multiple(probs, n)
        
        # 回退：合并所有诊断的该症状共现
        all_probs = {}
        for diag_data in cooccurrence.values():
            if isinstance(diag_data, dict) and symptom in diag_data:
                for k, v in diag_data[symptom].items():
                    all_probs[k] = all_probs.get(k, 0) + v
        
        if all_probs:
            return self._sample_multiple(all_probs, n)
        return []
    
    def sample_present_illness_keywords(
        self,
        diagnosis_code: str = None,
        n: int = 10
    ) -> List[str]:
        """
        采样现病史关键词
        
        Args:
            diagnosis_code: 诊断编码大类（如 F32.1, F41.2）
            n: 采样数量
            
        Returns:
            关键词列表
        """
        pi_keywords = self.mapping.get("present_illness_keywords", {})
        
        if diagnosis_code and diagnosis_code in pi_keywords:
            probs = pi_keywords[diagnosis_code]
        else:
            # 合并所有诊断编码的关键词
            all_probs = {}
            for diag_keywords in pi_keywords.values():
                if isinstance(diag_keywords, dict):
                    for k, v in diag_keywords.items():
                        all_probs[k] = all_probs.get(k, 0) + v
            probs = all_probs
        
        return self._sample_multiple(probs, n)
    
    def sample_triggers(self, diagnosis_code: str = None, n: int = 2) -> List[str]:
        """
        采样诱因（按诊断编码）
        
        Args:
            diagnosis_code: 诊断编码大类（如 F32.9）
            n: 采样数量
            
        Returns:
            诱因列表
        """
        trigger_info = self.mapping.get("trigger_keywords", {})
        
        # 优先使用诊断特定的诱因分布
        if diagnosis_code and diagnosis_code in trigger_info:
            probs = trigger_info[diagnosis_code]
            return self._sample_multiple(probs, n)
        
        # 回退：合并所有诊断的诱因
        all_probs = {}
        for diag_triggers in trigger_info.values():
            if isinstance(diag_triggers, dict):
                for k, v in diag_triggers.items():
                    all_probs[k] = all_probs.get(k, 0) + v
        
        if all_probs:
            return self._sample_multiple(all_probs, n)
        
        return ["无明显诱因"]
    
    def sample_time_expression(self, diagnosis_code: str = None) -> str:
        """
        采样时间表达式（按诊断编码）
        
        Args:
            diagnosis_code: 诊断编码大类（如 F32.9）
            
        Returns:
            时间表达式
        """
        time_templates = self.mapping.get("time_templates", {})
        
        probs = None
        # 优先使用诊断特定的时间模板
        if diagnosis_code and diagnosis_code in time_templates:
            probs = time_templates[diagnosis_code]
        else:
            # 回退：合并所有诊断的时间模板
            all_probs = {}
            for diag_times in time_templates.values():
                if isinstance(diag_times, dict):
                    for k, v in diag_times.items():
                        all_probs[k] = all_probs.get(k, 0) + v
            if all_probs:
                probs = all_probs
        
        if probs:
            items = list(probs.keys())
            weights = list(probs.values())
            return random.choices(items, weights=weights, k=1)[0]
        
        # 默认时间表达式
        return f"{random.randint(1, 5)}年前"
    
    def get_sentence_templates(self) -> List[str]:
        """获取句式模板"""
        return self.mapping.get("sentence_templates", [])
    
    def _sample_multiple(self, probabilities: Dict[str, float], n: int) -> List[str]:
        """采样多个不重复的项"""
        if not probabilities:
            return []
        
        items = list(probabilities.keys())
        probs = list(probabilities.values())
        
        if len(items) <= n:
            return items
        
        result = []
        remaining_items = items.copy()
        remaining_probs = probs.copy()
        
        for _ in range(n):
            if not remaining_items:
                break
            
            total = sum(remaining_probs)
            normalized = [p / total for p in remaining_probs]
            
            idx = random.choices(range(len(remaining_items)), weights=normalized, k=1)[0]
            result.append(remaining_items[idx])
            
            remaining_items.pop(idx)
            remaining_probs.pop(idx)
        
        return result
