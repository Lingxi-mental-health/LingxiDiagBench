"""
分布分析器 - 分析各字段的分布情况
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import Counter, defaultdict
import random

from ..extractors.rule_extractor import RuleExtractor
from ..extractors.schemas import PersonalHistorySlot
from ..config import Config


def parse_diagnosis_codes(diagnosis_code: str) -> List[str]:
    """
    解析诊断编码为大类列表
    
    例如:
    - "F28.x02,F39.x00,F42.000x011" -> ["F28", "F39", "F42.0"]
    - "F32.100" -> ["F32.1"]
    - "F41.200x002" -> ["F41.2"]
    - "F41.200x002,G47.900" -> ["F41.2", "G47.9"]
    
    规则:
    1. 按逗号分割多个编码
    2. 对于每个编码，提取主类别（如 F32, F41）
    3. 如果有小数点后的数字（非x开头），保留第一位作为亚类
    """
    if not diagnosis_code:
        return []
    
    codes = []
    # 按逗号分割
    raw_codes = [c.strip() for c in diagnosis_code.split(",")]
    
    for raw_code in raw_codes:
        if not raw_code:
            continue
        
        # 匹配模式: 字母+数字(.数字)?(x...)?
        # 例如: F32.100, F41.200x002, F28.x02
        match = re.match(r'^([A-Z]\d+)(?:\.(\d))?', raw_code)
        
        if match:
            main_code = match.group(1)  # 如 F32, F41, G47
            sub_code = match.group(2)   # 如 1, 2, 9 (可能为None)
            
            if sub_code:
                parsed = f"{main_code}.{sub_code}"
            else:
                parsed = main_code
            
            if parsed not in codes:
                codes.append(parsed)
    
    return codes


def get_primary_diagnosis_code(diagnosis_code: str) -> Optional[str]:
    """获取主要诊断编码（第一个）"""
    codes = parse_diagnosis_codes(diagnosis_code)
    return codes[0] if codes else None


def get_length_bin(length: int) -> str:
    """
    将文本长度映射到分布区间（用于主诉、对话等短文本）
    
    区间: 1-5, 5-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90, 90-100, 100+
    """
    if length < 1:
        return "0"
    elif length <= 5:
        return "1-5"
    elif length <= 10:
        return "5-10"
    elif length <= 20:
        return "10-20"
    elif length <= 30:
        return "20-30"
    elif length <= 40:
        return "30-40"
    elif length <= 50:
        return "40-50"
    elif length <= 60:
        return "50-60"
    elif length <= 70:
        return "60-70"
    elif length <= 80:
        return "70-80"
    elif length <= 90:
        return "80-90"
    elif length <= 100:
        return "90-100"
    else:
        return "100+"


def get_present_illness_length_bin(length: int) -> str:
    """
    将现病史文本长度映射到分布区间（较长文本）
    
    区间: 0-50, 50-100, 100-150, ..., 450-500, 500+（间隔50）
    """
    if length <= 50:
        return "0-50"
    elif length <= 100:
        return "50-100"
    elif length <= 150:
        return "100-150"
    elif length <= 200:
        return "150-200"
    elif length <= 250:
        return "200-250"
    elif length <= 300:
        return "250-300"
    elif length <= 350:
        return "300-350"
    elif length <= 400:
        return "350-400"
    elif length <= 450:
        return "400-450"
    elif length <= 500:
        return "450-500"
    else:
        return "500+"


def get_dialogue_turns_bin(turns: int) -> str:
    """
    将对话轮数映射到分布区间
    
    区间: 0-10, 10-20, 20-30, ..., 90-100, 100+（间隔10）
    """
    if turns <= 10:
        return "0-10"
    elif turns <= 20:
        return "10-20"
    elif turns <= 30:
        return "20-30"
    elif turns <= 40:
        return "30-40"
    elif turns <= 50:
        return "40-50"
    elif turns <= 60:
        return "50-60"
    elif turns <= 70:
        return "60-70"
    elif turns <= 80:
        return "70-80"
    elif turns <= 90:
        return "80-90"
    elif turns <= 100:
        return "90-100"
    else:
        return "100+"


def get_age_group(age: int) -> str:
    """
    将年龄映射到年龄组
    
    区间: 0-18 (未成年), 18-30 (青年), 30-45 (中青年), 45-60 (中年), 60+ (老年)
    """
    if age < 18:
        return "0-18"
    elif age < 30:
        return "18-30"
    elif age < 45:
        return "30-45"
    elif age < 60:
        return "45-60"
    else:
        return "60+"


def get_doctor_avg_chars_bin(length: int) -> str:
    """
    将医生平均发言字数映射到分布区间
    
    区间: 0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+（间隔5）
    """
    if length <= 5:
        return "0-5"
    elif length <= 10:
        return "5-10"
    elif length <= 15:
        return "10-15"
    elif length <= 20:
        return "15-20"
    elif length <= 25:
        return "20-25"
    elif length <= 30:
        return "25-30"
    else:
        return "30+"


def parse_dialogue(text: str) -> dict:
    """
    解析对话文本，提取医生和患者的对话信息
    
    Returns:
        dict: {
            "total_turns": 总轮数（医生+患者）,
            "interaction_turns": 互动轮数（总轮数//2）,
            "doctor_turns": 医生发言次数,
            "doctor_total_chars": 医生发言总字数,
            "patient_turns": 患者发言次数,
            "patient_total_chars": 患者发言总字数,
        }
    """
    if not text:
        return None
    
    doctor_turns = 0
    doctor_total_chars = 0
    patient_turns = 0
    patient_total_chars = 0
    
    # 按行分割
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 医生发言
        if line.startswith('医生：') or line.startswith('医生:'):
            doctor_turns += 1
            # 去除前缀后计算字数
            content = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
            doctor_total_chars += len(content.strip())
        # 患者发言
        elif line.startswith('患者：') or line.startswith('患者:'):
            patient_turns += 1
            content = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
            patient_total_chars += len(content.strip())
        # 家属发言也算作患者方
        elif line.startswith('家属') and ('：' in line or ':' in line):
            patient_turns += 1
            content = line.split('：', 1)[-1] if '：' in line else line.split(':', 1)[-1]
            patient_total_chars += len(content.strip())
    
    total_turns = doctor_turns + patient_turns
    
    return {
        "total_turns": total_turns,
        "interaction_turns": total_turns // 2,
        "doctor_turns": doctor_turns,
        "doctor_total_chars": doctor_total_chars,
        "doctor_avg_chars": doctor_total_chars // doctor_turns if doctor_turns > 0 else 0,
        "patient_turns": patient_turns,
        "patient_total_chars": patient_total_chars,
        "patient_avg_chars": patient_total_chars // patient_turns if patient_turns > 0 else 0,
    }


class DistributionAnalyzer:
    """分布分析器 - 统计各字段的分布"""
    
    def __init__(self):
        self.rule_extractor = RuleExtractor()
        
        # 存储各字段的分布
        self.distributions: Dict[str, Counter] = defaultdict(Counter)
        
        # 条件分布：{条件字段: {条件值: {目标字段: Counter}}}
        self.conditional_distributions: Dict[str, Dict[str, Dict[str, Counter]]] = \
            defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        
        # 诊断编码大类 -> 诊断名称 映射（从数据中提取）
        self.diagnosis_code_to_name: Dict[str, str] = {}
        
        # 诊断相关的症状分布
        self.diagnosis_symptom_dist: Dict[str, Counter] = defaultdict(Counter)
        
        # 年龄分布参数
        self.age_stats: Dict[str, Any] = {}
        
        # 诊断编码 -> 文本长度分布
        # {诊断编码: {字段名: {长度区间: 计数}}}
        self.diagnosis_length_dist: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        
        # 年龄组 -> 个人史字段分布
        # {年龄组: {字段名: Counter(值)}}
        self.age_personal_history_dist: Dict[str, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        
        # 年龄组+性别 -> 陪同人分布
        # {年龄组: {性别: {"has_accompanying": Counter, "relation": Counter}}}
        self.age_gender_accompanying_dist: Dict[str, Dict[str, Dict[str, Counter]]] = defaultdict(
            lambda: defaultdict(lambda: {"has_accompanying": Counter(), "relation": Counter()})
        )
        
        # ICD 编码列表 -> 文本长度分布（使用所有 ICD 编码）
        # key: tuple of icd codes (will be converted to list string in output)
        # {("F28", "F39", "F42.0"): {字段名: {长度区间: 计数}}}
        self.icd_codes_length_dist: Dict[tuple, Dict[str, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        
        # 共病分布统计
        # 诊断编码数量分布 {1: count, 2: count, 3: count, ...}
        self.comorbidity_count_dist: Counter = Counter()
        
        # 共病组合分布 - 按主诊断分组
        # {主诊断: {共病诊断: 计数}}
        self.comorbidity_pairs_dist: Dict[str, Counter] = defaultdict(Counter)
        
        # 共病组合分布 - 完整组合
        # {(诊断1, 诊断2, ...): 计数}
        self.comorbidity_combinations: Counter = Counter()
    
    def analyze_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析所有记录，构建分布
        
        Args:
            records: 原始记录列表
            
        Returns:
            分析结果摘要
        """
        for record in records:
            self._analyze_single_record(record)
        
        return self._build_summary()
    
    def _analyze_single_record(self, record: Dict[str, Any]):
        """分析单条记录"""
        # 提取所有规则信息
        extracted = self.rule_extractor.extract_all(record)
        
        # 解析诊断编码为大类
        diagnosis_code = record.get("DiagnosisCode", "")
        diagnosis_categories = parse_diagnosis_codes(diagnosis_code)
        primary_diagnosis = diagnosis_categories[0] if diagnosis_categories else None
        
        # 统计共病分布
        num_diagnoses = len(diagnosis_categories)
        self.comorbidity_count_dist[num_diagnoses] += 1
        
        # 统计共病组合
        if num_diagnoses >= 2:
            # 按主诊断分组的共病对
            for secondary in diagnosis_categories[1:]:
                self.comorbidity_pairs_dist[primary_diagnosis][secondary] += 1
            
            # 完整共病组合（排序后作为key，保证唯一性）
            sorted_codes = tuple(sorted(diagnosis_categories))
            self.comorbidity_combinations[sorted_codes] += 1
        
        # 提取诊断编码大类 -> 诊断名称 的映射
        # 处理共病情况：将诊断名称按逗号分割，与编码一一对应
        diagnosis_name = record.get("Diagnosis", "")
        diagnosis_names = [n.strip() for n in diagnosis_name.split(",") if n.strip()]
        
        # 为每个诊断编码建立映射
        for i, diag_code in enumerate(diagnosis_categories):
            if diag_code not in self.diagnosis_code_to_name:
                # 如果有对应的诊断名称，使用它；否则跳过
                if i < len(diagnosis_names):
                    self.diagnosis_code_to_name[diag_code] = diagnosis_names[i]
                elif diagnosis_names:
                    # 如果编码数量多于名称数量，使用第一个名称
                    self.diagnosis_code_to_name[diag_code] = diagnosis_names[0]
        
        # 基础字段分布
        if extracted.get("age"):
            self.distributions["age"][extracted["age"]] += 1
        
        if extracted.get("gender"):
            self.distributions["gender"][extracted["gender"]] += 1
        
        if extracted.get("department"):
            self.distributions["department"][extracted["department"]] += 1
        
        if extracted.get("family_history"):
            self.distributions["family_history"][extracted["family_history"]] += 1
        
        if extracted.get("drug_allergy"):
            self.distributions["drug_allergy"][extracted["drug_allergy"]] += 1
        
        # 陪同人分布（全局 + 按年龄和性别）
        age = extracted.get("age")
        gender = extracted.get("gender")
        age_group = get_age_group(age) if age else None
        
        if extracted.get("has_accompanying_person"):
            self.distributions["has_accompanying_person"]["有"] += 1
            if extracted.get("accompanying_relation"):
                relation = extracted["accompanying_relation"]
                self.distributions["accompanying_relation"][relation] += 1
                
                # 按年龄组和性别统计
                if age_group and gender:
                    self.age_gender_accompanying_dist[age_group][gender]["has_accompanying"]["有"] += 1
                    self.age_gender_accompanying_dist[age_group][gender]["relation"][relation] += 1
        else:
            self.distributions["has_accompanying_person"]["自来"] += 1
            # 按年龄组和性别统计
            if age_group and gender:
                self.age_gender_accompanying_dist[age_group][gender]["has_accompanying"]["自来"] += 1
        
        # 诊断编码大类分布
        for diag_cat in diagnosis_categories:
            self.distributions["diagnosis_code_category"][diag_cat] += 1
        
        # 个人史槽位分布
        personal_history = extracted.get("personal_history")
        if personal_history:
            self._analyze_personal_history(personal_history, extracted, primary_diagnosis)
        
        # 主诉槽位分布 - 使用诊断编码大类
        chief_complaint = extracted.get("chief_complaint")
        if chief_complaint:
            self._analyze_chief_complaint(chief_complaint, extracted, diagnosis_categories)
        
        # 躯体疾病史分布
        physical_illness = extracted.get("physical_illness")
        if physical_illness:
            if physical_illness.has_illness:
                self.distributions["has_physical_illness"]["有"] += 1
                for illness in physical_illness.illnesses:
                    self.distributions["physical_illnesses"][illness] += 1
            else:
                self.distributions["has_physical_illness"]["无"] += 1
        
        # 按诊断编码统计文本长度分布
        if primary_diagnosis:
            self._analyze_text_lengths(record, primary_diagnosis)
        
        # 按 ICD 编码列表统计文本长度分布（使用所有编码）
        if diagnosis_categories:
            codes_key = tuple(diagnosis_categories)  # 使用 tuple 作为 key
            self._analyze_text_lengths_for_icd_codes(record, codes_key)
    
    def _analyze_text_lengths(self, record: Dict[str, Any], diagnosis_code: str):
        """分析文本长度分布（按诊断编码分类）"""
        # 主诉长度
        chief_complaint = record.get("ChiefComplaint", "")
        if chief_complaint:
            # 去除前缀 "主诉："
            text = chief_complaint.replace("主诉：", "").replace("主诉:", "").strip()
            length = len(text)
            length_bin = get_length_bin(length)
            self.diagnosis_length_dist[diagnosis_code]["chief_complaint"][length_bin] += 1
        
        # 现病史长度（使用较大区间）
        present_illness = record.get("PresentIllnessHistory", "")
        if present_illness:
            # 去除前缀 "现病史："
            text = present_illness.replace("现病史：", "").replace("现病史:", "").strip()
            length = len(text)
            length_bin = get_present_illness_length_bin(length)
            self.diagnosis_length_dist[diagnosis_code]["present_illness"][length_bin] += 1
        
        # 对话分析 (cleaned_text)
        cleaned_text = record.get("cleaned_text", "")
        if cleaned_text:
            dialogue_info = parse_dialogue(cleaned_text)
            if dialogue_info:
                # 互动轮数分布
                turns_bin = get_dialogue_turns_bin(dialogue_info["interaction_turns"])
                self.diagnosis_length_dist[diagnosis_code]["dialogue_turns"][turns_bin] += 1
                
                # 医生平均发言字数分布
                doctor_avg = dialogue_info["doctor_avg_chars"]
                doctor_bin = get_doctor_avg_chars_bin(doctor_avg)
                self.diagnosis_length_dist[diagnosis_code]["doctor_avg_chars"][doctor_bin] += 1
                
                # 患者平均发言字数分布
                patient_avg = dialogue_info["patient_avg_chars"]
                patient_bin = get_length_bin(patient_avg)
                self.diagnosis_length_dist[diagnosis_code]["patient_avg_chars"][patient_bin] += 1
    
    def _analyze_text_lengths_for_icd_codes(self, record: Dict[str, Any], codes_key: tuple):
        """
        分析文本长度分布（按 ICD 编码列表分类）
        
        Args:
            record: 原始记录
            codes_key: ICD 编码元组，如 ("F28", "F39", "F42.0")
        """
        # 主诉长度
        chief_complaint = record.get("ChiefComplaint", "")
        if chief_complaint:
            text = chief_complaint.replace("主诉：", "").replace("主诉:", "").strip()
            length = len(text)
            length_bin = get_length_bin(length)
            self.icd_codes_length_dist[codes_key]["chief_complaint"][length_bin] += 1
        
        # 现病史长度（使用较大区间）
        present_illness = record.get("PresentIllnessHistory", "")
        if present_illness:
            text = present_illness.replace("现病史：", "").replace("现病史:", "").strip()
            length = len(text)
            length_bin = get_present_illness_length_bin(length)
            self.icd_codes_length_dist[codes_key]["present_illness"][length_bin] += 1
        
        # 对话分析 (cleaned_text)
        cleaned_text = record.get("cleaned_text", "")
        if cleaned_text:
            dialogue_info = parse_dialogue(cleaned_text)
            if dialogue_info:
                # 互动轮数分布
                turns_bin = get_dialogue_turns_bin(dialogue_info["interaction_turns"])
                self.icd_codes_length_dist[codes_key]["dialogue_turns"][turns_bin] += 1
                
                # 医生平均发言字数分布
                doctor_avg = dialogue_info["doctor_avg_chars"]
                doctor_bin = get_doctor_avg_chars_bin(doctor_avg)
                self.icd_codes_length_dist[codes_key]["doctor_avg_chars"][doctor_bin] += 1
                
                # 患者平均发言字数分布
                patient_avg = dialogue_info["patient_avg_chars"]
                patient_bin = get_length_bin(patient_avg)
                self.icd_codes_length_dist[codes_key]["patient_avg_chars"][patient_bin] += 1
    
    def _analyze_personal_history(
        self,
        personal_history: PersonalHistorySlot,
        extracted: Dict[str, Any],
        primary_diagnosis: Optional[str] = None
    ):
        """分析个人史槽位"""
        gender = extracted.get("gender")
        age = extracted.get("age")
        age_group = get_age_group(age) if age else None
        
        # 孕产情况
        if personal_history.pregnancy_status:
            self.distributions["pregnancy_status"][personal_history.pregnancy_status] += 1
            # 按年龄组统计
            if age_group:
                self.age_personal_history_dist[age_group]["pregnancy_status"][personal_history.pregnancy_status] += 1
        
        # 发育情况
        if personal_history.development_status:
            self.distributions["development_status"][personal_history.development_status] += 1
            if age_group:
                self.age_personal_history_dist[age_group]["development_status"][personal_history.development_status] += 1
        
        # 婚恋情况
        if personal_history.marriage_status:
            self.distributions["marriage_status"][personal_history.marriage_status] += 1
            # 按年龄组统计
            if age_group:
                self.age_personal_history_dist[age_group]["marriage_status"][personal_history.marriage_status] += 1
            
            # 条件分布：性别 -> 婚恋
            if gender:
                self.conditional_distributions["gender"][gender]["marriage_status"][
                    personal_history.marriage_status
                ] += 1
        
        # 职业
        if personal_history.occupation:
            self.distributions["occupation"][personal_history.occupation] += 1
            if age_group:
                self.age_personal_history_dist[age_group]["occupation"][personal_history.occupation] += 1
        
        # 月经情况（仅女性）
        if personal_history.menstrual_status and gender == "女":
            self.distributions["menstrual_status"][personal_history.menstrual_status] += 1
            if age_group:
                self.age_personal_history_dist[age_group]["menstrual_status"][personal_history.menstrual_status] += 1
        
        # 性格
        if personal_history.premorbid_personality:
            for p in personal_history.premorbid_personality.split(","):
                p = p.strip()
                self.distributions["personality"][p] += 1
                if age_group:
                    self.age_personal_history_dist[age_group]["personality"][p] += 1
        
        # 嗜好
        if personal_history.special_habits:
            self.distributions["special_habits"][personal_history.special_habits] += 1
            if age_group:
                self.age_personal_history_dist[age_group]["special_habits"][personal_history.special_habits] += 1
    
    def _analyze_chief_complaint(
        self,
        chief_complaint,
        extracted: Dict[str, Any],
        diagnosis_categories: List[str] = None
    ):
        """分析主诉槽位"""
        # 症状分布
        for symptom in chief_complaint.symptoms:
            self.distributions["symptoms"][symptom] += 1
            
            # 条件分布：诊断编码大类 -> 症状
            if diagnosis_categories:
                for diag_cat in diagnosis_categories:
                    self.diagnosis_symptom_dist[diag_cat][symptom] += 1
        
        # 病程分布
        if chief_complaint.duration:
            self.distributions["duration"][chief_complaint.duration] += 1
    
    def _build_summary(self) -> Dict[str, Any]:
        """构建分析摘要 - 只输出概率分布"""
        summary = {
            "total_records": sum(self.distributions["gender"].values()),
            "distributions": {},
            "conditional_distributions": {},
        }
        
        # 转换为概率分布（不保留计数）
        for field, counter in self.distributions.items():
            total = sum(counter.values())
            if total > 0:
                summary["distributions"][field] = {k: v / total for k, v in counter.items()}
        
        # 条件分布（只保留概率）
        for cond_field, cond_values in self.conditional_distributions.items():
            summary["conditional_distributions"][cond_field] = {}
            for cond_value, target_fields in cond_values.items():
                summary["conditional_distributions"][cond_field][cond_value] = {}
                for target_field, counter in target_fields.items():
                    total = sum(counter.values())
                    if total > 0:
                        summary["conditional_distributions"][cond_field][cond_value][target_field] = {
                            k: v / total for k, v in counter.items()
                        }
        
        # 诊断编码大类-症状分布（只保留概率）
        summary["diagnosis_code_symptoms"] = {}
        for diagnosis, counter in self.diagnosis_symptom_dist.items():
            total = sum(counter.values())
            if total > 0:
                summary["diagnosis_code_symptoms"][diagnosis] = {
                    k: v / total for k, v in counter.items()
                }
        
        # 年龄统计（保留分布概率）
        ages = list(self.distributions["age"].elements())
        if ages:
            total_ages = len(ages)
            summary["age_stats"] = {
                "min": min(ages),
                "max": max(ages),
                "mean": round(sum(ages) / len(ages), 1),
                "distribution": {k: v / total_ages for k, v in self.distributions["age"].items()},
            }
        
        # 诊断编码 -> 文本长度分布（只保留概率）
        # 按长度区间排序的键顺序（不同字段使用不同区间）
        default_length_bins = ["0", "1-5", "5-10", "10-20", "20-30", "30-40", "40-50", 
                               "50-60", "60-70", "70-80", "80-90", "90-100", "100+"]
        length_bin_orders = {
            "chief_complaint": default_length_bins,
            "present_illness": ["0-50", "50-100", "100-150", "150-200", "200-250", "250-300", 
                               "300-350", "350-400", "400-450", "450-500", "500+"],
            "dialogue_turns": ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", 
                              "60-70", "70-80", "80-90", "90-100", "100+"],
            "doctor_avg_chars": ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"],
            "patient_avg_chars": default_length_bins,
        }
        
        # 最小样本数阈值
        MIN_SAMPLES = 5
        
        summary["diagnosis_length_distributions"] = {}
        for diagnosis_code, field_counters in self.diagnosis_length_dist.items():
            # 检查该诊断编码的样本数（以任一字段为准）
            sample_count = max(sum(counter.values()) for counter in field_counters.values()) if field_counters else 0
            if sample_count < MIN_SAMPLES:
                continue  # 跳过样本数不足的诊断编码
            
            summary["diagnosis_length_distributions"][diagnosis_code] = {}
            for field, counter in field_counters.items():
                total = sum(counter.values())
                if total > 0:
                    # 按区间顺序排序（使用对应字段的区间顺序）
                    bin_order = length_bin_orders.get(field, length_bin_orders["chief_complaint"])
                    sorted_dist = {}
                    for bin_name in bin_order:
                        if bin_name in counter:
                            sorted_dist[bin_name] = counter[bin_name] / total
                    summary["diagnosis_length_distributions"][diagnosis_code][field] = sorted_dist
        
        # ICD 编码列表 -> 文本长度分布（只保留概率）
        # key 为 JSON 格式的列表字符串，如 '["F28", "F39", "F42.0"]'
        summary["icd_codes_length_distribution"] = {}
        for codes_tuple, field_counters in self.icd_codes_length_dist.items():
            # 检查该编码组合的样本数
            sample_count = max(sum(counter.values()) for counter in field_counters.values()) if field_counters else 0
            if sample_count < MIN_SAMPLES:
                continue  # 跳过样本数不足的编码组合
            
            # 将 tuple 转换为 JSON 列表字符串，保持原始排序
            codes_key = json.dumps(list(codes_tuple), ensure_ascii=False)
            
            summary["icd_codes_length_distribution"][codes_key] = {}
            for field, counter in field_counters.items():
                total = sum(counter.values())
                if total > 0:
                    # 按区间顺序排序
                    bin_order = length_bin_orders.get(field, length_bin_orders["chief_complaint"])
                    sorted_dist = {}
                    for bin_name in bin_order:
                        if bin_name in counter:
                            sorted_dist[bin_name] = counter[bin_name] / total
                    summary["icd_codes_length_distribution"][codes_key][field] = sorted_dist
        
        # 年龄组 -> 个人史字段分布（只保留概率）
        # 年龄组顺序
        age_group_order = ["0-18", "18-30", "30-45", "45-60", "60+"]
        summary["age_personal_history"] = {}
        for age_group in age_group_order:
            if age_group in self.age_personal_history_dist:
                summary["age_personal_history"][age_group] = {}
                for field, counter in self.age_personal_history_dist[age_group].items():
                    total = sum(counter.values())
                    if total > 0:
                        # 只保留概率前20的值
                        top_items = counter.most_common(20)
                        summary["age_personal_history"][age_group][field] = {
                            k: v / total for k, v in top_items
                        }
        
        # 年龄组+性别 -> 陪同人分布（只保留概率）
        summary["age_gender_accompanying"] = {}
        for age_group in age_group_order:
            if age_group in self.age_gender_accompanying_dist:
                summary["age_gender_accompanying"][age_group] = {}
                for gender in ["男", "女"]:
                    if gender in self.age_gender_accompanying_dist[age_group]:
                        gender_data = self.age_gender_accompanying_dist[age_group][gender]
                        summary["age_gender_accompanying"][age_group][gender] = {}
                        
                        # 是否有陪同
                        has_total = sum(gender_data["has_accompanying"].values())
                        if has_total > 0:
                            summary["age_gender_accompanying"][age_group][gender]["has_accompanying"] = {
                                k: v / has_total for k, v in gender_data["has_accompanying"].items()
                            }
                        
                        # 陪同人关系（保留前15个）
                        rel_total = sum(gender_data["relation"].values())
                        if rel_total > 0:
                            top_relations = gender_data["relation"].most_common(15)
                            summary["age_gender_accompanying"][age_group][gender]["relation"] = {
                                k: v / rel_total for k, v in top_relations
                            }
        
        # 共病分布统计
        # 诊断数量分布（只保留概率）
        total_records = sum(self.comorbidity_count_dist.values())
        if total_records > 0:
            summary["comorbidity"] = {
                "count_distribution": {
                    str(k): v / total_records for k, v in sorted(self.comorbidity_count_dist.items())
                }
            }
            
            # 按主诊断的共病对分布（只保留概率，取前30个主诊断）
            summary["comorbidity"]["pairs_by_primary"] = {}
            top_primaries = sorted(
                self.comorbidity_pairs_dist.items(),
                key=lambda x: sum(x[1].values()),
                reverse=True
            )[:30]
            
            for primary, secondary_counter in top_primaries:
                total_pairs = sum(secondary_counter.values())
                if total_pairs > 0:
                    # 取前10个共病诊断
                    top_secondaries = secondary_counter.most_common(10)
                    summary["comorbidity"]["pairs_by_primary"][primary] = {
                        k: v / total_pairs for k, v in top_secondaries
                    }
            
            # 完整共病组合分布（取前50个最常见组合）
            # 将 tuple key 转换为 JSON 字符串
            top_combinations = self.comorbidity_combinations.most_common(50)
            total_combo = sum(self.comorbidity_combinations.values())
            if total_combo > 0:
                summary["comorbidity"]["combinations"] = {
                    json.dumps(list(codes), ensure_ascii=False): count / total_combo
                    for codes, count in top_combinations
                }
        
        return summary
    
    def save_mapping(self, filepath: Path = None):
        """保存分布映射到文件"""
        filepath = filepath or Config.DISTRIBUTION_MAPPING_FILE
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self._build_summary()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"分布映射已保存到: {filepath}")
        return summary
    
    def save_diagnosis_mapping(self, filepath: Path = None):
        """保存诊断编码映射到单独的文件"""
        filepath = filepath or Config.DIAGNOSIS_CODE_MAPPING_FILE
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 诊断编码大类 -> 诊断名称 的一一对应映射
        mapping = {
            "diagnosis_code_to_name": self.diagnosis_code_to_name,
            "name_to_diagnosis_code": {v: k for k, v in self.diagnosis_code_to_name.items()},
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        
        print(f"诊断编码映射已保存到: {filepath}")
        return mapping
    
    @classmethod
    def load_mapping(cls, filepath: Path = None) -> Dict[str, Any]:
        """从文件加载分布映射"""
        filepath = filepath or Config.DISTRIBUTION_MAPPING_FILE
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


class DistributionSampler:
    """分布采样器 - 根据分布进行采样"""
    
    def __init__(self, mapping: Dict[str, Any] = None, mapping_file: Path = None):
        """
        初始化采样器
        
        Args:
            mapping: 分布映射字典
            mapping_file: 分布映射文件路径
        """
        if mapping:
            self.mapping = mapping
        elif mapping_file:
            self.mapping = DistributionAnalyzer.load_mapping(mapping_file)
        else:
            self.mapping = DistributionAnalyzer.load_mapping()
    
    def sample(self, field: str, condition: Dict[str, Any] = None) -> Optional[Any]:
        """
        从分布中采样
        
        Args:
            field: 字段名
            condition: 条件字典，如 {"gender": "女"}
            
        Returns:
            采样结果
        """
        # 尝试条件采样
        if condition:
            for cond_field, cond_value in condition.items():
                cond_dist = self.mapping.get("conditional_distributions", {})
                if cond_field in cond_dist and cond_value in cond_dist[cond_field]:
                    field_dist = cond_dist[cond_field][cond_value].get(field)
                    if field_dist:
                        return self._sample_from_prob(field_dist)
        
        # 无条件采样
        dist = self.mapping.get("distributions", {}).get(field)
        if dist:
            return self._sample_from_prob(dist)
        
        return None
    
    def sample_symptoms(
        self,
        diagnosis_code: str = None,
        n: int = 3,
    ) -> List[str]:
        """
        采样症状
        
        Args:
            diagnosis_code: 诊断编码大类（如 F32.1, F41.2）
            n: 采样数量
            
        Returns:
            症状列表
        """
        if diagnosis_code:
            # 优先使用诊断编码大类-症状分布
            diagnosis_symptoms = self.mapping.get("diagnosis_code_symptoms", {})
            if diagnosis_code in diagnosis_symptoms:
                probs = diagnosis_symptoms[diagnosis_code]
                return self._sample_multiple(probs, n)
        
        # 从通用症状分布采样
        symptoms_dist = self.mapping.get("distributions", {}).get("symptoms")
        if symptoms_dist:
            return self._sample_multiple(symptoms_dist, n)
        
        return []
    
    def sample_age(self) -> int:
        """
        采样年龄
        
        年龄映射关系：
            10岁 → 10-18岁
            20岁 → 18-25岁
            30岁 → 25-35岁
            40岁 → 35-45岁
            50岁 → 45-55岁
            60岁 → 55-65岁
            70岁 → 65-75岁
            80岁 → 75-95岁
        """
        # 定义年龄映射区间
        age_mapping = {
            10: (10, 18),
            20: (18, 25),
            30: (25, 35),
            40: (35, 45),
            50: (45, 55),
            60: (55, 65),
            70: (65, 75),
            80: (75, 95),
        }
        
        age_stats = self.mapping.get("age_stats")
        if age_stats and "distribution" in age_stats:
            decade = self._sample_from_prob_as_int(age_stats["distribution"])
            # 使用映射区间随机选择具体年龄
            if decade in age_mapping:
                min_age, max_age = age_mapping[decade]
                return random.randint(min_age, max_age)
            elif decade < 10:
                return random.randint(0, 9)
            elif decade > 80:
                return random.randint(75, 95)
            else:
                # 其他情况（理论上不会发生）
                return random.randint(decade, decade + 9)
        return random.randint(18, 60)
    
    def sample_personal_history_field(self, field: str, age: int = None, gender: str = None) -> Optional[str]:
        """
        按年龄组采样个人史字段
        
        Args:
            field: 字段名（如 marriage_status, occupation, pregnancy_status 等）
            age: 年龄
            gender: 性别（某些字段如 menstrual_status 需要性别过滤）
            
        Returns:
            采样结果
        """
        # 获取年龄组
        age_group = get_age_group(age) if age else None
        
        # 尝试按年龄组采样
        if age_group:
            age_ph = self.mapping.get("age_personal_history", {})
            if age_group in age_ph and field in age_ph[age_group]:
                probs = age_ph[age_group][field]
                return self._sample_from_prob(probs)
        
        # 回退到通用分布
        dist = self.mapping.get("distributions", {}).get(field)
        if dist:
            return self._sample_from_prob(dist)
        
        return None
    
    def sample_accompanying_person(self, age: int = None, gender: str = None) -> tuple:
        """
        按年龄和性别采样陪同人信息
        
        Args:
            age: 年龄
            gender: 性别
            
        Returns:
            (has_accompanying, relation): (是否有陪同, 陪同人关系)
            - 如果自来，relation 为 None
        """
        age_group = get_age_group(age) if age else None
        
        # 尝试按年龄组和性别采样
        age_gender_acc = self.mapping.get("age_gender_accompanying", {})
        
        has_dist = None
        rel_dist = None
        
        if age_group and gender:
            if age_group in age_gender_acc and gender in age_gender_acc[age_group]:
                gender_data = age_gender_acc[age_group][gender]
                has_dist = gender_data.get("has_accompanying")
                rel_dist = gender_data.get("relation")
        
        # 采样是否有陪同
        if has_dist:
            has_accompanying = self._sample_from_prob(has_dist)
        else:
            # 回退到全局分布
            global_has = self.mapping.get("distributions", {}).get("has_accompanying_person")
            if global_has:
                has_accompanying = self._sample_from_prob(global_has)
            else:
                has_accompanying = "有" if random.random() < 0.85 else "自来"
        
        # 如果自来，不需要采样关系
        if has_accompanying == "自来":
            return ("自来", None)
        
        # 采样陪同人关系
        if rel_dist:
            relation = self._sample_from_prob(rel_dist)
        else:
            # 回退到全局分布
            global_rel = self.mapping.get("distributions", {}).get("accompanying_relation")
            if global_rel:
                relation = self._sample_from_prob(global_rel)
            else:
                # 基于年龄的默认关系
                if age and age < 18:
                    relation = random.choice(["母亲", "父亲", "父母"])
                elif age and age > 60:
                    relation = random.choice(["子女", "配偶", "儿子", "女儿"])
                else:
                    relation = random.choice(["母亲", "配偶", "朋友", "家属"])
        
        return ("有", relation)
    
    def _sample_from_prob_as_int(self, probabilities: Dict[str, float]) -> int:
        """从概率分布采样并转换为整数"""
        items = list(probabilities.keys())
        probs = list(probabilities.values())
        result = random.choices(items, weights=probs, k=1)[0]
        try:
            return int(result)
        except:
            return 30  # 默认年龄
    
    def _sample_from_prob(self, probabilities: Dict[str, float]) -> Any:
        """从概率分布采样"""
        items = list(probabilities.keys())
        probs = list(probabilities.values())
        return random.choices(items, weights=probs, k=1)[0]
    
    def _sample_from_counts(self, counts: Dict[str, int]) -> Any:
        """从计数分布采样"""
        items = list(counts.keys())
        weights = list(counts.values())
        result = random.choices(items, weights=weights, k=1)[0]
        # 如果是数字字符串，转为整数
        try:
            return int(result)
        except:
            return result
    
    def _sample_multiple(self, probabilities: Dict[str, float], n: int) -> List[str]:
        """采样多个不重复的项"""
        items = list(probabilities.keys())
        probs = list(probabilities.values())
        
        if len(items) <= n:
            return items
        
        # 加权随机采样（不放回）
        result = []
        remaining_items = items.copy()
        remaining_probs = probs.copy()
        
        for _ in range(n):
            if not remaining_items:
                break
            
            # 归一化概率
            total = sum(remaining_probs)
            normalized = [p / total for p in remaining_probs]
            
            # 采样
            idx = random.choices(range(len(remaining_items)), weights=normalized, k=1)[0]
            result.append(remaining_items[idx])
            
            # 移除已选项
            remaining_items.pop(idx)
            remaining_probs.pop(idx)
        
        return result
    
    def sample_length_bin(
        self, 
        diagnosis_code: str, 
        field: str
    ) -> Optional[str]:
        """
        根据诊断编码采样文本长度区间
        
        Args:
            diagnosis_code: 诊断编码大类（如 F32.9）
            field: 字段名（chief_complaint, present_illness）
            
        Returns:
            长度区间字符串（如 "10-20", "100-150"）
        """
        length_dists = self.mapping.get("diagnosis_length_distributions", {})
        
        if diagnosis_code in length_dists:
            field_dist = length_dists[diagnosis_code].get(field, {})
            if field_dist:
                return self._sample_from_prob(field_dist)
        
        return None
    
    def sample_target_length(
        self, 
        diagnosis_code: str, 
        field: str
    ) -> Optional[int]:
        """
        根据诊断编码采样目标文本长度（取区间中点）
        
        Args:
            diagnosis_code: 诊断编码大类（如 F32.9）
            field: 字段名（chief_complaint, present_illness）
            
        Returns:
            目标长度（整数）
        """
        length_bin = self.sample_length_bin(diagnosis_code, field)
        
        if not length_bin:
            return None
        
        return self._bin_to_target_length(length_bin)
    
    def _bin_to_target_length(self, length_bin: str) -> int:
        """
        将长度区间转换为目标长度（取区间中点或随机值）
        
        Args:
            length_bin: 长度区间字符串（如 "10-20", "100+"）
            
        Returns:
            目标长度（整数）
        """
        if "+" in length_bin:
            # 如 "100+", "500+"
            base = int(length_bin.replace("+", ""))
            # 在base到base*1.5之间随机取值
            return random.randint(base, int(base * 1.3))
        
        if "-" in length_bin:
            parts = length_bin.split("-")
            low = int(parts[0])
            high = int(parts[1])
            # 在区间内随机取值
            return random.randint(low, high)
        
        # 单个数字
        try:
            return int(length_bin)
        except:
            return 50  # 默认值
