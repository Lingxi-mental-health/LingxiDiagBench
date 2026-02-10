"""
规则提取器 - 使用规则和正则表达式提取结构化信息
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter

from .schemas import (
    PersonalHistorySlot,
    ChiefComplaintSlot,
    PhysicalIllnessSlot,
)


class RuleExtractor:
    """基于规则的信息提取器"""
    
    def __init__(self):
        """初始化规则提取器"""
        # 性别模式
        self.gender_patterns = ["男", "女"]
        
        # 科室列表
        self.departments = [
            "普通精神科", "精神科配药", "心理咨询", "特需 (精神2)",
            "特需（咨询）", "特需（精神1）"
        ]
        
        # 陪同人关系模式
        self.accompanying_patterns = {
            "自来": "自来",
            "丈夫": "丈夫",
            "母亲": "母亲",
            "父亲": "父亲",
            "奶奶": "奶奶",
            "家属": "家属",
        }
        
        # 家族史模式
        self.family_history_patterns = {
            "阴性": "阴性",
            "无": "阴性",
        }
        
        # 药物过敏史模式
        self.drug_allergy_patterns = {
            "无": "无",
        }
        
        # 婚恋状态模式
        self.marriage_patterns = [
            ("已婚", r"已婚"),
            ("未婚", r"未婚"),
            ("离异", r"离[异婚]"),
            ("未恋", r"未恋"),
            ("单身", r"单身"),
        ]
        
        # 孕产情况模式
        self.pregnancy_patterns = [
            ("足月顺产", r"足月顺产"),
            ("剖腹产", r"剖[腹宫]产"),
            ("早产", r"早产"),
            ("不详", r"不详"),
        ]
        
        # 发育情况模式
        self.development_patterns = [
            ("正常", r"发育[情况：]*正常"),
            ("迟缓", r"发育[情况：]*迟缓"),
            ("不详", r"发育[情况：]*不详"),
        ]
        
        # 性格模式
        self.personality_patterns = [
            "内向", "外向", "认真", "急躁", "敏感", "温和", "要强"
        ]
        
        # 嗜好模式
        self.habit_patterns = [
            ("吸烟史", r"吸烟[史]?"),
            ("饮酒", r"[饮喝]酒"),
            ("无特殊嗜好", r"无特殊嗜好"),
        ]
        
        # 症状关键词
        self.symptom_keywords = [
            "入睡困难", "睡眠差", "眠浅", "多梦", "早醒", "失眠",
            "情绪低落", "闷闷不乐", "不愉快", "抑郁",
            "焦虑", "紧张", "烦躁", "不安", "心慌", "易激惹",
            "兴趣减退", "兴趣下降",
            "自伤", "自残", "自杀",
            "幻听", "耳闻人语", "幻觉",
            "被监视感", "被害妄想",
            "强迫", "反复行为",
            "注意力不集中", "多虑", "多思",
        ]
        
        # 病程时间模式
        self.duration_pattern = re.compile(
            r'(\d+)\s*(年|月|周|天|个月|半年)'
        )
    
    def extract_age(self, age_str: str) -> Optional[int]:
        """提取年龄"""
        try:
            return int(age_str)
        except (ValueError, TypeError):
            return None
    
    def extract_gender(self, gender_str: str) -> Optional[str]:
        """提取性别"""
        if gender_str in self.gender_patterns:
            return gender_str
        return None
    
    def extract_department(self, dept_str: str) -> str:
        """提取科室"""
        return dept_str
    
    def extract_accompanying_person(self, text: str) -> Tuple[bool, Optional[str]]:
        """提取陪同人信息"""
        if "自来" in text:
            return (False, None)
        
        for relation in self.accompanying_patterns:
            if relation in text:
                return (True, relation)
        
        if "有" in text:
            # 尝试提取关系
            match = re.search(r'关系[：:]\s*(\S+)', text)
            if match:
                return (True, match.group(1))
            return (True, "家属")
        
        return (False, None)
    
    def extract_family_history(self, text: str) -> str:
        """提取家族史"""
        if "阴性" in text or "无" in text:
            return "阴性"
        # 去除所有句号（包括括号内的）
        text = text.replace('。', '').replace('.', '')
        return text.strip()
    
    def extract_drug_allergy(self, text: str) -> str:
        """提取药物过敏史"""
        if "无" in text:
            return "无"
        # 提取具体过敏药物
        match = re.search(r'(?:药物过敏史[：:])?\s*(.+)', text)
        if match:
            return match.group(1)
        return text
    
    def extract_personal_history_slots(self, text: str, gender: str = None) -> PersonalHistorySlot:
        """从个人史文本中提取结构化槽位"""
        slots = PersonalHistorySlot()
        
        # 辅助函数：清理提取的值（去除句号等）
        def clean_value(value: str) -> str:
            if not value:
                return value
            return value.strip().rstrip('。，,. ')
        
        # 提取孕产情况（仅女性）
        if gender != "男":
            for status, pattern in self.pregnancy_patterns:
                if re.search(pattern, text):
                    slots.pregnancy_status = status
                    break
        
        # 提取发育情况
        for status, pattern in self.development_patterns:
            if re.search(pattern, text):
                slots.development_status = status
                break
        
        # 提取婚恋情况
        # 支持 "婚恋情况：已婚" 或直接 "已婚"，但要排除后续的 "工作、学习情况"
        marriage_match = re.search(r'婚恋情况[：:]\s*([^，,。\s工月病特]+)', text)
        if marriage_match:
            slots.marriage_status = clean_value(marriage_match.group(1))
        else:
            # 直接匹配婚恋关键词
            for status, pattern in self.marriage_patterns:
                if re.search(pattern, text):
                    slots.marriage_status = status
                    break
        
        # 提取子女情况
        children_match = re.search(r'育有([一二三四五六七八九十\d]+)孩', text)
        if children_match:
            slots.children_info = f"育有{children_match.group(1)}孩"
        
        # 提取工作/学习情况
        # 支持多种格式："工作、学习情况：公司职员" 或 "工作 学习情况：公司职员"
        # 使用更精确的结束标记，排除无效字符
        work_match = re.search(r'工作[、和]?\s*学习情况[：:]\s*([^，,。婚月病特\s]+)', text)
        if work_match:
            occupation = clean_value(work_match.group(1))
            # 过滤掉无效的职业值
            invalid_occupations = {'可', '本科', '大学', '无殊', '不详', '无', '已', '未', '正常', '一般'}
            if occupation and occupation not in invalid_occupations and len(occupation) > 1:
                slots.occupation = occupation
        
        # 提取月经情况（仅女性）
        if gender != "男":
            menstrual_match = re.search(r'月经情况[：:]\s*([^，,。病\s]+)', text)
            if menstrual_match:
                menstrual = clean_value(menstrual_match.group(1))
                # 过滤掉无效值，"无特殊"应该被视为正常
                if menstrual and menstrual not in {'无', '无特殊'}:
                    slots.menstrual_status = menstrual
                elif menstrual in {'无特殊'}:
                    slots.menstrual_status = "正常"
        
        # 提取病前性格
        personality_match = re.search(r'病前性格[：:]\s*([^，,。特\s]+)', text)
        if personality_match:
            slots.premorbid_personality = clean_value(personality_match.group(1))
        else:
            # 回退到关键词匹配
            personalities = []
            for p in self.personality_patterns:
                if p in text:
                    personalities.append(p)
            if personalities:
                slots.premorbid_personality = ",".join(personalities)
        
        # 提取特殊嗜好（支持带括号的内容，如"嗜烟（时间：5年，数量：10支/天）"）
        # 先尝试匹配带括号的完整内容
        habit_match = re.search(r'特殊嗜好[：:]\s*([^（，,。]+（[^）]*）)', text)
        if not habit_match:
            # 没有括号的情况
            habit_match = re.search(r'特殊嗜好[：:]\s*([^，,。\s]+)', text)
        if habit_match:
            habit = clean_value(habit_match.group(1))
            if habit:
                slots.special_habits = habit
        else:
            for habit, pattern in self.habit_patterns:
                if re.search(pattern, text):
                    slots.special_habits = habit
                    break
        
        # 提取特殊经历
        special_exp_patterns = [
            r'(\d+岁时[^，,。]+)',
            r'(从小[^，,。]+)',
            r'(家中独[女子男])',
        ]
        for pattern in special_exp_patterns:
            match = re.search(pattern, text)
            if match:
                slots.special_experience = clean_value(match.group(1))
                break
        
        return slots
    
    def extract_chief_complaint_slots(self, text: str) -> ChiefComplaintSlot:
        """从主诉文本中提取结构化槽位"""
        slots = ChiefComplaintSlot()
        
        # 移除"主诉："前缀
        text = re.sub(r'^主诉[：:]\s*', '', text)
        
        # 提取症状
        symptoms = []
        for symptom in self.symptom_keywords:
            if symptom in text:
                symptoms.append(symptom)
        
        # 如果规则没有匹配到，使用分割方法
        if not symptoms:
            # 尝试按逗号分割并提取症状部分
            parts = re.split(r'[，,]', text)
            for part in parts:
                # 排除时间部分
                if not re.search(r'\d+\s*(年|月|周|天)', part):
                    part = part.strip()
                    if part and len(part) > 1:
                        symptoms.append(part)
        
        slots.symptoms = symptoms[:5]  # 最多保留5个症状
        
        # 提取病程时长
        duration_matches = self.duration_pattern.findall(text)
        if duration_matches:
            num, unit = duration_matches[0]
            slots.duration = f"{num}{unit}"
        
        # 提取总病程
        total_match = re.search(r'总病程\s*(\d+)\s*(年|月)', text)
        if total_match:
            slots.total_duration = f"{total_match.group(1)}{total_match.group(2)}"
        
        # 提取加重时间
        exac_match = re.search(r'加重\s*(\d+)\s*(年|月|周|天|个月)', text)
        if exac_match:
            slots.exacerbation_duration = f"{exac_match.group(1)}{exac_match.group(2)}"
        
        return slots
    
    def extract_physical_illness(self, text: str) -> PhysicalIllnessSlot:
        """提取躯体疾病史"""
        slots = PhysicalIllnessSlot()
        
        # 移除前缀
        text = re.sub(r'^重要或相关躯体疾病史[：:]\s*', '', text)
        
        if "无" in text and len(text) < 10:
            slots.has_illness = False
            slots.illnesses = []
        else:
            slots.has_illness = True
            # 提取疾病列表（保护括号内容不被分割）
            # 先替换括号内的分隔符为临时标记
            protected_text = text
            # 匹配括号内容并保护（包括句号）
            def protect_parentheses(match):
                content = match.group(0)
                content = content.replace('，', '<<COMMA>>').replace(',', '<<COMMA>>')
                content = content.replace('；', '<<SEMI>>').replace('。', '<<PERIOD>>')
                return content
            protected_text = re.sub(r'（[^）]*）', protect_parentheses, protected_text)
            protected_text = re.sub(r'\([^)]*\)', protect_parentheses, protected_text)
            
            # 按分隔符分割
            illnesses = re.split(r'[；;,，。]', protected_text)
            # 恢复临时标记并清理
            cleaned = []
            for i in illnesses:
                i = i.replace('<<COMMA>>', '，').replace('<<SEMI>>', '；').replace('<<PERIOD>>', '。').strip()
                # 去除括号内的尾部句号
                i = re.sub(r'。）$', '）', i)
                if i and i != "无" and i != "）":
                    cleaned.append(i)
            slots.illnesses = cleaned
        
        return slots
    
    def extract_diagnosis_info(self, record: Dict) -> Dict[str, Any]:
        """提取诊断相关信息"""
        return {
            "diagnosis_code": record.get("DiagnosisCode", ""),
            "overall_diagnosis": record.get("OverallDiagnosis", ""),
            "diagnosis": record.get("Diagnosis", ""),
        }
    
    def extract_all(self, record: Dict) -> Dict[str, Any]:
        """从一条记录中提取所有可用规则提取的信息"""
        result = {
            "patient_id": record.get("patient_id"),
            "age": self.extract_age(record.get("Age", "")),
            "gender": self.extract_gender(record.get("Gender", "")),
            "department": self.extract_department(record.get("Department", "")),
        }
        
        # 陪同人
        has_accompany, relation = self.extract_accompanying_person(
            record.get("AccompanyingPerson", "")
        )
        result["has_accompanying_person"] = has_accompany
        result["accompanying_relation"] = relation
        
        # 家族史和药物过敏
        result["family_history"] = self.extract_family_history(
            record.get("FamilyHistory", "")
        )
        result["drug_allergy"] = self.extract_drug_allergy(
            record.get("DrugAllergyHistory", "")
        )
        
        # 结构化槽位（传入性别用于判断是否提取月经/孕产情况）
        result["personal_history"] = self.extract_personal_history_slots(
            record.get("PersonalHistory", ""),
            gender=result.get("gender")
        )
        result["chief_complaint"] = self.extract_chief_complaint_slots(
            record.get("ChiefComplaint", "")
        )
        result["physical_illness"] = self.extract_physical_illness(
            record.get("ImportantRelevantPhysicalIllnessHistory", "")
        )
        
        # 诊断信息
        result.update(self.extract_diagnosis_info(record))
        
        # 原始文本（供后续LLM处理）
        result["raw_personal_history"] = record.get("PersonalHistory", "")
        result["raw_chief_complaint"] = record.get("ChiefComplaint", "")
        result["raw_present_illness"] = record.get("PresentIllnessHistory", "")
        
        return result


class DistributionCounter:
    """分布统计器 - 统计各字段的分布"""
    
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
    
    def add(self, field: str, value: Any):
        """添加一个值到统计"""
        if value is None:
            return
        
        if field not in self.counters:
            self.counters[field] = Counter()
        
        if isinstance(value, (list, tuple)):
            for v in value:
                self.counters[field][v] += 1
        else:
            self.counters[field][value] += 1
    
    def get_distribution(self, field: str) -> Dict[str, float]:
        """获取字段的分布（归一化为概率）"""
        if field not in self.counters:
            return {}
        
        counter = self.counters[field]
        total = sum(counter.values())
        
        if total == 0:
            return {}
        
        return {k: v / total for k, v in counter.items()}
    
    def get_all_distributions(self) -> Dict[str, Dict[str, float]]:
        """获取所有字段的分布"""
        return {field: self.get_distribution(field) for field in self.counters}
