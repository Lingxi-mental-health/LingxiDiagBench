"""
完整病例生成器 - 生成完整的虚拟电子病历
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..config import Config
from ..analyzers.distribution_analyzer import DistributionSampler
from ..analyzers.keyword_analyzer import KeywordSampler
from ..utils.llm_client import LLMClient
from ..extractors.schemas import (
    GenerationContext,
    PersonalHistorySlot,
    ChiefComplaintSlot,
    EMRRecord,
)
from .chief_complaint_generator import ChiefComplaintGenerator
from .present_illness_generator import PresentIllnessGenerator


class EMRGenerator:
    """完整病例生成器"""
    
    def __init__(
        self,
        distribution_mapping: Dict[str, Any] = None,
        keyword_mapping: Dict[str, Any] = None,
        diagnosis_mapping: Dict[str, Any] = None,
        llm_host: str = None,
        llm_port: int = None,
        llm_model: str = None,
        use_llm: bool = True,
    ):
        """
        初始化病例生成器
        
        Args:
            distribution_mapping: 分布映射（如果为None则从文件加载）
            keyword_mapping: 关键词映射（如果为None则从文件加载）
            diagnosis_mapping: 诊断编码映射（如果为None则从文件加载）
            llm_host: LLM服务地址
            llm_port: LLM服务端口
            llm_model: LLM模型名称
            use_llm: 是否使用LLM
        """
        # 加载分布映射
        if distribution_mapping:
            self.dist_sampler = DistributionSampler(mapping=distribution_mapping)
        else:
            try:
                self.dist_sampler = DistributionSampler()
            except FileNotFoundError:
                print("警告：未找到分布映射文件，使用默认分布")
                self.dist_sampler = None
        
        # 加载关键词映射
        if keyword_mapping:
            self.keyword_sampler = KeywordSampler(mapping=keyword_mapping)
        else:
            try:
                self.keyword_sampler = KeywordSampler()
            except FileNotFoundError:
                print("警告：未找到关键词映射文件，使用默认关键词")
                self.keyword_sampler = None
        
        # 加载诊断编码映射
        if diagnosis_mapping:
            self.diagnosis_mapping = diagnosis_mapping
        else:
            try:
                with open(Config.DIAGNOSIS_CODE_MAPPING_FILE, 'r', encoding='utf-8') as f:
                    self.diagnosis_mapping = json.load(f)
            except FileNotFoundError:
                print("警告：未找到诊断编码映射文件")
                self.diagnosis_mapping = {"diagnosis_code_to_name": {}, "name_to_diagnosis_code": {}}
        
        # 初始化LLM客户端
        self.use_llm = use_llm
        self.llm_client = None
        if use_llm:
            try:
                self.llm_client = LLMClient(
                    host=llm_host,
                    port=llm_port,
                    model=llm_model,
                )
            except Exception as e:
                print(f"警告：LLM客户端初始化失败: {e}")
                self.use_llm = False
        
        # 初始化子生成器
        self.chief_complaint_generator = ChiefComplaintGenerator(
            distribution_sampler=self.dist_sampler,
            keyword_sampler=self.keyword_sampler,
            llm_client=self.llm_client,
        )
        
        self.present_illness_generator = PresentIllnessGenerator(
            distribution_sampler=self.dist_sampler,
            keyword_sampler=self.keyword_sampler,
            llm_client=self.llm_client,
        )
    
    def generate(
        self,
        constraints: Dict[str, Any] = None,
        use_llm: bool = None,
    ) -> Dict[str, Any]:
        """
        生成一条完整的虚拟病例
        
        Args:
            constraints: 约束条件，如 {"diagnosis": "Depression", "gender": "女"}
            use_llm: 是否使用LLM（覆盖默认设置）
            
        Returns:
            生成的病例字典
        """
        constraints = constraints or {}
        use_llm = use_llm if use_llm is not None else self.use_llm
        
        # 创建生成上下文
        context = self._create_context(constraints)
        
        # 生成各部分
        emr = {}
        
        # 基础信息
        emr["patient_id"] = self._generate_patient_id()
        emr["Age"] = str(context.age)
        emr["Gender"] = context.gender
        emr["Department"] = context.department
        
        # 陪同人
        emr["AccompanyingPerson"] = self._generate_accompanying_person(context)
        
        # 个人史
        emr["PersonalHistory"] = self._generate_personal_history(context, use_llm)
        
        # 主诉
        emr["ChiefComplaint"] = self.chief_complaint_generator.generate(
            context=context,
            use_llm=use_llm,
        )
        
        # 更新上下文中的主诉信息
        context.chief_complaint = ChiefComplaintSlot(
            symptoms=context.selected_symptoms,
            duration=self._extract_duration_from_text(emr["ChiefComplaint"]),
        )
        
        # 躯体疾病史
        emr["ImportantRelevantPhysicalIllnessHistory"] = self._generate_physical_illness(context)
        
        # 现病史
        emr["PresentIllnessHistory"] = self.present_illness_generator.generate(
            context=context,
            use_llm=use_llm,
        )
        
        # 其他固定字段
        emr["DrugAllergyHistory"] = self._generate_drug_allergy(context)
        emr["FamilyHistory"] = self._generate_family_history(context)
        
        # 诊断信息
        emr["DiagnosisCode"] = self._generate_diagnosis_code(context)
        emr["OverallDiagnosis"] = self._get_overall_diagnosis(context)
        emr["Diagnosis"] = self._generate_diagnosis_name(context)
        
        # 量表信息（可选）
        emr["Scale_name"] = ""
        emr["score"] = ""
        emr["AuxiliaryExamination"] = "辅助检查:暂缺"
        
        return emr
    
    def generate_batch(
        self,
        n: int,
        constraints: Dict[str, Any] = None,
        use_llm: bool = None,
        progress_callback=None,
        num_workers: int = None,
    ) -> List[Dict[str, Any]]:
        """
        批量生成虚拟病例（支持并行）
        
        Args:
            n: 生成数量
            constraints: 约束条件
            use_llm: 是否使用LLM
            progress_callback: 进度回调
            num_workers: 并行工作线程数（None表示串行，默认为CPU核心数）
            
        Returns:
            病例列表
        """
        if num_workers is None or num_workers <= 1:
            # 串行生成
            return self._generate_batch_sequential(n, constraints, use_llm, progress_callback)
        else:
            # 并行生成
            return self._generate_batch_parallel(n, constraints, use_llm, progress_callback, num_workers)
    
    def _generate_batch_sequential(
        self,
        n: int,
        constraints: Dict[str, Any] = None,
        use_llm: bool = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """串行批量生成"""
        results = []
        
        for i in range(n):
            try:
                emr = self.generate(constraints=constraints, use_llm=use_llm)
                results.append(emr)
            except Exception as e:
                print(f"生成第 {i+1} 条病例失败: {e}")
            
            if progress_callback:
                progress_callback(i + 1, n)
        
        return results
    
    def _generate_batch_parallel(
        self,
        n: int,
        constraints: Dict[str, Any] = None,
        use_llm: bool = None,
        progress_callback=None,
        num_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """并行批量生成"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        results = []
        completed_count = 0
        lock = threading.Lock()
        
        def generate_one(idx: int) -> Dict[str, Any]:
            """生成单条病例"""
            nonlocal completed_count
            try:
                emr = self.generate(constraints=constraints, use_llm=use_llm)
                with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, n)
                return emr
            except Exception as e:
                print(f"生成第 {idx+1} 条病例失败: {e}")
                with lock:
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, n)
                return None
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(generate_one, i) for i in range(n)]
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
    
    def _create_context(self, constraints: Dict[str, Any]) -> GenerationContext:
        """创建生成上下文"""
        context = GenerationContext()
        
        # 诊断编码大类（优先使用约束）
        if "diagnosis_code" in constraints:
            context.diagnosis = constraints["diagnosis_code"]
        elif self.dist_sampler:
            context.diagnosis = self.dist_sampler.sample("diagnosis_code_category")
        else:
            context.diagnosis = random.choice(["F32.9", "F41.2", "F41.1", "F39"])
        
        # 性别
        if "gender" in constraints:
            context.gender = constraints["gender"]
        elif self.dist_sampler:
            context.gender = self.dist_sampler.sample("gender")
        else:
            context.gender = random.choice(["男", "女"])
        
        # 年龄
        if "age" in constraints:
            context.age = constraints["age"]
        elif self.dist_sampler:
            context.age = self.dist_sampler.sample_age()
        else:
            context.age = random.randint(18, 60)
        
        # 科室
        if "department" in constraints:
            context.department = constraints["department"]
        elif self.dist_sampler:
            context.department = self.dist_sampler.sample("department")
        else:
            context.department = "普通精神科"
        
        # 采样症状（使用诊断编码大类）
        if self.dist_sampler:
            context.selected_symptoms = self.dist_sampler.sample_symptoms(
                diagnosis_code=context.diagnosis,
                n=random.randint(2, 4)
            )
        else:
            context.selected_symptoms = []
        
        # 采样诱因
        if self.keyword_sampler:
            context.selected_triggers = self.keyword_sampler.sample_triggers(n=2)
        else:
            context.selected_triggers = []
        
        return context
    
    def _generate_patient_id(self) -> str:
        """生成患者ID"""
        return f"{random.randint(300000000, 399999999)}"
    
    def _generate_accompanying_person(self, context: GenerationContext) -> str:
        """生成陪同人信息（按年龄和性别采样）"""
        if self.dist_sampler:
            # 使用新的按年龄和性别采样方法
            has_status, relation = self.dist_sampler.sample_accompanying_person(
                age=context.age,
                gender=context.gender
            )
            if has_status == "自来":
                return "自来"
            else:
                return f"有 关系：{relation}" if relation else "有 关系：家属"
        
        # 默认逻辑（基于年龄的合理关系）
        if random.random() < 0.15:  # 15% 自来
            return "自来"
        
        # 根据年龄选择合理的陪同人
        age = context.age
        gender = context.gender
        
        if age < 18:
            # 未成年人主要由父母陪同
            relations = ["母亲", "父亲", "父母", "家属"]
        elif age < 30:
            # 青年可能由父母、朋友、配偶陪同
            relations = ["母亲", "父亲", "朋友", "配偶", "家属"]
        elif age < 60:
            # 中年人主要由配偶、子女陪同
            if gender == "男":
                relations = ["妻子", "配偶", "家属", "子女"]
            else:
                relations = ["丈夫", "配偶", "家属", "子女"]
        else:
            # 老年人主要由子女、配偶陪同
            relations = ["子女", "儿子", "女儿", "配偶", "家属"]
        
            return f"有 关系：{random.choice(relations)}"
    
    def _generate_personal_history(
        self,
        context: GenerationContext,
        use_llm: bool = False,
    ) -> str:
        """生成个人史（按年龄采样各字段）"""
        slots = PersonalHistorySlot()
        age = context.age
        gender = context.gender
        
        # 采样各槽位（优先按年龄组采样）
        if self.dist_sampler:
            # 孕产情况（仅女性，按年龄采样）
            if gender == "女":
                slots.pregnancy_status = self.dist_sampler.sample_personal_history_field(
                    "pregnancy_status", age=age, gender=gender
                ) or "足月顺产"
            
            # 发育情况（按年龄采样）
            slots.development_status = self.dist_sampler.sample_personal_history_field(
                "development_status", age=age
            ) or "正常"
            
            # 婚恋情况（按年龄采样）
            slots.marriage_status = self.dist_sampler.sample_personal_history_field(
                "marriage_status", age=age
            ) or ("已婚" if age > 25 else "未婚")
            
            # 职业（按年龄采样）
            slots.occupation = self.dist_sampler.sample_personal_history_field(
                "occupation", age=age
            )
            
            # 月经情况（仅女性，按年龄采样）
            if gender == "女":
                slots.menstrual_status = self.dist_sampler.sample_personal_history_field(
                    "menstrual_status", age=age, gender=gender
                ) or "正常"
            
            # 性格（按年龄采样）
            personalities = []
            personality_pool = ["内向", "外向", "认真", "急躁", "敏感", "温和"]
            for _ in range(random.randint(1, 2)):
                p = self.dist_sampler.sample_personal_history_field("personality", age=age)
                if p and p not in personalities:
                    personalities.append(p)
            if not personalities:
                personalities = random.sample(personality_pool, random.randint(1, 2))
            slots.premorbid_personality = ",".join(personalities)
            
            # 嗜好（按年龄采样）
            slots.special_habits = self.dist_sampler.sample_personal_history_field(
                "special_habits", age=age
            ) or "无特殊嗜好"
        else:
            # 默认值
            if gender == "女":
                slots.pregnancy_status = "足月顺产"
            slots.development_status = "正常"
            slots.marriage_status = "已婚" if age > 25 else "未婚"
            slots.occupation = random.choice(["学生", "职员", "无业"])
            if gender == "女":
                slots.menstrual_status = "正常"
            slots.premorbid_personality = random.choice(["内向", "外向"])
            slots.special_habits = "无特殊嗜好"
        
        # 构建文本
        text_parts = []
        
        # 孕产情况（仅女性）
        if context.gender == "女" and slots.pregnancy_status:
            text_parts.append(f"孕产情况：{slots.pregnancy_status}")
        
        text_parts.append(f"发育情况：{slots.development_status}")
        text_parts.append(f"婚恋情况：{slots.marriage_status}")
        
        if slots.occupation:
            text_parts.append(f"工作、学习情况：{slots.occupation}")
        
        # 月经情况（仅女性）
        if context.gender == "女" and slots.menstrual_status:
            text_parts.append(f"月经情况：{slots.menstrual_status}")
        
        text_parts.append(f"病前性格：{slots.premorbid_personality}")
        text_parts.append(f"特殊嗜好：{slots.special_habits}")
        
        return "，".join(text_parts)
    
    def _generate_physical_illness(self, context: GenerationContext) -> str:
        """生成躯体疾病史"""
        if self.dist_sampler:
            has_illness = self.dist_sampler.sample("has_physical_illness")
            if has_illness == "有":
                illness = self.dist_sampler.sample("physical_illnesses")
                if illness:
                    return f"重要或相关躯体疾病史：{illness}"
        
        # 大部分无躯体疾病
        if random.random() < 0.8:
            return "重要或相关躯体疾病史：无"
        else:
            illnesses = ["高血压", "糖尿病", "甲状腺功能异常", "胃炎"]
            return f"重要或相关躯体疾病史：{random.choice(illnesses)}"
    
    def _generate_drug_allergy(self, context: GenerationContext) -> str:
        """生成药物过敏史（从映射分布采样）"""
        if self.dist_sampler:
            allergy = self.dist_sampler.sample("drug_allergy")
            if allergy:
                if allergy == "无":
                    return "药物过敏史：无"
                else:
                    return f"药物过敏史：{allergy}"
        
        # 默认无过敏
        return "药物过敏史：无"
    
    def _generate_family_history(self, context: GenerationContext) -> str:
        """生成家族史（从映射分布采样）"""
        if self.dist_sampler:
            history = self.dist_sampler.sample("family_history")
            if history:
                return f"家族史：{history}"
        
        # 默认阴性
        return "家族史：阴性"
    
    def _generate_diagnosis_code(self, context: GenerationContext) -> str:
        """生成诊断编码（基于诊断编码大类生成完整编码）"""
        # context.diagnosis 现在是诊断编码大类，如 F32.9, F41.2
        base_code = context.diagnosis or "F32.9"
        
        # 添加随机的细分编码
        if "." in base_code:
            parts = base_code.split(".")
            main = parts[0]
            sub = parts[1]
            # 生成完整编码，如 F32.900, F41.200x002
            full_code = f"{main}.{sub}00"
        else:
            full_code = f"{base_code}.900"
        
        return full_code
    
    def _generate_diagnosis_name(self, context: GenerationContext) -> str:
        """生成诊断名称（基于诊断编码大类，从诊断映射文件读取）"""
        diag_code = context.diagnosis or "F32.9"
        
        # 从诊断编码映射获取名称
        code_to_name = self.diagnosis_mapping.get("diagnosis_code_to_name", {})
        if diag_code in code_to_name:
            return code_to_name[diag_code]
        
        # 如果映射中没有，返回编码本身
        return f"诊断{diag_code}"
    
    def _get_overall_diagnosis(self, context: GenerationContext) -> str:
        """获取总体诊断分类（根据诊断编码推断）"""
        diag_code = context.diagnosis or "F32.9"
        
        # 根据 ICD-10 编码前缀推断 OverallDiagnosis
        if diag_code.startswith("F32") or diag_code.startswith("F33"):
            return "Depression"
        elif diag_code.startswith("F41.2"):
            return "Mix"  # 焦虑抑郁状态
        elif diag_code.startswith("F41"):
            return "Anxiety"
        elif diag_code.startswith("F42"):
            return "OCD"
        elif diag_code.startswith("F51") or diag_code.startswith("G47"):
            return "Sleep"
        elif diag_code.startswith("F20"):
            return "Schizophrenia"
        elif diag_code.startswith("F39"):
            return "Mood"
        else:
            return "Other"
    
    def _extract_duration_from_text(self, text: str) -> Optional[str]:
        """从文本中提取病程"""
        import re
        match = re.search(r'(\d+)\s*(年|月|周)', text)
        if match:
            return f"{match.group(1)}{match.group(2)}"
        return None
    
    def save_generated_emrs(
        self,
        emrs: List[Dict[str, Any]],
        filepath: Path = None,
    ):
        """保存生成的病例"""
        if filepath is None:
            filepath = Config.OUTPUTS_DIR / "generated_emrs.json"
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(emrs, f, ensure_ascii=False, indent=2)
        
        print(f"生成的病例已保存到: {filepath}")
