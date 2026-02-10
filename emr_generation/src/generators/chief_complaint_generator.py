"""
主诉生成器 - 根据分布和关键词生成主诉
"""

import random
from typing import List, Optional, Dict, Any

from ..analyzers.distribution_analyzer import DistributionSampler
from ..analyzers.keyword_analyzer import KeywordSampler
from ..utils.llm_client import LLMClient
from ..extractors.schemas import GenerationContext


class ChiefComplaintGenerator:
    """主诉生成器"""
    
    def __init__(
        self,
        distribution_sampler: DistributionSampler = None,
        keyword_sampler: KeywordSampler = None,
        llm_client: LLMClient = None,
    ):
        """
        初始化主诉生成器
        
        Args:
            distribution_sampler: 分布采样器
            keyword_sampler: 关键词采样器
            llm_client: LLM客户端（可选，用于润色）
        """
        self.dist_sampler = distribution_sampler
        self.keyword_sampler = keyword_sampler
        self.llm_client = llm_client
    
    def generate(
        self,
        context: GenerationContext,
        use_llm: bool = False,
    ) -> str:
        """
        生成主诉
        
        Args:
            context: 生成上下文
            use_llm: 是否使用LLM润色
            
        Returns:
            生成的主诉文本
        """
        # 采样目标长度
        target_length = None
        if self.dist_sampler:
            target_length = self.dist_sampler.sample_target_length(
                diagnosis_code=context.diagnosis,
                field="chief_complaint"
            )
        
        # 采样症状
        symptoms = self._sample_symptoms(context)
        
        # 采样病程（按诊断编码）
        duration = self._sample_duration(diagnosis_code=context.diagnosis)
        
        # 采样加重时间（可选）
        exacerbation = self._sample_exacerbation() if random.random() < 0.3 else None
        
        # 构建主诉
        chief_complaint = self._build_chief_complaint(symptoms, duration, exacerbation)
        
        # 如果启用LLM，进行润色（传入目标长度）
        if use_llm and self.llm_client:
            chief_complaint = self._polish_with_llm(chief_complaint, context, target_length)
        
        return chief_complaint
    
    def _sample_symptoms(self, context: GenerationContext) -> List[str]:
        """采样症状（从映射分布采样）"""
        # 根据诊断编码大类采样症状
        diagnosis_code = context.diagnosis  # 诊断编码大类，如 F32.9
        
        if self.dist_sampler:
            # 优先从诊断编码对应的症状分布采样
            symptoms = self.dist_sampler.sample_symptoms(
                diagnosis_code=diagnosis_code,
                n=random.randint(2, 7)
            )
            if symptoms:
                return symptoms
            
            # 如果没有诊断特定的症状，从通用症状分布采样
            symptoms = self.dist_sampler.sample_symptoms(
                diagnosis_code=None,
                n=random.randint(2, 7)
            )
            if symptoms:
                return symptoms
        
        # 最后的回退：使用通用默认症状
        default_symptoms = ["情绪问题", "睡眠问题", "紧张"]
        n = random.randint(2, 7)
        return random.sample(default_symptoms, n)
    
    def _sample_duration(self, diagnosis_code: str = None) -> str:
        """采样病程时长（按诊断编码）"""
        if self.keyword_sampler:
            return self.keyword_sampler.sample_time_expression(diagnosis_code=diagnosis_code)
        
        # 默认病程
        units = ["年", "月"]
        unit = random.choice(units)
        
        if unit == "年":
            num = random.randint(1, 5)
        else:
            num = random.randint(1, 12)
        
        return f"{num}{unit}"
    
    def _sample_exacerbation(self) -> Optional[str]:
        """采样加重时间"""
        units = ["月", "周"]
        unit = random.choice(units)
        
        if unit == "月":
            num = random.randint(1, 3)
        else:
            num = random.randint(1, 4)
        
        return f"加重{num}{unit}"
    
    def _build_chief_complaint(
        self,
        symptoms: List[str],
        duration: str,
        exacerbation: Optional[str] = None,
    ) -> str:
        """构建主诉文本"""
        # 症状描述（处理空症状列表）
        if not symptoms:
            symptom_text = "情绪问题"  # 默认症状
        elif len(symptoms) == 1:
            symptom_text = symptoms[0]
        elif len(symptoms) == 2:
            symptom_text = f"{symptoms[0]}、{symptoms[1]}"
        else:
            symptom_text = "、".join(symptoms[:-1]) + f"伴{symptoms[-1]}"
        
        # 构建主诉
        chief_complaint = f"主诉：{symptom_text} {duration}"
        
        # 添加加重描述
        if exacerbation:
            chief_complaint += f"，{exacerbation}"
        
        return chief_complaint
    
    def _polish_with_llm(
        self,
        chief_complaint: str,
        context: GenerationContext,
        target_length: int = None,
    ) -> str:
        """使用LLM润色主诉"""
        # 构建长度要求
        length_requirement = ""
        if target_length:
            length_requirement = f"\n5. 主诉内容（不含'主诉：'前缀）控制在{target_length-5}到{target_length+5}个字左右"
        
        prompt = f"""/no_think 请将以下主诉润色成更自然、更符合临床规范的表述。

原始主诉：{chief_complaint}

患者信息：
- 年龄：{context.age}岁
- 性别：{context.gender}
- 诊断方向：{context.diagnosis}

要求：
1. 保持主诉的核心信息不变
2. 使表述更加专业、规范
3. 只输出润色后的主诉，以"主诉："开头
4. 不要添加额外的信息{length_requirement}"""

        try:
            result = self.llm_client.generate_text(
                prompt=prompt,
                temperature=0.5,
                max_tokens=200,
            )
            
            # 如果结果为空，回退到原始主诉
            if not result or not result.strip():
                return chief_complaint
            
            # 确保以"主诉："开头
            result = result.strip()
            
            # 检查是否返回了无效的 HTML 或错误响应
            if result.startswith("<!DOCTYPE") or result.startswith("<html") or "ERROR" in result[:100]:
                print(f"LLM返回无效响应，回退到原始主诉")
                return chief_complaint
            
            # 检查润色后的内容是否有效（不能只是"主诉："）
            if not result.startswith("主诉"):
                result = f"主诉：{result}"
            
            # 验证内容是否有效
            cleaned_result = result.replace("主诉：", "").replace("主诉:", "").strip()
            if not cleaned_result:
                return chief_complaint
            
            return result
        except Exception as e:
            # 只打印错误的前100个字符，避免日志过长
            error_msg = str(e)[:100] if len(str(e)) > 100 else str(e)
            print(f"LLM润色失败: {error_msg}")
            return chief_complaint


class ChiefComplaintTemplates:
    """主诉模板库"""
    
    # 常见主诉模板
    TEMPLATES = [
        "主诉：{症状1}、{症状2} {时长}",
        "主诉：{症状1}伴{症状2} {时长}",
        "主诉：{症状1}、{症状2}伴{症状3} {时长}",
        "主诉：{症状1}，{症状2} {时长}，加重{加重时长}",
        "主诉：{症状1}、{症状2} {时长}，总病程{总病程}",
    ]
    
    # 症状-诊断对应关系
    DIAGNOSIS_SYMPTOMS = {
        "Depression": [
            ["情绪低落", "兴趣减退"],
            ["闷闷不乐", "有对生活没意思"],
            ["情绪低落", "睡眠差"],
            ["不开心", "乏力", "不想动"],
        ],
        "Anxiety": [
            ["紧张不安", "入睡困难"],
            ["焦虑", "心慌"],
            ["紧张心慌", "眠差"],
            ["多思多虑", "睡眠差"],
        ],
        "Mix": [
            ["情绪低落", "焦虑烦躁", "睡眠不好"],
            ["闷闷不乐", "多虑", "入睡困难"],
            ["焦虑抑郁", "睡眠障碍"],
        ],
        "Other": [
            ["情绪时高时低", "反复行为"],
            ["耳闻人语", "被监视感"],
            ["行为异常", "睡眠问题"],
        ],
    }
    
    @classmethod
    def get_random_template(cls, diagnosis: str = None) -> tuple:
        """
        获取随机模板和症状组合
        
        Returns:
            (模板, 症状列表)
        """
        template = random.choice(cls.TEMPLATES)
        
        symptom_pool = cls.DIAGNOSIS_SYMPTOMS.get(diagnosis, cls.DIAGNOSIS_SYMPTOMS["Other"])
        symptoms = random.choice(symptom_pool)
        
        return template, symptoms
