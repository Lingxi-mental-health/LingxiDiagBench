"""
现病史生成器 - 根据上下文信息生成现病史
"""

import random
from typing import List, Optional, Dict, Any

from ..analyzers.distribution_analyzer import DistributionSampler
from ..analyzers.keyword_analyzer import KeywordSampler
from ..utils.llm_client import LLMClient
from ..extractors.schemas import GenerationContext


class PresentIllnessGenerator:
    """现病史生成器"""
    
    def __init__(
        self,
        distribution_sampler: DistributionSampler = None,
        keyword_sampler: KeywordSampler = None,
        llm_client: LLMClient = None,
    ):
        """
        初始化现病史生成器
        
        Args:
            distribution_sampler: 分布采样器
            keyword_sampler: 关键词采样器
            llm_client: LLM客户端
        """
        self.dist_sampler = distribution_sampler
        self.keyword_sampler = keyword_sampler
        self.llm_client = llm_client
    
    def generate(
        self,
        context: GenerationContext,
        use_llm: bool = True,
    ) -> str:
        """
        生成现病史
        
        Args:
            context: 生成上下文
            use_llm: 是否使用LLM生成
            
        Returns:
            生成的现病史文本
        """
        # 采样目标长度
        target_length = None
        if self.dist_sampler:
            target_length = self.dist_sampler.sample_target_length(
                diagnosis_code=context.diagnosis,
                field="present_illness"
            )
        
        if use_llm and self.llm_client:
            return self._generate_with_llm(context, target_length)
        else:
            return self._generate_with_templates(context)
    
    def _generate_with_llm(self, context: GenerationContext, target_length: int = None) -> str:
        """使用LLM生成现病史"""
        # 采样关键词
        keywords = self._sample_keywords(context)
        triggers = self._sample_triggers(diagnosis_code=context.diagnosis)
        
        # 构建提示
        prompt = self._build_llm_prompt(context, keywords, triggers, target_length)
        
        # 根据目标长度调整描述
        length_instruction = ""
        if target_length:
            length_instruction = f"\n7. 现病史内容（不含'现病史：'前缀）控制在{target_length-20}到{target_length+20}个字左右"
        
        system_prompt = f"""你是一个专业的精神科医生，擅长撰写规范的电子病历。
请根据提供的信息生成现病史，要求：
1. 符合临床病历书写规范
2. 内容连贯、逻辑清晰
3. 包含发病时间、诱因、主要症状、伴随症状、诊治经过等
4. 适当提及睡眠、饮食情况
5. 结尾说明就诊原因
6. 只输出现病史内容，以"现病史："开头{length_instruction}"""

        try:
            result = self.llm_client.generate_text(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=800,
            )
            
            if result is None:
                return self._generate_with_templates(context)
            
            result = result.strip()
            if not result.startswith("现病史"):
                result = f"现病史：{result}"
            
            return result
        except Exception as e:
            print(f"LLM生成失败: {e}")
            return self._generate_with_templates(context)
    
    def _build_llm_prompt(
        self,
        context: GenerationContext,
        keywords: List[str],
        triggers: List[str],
        target_length: int = None,
    ) -> str:
        """构建LLM提示"""
        # 主诉信息
        chief_complaint = context.chief_complaint
        symptoms = context.selected_symptoms or []
        if chief_complaint:
            symptoms = chief_complaint.symptoms or symptoms
        
        # 病程信息
        duration = ""
        if chief_complaint and chief_complaint.duration:
            duration = chief_complaint.duration
        
        # 长度要求
        length_requirement = ""
        if target_length:
            length_requirement = f"\n7. 整体字数控制在{target_length-20}到{target_length+20}个字左右"
        
        prompt = f"""请根据以下信息生成一段精神科现病史：

患者基本信息：
- 年龄：{context.age}岁
- 性别：{context.gender}
- 诊断方向：{context.diagnosis}

主诉症状：{', '.join(symptoms) if symptoms else '待定'}
病程时长：{duration or '待定'}
可能的诱因：{', '.join(triggers) if triggers else '无明显诱因'}

请使用以下关键词（可选择性使用）：
{', '.join(keywords)}

要求：
1. 以"现病史：患者..."开头
2. 描述发病时间和诱因
3. 描述主要症状及其演变
4. 提及睡眠、饮食情况
5. 如有既往诊疗，简单提及
6. 以就诊原因结尾{length_requirement}"""

        return prompt
    
    def _generate_with_templates(self, context: GenerationContext) -> str:
        """使用模板生成现病史"""
        templates = PresentIllnessTemplates()
        
        # 获取症状
        symptoms = context.selected_symptoms or []
        if context.chief_complaint:
            symptoms = context.chief_complaint.symptoms or symptoms
        
        if not symptoms:
            symptoms = ["情绪问题", "睡眠问题"]
        
        # 获取病程
        duration = "数月"
        if context.chief_complaint and context.chief_complaint.duration:
            duration = context.chief_complaint.duration
        
        # 采样诱因（按诊断编码）
        triggers = self._sample_triggers(diagnosis_code=context.diagnosis)
        
        # 构建现病史
        present_illness = templates.build(
            diagnosis=context.diagnosis or "Other",
            symptoms=symptoms,
            duration=duration,
            triggers=triggers,
            gender=context.gender,
        )
        
        return present_illness
    
    def _sample_keywords(self, context: GenerationContext) -> List[str]:
        """采样关键词"""
        if self.keyword_sampler:
            return self.keyword_sampler.sample_present_illness_keywords(
                diagnosis_code=context.diagnosis,
                n=10
            )
        return []
    
    def _sample_triggers(self, diagnosis_code: str = None) -> List[str]:
        """采样诱因（按诊断编码）"""
        if self.keyword_sampler:
            triggers = self.keyword_sampler.sample_triggers(diagnosis_code=diagnosis_code, n=2)
            if triggers:
                return triggers
        
        # 默认诱因
        default_triggers = [
            "无明显诱因",
            "学习压力大",
            "工作压力大",
            "家庭问题",
            "感情问题",
            "人际关系问题",
        ]
        
        if random.random() < 0.3:
            return ["无明显诱因"]
        else:
            return random.sample(default_triggers[1:], random.randint(1, 2))


class PresentIllnessTemplates:
    """现病史模板库"""
    
    # 开头模板
    ONSET_TEMPLATES = [
        "患者{duration}无明显诱因下出现{main_symptom}",
        "患者{duration}因{trigger}出现{main_symptom}",
        "患者近{duration}来逐渐出现{main_symptom}",
        "{duration}，患者开始出现{main_symptom}",
    ]
    
    # 症状描述模板
    SYMPTOM_TEMPLATES = [
        "表现为{symptoms}",
        "主要表现为{symptoms}",
        "伴有{symptoms}",
        "同时有{symptoms}",
    ]
    
    # 睡眠描述
    SLEEP_TEMPLATES = {
        "Depression": ["眠差", "入睡困难", "早醒", "眠浅多梦", "睡眠减少"],
        "Anxiety": ["入睡困难", "睡眠不安", "眠浅", "难以入睡"],
        "Mix": ["睡眠差", "眠浅多梦", "入睡困难"],
        "Other": ["睡眠不规律", "日夜颠倒", "睡眠问题"],
    }
    
    # 饮食描述
    APPETITE_TEMPLATES = {
        "Depression": ["胃口差", "食欲减退", "不想吃饭"],
        "Anxiety": ["胃口一般", "进食尚可"],
        "Mix": ["食欲下降", "胃口差"],
        "Other": ["胃口一般", "进食尚可"],
    }
    
    # 结尾模板
    ENDING_TEMPLATES = [
        "故来我院门诊就诊。",
        "为进一步治疗来我院门诊。",
        "故本次求助我院门诊。",
        "遂来我院就诊。",
    ]
    
    # 诊治情况模板
    TREATMENT_TEMPLATES = [
        "诊治情况:无",
        "诊治情况:曾在外院就诊，效果不佳。",
        "既往未系统诊治。",
    ]
    
    def build(
        self,
        diagnosis: str,
        symptoms: List[str],
        duration: str,
        triggers: List[str] = None,
        gender: str = None,
    ) -> str:
        """
        构建现病史
        
        Args:
            diagnosis: 诊断类型
            symptoms: 症状列表
            duration: 病程时长
            triggers: 诱因列表
            gender: 性别
            
        Returns:
            现病史文本
        """
        parts = []
        
        # 开头
        main_symptom = symptoms[0] if symptoms else "情绪问题"
        trigger = triggers[0] if triggers and triggers[0] != "无明显诱因" else None
        
        if trigger:
            onset = random.choice(self.ONSET_TEMPLATES[:2])
            onset = onset.format(
                duration=duration,
                trigger=trigger,
                main_symptom=main_symptom
            )
        else:
            onset = self.ONSET_TEMPLATES[0].format(
                duration=duration,
                main_symptom=main_symptom
            )
        
        parts.append("现病史：" + onset)
        
        # 其他症状
        if len(symptoms) > 1:
            other_symptoms = "、".join(symptoms[1:])
            symptom_desc = random.choice(self.SYMPTOM_TEMPLATES)
            parts.append(symptom_desc.format(symptoms=other_symptoms))
        
        # 睡眠情况
        sleep_options = self.SLEEP_TEMPLATES.get(diagnosis, self.SLEEP_TEMPLATES["Other"])
        sleep_desc = random.choice(sleep_options)
        parts.append(f"起病来，{sleep_desc}")
        
        # 饮食情况
        appetite_options = self.APPETITE_TEMPLATES.get(diagnosis, self.APPETITE_TEMPLATES["Other"])
        appetite_desc = random.choice(appetite_options)
        parts.append(f"，{appetite_desc}")
        
        # 否认危险行为（大部分情况）
        if random.random() < 0.7:
            parts.append("。否认自伤、自杀等行为异常")
        
        # 结尾
        ending = random.choice(self.ENDING_TEMPLATES)
        parts.append("。" + ending)
        
        # 诊治情况
        treatment = random.choice(self.TREATMENT_TEMPLATES)
        parts.append(" " + treatment)
        
        return "".join(parts)
