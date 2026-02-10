"""
LLM 提取器 - 使用大语言模型提取复杂结构化信息
"""

from typing import Optional, List, Dict, Any

from .schemas import PersonalHistorySlot, ChiefComplaintSlot, PresentIllnessSlot
from ..utils.llm_client import LLMClient


class LLMExtractor:
    """基于 LLM 的信息提取器"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        model: str = None,
    ):
        """
        初始化 LLM 提取器
        
        Args:
            host: LLM 服务地址
            port: LLM 服务端口
            model: 模型名称
        """
        self.client = LLMClient(host=host, port=port, model=model)
    
    def extract_personal_history(self, text: str) -> Optional[PersonalHistorySlot]:
        """
        从个人史文本中提取结构化信息
        
        Args:
            text: 个人史原始文本
            
        Returns:
            PersonalHistorySlot 实例
        """
        instruction = """
请仔细分析个人史文本，提取以下信息：
- pregnancy_status: 孕产情况（如足月顺产、早产、剖腹产等）
- development_status: 发育情况（正常、迟缓等）
- marriage_status: 婚恋情况（已婚、未婚、离异、未恋等）
- children_info: 子女情况
- occupation: 工作/学习情况
- menstrual_status: 月经情况（仅女性）
- premorbid_personality: 病前性格
- special_habits: 特殊嗜好
- special_experience: 特殊经历（如父母离异、从小跟奶奶生活等）
"""
        return self.client.extract_structured(
            text=text,
            output_schema=PersonalHistorySlot,
            instruction=instruction,
            temperature=0.2,
        )
    
    def extract_chief_complaint(self, text: str) -> Optional[ChiefComplaintSlot]:
        """
        从主诉文本中提取结构化信息
        
        Args:
            text: 主诉原始文本
            
        Returns:
            ChiefComplaintSlot 实例
        """
        instruction = """
请仔细分析主诉文本，提取以下信息：
- symptoms: 主要症状列表（如入睡困难、紧张不安、情绪低落等）
- duration: 病程时长（如1年、2月）
- total_duration: 总病程（如有）
- exacerbation_duration: 加重时长（如有）

注意：症状应该是具体的医学症状描述。
"""
        return self.client.extract_structured(
            text=text,
            output_schema=ChiefComplaintSlot,
            instruction=instruction,
            temperature=0.2,
        )
    
    def extract_present_illness(self, text: str) -> Optional[PresentIllnessSlot]:
        """
        从现病史文本中提取结构化信息
        
        Args:
            text: 现病史原始文本
            
        Returns:
            PresentIllnessSlot 实例
        """
        instruction = """
请仔细分析现病史文本，提取以下信息：
- onset_time: 发病时间（如1年前、2022年3月）
- triggers: 发病诱因列表（如学习压力、家庭问题、工作压力、感情问题等）
- main_symptoms: 主要症状表现列表
- accompanying_symptoms: 伴随症状列表
- sleep_status: 睡眠情况描述
- appetite_status: 饮食/胃口情况
- previous_treatment: 既往诊治情况
- self_harm_info: 自杀/自残相关信息
- visit_reason: 本次就诊原因

注意：
1. 诱因应该是导致发病或加重的原因
2. 主要症状和伴随症状应该分开
3. 如果提到自杀、自残、消极想法，请提取相关信息
"""
        return self.client.extract_structured(
            text=text,
            output_schema=PresentIllnessSlot,
            instruction=instruction,
            temperature=0.2,
        )
    
    def extract_all(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        从一条记录中提取所有需要LLM处理的信息
        
        Args:
            record: 原始记录字典
            
        Returns:
            提取结果字典
        """
        result = {}
        
        # 提取个人史
        personal_history_text = record.get("PersonalHistory", "")
        if personal_history_text:
            result["personal_history_llm"] = self.extract_personal_history(
                personal_history_text
            )
        
        # 提取主诉
        chief_complaint_text = record.get("ChiefComplaint", "")
        if chief_complaint_text:
            result["chief_complaint_llm"] = self.extract_chief_complaint(
                chief_complaint_text
            )
        
        # 提取现病史
        present_illness_text = record.get("PresentIllnessHistory", "")
        if present_illness_text:
            result["present_illness_llm"] = self.extract_present_illness(
                present_illness_text
            )
        
        return result


class BatchLLMExtractor:
    """批量 LLM 提取器"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        model: str = None,
    ):
        self.extractor = LLMExtractor(host=host, port=port, model=model)
    
    def extract_batch(
        self,
        records: List[Dict[str, Any]],
        fields: List[str] = None,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        批量提取记录
        
        Args:
            records: 记录列表
            fields: 要提取的字段列表，默认全部
            progress_callback: 进度回调函数
            
        Returns:
            提取结果列表
        """
        if fields is None:
            fields = ["personal_history", "chief_complaint", "present_illness"]
        
        results = []
        
        for i, record in enumerate(records):
            result = {"patient_id": record.get("patient_id")}
            
            try:
                if "personal_history" in fields:
                    text = record.get("PersonalHistory", "")
                    if text:
                        result["personal_history"] = self.extractor.extract_personal_history(text)
                
                if "chief_complaint" in fields:
                    text = record.get("ChiefComplaint", "")
                    if text:
                        result["chief_complaint"] = self.extractor.extract_chief_complaint(text)
                
                if "present_illness" in fields:
                    text = record.get("PresentIllnessHistory", "")
                    if text:
                        result["present_illness"] = self.extractor.extract_present_illness(text)
                        
            except Exception as e:
                print(f"提取记录 {i} 失败: {e}")
                result["error"] = str(e)
            
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(records))
        
        return results
