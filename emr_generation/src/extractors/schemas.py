"""
Pydantic 数据模型定义 - 定义病例各字段的结构化槽位
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class PersonalHistorySlot(BaseModel):
    """个人史结构化槽位"""
    
    # 孕产情况
    pregnancy_status: Optional[str] = Field(
        None, 
        description="孕产情况，如：足月顺产、早产、剖腹产等"
    )
    
    # 发育情况
    development_status: Optional[str] = Field(
        None,
        description="发育情况，如：正常、迟缓等"
    )
    
    # 婚恋情况
    marriage_status: Optional[str] = Field(
        None,
        description="婚恋情况，如：已婚、未婚、离异、未恋等"
    )
    
    # 子女情况
    children_info: Optional[str] = Field(
        None,
        description="子女情况，如：育有二孩、无子女等"
    )
    
    # 工作/学习情况
    occupation: Optional[str] = Field(
        None,
        description="工作学习情况，如：学生、印刷、审计工作、无业等"
    )
    
    # 月经情况（女性）
    menstrual_status: Optional[str] = Field(
        None,
        description="月经情况，如：正常、无特殊、月经容易推迟等"
    )
    
    # 病前性格
    premorbid_personality: Optional[str] = Field(
        None,
        description="病前性格，如：内向、外向、认真、急躁、敏感等"
    )
    
    # 特殊嗜好
    special_habits: Optional[str] = Field(
        None,
        description="特殊嗜好，如：吸烟史、饮酒、无特殊嗜好等"
    )
    
    # 特殊经历
    special_experience: Optional[str] = Field(
        None,
        description="特殊经历，如：5岁时父母离异、从小跟着奶奶生活等"
    )


class ChiefComplaintSlot(BaseModel):
    """主诉结构化槽位"""
    
    # 主要症状列表
    symptoms: List[str] = Field(
        default_factory=list,
        description="主要症状列表，如：入睡困难、紧张不安、情绪低落等"
    )
    
    # 病程时长
    duration: Optional[str] = Field(
        None,
        description="病程时长，如：1年、2月、3年等"
    )
    
    # 总病程（如有）
    total_duration: Optional[str] = Field(
        None,
        description="总病程时长"
    )
    
    # 加重时间（如有）
    exacerbation_duration: Optional[str] = Field(
        None,
        description="加重时长，如：加重2月"
    )


class PresentIllnessSlot(BaseModel):
    """现病史结构化槽位"""
    
    # 发病时间
    onset_time: Optional[str] = Field(
        None,
        description="发病时间，如：1年前、2022年3月"
    )
    
    # 诱因
    triggers: List[str] = Field(
        default_factory=list,
        description="发病诱因列表，如：学习压力、家庭问题、工作压力等"
    )
    
    # 主要症状表现
    main_symptoms: List[str] = Field(
        default_factory=list,
        description="主要症状表现列表"
    )
    
    # 伴随症状
    accompanying_symptoms: List[str] = Field(
        default_factory=list,
        description="伴随症状列表"
    )
    
    # 睡眠情况
    sleep_status: Optional[str] = Field(
        None,
        description="睡眠情况描述"
    )
    
    # 饮食情况
    appetite_status: Optional[str] = Field(
        None,
        description="饮食/胃口情况"
    )
    
    # 既往诊疗情况
    previous_treatment: Optional[str] = Field(
        None,
        description="既往诊治情况"
    )
    
    # 自杀/自残相关
    self_harm_info: Optional[str] = Field(
        None,
        description="自杀/自残相关信息"
    )
    
    # 就诊原因
    visit_reason: Optional[str] = Field(
        None,
        description="本次就诊原因"
    )


class PhysicalIllnessSlot(BaseModel):
    """躯体疾病史槽位"""
    
    has_illness: bool = Field(
        False,
        description="是否有躯体疾病史"
    )
    
    illnesses: List[str] = Field(
        default_factory=list,
        description="躯体疾病列表"
    )


class EMRRecord(BaseModel):
    """完整电子病历记录"""
    
    # 基础信息
    patient_id: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[Literal["男", "女"]] = None
    department: Optional[str] = None
    
    # 陪同人
    accompanying_person: Optional[str] = None
    
    # 结构化字段
    personal_history: Optional[PersonalHistorySlot] = None
    chief_complaint: Optional[ChiefComplaintSlot] = None
    present_illness: Optional[PresentIllnessSlot] = None
    physical_illness: Optional[PhysicalIllnessSlot] = None
    
    # 其他信息
    drug_allergy: Optional[str] = None
    family_history: Optional[str] = None
    
    # 诊断信息
    diagnosis_code: Optional[str] = None
    overall_diagnosis: Optional[str] = None
    diagnosis: Optional[str] = None
    
    # 量表信息
    scale_name: Optional[str] = None
    score: Optional[str] = None


class GenerationContext(BaseModel):
    """生成上下文 - 用于传递生成过程中的信息"""
    
    # 已确定的基础信息
    age: Optional[int] = None
    gender: Optional[str] = None
    department: Optional[str] = None
    diagnosis: Optional[str] = None
    
    # 已生成的结构化信息
    personal_history: Optional[PersonalHistorySlot] = None
    chief_complaint: Optional[ChiefComplaintSlot] = None
    
    # 关键词信息
    selected_symptoms: List[str] = Field(default_factory=list)
    selected_triggers: List[str] = Field(default_factory=list)
