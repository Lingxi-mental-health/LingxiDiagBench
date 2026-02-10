"""
Patient Agent模块

包含不同版本的患者代理实现：
- patient_v1: 单阶段回复逻辑
- patient_cot: 两阶段回复逻辑（分类+生成）
- patient_api: 患者API工具函数
"""

from .patient_v1 import Patient as PatientV1
from .patient_cot import Patient as PatientCoT

__all__ = ['PatientV1', 'PatientCoT']

