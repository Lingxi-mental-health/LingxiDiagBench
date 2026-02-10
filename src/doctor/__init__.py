"""
Doctor Agent模块

包含不同版本的医生代理实现：
- doctor_base: 基础版本（无诊断树）
- doctor_v1: 传统诊断树版本
- doctor_v2: 阶段式诊断树版本
- diagtree_v1: V1版本的诊断树实现
- diagtree_v2: V2版本的诊断树实现
"""

from .doctor_base import DoctorBase
from .doctor_v1 import Doctor as DoctorV1
from .doctor_v2 import Doctor as DoctorV2

__all__ = ['DoctorBase', 'DoctorV1', 'DoctorV2']

