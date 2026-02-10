"""
统一阶段式诊断树 - 替代多个疾病特定的诊断树
基于临床实践的阶段式问诊流程
"""

import json
import random
import time
import os
import sys
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import llm_tools_api

class DiagnosticPhase(Enum):
    """诊断阶段 - 基于APA临床指南"""
    SCREENING = "screening"       # 筛查阶段 (3-5个问题)
    ASSESSMENT = "assessment"     # 评估阶段 (5-8个问题) 
    DEEPDIVE = "deepdive"        # 深入阶段 (3-6个问题)
    RISK = "risk"                # 风险评估 (2-4个问题)
    CLOSURE = "closure"          # 总结阶段 (1-2个问题)

@dataclass
class PhaseConfig:
    """阶段配置"""
    name: str
    max_questions: int
    min_questions: int
    mandatory_topics: List[str]
    optional_topics: List[str]
    exit_conditions: List[str]
    next_phase_triggers: Dict[str, str]

class DiagTree:
    """统一诊断树 - 基于阶段的问诊流程"""
    
    def __init__(self, model_name, prompts={}):
        # 保持与原有接口兼容
        self.model_name = model_name
        self.doctor_promot_path = prompts.get('doctor', '')
        self.diagtree_path = prompts.get('diagtree', '')
        
        # 新的阶段式属性
        self.current_phase = DiagnosticPhase.SCREENING
        self.phase_configs = self._initialize_phases()
        self.conversation_data = {
            "patient_responses": [],
            "identified_issues": set(),
            "risk_level": 0,
            "confidence_score": 0.0
        }
        self.completed_phases = set()
        self.dialstate = []  # 保持与原接口兼容
        self.topic_end = []  # 保持与原接口兼容
        self.topic_sequence = []
        
    def _initialize_phases(self) -> Dict[DiagnosticPhase, PhaseConfig]:
        """初始化各阶段配置"""
        configs = {
            DiagnosticPhase.SCREENING: PhaseConfig(
                name="初步筛查",
                max_questions=5,
                min_questions=3,
                mandatory_topics=[
                    "主要困扰症状",
                    "症状持续时间和严重程度"
                ],
                optional_topics=[
                    "既往求助经历",
                    "症状变化趋势"
                ],
                exit_conditions=["明确诊断方向", "发现紧急情况"],
                next_phase_triggers={
                    "情绪问题": "assessment",
                    "精神病性症状": "risk", 
                    "自杀风险": "risk"
                }
            ),
            
            DiagnosticPhase.ASSESSMENT: PhaseConfig(
                name="详细评估", 
                max_questions=8,
                min_questions=5,
                mandatory_topics=[
                    "核心症状详细评估",
                    "症状严重程度",
                    "功能损害评估"
                ],
                optional_topics=[
                    "伴随症状",
                    "诱发因素",
                    "既往病史",
                    "治疗反应",
                    "社会支持"
                ],
                exit_conditions=["症状评估完整", "诊断明确"],
                next_phase_triggers={
                    "需要深入了解": "deepdive",
                    "风险因素存在": "risk",
                    "评估完整": "closure"
                }
            ),
            
            DiagnosticPhase.DEEPDIVE: PhaseConfig(
                name="深入探讨",
                max_questions=6, 
                min_questions=3,
                mandatory_topics=[
                    "特殊症状详询",
                    "深层病因探讨"
                ],
                optional_topics=[
                    "家族史",
                    "童年创伤",
                    "人格特征", 
                    "应对方式",
                    "环境因素",
                    "共病评估"
                ],
                exit_conditions=["深层原因明确", "特殊症状确认"],
                next_phase_triggers={
                    "发现风险": "risk",
                    "评估完成": "closure"
                }
            ),
            
            DiagnosticPhase.RISK: PhaseConfig(
                name="风险评估",
                max_questions=4,
                min_questions=2, 
                mandatory_topics=[
                    "自杀风险评估",
                    "自伤行为评估"
                ],
                optional_topics=[
                    "冲动控制",
                    "支持系统",
                    "安全计划",
                    "紧急联系人"
                ],
                exit_conditions=["风险评估完成"],
                next_phase_triggers={
                    "低风险": "closure",
                    "中高风险": "closure"
                }
            ),
            
            DiagnosticPhase.CLOSURE: PhaseConfig(
                name="诊断总结",
                max_questions=2,
                min_questions=1,
                mandatory_topics=[
                    "关键信息确认"
                ],
                optional_topics=[
                    "患者关切",
                    "治疗期望"
                ],
                exit_conditions=["诊断完成"],
                next_phase_triggers={}
            )
        }
        
        return configs
    
    def load_tree(self):
        """加载树 - 现在使用阶段式配置而不是JSON文件"""
        self.topic_sequence = self.dynamic_select()
        print(f"已加载统一阶段式诊断树，共{len(self.topic_sequence)}个话题")
    
    def dynamic_select(self) -> List[str]:
        """生成动态话题序列 - 替代原有的基于疾病的方法"""
        sequence = []
        
        # 按阶段生成话题
        for phase in DiagnosticPhase:
            config = self.phase_configs[phase]
            
            # 添加必问话题
            for topic in config.mandatory_topics:
                prompt = self.prompt_gen(topic)
                sequence.append(prompt)
            
            # 选择部分可选话题
            optional_count = min(
                config.max_questions - len(config.mandatory_topics),
                len(config.optional_topics)
            )
            
            if optional_count > 0:
                selected_optional = self._select_optional_topics(
                    config.optional_topics, 
                    optional_count,
                    phase
                )
                for topic in selected_optional:
                    prompt = self.prompt_gen(topic)
                    sequence.append(prompt)
        
        self.dialstate = sequence  # 保持兼容性
        return sequence
    
    def _select_optional_topics(self, optional_topics: List[str], count: int, phase: DiagnosticPhase) -> List[str]:
        """智能选择可选话题"""
        if count >= len(optional_topics):
            return optional_topics
        
        # 根据阶段特点选择
        if phase == DiagnosticPhase.SCREENING:
            # 筛查阶段优先快速识别
            priority = [t for t in optional_topics if any(word in t for word in ["经历", "趋势"])]
            others = [t for t in optional_topics if t not in priority]
            selected = priority[:count]
            if len(selected) < count:
                selected.extend(random.sample(others, min(count - len(selected), len(others))))
        elif phase == DiagnosticPhase.RISK:
            # 风险阶段优先安全相关
            priority = [t for t in optional_topics if any(word in t for word in ["支持", "安全", "联系"])]
            others = [t for t in optional_topics if t not in priority]
            selected = priority[:count]
            if len(selected) < count:
                selected.extend(random.sample(others, min(count - len(selected), len(others))))
        else:
            # 其他阶段随机选择
            selected = random.sample(optional_topics, count)
            
        return selected
    
    def prompt_gen(self, option):
        """根据话题生成LLM提示 - 保持接口兼容"""
        return f"询问患者有关{option}，不要包含其他话题和问题"
    
    def is_topic_end(self, current_state, input_history):
        """判断话题是否结束 - 保持接口兼容但使用新逻辑"""
        if not input_history:
            return False, 0, 0
            
        # 基于回应质量判断
        last_response = input_history[-1] if isinstance(input_history, list) else str(input_history)
        
        # 分析回应内容
        analysis = self._analyze_response(last_response)
        
        # 检测紧急情况立即结束当前话题
        if analysis["risk_level"] >= 2:
            self.topic_end.append(True)
            return True, 0, 0
        
        # 基于回应长度和质量判断
        if len(last_response) < 15:
            should_end = False  # 回应太短，继续
        elif len(last_response) > 80:
            should_end = True   # 回应充分，可以结束
        else:
            should_end = random.random() > 0.4  # 中等长度，倾向结束
        
        self.topic_end.append(should_end)
        
        # 应用强制结束逻辑
        final_decision = self.force_topic_end() if len(self.topic_end) >= 6 else should_end
        
        return final_decision, 0, 0
    
    def _analyze_response(self, response: str) -> Dict:
        """分析患者回应"""
        issues = set()
        risk_level = 0
        confidence = 0.5
        
        # 风险关键词检测
        high_risk = ["自杀", "想死", "结束生命", "不想活了"]
        medium_risk = ["绝望", "无助", "痛苦得不行", "受不了"]
        
        response_lower = response.lower()
        
        for keyword in high_risk:
            if keyword in response_lower:
                risk_level = 3
                issues.add("高风险")
                break
                
        if risk_level < 3:
            for keyword in medium_risk:
                if keyword in response_lower:
                    risk_level = max(risk_level, 2)
                    issues.add("中等风险")
                    
        # 症状关键词检测
        symptoms = {
            "抑郁": "抑郁症状", "焦虑": "焦虑症状", 
            "失眠": "睡眠问题", "幻觉": "精神病性症状"
        }
        
        for keyword, issue in symptoms.items():
            if keyword in response_lower:
                issues.add(issue)
                confidence += 0.1
                
        return {
            "issues": issues,
            "risk_level": risk_level,
            "confidence": min(confidence, 1.0)
        }
    
    def force_topic_end(self):
        """强制话题结束逻辑 - 保持原有逻辑"""
        if len(self.topic_end) < 6:
            return self.topic_end[-1]
        else:
            if self.topic_end[-2] == False:
                if self.topic_end[-3] == False:
                    self.topic_end[-1] = True
                    return True
                elif random.randint(0, 2) == 1:
                    return self.topic_end[-1]
                else:
                    self.topic_end[-1] = True
                    return True
            else:
                if self.topic_end[-1] == True:
                    return True
                else:
                    if random.randint(0, 2) == 1:
                        self.topic_end[-1] = True
                        return True
                    else:
                        return self.topic_end[-1]
    
    def is_end(self, current_state):
        """判断整个问诊是否结束 - 保持接口兼容"""
        # 基于话题序列判断
        if hasattr(self, 'dialstate') and self.dialstate and current_state == self.dialstate[-1]:
            return True
            
        return False
    
    # 保持原有接口兼容的方法
    def parse_experience(self, input):
        """解析患者经历 - 保持接口兼容"""
        return llm_tools_api.api_parse_experience(self.model_name, input)
    
    def topic_detection(self, topic_seq, parse_topic):
        """话题检测 - 保持接口兼容"""
        result = []
        for topic in topic_seq:
            prompt = f"判断在话题集合'{topic}'中是否有表达意思与'{parse_topic}'相似或者相同的，如果包含输出'是'，如果不包含输出'否'。"
            answer, x = llm_tools_api.api_topic_detection(self.model_name, prompt)
            if '否' in answer:
                result.append(False)
            elif '是' in answer:
                result.append(True)
            else:
                result.append(False)  # 默认为否
        return result, 0, 0  # 保持返回格式兼容
