# -*- coding: utf-8 -*-
"""
Reasoning生成器 - 用于在对话生成过程中实时生成reasoning数据

采用qwen3格式，每个对话轮次包含：
- turn: 轮次编号
- role: doctor/patient
- content: 对话内容
- think_content: 思考过程（从模型响应中提取）
- full_response: 完整响应（包含think和answer）
- tokens, prompt_tokens: token使用量
- is_greeting, is_complaint, is_diagnosis: 特殊标志
"""

import json
import re
from typing import List, Dict, Any, Optional


class ReasoningGenerator:
    """Reasoning数据生成器（qwen3格式）"""
    
    # 系统提示词模板（用于RAG/CoT阶段）
    SYSTEM_MESSAGE_TEMPLATE = """你是一位专业的心理健康顾问（医生）。你的主要任务是根据患者的描述，利用你提供的内部知识库工具（InternalKnowledgeBase）检索相关的精神疾病诊断标准和治疗建议，然后给出专业、同理心的回复。

你被授权使用以下函数：

[
  {
    "name": "InternalKnowledgeBase",
    "description": "检索内部精神病学知识库、DSM-5诊断标准或临床治疗指南，以获取特定症状或疾病的信息。",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "用于检索知识库的关键词，例如：'持续心境低落的诊断标准' 或 '双相情感障碍的最新治疗方案'"
        }
      },
      "required": [
        "query"
      ]
    }
  }
]"""
    
    def __init__(self, patient_id: str):
        """
        初始化Reasoning生成器
        
        Args:
            patient_id: 患者ID
        """
        self.patient_id = patient_id
        self.simulation_dialogue = []  # 存储所有对话轮次（qwen3格式）
        self.current_system_prompt = None  # 当前使用的system prompt
        self.turn_counter = 0  # 对话轮次计数器
    
    def extract_think_content(self, content: str) -> str:
        """从content中提取<think>标签内的内容，并清理患者部分"""
        if not content:
            return ""
        
        think_pattern = r'<think>(.*?)</think>'
        matches = re.findall(think_pattern, content, re.DOTALL)
        
        if matches:
            think_content = matches[0].strip()
            # 清理患者部分
            think_content = self._clean_patient_content_from_reasoning(think_content)
            return think_content
        return ""
    
    def _clean_patient_content_from_reasoning(self, reasoning: str) -> str:
        """
        清理 reasoning 中的患者部分，只保留医生的 reasoning
        
        Args:
            reasoning: 原始 reasoning 内容
            
        Returns:
            清理后的 reasoning（只包含医生部分），如果整个 reasoning 都是患者的则返回空字符串
        """
        if not reasoning:
            return ""
        
        reasoning = reasoning.strip()
        if not reasoning:
            return ""
        
        # 患者角色指示词
        patient_indicators = [
            "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
            "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
            "我是一名", "我是患者", "作为患者", "患者角色"
        ]
        
        # 检查是否包含患者指示词
        contains_patient_content = any(indicator in reasoning for indicator in patient_indicators)
        
        if not contains_patient_content:
            # 不包含患者内容，直接返回
            return reasoning
        
        # 包含患者内容，尝试分离医生和患者的部分
        # 策略1: 查找患者内容的分隔符（如"患者："、"患者说："等）
        patient_markers = [
            r'患者[：:]\s*',
            r'患者说[：:]\s*',
            r'作为患者[，,]\s*',
            r'我是一名.*?患者[，,。]\s*',
            r'扮演.*?患者[，,。]\s*',
            r'用户希望我扮演[，,。]\s*',
        ]
        
        # 尝试找到患者内容的开始位置
        patient_start_pos = len(reasoning)  # 默认认为患者内容在最后
        for marker in patient_markers:
            match = re.search(marker, reasoning, re.IGNORECASE)
            if match:
                patient_start_pos = min(patient_start_pos, match.start())
        
        # 如果找到了患者内容的开始位置，提取之前的部分（医生的部分）
        if patient_start_pos < len(reasoning):
            doctor_reasoning = reasoning[:patient_start_pos].strip()
            
            # 验证提取的部分是否还包含患者内容
            if doctor_reasoning and not any(indicator in doctor_reasoning for indicator in patient_indicators):
                return doctor_reasoning
            elif doctor_reasoning:
                # 提取的部分仍然包含患者内容，尝试更激进的清理
                # 按句子分割，只保留不包含患者指示词的句子
                sentences = re.split(r'[。！？\n]', doctor_reasoning)
                clean_sentences = []
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and not any(indicator in sentence for indicator in patient_indicators):
                        clean_sentences.append(sentence)
                
                if clean_sentences:
                    return '。'.join(clean_sentences)
                else:
                    # 所有句子都包含患者内容
                    return ""
            else:
                # 患者内容在开头，整个 reasoning 都是患者的
                return ""
        else:
            # 没有找到明确的分隔符，但包含患者指示词
            # 尝试按句子过滤
            sentences = re.split(r'[。！？\n]', reasoning)
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and not any(indicator in sentence for indicator in patient_indicators):
                    clean_sentences.append(sentence)
            
            if clean_sentences:
                return '。'.join(clean_sentences)
            else:
                # 所有句子都包含患者内容
                return ""
        
    
    def extract_answer_content(self, content: str) -> str:
        """从content中提取<answer>标签内的内容，如果没有则返回原始内容（去除think标签和answer标签）"""
        if not content:
            return ""
        
        # 先尝试提取<answer>标签（不区分大小写，使用非贪婪匹配）
        # 查找所有匹配的<answer>标签
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # 提取最后一个匹配的answer内容（防止有多个answer标签，取最后一个）
            extracted = matches[-1].strip()
            # 清理提取的内容：去除所有标签
            extracted = re.sub(r'<think>.*?</think>', '', extracted, flags=re.DOTALL | re.IGNORECASE)
            extracted = re.sub(r'<answer>.*?</answer>', '', extracted, flags=re.DOTALL | re.IGNORECASE)
            extracted = re.sub(r'</think>', '', extracted, flags=re.IGNORECASE)
            extracted = re.sub(r'<think>', '', extracted, flags=re.IGNORECASE)
            # 再次检查提取的内容中是否还有answer标签，如果有则递归提取
            if '<answer>' in extracted or '<Answer>' in extracted:
                extracted = self.extract_answer_content(extracted)
            return extracted.strip()
        
        # 如果没有answer标签，去除think标签和answer标签后返回
        content_cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content_cleaned = re.sub(r'<answer>.*?</answer>', '', content_cleaned, flags=re.DOTALL | re.IGNORECASE)
        # 去除<box>标签
        content_cleaned = re.sub(r'<box>.*?</box>', '', content_cleaned, flags=re.DOTALL | re.IGNORECASE)
        # 去除残留的标签
        content_cleaned = re.sub(r'</think>', '', content_cleaned, flags=re.IGNORECASE)
        content_cleaned = re.sub(r'<think>', '', content_cleaned, flags=re.IGNORECASE)
        return content_cleaned.strip()
    
    def clean_messages_for_input_prompt(self, messages: List[Dict]) -> List[Dict]:
        """清理messages，去除assistant消息中的think标签内容，只保留answer部分"""
        cleaned_messages = []
        for msg in messages:
            cleaned_msg = msg.copy()
            if msg.get('role') == 'assistant' and 'content' in msg:
                content = msg['content']
                # 提取answer部分
                answer_content = self.extract_answer_content(content)
                cleaned_msg['content'] = answer_content
            cleaned_messages.append(cleaned_msg)
        return cleaned_messages
    
    def add_turn_without_rag(
        self,
        turn_number: int,
        patient_message: str,
        doctor_message: str,
        doctor_full_response: Optional[str] = None,
        tokens: int = 0,
        prompt_tokens: int = 0,
        input_prompt: Optional[str] = None,
        is_greeting: bool = False,
        is_diagnosis: bool = False
    ):
        """
        添加无RAG/CoT阶段的reasoning轮次（qwen3格式）
        
        Args:
            turn_number: 轮次编号
            patient_message: 患者消息
            doctor_message: 医生回复（已提取的纯文本）
            doctor_full_response: 医生完整响应（包含think标签，可选）
            tokens: 生成的token数
            prompt_tokens: prompt的token数
            input_prompt: 输入prompt（可选）
            is_greeting: 是否是问候语
            is_diagnosis: 是否是诊断
        """
        # 设置system prompt（只在第一次设置）
        if self.current_system_prompt is None:
            self.current_system_prompt = "你是一位经验丰富的精神科医生，正在对患者进行问诊。"
        
        # 添加患者消息（移除主诉功能）
        if patient_message:
            # 去除 patient 回复中的 <think> 标签，只保留实际回复内容
            patient_content = self.extract_answer_content(patient_message)
            # 如果提取失败，尝试直接去除 think 标签
            if not patient_content or not patient_content.strip():
                patient_content = re.sub(r'<think>.*?</think>', '', patient_message, flags=re.DOTALL | re.IGNORECASE)
                patient_content = re.sub(r'<answer>.*?</answer>', '', patient_content, flags=re.DOTALL | re.IGNORECASE)
                patient_content = patient_content.strip()
            # 如果还是为空，使用原始消息（但去除标签）
            if not patient_content or not patient_content.strip():
                patient_content = re.sub(r'<think>.*?</think>', '', patient_message, flags=re.DOTALL | re.IGNORECASE)
                patient_content = re.sub(r'<answer>.*?</answer>', '', patient_content, flags=re.DOTALL | re.IGNORECASE)
                patient_content = re.sub(r'</?think>', '', patient_content, flags=re.IGNORECASE)
                patient_content = re.sub(r'</?answer>', '', patient_content, flags=re.IGNORECASE)
                patient_content = patient_content.strip()
            
            patient_turn = {
                "turn": turn_number,
                "role": "patient",
                "content": patient_content  # 只保存去除 think 标签后的内容
            }
            self.simulation_dialogue.append(patient_turn)
        
        # 添加医生回复
        if doctor_message:
            # 验证 doctor_full_response 不是患者响应（检查是否包含患者角色的思考）
            if doctor_full_response:
                # 检查是否包含患者角色的标识（更全面的检查）
                patient_indicators = [
                    "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                    "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                    "我是一名", "我是患者", "作为患者", "患者角色"
                ]
                is_patient_response = any(indicator in doctor_full_response for indicator in patient_indicators)
                
                if is_patient_response:
                    # 如果 doctor_full_response 包含患者响应，使用 doctor_message 代替
                    import logging
                    logging.warning(f"[ReasoningGenerator] 检测到 doctor_full_response 包含患者响应，使用 doctor_message 代替")
                    doctor_full_response = None  # 强制使用 doctor_message
            
            # 优先从完整响应中提取thinking和answer
            if doctor_full_response:
                think_content = self.extract_think_content(doctor_full_response)
                answer_content = self.extract_answer_content(doctor_full_response)
                # 如果从full_response提取失败，尝试从doctor_message中提取
                if not answer_content or not answer_content.strip():
                    answer_content = self.extract_answer_content(doctor_message)
                # 如果还是为空，使用doctor_message（但需要清理标签）
                if not answer_content or not answer_content.strip():
                    # 清理doctor_message中的标签
                    answer_content = re.sub(r'<think>.*?</think>', '', doctor_message, flags=re.DOTALL | re.IGNORECASE)
                    answer_content = re.sub(r'<answer>.*?</answer>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
                    answer_content = answer_content.strip()
            else:
                # 如果没有完整响应，尝试从doctor_message中提取
                think_content = self.extract_think_content(doctor_message)
                answer_content = self.extract_answer_content(doctor_message)
                if not answer_content or not answer_content.strip():
                    # 清理doctor_message中的标签
                    answer_content = re.sub(r'<think>.*?</think>', '', doctor_message, flags=re.DOTALL | re.IGNORECASE)
                    answer_content = re.sub(r'<answer>.*?</answer>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
                    answer_content = answer_content.strip()
            
            # 清理模型生成的thinking内容，去除结构化标记
            if think_content:
                # 去除各种格式的标号：1）2）3）、1. 2. 3.、1、2、3、等
                think_content = re.sub(r'\d+\)\s*', '', think_content)  # 1）2）3）
                think_content = re.sub(r'\d+\.\s+', '', think_content)  # 1. 2. 3.
                think_content = re.sub(r'\d+、\s*', '', think_content)  # 1、2、3、
                think_content = re.sub(r'^\d+[\.\)、]\s*', '', think_content, flags=re.MULTILINE)  # 行首的标号
                # 去除"问诊策略："、"问诊目的："等标签
                think_content = re.sub(r'问诊策略[：:]\s*', '', think_content)
                think_content = re.sub(r'问诊目的[：:]\s*', '', think_content)
                # 去除其他结构化标题
                think_content = re.sub(r'\*\*[^：:]+[：:]\*\*', '', think_content)
                think_content = re.sub(r'\d+\.\s*\*\*[^：:]+[：:]\*\*', '', think_content)
                # 去除多余的换行和空格
                think_content = re.sub(r'\n\s*\n', '\n', think_content)
                think_content = think_content.strip()
            
            # 确保每一轮都有think_content
            if not think_content:
                # 如果没有think_content，生成一个默认的（基于患者消息）
                if patient_message:
                    # 提取关键信息生成简单的thinking
                    think_content = f"患者描述了相关情况。我需要继续了解相关信息，以便更好地帮助患者。"
                else:
                    think_content = "我需要继续问诊，了解患者的情况。"
            
            # 确保answer_content不包含任何标签，只保留纯文本
            if answer_content:
                # 去除所有可能的标签
                answer_content = re.sub(r'<think>.*?</think>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
                answer_content = re.sub(r'<answer>.*?</answer>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
                answer_content = re.sub(r'<box>.*?</box>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
                # 去除残留的标签标记
                answer_content = re.sub(r'</?think>', '', answer_content, flags=re.IGNORECASE)
                answer_content = re.sub(r'</?answer>', '', answer_content, flags=re.IGNORECASE)
                answer_content = re.sub(r'</?box>', '', answer_content, flags=re.IGNORECASE)
                answer_content = answer_content.strip()
            
            doctor_turn = {
                "turn": turn_number,
                "role": "doctor",
                "content": answer_content,
                "tokens": tokens,
                "prompt_tokens": prompt_tokens
            }
            
            # 添加thinking内容（确保每一轮都有）
            doctor_turn["think_content"] = think_content
            # 构建full_response（确保使用<think>和<answer>标签，并清理标号）
            # 总是使用清理后的think_content和answer_content来构建，确保没有标号
            doctor_turn["full_response"] = f"<think>\n{think_content}\n</think>\n\n<answer>{answer_content}</answer>"
            
            # 添加input_prompt（如果有）
            if input_prompt:
                # 如果 input_prompt 是 JSON 字符串，先解析为对象
                try:
                    if isinstance(input_prompt, str):
                        # 尝试解析 JSON 字符串
                        parsed_prompt = json.loads(input_prompt)
                    else:
                        # 如果已经是对象，直接使用
                        parsed_prompt = input_prompt
                    
                    # 从消息列表中提取 user 消息的 content
                    user_content = None
                    if isinstance(parsed_prompt, list):
                        for msg in parsed_prompt:
                            if msg.get('role') == 'user':
                                user_content = msg.get('content', '')
                                break
                    
                    # 保存 user 消息的 content（如果没有找到 user 消息，保存整个 input_prompt）
                    if user_content is not None:
                        doctor_turn["input_prompt"] = user_content
                    else:
                        # 如果没有找到 user 消息，保存原始格式（向后兼容）
                        doctor_turn["input_prompt"] = parsed_prompt if isinstance(parsed_prompt, list) else input_prompt
                except (json.JSONDecodeError, TypeError):
                    # 如果解析失败，保存为字符串（向后兼容）
                    doctor_turn["input_prompt"] = input_prompt
            
            # 添加特殊标志
            if is_greeting:
                doctor_turn["is_greeting"] = True
            if is_diagnosis:
                doctor_turn["is_diagnosis"] = True
            
            self.simulation_dialogue.append(doctor_turn)
    
    def add_turn_with_rag(
        self,
        turn_number: int,
        patient_message: str,
        doctor_message: str,
        doctor_full_response: Optional[str] = None,
        rag_search_step: Optional[Dict] = None,
        disease_extraction_step: Optional[Dict] = None,
        cot_reasoning_step: Optional[Dict] = None,
        tokens: int = 0,
        prompt_tokens: int = 0,
        input_prompt: Optional[str] = None,
        is_diagnosis: bool = False,
        current_topic: Optional[str] = None,
        converted_think_content: Optional[str] = None
    ):
        """
        添加有RAG/CoT阶段的reasoning轮次（qwen3格式）
        
        Args:
            turn_number: 轮次编号
            patient_message: 患者消息
            doctor_message: 医生回复（已提取的纯文本）
            doctor_full_response: 医生完整响应（包含think标签，可选）
            rag_search_step: RAG检索步骤信息
            disease_extraction_step: 疾病提取步骤信息
            cot_reasoning_step: CoT推理步骤信息
            tokens: 生成的token数
            prompt_tokens: prompt的token数
            input_prompt: 输入prompt（可选）
            is_diagnosis: 是否是诊断
            current_topic: 当前话题（可选，用于生成更准确的thinking）
            converted_think_content: 通过大模型转换后的口语化think内容（优先使用）
        """
        # 设置system prompt（只在第一次设置，或切换到RAG模式时更新）
        if self.current_system_prompt != self.SYSTEM_MESSAGE_TEMPLATE:
            self.current_system_prompt = self.SYSTEM_MESSAGE_TEMPLATE
        
        # 添加患者消息
        if patient_message:
            # 去除 patient 回复中的 <think> 标签，只保留实际回复内容
            patient_content = self.extract_answer_content(patient_message)
            # 如果提取失败，尝试直接去除 think 标签
            if not patient_content or not patient_content.strip():
                patient_content = re.sub(r'<think>.*?</think>', '', patient_message, flags=re.DOTALL | re.IGNORECASE)
                patient_content = re.sub(r'<answer>.*?</answer>', '', patient_content, flags=re.DOTALL | re.IGNORECASE)
                patient_content = patient_content.strip()
            # 如果还是为空，使用原始消息（但去除标签）
            if not patient_content or not patient_content.strip():
                patient_content = re.sub(r'<think>.*?</think>', '', patient_message, flags=re.DOTALL | re.IGNORECASE)
                patient_content = re.sub(r'<answer>.*?</answer>', '', patient_content, flags=re.DOTALL | re.IGNORECASE)
                patient_content = re.sub(r'</?think>', '', patient_content, flags=re.IGNORECASE)
                patient_content = re.sub(r'</?answer>', '', patient_content, flags=re.IGNORECASE)
                patient_content = patient_content.strip()
            
            patient_turn = {
                "turn": turn_number,
                "role": "patient",
                "content": patient_content  # 只保存去除 think 标签后的内容
            }
            # 第一个患者消息标记为主诉
            if turn_number == 1:
                patient_turn["is_complaint"] = True
            self.simulation_dialogue.append(patient_turn)
        
        # 优先使用转换后的think内容（如果提供）
        think_content = None
        if converted_think_content:
            think_content = converted_think_content.strip()
        
        # 验证 doctor_full_response 不是患者响应（检查是否包含患者角色的思考）
        if doctor_full_response:
            # 检查是否包含患者角色的标识（更全面的检查）
            patient_indicators = [
                "扮演一位", "扮演", "患者", "因为情绪低落", "用户希望我扮演",
                "20岁的男性患者", "30岁的女性患者", "岁", "性患者",
                "我是一名", "我是患者", "作为患者", "患者角色"
            ]
            is_patient_response = any(indicator in doctor_full_response for indicator in patient_indicators)
            
            if is_patient_response:
                # 如果 doctor_full_response 包含患者响应，使用 doctor_message 代替
                import logging
                logging.warning(f"[ReasoningGenerator] 检测到 doctor_full_response 包含患者响应（RAG模式），使用 doctor_message 代替")
                doctor_full_response = None  # 强制使用 doctor_message
        
        # 从完整响应中提取answer内容
        if doctor_full_response:
            answer_content = self.extract_answer_content(doctor_full_response)
            if not answer_content:
                answer_content = doctor_message
        else:
            answer_content = self.extract_answer_content(doctor_message)
            if not answer_content:
                answer_content = doctor_message
        
        # 如果没有转换后的think内容，从模型响应中提取（作为fallback）
        if not think_content:
            if doctor_full_response:
                model_think_content = self.extract_think_content(doctor_full_response)
            else:
                model_think_content = self.extract_think_content(doctor_message)
            
            # 清理模型生成的thinking内容，去除结构化标记
            if model_think_content:
                # 去除各种格式的标号：1）2）3）、1. 2. 3.、1、2、3、等
                model_think_content = re.sub(r'\d+\)\s*', '', model_think_content)  # 1）2）3）
                model_think_content = re.sub(r'\d+\.\s+', '', model_think_content)  # 1. 2. 3.
                model_think_content = re.sub(r'\d+、\s*', '', model_think_content)  # 1、2、3、
                model_think_content = re.sub(r'^\d+[\.\)、]\s*', '', model_think_content, flags=re.MULTILINE)  # 行首的标号
                # 去除"问诊策略："、"问诊目的："等标签
                model_think_content = re.sub(r'问诊策略[：:]\s*', '', model_think_content)
                model_think_content = re.sub(r'问诊目的[：:]\s*', '', model_think_content)
                # 去除其他结构化标题
                model_think_content = re.sub(r'\*\*[^：:]+[：:]\*\*', '', model_think_content)
                model_think_content = re.sub(r'\d+\.\s*\*\*[^：:]+[：:]\*\*', '', model_think_content)
                # 去除多余的换行和空格
                model_think_content = re.sub(r'\n\s*\n', '\n', model_think_content)
                model_think_content = model_think_content.strip()
                think_content = model_think_content
        
            # 在RAG/COT阶段，提取检索到的知识内容（用于后续整合到thinking中，不单独列出）
        # 提取RAG检索的知识内容（如果有），用于后续整合到自然语言的thinking中
        rag_knowledge_summary = ""
        if rag_search_step and rag_search_step.get('retrieved_knowledge'):
            retrieved_knowledge = rag_search_step.get('retrieved_knowledge', [])
            if retrieved_knowledge:
                # 提取关键信息，用于后续整合到thinking中
                knowledge_texts = []
                for item in retrieved_knowledge[:3]:  # 只取前3条
                    text = item.get('text', '')
                    if text:
                        knowledge_texts.append(text)
                # 将知识内容合并，用于后续提取关键信息
                if knowledge_texts:
                    rag_knowledge_summary = " ".join(knowledge_texts)
        
        # 如果已经有转换后的think内容，直接使用，不再构建
        if think_content:
            # 使用转换后的think内容，跳过后续的构建逻辑
            pass
        # 如果模型没有生成thinking，基于RAG/CoT步骤构建自然语言的thinking
        elif not model_think_content:
            # 构建自然语言的thinking内容（基于RAG/CoT步骤）
            # 提取疾病信息（如果有）
            disease_info = []
            if disease_extraction_step:
                possible_diseases = disease_extraction_step.get('output', '')
                if possible_diseases:
                    # 解析疾病信息，提取关键点
                    # 格式通常是：疾病A: F41.1 - 广泛性焦虑障碍（...，可能性最高）
                    disease_pattern = r'疾病([ABC]):\s*([F\d\.]+)\s*-\s*([^（]+)（([^）]+)）'
                    matches = re.findall(disease_pattern, possible_diseases)
                    for match in matches:
                        disease_letter, icd_code, disease_name, reason = match
                        disease_info.append({
                            'letter': disease_letter,
                            'name': disease_name.strip(),
                            'reason': reason.strip()
                        })
            
            # 提取CoT推理内容（如果有）
            cot_reasoning = ""
            if cot_reasoning_step:
                cot_reasoning = cot_reasoning_step.get('reasoning_output', '')
            
            # 构建自然语言的thinking（参考用户期望的格式，完全口语化，无结构化标题）
            # 目标格式：患者描述了...，根据诊疗指南，目前可能是...。但现在缺的关键信息是...。所以下一步我需要围绕"..."发问，只问这一点，不涉及其他内容，语言要自然、口语化。
            
            if disease_info or cot_reasoning or rag_knowledge_summary:
                thinking_parts = []
                
                # 第一部分：症状描述（从对话历史中提取关键症状，口语化）
                symptom_keywords = []
                if patient_message:
                    # 提取关键症状词
                    symptom_patterns = ['紧张', '焦虑', '担忧', '害怕', '情绪低落', '没劲', '提不起精神', '睡眠', '失眠', '睡不好', '多思多虑', '心慌', '胸口发紧', '手心出汗', '胸口被勒住', '眼前发黑', '手脚麻木', '不安', '绷不住']
                    for pattern in symptom_patterns:
                        if pattern in patient_message:
                            symptom_keywords.append(pattern)
                
                symptom_desc = ""
                if symptom_keywords:
                    # 更口语化的描述
                    if len(symptom_keywords) >= 3:
                        symptom_desc = f"患者描述了{symptom_keywords[0]}、{symptom_keywords[1]}、{symptom_keywords[2]}"
                        if len(symptom_keywords) > 3:
                            symptom_desc += "等"
                        symptom_desc += "相关症状"
                    else:
                        symptom_desc = f"患者描述了{', '.join(symptom_keywords)}等相关症状"
                else:
                    symptom_desc = "患者描述了相关症状"
                
                # 第二部分：疾病可能性（用自然语言描述，口语化）
                # 如果有RAG知识，可以从中提取疾病相关信息
                disease_context = ""
                if rag_knowledge_summary:
                    # 从RAG知识中提取疾病相关信息（简化）
                    # 查找常见的疾病名称
                    disease_keywords = ['广泛性焦虑障碍', '抑郁', '适应障碍', '焦虑障碍', '睡眠障碍', '抑郁发作']
                    found_diseases = []
                    for keyword in disease_keywords:
                        if keyword in rag_knowledge_summary:
                            found_diseases.append(keyword)
                    if found_diseases:
                        # 去重并限制数量
                        found_diseases = list(dict.fromkeys(found_diseases))[:3]  # 保持顺序并去重
                        if len(found_diseases) == 1:
                            disease_context = f"，这些症状都和{found_diseases[0]}有关"
                        elif len(found_diseases) == 2:
                            disease_context = f"，这些症状都和{found_diseases[0]}、{found_diseases[1]}有关"
                        else:
                            disease_context = f"，这些症状都和{found_diseases[0]}、{found_diseases[1]}或{found_diseases[2]}有关"
                
                if disease_info:
                    disease_names = [d['name'] for d in disease_info]
                    # 如果RAG知识中找到了疾病信息，优先使用RAG的（更自然）
                    if disease_context:
                        symptom_desc += disease_context
                    if len(disease_names) == 1:
                        if not disease_context:  # 如果RAG没有提供，使用disease_info
                            symptom_desc += f"，根据诊疗指南，目前可能是{disease_names[0]}"
                    elif len(disease_names) == 2:
                        if not disease_context:
                            symptom_desc += f"，根据诊疗指南，目前可能是{disease_names[0]}或{disease_names[1]}"
                    else:
                        if not disease_context:
                            symptom_desc += f"，根据诊疗指南，目前可能是{disease_names[0]}、{disease_names[1]}或{disease_names[2]}"
                    symptom_desc += "。"
                elif disease_context:
                    symptom_desc += disease_context + "。"
                else:
                    symptom_desc += "。"
                
                thinking_parts.append(symptom_desc)
                
                # 第三部分：关键缺失信息和下一步策略（从CoT推理中提取，用更口语化的方式）
                missing_info = ""
                
                # 首先尝试从current_topic中提取缺失信息
                if current_topic:
                    # 根据current_topic转换为自然语言的问题
                    if "既往" in current_topic or "病史" in current_topic:
                        missing_info = "患者以前有没有类似的情况？有没有看过医生、吃过药？这些能帮我判断这是首次发生、复发，还是长期慢性问题"
                    elif "严重程度" in current_topic:
                        missing_info = "这些症状的严重程度如何，对患者的生活和工作有什么影响"
                    elif "功能" in current_topic or "损害" in current_topic:
                        missing_info = "这些症状对患者的工作、生活、社交等方面造成了什么影响"
                    elif "核心症状" in current_topic:
                        missing_info = "核心症状的具体表现和特点"
                    elif "变化" in current_topic:
                        missing_info = "症状是如何变化的，有没有加重的趋势"
                
                # 使用传入的current_topic，如果没有则从CoT推理中提取
                if not current_topic and cot_reasoning:
                    # 提取当前话题（从CoT推理中）
                    topic_patterns = [
                        r'围绕"([^"]+)"',
                        r'询问([^，。\n]+)',
                        r'关于([^，。\n]+)',
                    ]
                    for pattern in topic_patterns:
                        match = re.search(pattern, cot_reasoning)
                        if match:
                            current_topic = match.group(1).strip()
                            break
                    
                    # 提取关键缺失信息（更口语化的提取方式）
                    # 去除所有结构化标记（###、---、*、数字标号、标题等）
                    cot_clean = cot_reasoning
                    cot_clean = re.sub(r'#{1,6}\s+', '', cot_clean)  # 去除markdown标题
                    cot_clean = re.sub(r'-{3,}', '', cot_clean)  # 去除分隔线
                    cot_clean = re.sub(r'^\s*[\*\-\+]\s+', '', cot_clean, flags=re.MULTILINE)  # 去除列表标记
                    cot_clean = re.sub(r'^\s*\d+[\.\)]\s+', '', cot_clean, flags=re.MULTILINE)  # 去除数字标号（包括 1. 和 1)）
                    cot_clean = re.sub(r'\d+\)\s*', '', cot_clean)  # 去除标号 1）2）3）等（单独处理）
                    cot_clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', cot_clean)  # 去除粗体标记
                    # 去除所有结构化标题
                    cot_clean = re.sub(r'\d+\)\s*\*\*[^：:]+[：:]\*\*', '', cot_clean)  # 去除 "1) **标题**："
                    cot_clean = re.sub(r'\*\*[^：:]+[：:]\*\*', '', cot_clean)  # 去除 "**标题**："
                    cot_clean = re.sub(r'\*\*下一轮问诊目标\*\*[：:]', '', cot_clean)  # 去除 "**下一轮问诊目标**："
                    cot_clean = re.sub(r'问诊策略[：:]\s*', '', cot_clean)  # 去除 "问诊策略："
                    cot_clean = re.sub(r'问诊目的[：:]\s*', '', cot_clean)  # 去除 "问诊目的："
                    cot_clean = re.sub(r'^\s*\*\s+', '', cot_clean, flags=re.MULTILINE)  # 去除列表项标记
                    cot_clean = re.sub(r'疾病([ABC])', lambda m: disease_info[ord(m.group(1)) - ord('A')]['name'] if ord(m.group(1)) - ord('A') < len(disease_info) else m.group(0), cot_clean)  # 替换疾病A/B/C为疾病名
                    
                    # 提取关键缺失信息（查找自然语言表达）
                    missing_patterns = [
                        r'需要(进一步|了解|询问|评估|知道)([^。\n，；]+)',
                        r'缺(少|乏)([^。\n，；]+)',
                        r'关键(信息|点|是)([^。\n，；]+)',
                        r'尚未(获得|了解|知道)([^。\n，；]+)',
                        r'还没有([^。\n，；]+)',
                        r'不清楚([^。\n，；]+)',
                        r'不知道([^。\n，；]+)',
                    ]
                    for pattern in missing_patterns:
                        match = re.search(pattern, cot_clean)
                        if match:
                            missing_info = match.group(2) if len(match.groups()) > 1 else match.group(1)
                            missing_info = missing_info.strip()
                            # 清理提取的内容，去除"关于"、"有关"、"的"等前缀
                            missing_info = re.sub(r'^(关于|有关|的|患者|他|她)', '', missing_info)
                            # 去除过长或过于技术性的描述
                            if len(missing_info) > 50:
                                # 取前50个字符
                                missing_info = missing_info[:50]
                            break
                    
                    # 如果没有找到，尝试从"需要询问"中提取
                    if not missing_info:
                        ask_match = re.search(r'需要询问([^。\n，；]+)', cot_clean)
                        if ask_match:
                            missing_info = ask_match.group(1).strip()
                            missing_info = re.sub(r'^(关于|有关|的|患者|他|她)', '', missing_info)
                            if len(missing_info) > 50:
                                missing_info = missing_info[:50]
                
                # 构建缺失信息描述（更口语化）
                if missing_info:
                    # 如果missing_info已经是自然语言形式（从current_topic转换的），直接使用
                    if "？" in missing_info or "什么" in missing_info or "如何" in missing_info:
                        # 已经是自然语言形式
                        pass
                    else:
                        # 进一步简化，使其更口语化
                        missing_info = missing_info.replace("询问患者有关", "").replace("不要包含其他话题和问题", "").strip()
                        # 将缺失信息转换为更自然的问题形式
                        if "既往病史" in missing_info or "病史" in missing_info:
                            missing_info = "患者以前有没有类似的情况？有没有看过医生、吃过药？这些能帮我判断这是首次发生、复发，还是长期慢性问题"
                        elif "症状严重程度" in missing_info or "严重程度" in missing_info:
                            missing_info = "这些症状的严重程度如何，对患者的生活和工作有什么影响"
                        elif "功能损害" in missing_info or "功能" in missing_info:
                            missing_info = "这些症状对患者的工作、生活、社交等方面造成了什么影响"
                        elif "核心症状" in missing_info:
                            missing_info = "核心症状的具体表现和特点"
                        elif "症状变化" in missing_info or "变化" in missing_info:
                            missing_info = "症状是如何变化的，有没有加重的趋势"
                    
                    if missing_info:
                        # 根据疾病数量调整描述
                        if disease_info and len(disease_info) >= 3:
                            thinking_parts.append(f"但现在缺的关键信息是：{missing_info}，对三种诊断的区分很重要。")
                        elif disease_info and len(disease_info) == 2:
                            thinking_parts.append(f"但现在缺的关键信息是：{missing_info}，对两种诊断的区分很重要。")
                        else:
                            thinking_parts.append(f"但现在缺的关键信息是：{missing_info}，这对诊断很重要。")
                    else:
                        thinking_parts.append("但现在还需要进一步了解相关信息，这对诊断很重要。")
                else:
                    thinking_parts.append("但现在还需要进一步了解相关信息，这对诊断很重要。")
                
                # 第四部分：下一步策略（更口语化）
                if current_topic:
                    # 清理current_topic，去除"询问患者有关"等前缀
                    clean_topic = current_topic.replace("询问患者有关", "").replace("不要包含其他话题和问题", "").strip()
                    if clean_topic:
                        thinking_parts.append(f"所以下一步我需要围绕\"{clean_topic}\"发问，只问这一点，不涉及其他内容，语言要自然、口语化。")
                    else:
                        thinking_parts.append("所以下一步我需要围绕当前话题进行有针对性的问诊，只问这一点，不涉及其他内容，语言要自然、口语化。")
                else:
                    thinking_parts.append("所以下一步我需要围绕当前话题进行有针对性的问诊，只问这一点，不涉及其他内容，语言要自然、口语化。")
                
                rag_thinking_content = "".join(thinking_parts)
            else:
                rag_thinking_content = "患者描述了相关症状。我需要检索诊疗指南来帮助诊断，然后进行有针对性的问诊，只问当前话题，不涉及其他内容，语言要自然、口语化。"
            
            # 如果没有转换后的think内容，使用构建的自然语言thinking（已经整合了RAG知识信息）
            if not think_content:
                think_content = rag_thinking_content
        else:
            # 使用模型生成的thinking（如果没有转换后的think内容）
            if not think_content:
                think_content = model_think_content
            
            # 如果有RAG知识但模型没有生成thinking，将RAG知识信息整合到thinking中
            if rag_knowledge_summary and not model_think_content and not think_content:
                # 从RAG知识中提取关键信息，整合到thinking中
                if disease_info:
                    disease_names = [d['name'] for d in disease_info]
                    if len(disease_names) >= 2:
                        think_content = f"患者描述了相关症状，根据诊疗指南，目前可能是{disease_names[0]}、{disease_names[1]}或{disease_names[2] if len(disease_names) > 2 else disease_names[0]}。"
                    elif len(disease_names) == 1:
                        think_content = f"患者描述了相关症状，根据诊疗指南，目前可能是{disease_names[0]}。"
                    else:
                        think_content = "患者描述了相关症状，根据诊疗指南，需要进一步了解相关信息。"
                else:
                    think_content = "患者描述了相关症状，根据诊疗指南，需要进一步了解相关信息。"
        
        # 确保think_content不为空
        if not think_content:
            if patient_message:
                think_content = f"患者描述了相关情况。我需要继续了解相关信息，以便更好地帮助患者。"
            else:
                think_content = "我需要继续问诊，了解患者的情况。"
        
        # 确保answer_content不包含任何标签，只保留纯文本
        if answer_content:
            # 去除所有可能的标签
            answer_content = re.sub(r'<think>.*?</think>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
            answer_content = re.sub(r'<answer>.*?</answer>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
            answer_content = re.sub(r'<box>.*?</box>', '', answer_content, flags=re.DOTALL | re.IGNORECASE)
            # 去除残留的标签标记
            answer_content = re.sub(r'</?think>', '', answer_content, flags=re.IGNORECASE)
            answer_content = re.sub(r'</?answer>', '', answer_content, flags=re.IGNORECASE)
            answer_content = re.sub(r'</?box>', '', answer_content, flags=re.IGNORECASE)
            answer_content = answer_content.strip()
        
        # 添加医生回复
        if doctor_message or answer_content:
            doctor_turn = {
                "turn": turn_number,
                "role": "doctor",
                "content": answer_content,
                "tokens": tokens,
                "prompt_tokens": prompt_tokens
            }
            
            # 添加thinking内容（确保每一轮都有）
            doctor_turn["think_content"] = think_content
            # 构建full_response（确保使用<think>和<answer>标签，并清理标号）
            # 总是使用清理后的think_content和answer_content来构建，确保没有标号
            doctor_turn["full_response"] = f"<think>\n{think_content}\n</think>\n\n<answer>{answer_content}</answer>"
            
            # 添加input_prompt（如果有）
            if input_prompt:
                # 如果 input_prompt 是 JSON 字符串，先解析为对象
                try:
                    if isinstance(input_prompt, str):
                        # 尝试解析 JSON 字符串
                        parsed_prompt = json.loads(input_prompt)
                    else:
                        # 如果已经是对象，直接使用
                        parsed_prompt = input_prompt
                    
                    # 从消息列表中提取 user 消息的 content
                    user_content = None
                    if isinstance(parsed_prompt, list):
                        for msg in parsed_prompt:
                            if msg.get('role') == 'user':
                                user_content = msg.get('content', '')
                                break
                    
                    # 保存 user 消息的 content（如果没有找到 user 消息，保存整个 input_prompt）
                    if user_content is not None:
                        doctor_turn["input_prompt"] = user_content
                    else:
                        # 如果没有找到 user 消息，保存原始格式（向后兼容）
                        doctor_turn["input_prompt"] = parsed_prompt if isinstance(parsed_prompt, list) else input_prompt
                except (json.JSONDecodeError, TypeError):
                    # 如果解析失败，保存为字符串（向后兼容）
                    doctor_turn["input_prompt"] = input_prompt
            
            # 添加function_call信息（如果有RAG检索）
            if rag_search_step and rag_search_step.get('retrieved_knowledge'):
                # 构建query
                query = rag_search_step.get('query', '')
                # 简化query
                if '关于' in query:
                    topic_start = query.find('关于') + 2
                    topic_end = query.find('，结合')
                    if topic_end > topic_start:
                        topic = query[topic_start:topic_end]
                        query = f"{topic}的诊断标准和症状特征"
                    else:
                        query = "抑郁和焦虑相关症状的诊断标准"
                elif len(query) > 100:
                    query = query[:100] + "..."
                
                # 注意：qwen3格式中，function_call信息可以包含在full_response中，但不作为独立字段
                # 如果需要，可以在think_content中提及
            
            # 添加特殊标志
            if is_diagnosis:
                doctor_turn["is_diagnosis"] = True
            
            self.simulation_dialogue.append(doctor_turn)
    
    def _format_knowledge_context(self, retrieved_knowledge: List[Dict]) -> str:
        """格式化检索到的知识为字符串"""
        if not retrieved_knowledge:
            return ""
        
        context_parts = []
        for i, item in enumerate(retrieved_knowledge, 1):
            text = item.get('text', '')
            score = item.get('score', 0)
            source = item.get('source', '未知来源')
            context_parts.append(f"{i}. {text} (相似度: {score:.3f}, 来源: {source})")
        
        return "\n".join(context_parts)
    
    def get_all_turns(self) -> List[Dict]:
        """获取所有reasoning轮次（qwen3格式）"""
        return self.simulation_dialogue
    
    def clear(self):
        """清空所有reasoning轮次"""
        self.simulation_dialogue = []
        self.current_system_prompt = None
        self.turn_counter = 0
