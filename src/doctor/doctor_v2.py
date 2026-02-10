# -*- coding: utf-8 -*-
"""
医生角色AI模块 V2版本 - 基于阶段式诊断树

该模块实现了基于统一阶段式诊断树的医生AI对话系统：
- 模拟精神科医生进行患者问诊
- 支持API调用和本地模型两种方式
- 实现阶段式诊断流程（筛查→评估→深入→风险→总结）
- 提供成本统计功能

适配MDD5k数据结构，使用diagtree_v2
"""

import json
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import llm_tools_api
from .diagtree_v2 import DiagTree



class Doctor(llm_tools_api.DoctorCost):
    """
    基于阶段式诊断树的结构化问诊医生类 V2版本
    
    该类实现了一个能够按照统一阶段式诊断树结构进行问诊的AI医生，
    支持5个临床阶段的动态问诊流程。
    
    继承自DoctorCost类，具备成本统计功能。
    """
    def __init__(self, patient_template, doctor_prompt_path, diagtree_path, model_path, use_api) -> None:
        """
        初始化医生对象
        
        Args:
            patient_template (dict): 患者模板信息，包含Age、Gender、patient_id等（MDD5k格式）
            doctor_prompt_path (str): 医生角色配置文件路径
            diagtree_path (str): 诊断树配置文件路径（V2版本中不使用，保持兼容性）
            model_path (str): 模型路径或API模型名称，可能包含端口信息
            use_api (bool): 是否使用API方式调用模型
        """
        # 提取基础模型名称用于成本统计（先去掉@host:port，再取最后部分）
        temp_path = model_path.split('@')[0] if '@' in model_path else model_path
        base_model_name = temp_path.split(':')[0].split('/')[-1]
        super().__init__(base_model_name)
        
        # 存储初始化参数
        self.patient_template = patient_template          # 患者信息模板
        self.doctor_prompt_path = doctor_prompt_path      # 医生角色配置文件路径
        self.diagtree_path = diagtree_path               # 诊断树配置文件路径
        self.model_path = model_path                     # 模型路径（可能包含端口）
        self.model_name = model_path                     # 模型名称（完整，包含端口）
        
        # 提取基础模型名称用于API调用中的model参数（去掉@host:port部分）
        if '@' in model_path:
            self.api_model_name = model_path.split('@')[0]
        else:
            self.api_model_name = model_path.split(':')[0]
        self.use_openrouter = llm_tools_api.is_openrouter_model(model_path)
        self.reasoning_enabled = llm_tools_api.is_reasoning_enabled("doctor", default=True)
        self.reasoning_extra_body = llm_tools_api.build_reasoning_extra_body(
            self.reasoning_enabled, self.use_openrouter, self.api_model_name
        )
        
        # 模型相关属性初始化
        self.doctor_model = None                         # 本地模型对象
        self.doctor_tokenizer = None                     # 本地模型分词器
        self.doctor_prompt = None                        # 当前选中的医生角色配置
        self.client = None                               # API客户端对象
        
        # 对话状态管理
        self.messages = []                               # 对话消息历史
        self.dialbegin = True                           # 是否为对话开始标志
        self.use_api = use_api                          # 是否使用API模式
        
        # 话题和诊断流程管理
        self.current_idx = 0                            # 当前话题索引
        self.doctor_persona = None                      # 医生角色描述
        self.patient_persona = None                     # 患者角色描述
        self.topic_seq = []                             # 问诊话题序列
        self.topic_begin = 0                            # 当前话题开始位置
        self.diagtree = None                            # 诊断树对象（已弃用）
        

        
        # 初始化阶段式诊断树 V2版本
        # 使用统一的阶段式诊断树，不再依赖于疾病特定的JSON文件
        # 改为使用内置的阶段配置，适用于所有疾病类型
        patient_id = self.patient_template.get('patient_id', 'Unknown')
        print(f"[V2] 为患者 {patient_id} 初始化阶段式诊断树")
        self.diagnosis_tree = DiagTree(
            model_name=self.model_name, 
            prompts={
                'doctor': self.doctor_prompt_path, 
                'diagtree': self.diagtree_path  # 保持接口兼容，但新系统不使用此参数
            }
        )
        # 加载诊断树并生成问诊话题序列
        self.diagnosis_tree.load_tree()
        self.topic_seq = self.diagnosis_tree.dynamic_select()
        
    def _reasoning_kwargs(self):
        if self.reasoning_extra_body:
            return {"extra_body": self.reasoning_extra_body}
        return {}

    def _extract_answer_and_reasoning(self, chat_response):
        """
        从API响应中分离输出与reasoning，兼容多种形式：
        1) reasoning_content / reasoning 字段
        2) 文本中的 <think>...</think> 标签
        3) \no_think 模式下 content 为空，内容在 reasoning_content 中
        """
        if not chat_response or not hasattr(chat_response, "choices") or not chat_response.choices:
            return "", ""

        raw_content = chat_response.choices[0].message.content or ""
        reasoning = llm_tools_api.extract_reasoning_content(chat_response) or ""

        # 特殊情况：\no_think 模式下，content 为空但 reasoning_content 有内容
        # 此时 reasoning_content 实际上是模型的回复内容
        if not raw_content and reasoning:
            clean_content, _ = self._strip_think_tags(reasoning)
            return clean_content, ""

        clean_content, think_reasoning = self._strip_think_tags(raw_content)
        if not reasoning:
            reasoning = think_reasoning
        return clean_content, reasoning

    def _strip_think_tags(self, text):
        """移除 <think> 标签并返回 (纯文本, reasoning)。"""
        if text is None:
            return "", ""
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return cleaned, reasoning

        # === 调试输出：显示诊断树构造过程 ===
        print(f"[V2-DEBUG] 患者 {patient_id} 诊断树构造完成：")
        print(f"[V2-DEBUG] - 总话题数: {len(self.topic_seq)}")
        print(f"[V2-DEBUG] - 采用阶段式流程")
        print(f"[V2-DEBUG] - 话题序列预览:")
        for i, topic in enumerate(self.topic_seq[:5]):  # 只显示前5个话题
            print(f"[V2-DEBUG]   {i+1}. {topic}")
        if len(self.topic_seq) > 5:
            print(f"[V2-DEBUG]   ... 等共{len(self.topic_seq)}个话题")
        print(f"[V2-DEBUG] - 当前阶段: {self.diagnosis_tree.current_phase.value}")

    def doctorbot_init(self, first_topic):
        """
        初始化医生机器人
        
        该方法负责：
        1. 从配置文件中随机选择一个医生角色
        2. 构建医生和患者的角色描述
        3. 生成初始对话提示
        4. 初始化模型（API或本地模型）
        5. 设置初始对话消息
        
        Args:
            first_topic (str): 对话的第一个主题
        """
        # 1. 加载医生配置文件
        with open(self.doctor_prompt_path) as f:
            prompt = json.load(f)
        
        # 2. 随机选择一个医生配置
        doctor_num = random.randint(0, len(prompt)-1)
        self.doctor_prompt = prompt[doctor_num]
        
        # 3. 构建医生角色描述
        # 根据配置文件中的医生属性构建详细的角色描述
        self.doctor_persona = """# 角色定位
你是一名{age}的{gender}专业的精神卫生中心临床心理科主任医师，对一名患者进行问诊。

# 问诊习惯
在所有的对话过程中，你需要严格遵循以下问诊习惯：
- 专业特长：你尤其擅长诊断{special}
- 问诊节奏：你的问诊速度是{speed}的
- 交流方式：你的交流风格是{commu}的
- 共情能力：你{empathy}在适当的时候与患者进行共情对话
- 专业解释：你{explain}向患者解释一些专业名词术语

# 表达要求
- 使用口语化的表达方式，需要有标点符号。
- 保持专业性和亲和力的平衡，
- 不要生成思考过程, 额外的语气说明, 前缀，括号等和问诊对话无关的内容""".format(
            age=self.doctor_prompt['age'],
            gender=self.doctor_prompt['gender'],
            special=self.doctor_prompt['special'],
            speed=self.doctor_prompt['speed'],
            commu=self.doctor_prompt['commu'],
            empathy=self.doctor_prompt['empathy'],
            explain=self.doctor_prompt['explain']
        )
        # self.doctor_persona = llm_tools_api.apply_reasoning_prompt_prefix(
        #     self.doctor_persona, self.model_name, self.use_openrouter, self.reasoning_enabled
        # )
        
        # 4. 构建患者角色描述（适配MDD5k字段）
        self.patient_persona = "患者是一名{}岁的{}性。".format(
            self.patient_template['Age'], 
            self.patient_template['Gender']
        )
        
        # 5. 构建初始问诊要求提示
        self.req_prompt ="""\n现在你与患者的对话开始，你需要首先向患者问好，然后自然地询问有关{}的内容。
        
# 请严格遵循以下要求：
  1. 首先向患者问好（例如："你好"、"您好"等简短问候），然后再询问相关话题。
  2. 不要询问例如睡眠、食欲之类的具体症状。
  3. 使用口语化表达与患者交流，需要有标点符号。
  4. 不要输出类似'好的，我会按照您的要求开始问诊'之类的确认性话语。
  5. 提问尽量简洁，一句话不能多于2个问题。
  6. 直接输出问候和问诊对话，不要生成思考过程, 额外的语气说明, 前缀，括号等和问诊对话无关的内容。""".format(first_topic)
        
        # 6. 构建初始对话提示
        # 结合患者信息和第一个主题，生成对话开始的提示
        final_prompt = self.patient_persona + self.req_prompt

        # 7. 初始化模型（根据配置选择API或本地模型）
        if self.use_api:
            # 使用API方式初始化客户端
            self.client = llm_tools_api.doctor_client_init(self.model_name)
        else:
            # 使用本地模型方式初始化
            self.doctor_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.doctor_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 8. 初始化对话消息列表
        # 设置系统角色（医生角色）和用户消息（初始提示）
        self.messages.extend([
            {"role": "system", "content": self.doctor_persona},
            {"role": "user", "content": final_prompt}
        ])

    def doctor_response_gen(self, patient_response, dialogue_history):
        """
        生成医生的回复 V2版本
        
        该方法是核心的对话生成逻辑，基于统一阶段式诊断树，
        根据当前使用的模型类型（API或本地）和对话状态生成合适的医生回复。
        
        主要流程：
        1. 判断是否为对话开始，如果是则初始化医生
        2. 检查当前话题是否结束
        3. 如果话题结束，判断是否需要切换话题或结束诊断
        4. 生成针对当前话题的回复
        5. 处理特殊话题如'parse'（经验解析）
        
        Args:
            patient_response (str): 患者的回复内容
            dialogue_history (list): 对话历史记录
            
        Returns:
            tuple: (医生回复, 当前话题, 成本信息) 或 (诊断结果, None, 成本信息)
        """
        if self.use_api:
            # === API模式处理逻辑 ===
            if self.dialbegin == True:
                # 对话开始：初始化医生并生成第一个回复
                self.doctorbot_init(self.topic_seq[self.current_idx])
                # print(self.topic_seq)  # 调试用，显示话题序列
                self.current_idx += 1
                
                # 调用API生成初始回复（带重试机制）
                print(f"[V2-DEBUG] 开始第一轮对话，使用话题: {self.topic_seq[0]}")
                
                max_retries = 2
                doctor_response = None
                doctor_reasoning = ""
                chat_response = None
                
                for retry in range(max_retries):
                    try:
                        chat_response = self.client.chat.completions.create(
                            model=self.api_model_name,
                            messages=self.messages,
                            top_p=0.93,
                            max_tokens=llm_tools_api.get_max_tokens(),
                            **self._reasoning_kwargs()
                        )
                        
                        # 检查响应有效性
                        if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
                            if retry < max_retries - 1:
                                print(f"[Doctor V2] 警告: API返回无效响应（初始化），重试第 {retry + 1} 次...")
                                continue
                            else:
                                print(f"[Doctor V2] 错误: API返回无效响应（初始化），已达最大重试次数")
                                raise Exception("Doctor初始化API返回无效响应")
                        
                        # 提取响应内容
                        try:
                            content, reasoning = self._extract_answer_and_reasoning(chat_response)
                            if content is None or str(content).strip() == "":
                                if retry < max_retries - 1:
                                    print(f"[Doctor V2] 警告: API返回空内容（初始化），重试第 {retry + 1} 次...")
                                    continue
                                else:
                                    raise Exception("Doctor响应内容为空")
                            
                            doctor_response = content
                            doctor_reasoning = reasoning
                            break  # 成功获取响应
                            
                        except (AttributeError, IndexError) as e:
                            if retry < max_retries - 1:
                                print(f"[Doctor V2] 警告: 无法提取响应内容（初始化）: {e}，重试第 {retry + 1} 次...")
                                continue
                            else:
                                raise Exception(f"Doctor API返回格式异常: {e}")
                                
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"[Doctor V2] 警告: API调用异常（初始化）: {e}，重试第 {retry + 1} 次...")
                            continue
                        else:
                            print(f"[Doctor V2] 错误: API调用异常（初始化）: {e}，已达最大重试次数")
                            raise
                
                # 记录token使用量
                prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                    chat_response,
                    messages=self.messages,
                    response_text=doctor_response
                )
                super().money_cost(prompt_tokens, completion_tokens)
                
                # 输出医生问题
                print(f"Doctor_ask: {doctor_response}")
                
                # 清理临时消息并标记对话已开始
                self.messages.pop()
                self.dialbegin = False
                # 返回格式：(response, topic, reasoning)
                return doctor_response, None, doctor_reasoning
            else:   
                # 对话进行中：检查当前话题是否结束
                is_topic_end, pt, ct = self.diagnosis_tree.is_topic_end(self.topic_seq[self.current_idx], dialogue_history[self.topic_begin:])
                super().money_cost(pt, ct)
                # print("[V2] ********topic_end", is_topic_end)
                
                if is_topic_end:
                    # === 当前话题已结束，准备切换到下一个话题 ===
                    
                    # 更新话题开始位置为当前对话历史长度
                    self.topic_begin = len(dialogue_history)
                    
                    # 检查是否整个诊断流程结束
                    is_dialogue_end = self.diagnosis_tree.is_end(self.topic_seq[self.current_idx])
                    if is_dialogue_end:
                        # 诊断流程结束，生成最终诊断结果
                        # 适配MDD5k数据结构，简化诊断结果格式
                        diag_result, diag_reasoning = self._generate_final_diagnosis(dialogue_history)
                        return diag_result, None, diag_reasoning
                    else:
                        # 切换到下一个话题
                        self.current_idx += 1
                        # print("[V2] **********current_topic", self.topic_seq[self.current_idx])
                        if self.topic_seq[self.current_idx] == 'parse':
                            # === 处理特殊的'parse'话题：解析患者经验并动态生成新话题 ===
                            self._handle_parse_topic(dialogue_history)
                            
                        # === 根据医生的共情能力设置生成不同的回复提示 ===
                        doctor_prompt = self._build_topic_prompt(dialogue_history, is_empathy_enabled=True)
                        
                        # 将构建的提示添加到消息列表并调用API（带重试机制）
                        self.messages.append({"role": "user", "content": doctor_prompt})
                        print(f"[V2-DEBUG] 话题切换 -> 新话题: {self.topic_seq[self.current_idx]}")
                        
                        max_retries = 2
                        doctor_response = None
                        doctor_reasoning = ""
                        chat_response = None
                        
                        for retry in range(max_retries):
                            try:
                                chat_response = self.client.chat.completions.create(
                                    model=self.api_model_name,
                                    messages=self.messages,
                                    top_p=0.93,
                                    frequency_penalty=0.8,
                                    max_tokens=llm_tools_api.get_max_tokens(),
                                    **self._reasoning_kwargs()
                                )
                                
                                # 检查响应有效性
                                if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
                                    if retry < max_retries - 1:
                                        print(f"[Doctor V2] 警告: API返回无效响应（话题切换），重试第 {retry + 1} 次...")
                                        continue
                                    else:
                                        print(f"[Doctor V2] 错误: API返回无效响应（话题切换），已达最大重试次数")
                                        self.messages.pop()
                                        raise Exception("Doctor话题切换API返回无效响应")
                                
                                # 提取响应内容
                                try:
                                    content, reasoning = self._extract_answer_and_reasoning(chat_response)
                                    if content is None or str(content).strip() == "":
                                        if retry < max_retries - 1:
                                            print(f"[Doctor V2] 警告: API返回空内容（话题切换），重试第 {retry + 1} 次...")
                                            continue
                                        else:
                                            self.messages.pop()
                                            raise Exception("Doctor响应内容为空")
                                    
                                    doctor_response = content
                                    doctor_reasoning = reasoning
                                    break  # 成功获取响应
                                    
                                except (AttributeError, IndexError) as e:
                                    if retry < max_retries - 1:
                                        print(f"[Doctor V2] 警告: 无法提取响应内容（话题切换）: {e}，重试第 {retry + 1} 次...")
                                        continue
                                    else:
                                        self.messages.pop()
                                        raise Exception(f"Doctor API返回格式异常: {e}")
                                        
                            except Exception as e:
                                if retry < max_retries - 1:
                                    print(f"[Doctor V2] 警告: API调用异常（话题切换）: {e}，重试第 {retry + 1} 次...")
                                    continue
                                else:
                                    print(f"[Doctor V2] 错误: API调用异常（话题切换）: {e}，已达最大重试次数")
                                    self.messages.pop()
                                    raise
                        
                        # 记录token使用量
                        prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                            chat_response,
                            messages=self.messages,
                            response_text=doctor_response
                        )
                        super().money_cost(prompt_tokens, completion_tokens)
                        
                        # 输出医生问题
                        print(f"Doctor_ask: {doctor_response}")
                        
                        # 清理临时消息并返回结果
                        self.messages.pop()
                        return doctor_response, self.topic_seq[self.current_idx], doctor_reasoning
                else:
                    # === 当前话题未结束，继续围绕当前话题进行问诊 ===
                    print("[V2] **********current_topic2", self.topic_seq[self.current_idx])
                    
                    if self.topic_seq[self.current_idx] == 'parse':
                        # === 处理特殊的'parse'话题（话题未结束的情况） ===
                        self._handle_parse_topic(dialogue_history)
                        
                    # === 继续围绕当前话题生成医生回复（带重试机制） ===
                    doctor_prompt = self._build_continuation_prompt(dialogue_history)
                    self.messages.append({"role": "user", "content": doctor_prompt})
                    print(f"[V2-DEBUG] 继续当前话题: {self.topic_seq[self.current_idx]}")
                    
                    max_retries = 2
                    doctor_response = None
                    doctor_reasoning = ""
                    chat_response = None
                    
                    for retry in range(max_retries):
                        try:
                            chat_response = self.client.chat.completions.create(
                                model=self.api_model_name,
                                messages=self.messages,
                                top_p=0.93,
                                frequency_penalty=0.8,
                                max_tokens=llm_tools_api.get_max_tokens(),
                                **self._reasoning_kwargs()
                            )
                            
                            # 检查响应有效性
                            if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
                                if retry < max_retries - 1:
                                    print(f"[Doctor V2] 警告: API返回无效响应（继续话题），重试第 {retry + 1} 次...")
                                    continue
                                else:
                                    print(f"[Doctor V2] 错误: API返回无效响应（继续话题），已达最大重试次数")
                                    self.messages.pop()
                                    raise Exception("Doctor继续话题API返回无效响应")
                            
                            # 提取响应内容
                            try:
                                content, reasoning = self._extract_answer_and_reasoning(chat_response)
                                if content is None or str(content).strip() == "":
                                    if retry < max_retries - 1:
                                        print(f"[Doctor V2] 警告: API返回空内容（继续话题），重试第 {retry + 1} 次...")
                                        continue
                                    else:
                                        self.messages.pop()
                                        raise Exception("Doctor响应内容为空")
                                
                                doctor_response = content
                                doctor_reasoning = reasoning
                                break  # 成功获取响应
                                
                            except (AttributeError, IndexError) as e:
                                if retry < max_retries - 1:
                                    print(f"[Doctor V2] 警告: 无法提取响应内容（继续话题）: {e}，重试第 {retry + 1} 次...")
                                    continue
                                else:
                                    self.messages.pop()
                                    raise Exception(f"Doctor API返回格式异常: {e}")
                                    
                        except Exception as e:
                            if retry < max_retries - 1:
                                print(f"[Doctor V2] 警告: API调用异常（继续话题）: {e}，重试第 {retry + 1} 次...")
                                continue
                            else:
                                print(f"[Doctor V2] 错误: API调用异常（继续话题）: {e}，已达最大重试次数")
                                self.messages.pop()
                                raise
                    
                    # 记录token使用量
                    prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                        chat_response,
                        messages=self.messages,
                        response_text=doctor_response
                    )
                    super().money_cost(prompt_tokens, completion_tokens)
                    
                    # 输出医生问题
                    print(f"Doctor_ask: {doctor_response}")
                    
                    # 清理临时消息并返回结果
                    self.messages.pop()
                    return doctor_response, self.topic_seq[self.current_idx], doctor_reasoning
        else:
            # === 本地模型处理逻辑 ===
            # TODO: 本地模型的逻辑需要进一步完善，目前缺少诊断树的集成
            if self.dialbegin == True:
                # 对话开始：初始化医生并生成第一个回复
                self.doctorbot_init(self.topic_seq[self.current_idx])
                
                # 使用本地模型的聊天模板格式化消息
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 将文本转换为模型输入格式
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                
                # 使用本地模型生成回复
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=4096
                )
                
                # 提取生成的新内容（去除输入部分）
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                
                # 解码生成的文本并移除可能的 <think>
                raw_response = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                doctor_response, doctor_reasoning = self._strip_think_tags(raw_response)
                
                # 输出医生问题
                print(f"Doctor_ask: {doctor_response}")
                
                # 将医生回复添加到对话历史
                self.messages.append({"role": "assistant", "content": doctor_response})
                self.dialbegin = False
            else:
                # 对话进行中：处理患者回复并生成医生回复
                self.messages.append({"role": "user", "content": patient_response})
                
                # 格式化消息并生成回复（流程与对话开始相同）
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=4096
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                raw_response = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                doctor_response, doctor_reasoning = self._strip_think_tags(raw_response)
                
                # 输出医生问题
                print(f"Doctor_ask: {doctor_response}")
                
                # 将医生回复添加到对话历史
                self.messages.append({"role": "assistant", "content": doctor_response})
            # 返回格式：(response, topic, reasoning) - 本地模型返回 <think> 中的reasoning（若有）
            return doctor_response, None, doctor_reasoning

    def _generate_final_diagnosis(self, dialogue_history):
        """
        生成最终诊断结果，适配MDD5k格式
        
        Args:
            dialogue_history (list): 对话历史
            
        Returns:
            str: 诊断结果字符串
        """
        # MDD5k版本的诊断逻辑，基于对话历史生成细分诊断
        diagnosis_prompt = (
f"你是一位经验丰富的精神科医生。请阅读以下初次精神科门诊的问诊对话记录，并根据ICD-10国际疾病分类标准，仔细分析后输出患者诊断结束后的ICD-10诊断代码。\n\n"

"## 疾病分类说明\n"
"请仅从以下ICD-10标准中的10种疾病中选择最符合的诊断大类以及进一步细分的小类：\n"
"    - F32 抑郁发作：情绪持续低落、兴趣/愉快感下降、精力不足；伴睡眠/食欲改变、自责/无价值感等；可轻/中/重度（重度可伴精神病性症状）；无既往躁狂/轻躁狂。\n"
"        F32.0 轻度抑郁发作：症状轻，社会功能影响有限。\n"
"        F32.1 中度抑郁发作：症状更明显，日常活动受限。\n"
"        F32.2 重度抑郁发作，无精神病性症状：症状显著，丧失功能，但无妄想/幻觉。\n"
"        F32.3 重度抑郁发作，有精神病性症状：伴有抑郁性妄想、幻觉或木僵。\n"
"        F32.8 其他抑郁发作；F32.9 抑郁发作，未特指。\n"
"    - F41 其他焦虑障碍：恐慌发作或广泛性焦虑为主；过度担忧、紧张、心悸、胸闷、出汗、眩晕、濒死感/失控感；与特定情境无关或不成比例，造成显著痛苦/功能损害。\n"
"        F41.0 惊恐障碍：突发的强烈恐慌发作，常伴濒死感。\n"
"        F41.1 广泛性焦虑障碍：长期持续的过度担忧和紧张不安。\n"
"        F41.2 混合性焦虑与抑郁障碍：焦虑与抑郁并存但均不足以单独诊断。\n"
"        F41.3 其他混合性焦虑障碍：混合焦虑表现但未完全符合特定标准。\n"
"        F41.9 焦虑障碍，未特指：存在焦虑症状但资料不足以分类。\n"
"    - F39.9 未特指的心境（情感）障碍：存在心境障碍证据，但资料不足以明确归入抑郁或双相等具体亚型时选用。\n"
"    - F51 非器质性睡眠障碍：失眠、过度嗜睡、梦魇、昼夜节律紊乱等；非器质性原因；睡眠问题为主要主诉并致显著困扰/功能损害。\n"
"        F51.0 非器质性失眠：入睡困难、易醒或睡眠不恢复精力。\n"
"        F51.1 非器质性嗜睡：过度睡眠或难以保持清醒。\n"
"        F51.2 非器质性睡眠-觉醒节律障碍：昼夜节律紊乱导致睡眠异常。\n"
"        F51.3 梦魇障碍：频繁恶梦导致醒后强烈不安。\n"
"        F51.4 睡眠惊恐（夜惊）：夜间突然惊恐醒来伴强烈焦虑反应。\n"
"        F51.5 梦游症：睡眠中出现起床或行走等复杂行为。\n"
"        F51.9 非器质性睡眠障碍，未特指：睡眠异常但无具体分类。\n"
"    - F98 其他儿童和青少年行为与情绪障碍：多见于儿童期起病（如遗尿/遗粪、口吃、抽动相关习惯性问题等），以发育期特异表现为主。\n"
"        F98.0 非器质性遗尿症：儿童在不适当年龄仍有排尿失控。\n"
"        F98.1 非器质性遗粪症：儿童在不适当情境排便。\n"
"        F98.2 婴儿期或儿童期进食障碍：儿童进食行为异常影响营养或发育。\n"
"        F98.3 异食癖：持续摄入非食物性物质。\n"
"        F98.4 刻板性运动障碍：重复、无目的的运动习惯。\n"
"        F98.5 口吃：言语流利性障碍，表现为言语阻塞或重复。\n"
"        F98.6 习惯性动作障碍：如咬甲、吮指等持续存在的习惯。\n"
"        F98.8 其他特指的儿童行为和情绪障碍：符合儿童期特异但不归入上述类。\n"
"        F98.9 未特指的儿童行为和情绪障碍：症状存在但缺乏分类依据。\n"
"    - F42 强迫障碍：反复的强迫观念/行为，个体自知过度或不合理但难以抵抗，耗时或致显著困扰/损害。\n"
"        F42.0 以强迫观念为主：反复出现难以摆脱的思想或冲动。\n"
"        F42.1 以强迫行为为主：反复、仪式化的动作难以控制。\n"
"        F42.2 强迫观念与强迫行为混合：思想和动作同时反复困扰。\n"
"        F42.9 强迫障碍，未特指：存在强迫症状但分类不详。\n"
"    - F31 双相情感障碍：既往或目前存在躁狂/轻躁狂发作与抑郁发作的交替或混合；需有明确躁狂谱系证据。\n"
"        F31.0 躁狂期，无精神病性症状：躁狂明显但无妄想或幻觉。\n"
"        F31.1 躁狂期，有精神病性症状：躁狂发作伴妄想或幻觉。\n"
"        F31.2 抑郁期，无精神病性症状：抑郁发作但无精神病性特征。\n"
"        F31.3 抑郁期，有精神病性症状：抑郁伴妄想或幻觉。\n"
"        F31.4 混合状态：躁狂与抑郁症状同时或快速交替出现。\n"
"        F31.5 缓解期：既往双相障碍，当前症状缓解。\n"
"        F31.6 其他状态：不符合典型躁狂/抑郁/混合的表现。\n"
"        F31.9 未特指：双相障碍，但无法进一步分类。\n"
"    - F43 对严重应激反应和适应障碍：与明确应激事件有关；可为急性应激反应、PTSD或适应障碍；核心包含再体验、回避、警觉性增高或与应激源相关的情绪/行为改变。\n"
"        F43.0 急性应激反应：暴露于重大应激后立即出现短暂严重反应。\n"
"        F43.1 创伤后应激障碍：经历创伤事件后持续出现再体验、回避和警觉性增高。\n"
"        F43.2 适应障碍：对生活变故反应过度，伴情绪或行为异常。\n"
"        F43.8 其他反应性障碍：与应激相关但不符合特定诊断。\n"
"        F43.9 未特指：应激反应存在，但资料不足以分类。\n"
"    - F45 躯体形式障碍：反复或多样躯体症状为主（如疼痛、心悸、胃肠不适等），检查难以找到足以解释的器质性原因或与病因不相称，显著痛苦/就诊反复。\n"
"        F45.0 躯体化障碍：反复多样的身体症状无器质性解释。\n"
"        F45.1 未分化的躯体形式障碍：躯体症状存在但未达到躯体化标准。\n"
"        F45.2 疑病障碍：持续担忧患严重疾病。\n"
"        F45.3 自主神经功能紊乱型：以心悸、胸闷等自主神经症状为主。\n"
"        F45.4 持续性躯体疼痛障碍：慢性疼痛为主要表现。\n"
"        F45.8 其他躯体形式障碍：特殊类型躯体症状但不归入上述类。\n"
"        F45.9 未特指：存在躯体症状但无法分类。\n"
"    - F20 精神分裂症：在知觉、思维、情感及行为等方面的广泛障碍；常见持续性妄想、幻听、思维松弛/破裂、情感淡漠、阴性症状，病程≥1月（或依本地标准）。\n"
"        F20.0 偏执型：以妄想和幻听为主。\n"
"        F20.1 紊乱型：思维、情感和行为紊乱显著。\n"
"        F20.2 紧张型：以木僵、紧张性兴奋为主要表现。\n"
"        F20.3 未分化型：符合精神分裂症但不属特定亚型。\n"
"        F20.4 残留状态：阴性症状为主，病程长期。\n"
"        F20.5 精神分裂症后抑郁：精神分裂症后出现显著抑郁。\n"
"        F20.6 单纯型：逐渐出现阴性症状，无显著阳性症状。\n"
"        F20.8 其他类型：特殊表现但不属于前述类别。\n"
"        F20.9 未特指：存在精神分裂症特征但资料不足。\n"
"    - Z71 咨询和医疗建议相关因素：包括心理咨询、健康教育、生活方式指导等，当患者主要需要咨询服务而非特定疾病治疗时使用。\n"
"        Z71.9 未特指的咨询：提供咨询，但缺乏具体分类。\n\n"
f"## 对话历史：\n{dialogue_history}\n\n"
f"## 患者背景信息：\n{self.patient_template.get('cleaned_text', '')}\n\n"
"## 注意：\n"
"1. 问诊对话为初次问诊，在症状严重程度和细节不可判断的时候，请推荐未特指的icd code。\n"
"2. 诊断结果可能包含1至2个icd-10诊断结果，大多只包含一个但不超过2个。\n"
"3. 用分号分隔不同的代码。\n"
"4. 需要严格根据icd-10标准来进行诊断的分析, 避免猜测和无根据的诊断，避免诊断错误。\n\n"
"## 输出格式：\n"
"请用中文一步一步思考，并将思考过程放在<think>xxx</think>中输出，之后将最后诊断的ICD-10代码必须放在<box>xxx</box>中输出，用分号分隔，格式如：<think>xxx</think>xxx<box>Fxx.x;Fxx.x</box>。"
)


        diagnosis_messages = [
            {"role": "system", "content": self.doctor_persona},
            {"role": "user", "content": diagnosis_prompt}
        ]
        
        # 带重试机制的诊断生成
        max_retries = 2
        diag_result = None
        diag_reasoning = ""
        chat_response = None
        
        for retry in range(max_retries):
            try:
                chat_response = self.client.chat.completions.create(
                    model=self.api_model_name,
                    messages=diagnosis_messages,
                    temperature=0.3,
                    max_tokens=llm_tools_api.get_max_tokens(),
                    **self._reasoning_kwargs()
                )
                
                # 检查响应有效性
                if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
                    if retry < max_retries - 1:
                        print(f"[Doctor V2] 警告: API返回无效响应（诊断生成），重试第 {retry + 1} 次...")
                        continue
                    else:
                        print(f"[Doctor V2] 错误: API返回无效响应（诊断生成），已达最大重试次数")
                        raise Exception("Doctor诊断生成API返回无效响应")
                
                # 提取响应内容和reasoning（兼容 reasoning_content 与 <think>）
                try:
                    content, reasoning = self._extract_answer_and_reasoning(chat_response)
                    if content is None or str(content).strip() == "":
                        if retry < max_retries - 1:
                            print(f"[Doctor V2] 警告: API返回空内容（诊断生成），重试第 {retry + 1} 次...")
                            continue
                        else:
                            raise Exception("诊断内容为空")
                    
                    diag_result = content
                    diag_reasoning = reasoning
                    break  # 成功获取诊断
                    
                except (AttributeError, IndexError) as e:
                    if retry < max_retries - 1:
                        print(f"[Doctor V2] 警告: 无法提取诊断内容: {e}，重试第 {retry + 1} 次...")
                        continue
                    else:
                        raise Exception(f"诊断API返回格式异常: {e}")
                        
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"[Doctor V2] 警告: API调用异常（诊断生成）: {e}，重试第 {retry + 1} 次...")
                    continue
                else:
                    print(f"[Doctor V2] 错误: API调用异常（诊断生成）: {e}，已达最大重试次数")
                    raise
        
        # 记录token使用量
        prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
            chat_response,
            messages=diagnosis_messages,
            response_text=diag_result
        )
        super().money_cost(prompt_tokens, completion_tokens)
        
        # 移除可能的 xxx 占位符
        diag_result = re.sub(r'^\s*xxx\s*', '', diag_result).strip()
        
        diag_result = "诊断结束，你的诊断结果为：" + diag_result
        return diag_result, diag_reasoning

    def _handle_parse_topic(self, dialogue_history):
        """
        处理parse话题的逻辑
        
        Args:
            dialogue_history (list): 对话历史
        """
        parse_topic, loc, pt, ct = self.diagnosis_tree.parse_experience(dialogue_history)
        print("[V2] *******parse_topic", parse_topic)
        super().money_cost(pt, ct)
        assert loc == self.current_idx
        
        # 检测已解析的话题是否与后续话题重复
        topic_cover, pt, ct = self.diagnosis_tree.topic_detection(self.topic_seq[loc+1:], parse_topic)
        print("[V2] ************topic_cover:", topic_cover)
        super().money_cost(pt, ct)
        
        # 构建需要删除的重复话题列表
        delete_list = [self.topic_seq[loc+1+idx] for idx in range(len(topic_cover)) if topic_cover[idx] == True]
        
        # 将解析出的新话题插入到当前位置
        for i in range(len(parse_topic)):
            self.topic_seq.insert(loc+i+1, parse_topic[i])
        
        # 删除'parse'标记和重复的话题
        del self.topic_seq[loc]    # 删除'parse'
        for item in delete_list:
            self.topic_seq.remove(item)

    def _build_topic_prompt(self, dialogue_history, is_empathy_enabled=True):
        """
        构建话题切换时的提示词
        
        Args:
            dialogue_history (list): 对话历史
            is_empathy_enabled (bool): 是否启用共情能力
            
        Returns:
            str: 构建的提示词
        """
        empathy_text = ""
        if hasattr(self, 'doctor_prompt') and self.doctor_prompt and self.doctor_prompt.get('empathy') == '有':
            empathy_text = "在适当的时候提供与患者的共情"
        else:
            empathy_text = "简洁的生成"
            
        return self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history) + \
            "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。".format(self.topic_seq[self.current_idx]) + \
            f"\n3.你每次只能围绕1个话题询问。使用口语化的表达，{empathy_text}\n4.不要生成类似'谢谢'，'你的回答很有帮助'，'听到你的描述我很'，'你提到'之类的话。不要与历史对话使用相同的开头\n输出有标点符号的一段文字，不要换行。"

    def _build_continuation_prompt(self, dialogue_history):
        """
        构建话题继续时的提示词
        
        Args:
            dialogue_history (list): 对话历史
            
        Returns:
            str: 构建的提示词
        """
        empathy_text = ""
        if hasattr(self, 'doctor_prompt') and self.doctor_prompt and self.doctor_prompt.get('empathy') == '有':
            empathy_text = "在适当的时候提供与患者的共情"
        else:
            empathy_text = "简洁的，口语化的表达进行文本生成"
            
        return self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history) + \
            "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。。".format(self.topic_seq[self.current_idx]) + \
            f"3.\n你每次只能围绕1个话题询问。使用{empathy_text}\n4.不要生成类似'谢谢'，'你的回答很有帮助'，'听到你的描述我很'，'你提到'之类的话。不要与历史对话使用相同的开头\n输出有标点符号的一段文字，不要换行。"



        # diagnosis_prompt = (
        #     f"基于当前的对话历史：{dialogue_history}以及之前的对话信息：{self.patient_template.get('cleaned_text', '')}，"
        #     "输出最终的精神科诊断结果（仅二选一：F32 或 F41）。\n\n"
        #     "请作为一名专业精神健康科主任医师，根据以下 ICD-10 临床描述判断：\n"
        #     "## ICD-10临床描述\n"
        #     "F32 忧郁发作：在典型的轻度、中度或重度忧郁症发作中，患者通常有忧郁情绪、体力下降、活动减少，快乐感、兴趣、注意力均减低，稍微活动即可感到疲倦。常伴有睡眠障碍、食欲减低、自我评价与自信降低，有罪恶感与无用感的意念（即便在轻度时亦可存在）。情绪持续低潮，对生活情境反应减弱。部分患者出现身体性症状，如失去兴趣或快乐感、清晨早醒、晨间症状加重、精神运动迟滞或激动、食欲减退、体重下降、性欲降低。根据症状数量与严重度可分轻度、中度、重度，重度者可伴精神病性症状。\n"
        #     "F41 焦虑障碍：包括恐慌症和广泛性焦虑症。恐慌症的基本特征是反复发作的强烈焦虑（恐慌），发生与情境无关、不可预期，伴心悸、胸痛、窒息感、头晕、不真实感，以及害怕死亡、失控或发狂等。广泛性焦虑症表现为广泛且持续的焦虑，与特定情境无关，常伴紧张、颤抖、肌肉紧张、出汗、头轻飘感、心悸、头晕、上腹不适等，并可能担心自己或亲人的健康或安全。症状可造成明显困扰或功能受损。\n"
        #     "## 判别参考（内部使用，不在输出中展示）\n"
        #     "1) 当抑郁症状为主时，选 F32。\n"
        #     "2) 当焦虑症状为主时，选 F41。\n"
        #     "3) 如两类症状均存在但一方更为突出，则选该方对应的 ICD 代码。\n"
        #     "## 输出要求\n"
        #     "严格按照以下格式输出诊断结果：icd_code{F32} 或 icd_code{F41}（二选一），不要输出其他任何文字。\n"
        # )