# -*- coding: utf-8 -*-
"""
医生角色AI模块 - 基础版本（无诊断树）

该模块实现了基于模型完全自主判断的医生AI对话系统：
- 模拟精神科医生进行患者问诊
- 支持API调用和本地模型两种方式
- 仅凭模型本身的问诊能力，无需诊断树引导
- 模型完全自主判断问诊方向、深度和结束时机
- 无任何外部限制，纯粹测试模型能力
- 提供成本统计功能

适配MDD5k数据结构，测试模型纯粹的基础问诊能力
"""

import json
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import llm_tools_api

class DoctorBase(llm_tools_api.DoctorCost):
    """
    基于模型完全自主判断的基础问诊医生类
    
    该类实现了一个完全依靠模型本身能力进行问诊的AI医生，
    不使用任何预设的诊断树、话题序列或外部限制，让模型自己判断：
    - 问诊方向和策略
    - 问题的深度和广度  
    - 何时结束问诊并给出诊断
    
    继承自DoctorCost类，具备成本统计功能。
    """
    def __init__(self, patient_template, doctor_prompt_path, diagtree_path, model_path, use_api) -> None:
        """
        初始化医生对象
        
        Args:
            patient_template (dict): 患者模板信息，包含Age、Gender、patient_id等（MDD5k格式）
            doctor_prompt_path (str): 医生角色配置文件路径
            diagtree_path (str): 诊断树配置文件路径（基础版本中不使用，仅保持接口兼容性）
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
        self.diagtree_path = diagtree_path               # 诊断树配置文件路径（保持兼容性）
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
        
        # 基础问诊状态管理
        self.question_count = 0                         # 问诊问题计数
        self.MAX_QUESTIONS = 20                         # 最大问诊轮数
        self.doctor_persona = None                      # 医生角色描述
        self.patient_persona = None                     # 患者角色描述
        
        # 输出患者ID用于调试
        patient_id = self.patient_template.get('patient_id', 'Unknown')
        print(f"[BASE] 为患者 {patient_id} 初始化基础问诊系统")
        print(f"[BASE] - 使用模型: {self.model_name}")
        print(f"[BASE] - 最大问诊轮数: {self.MAX_QUESTIONS}")
        print(f"[BASE] - 模型决定问诊流程和结束时机")
        
        # 对话状态管理
        self.expecting_chief_complaint = True           # 是否期待患者主诉回复

    def doctorbot_init(self):
        """
        初始化医生机器人
        
        该方法负责：
        1. 从配置文件中随机选择一个医生角色
        2. 构建医生和患者的角色描述
        3. 生成初始对话提示（包含自主判断指令）
        4. 初始化模型（API或本地模型）
        5. 设置初始对话消息
        """
        # 1. 加载医生配置文件
        with open(self.doctor_prompt_path) as f:
            prompt = json.load(f)
        
        # 2. 随机选择一个医生配置
        doctor_num = random.randint(0, len(prompt)-1)
        self.doctor_prompt = prompt[doctor_num]
        
        # 3. 构建医生角色描述（包含自主判断能力）
        self.doctor_persona = """# 角色定位
你是一名{age}的{gender}专业的精神卫生中心临床心理科主任医师，正在对一名患者进行问诊。

# 问诊习惯
在所有的对话过程中，你需要严格遵循以下问诊习惯：
- 专业特长：你尤其擅长诊断{special}
- 问诊节奏：你的问诊速度是{speed}的
- 交流方式：你的交流风格是{commu}的
- 共情能力：你{empathy}在适当的时候与患者进行共情对话
- 专业解释：你{explain}向患者解释一些专业名词术语

# 自主问诊规则
1. 根据患者回答自主选择下一个问题方向
2. 判断何时信息足够可以诊断
3. 在适当时候标记可以诊断

# 诊断时机判断
请严格按照《国际疾病分类 ICD-10 精神与行为障碍诊断标准》进行判断
- 当你认为收集到足够信息可以做出准确诊断时，请输出："可以诊断：信息已充分"
- 在此之前，继续进行必要的问诊以收集诊断所需信息

# 表达要求
- 使用口语化的表达方式
- 保持专业性和亲和力的平衡
- 不要输出思考过程或前缀“医生：”等任何前缀标识
- 每次可以问1-2个具体的问题

""".format(
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
        
        # 5. 初始化模型（根据配置选择API或本地模型）
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
        
        # 6. 初始化对话消息列表
        self.messages.extend([
            {"role": "system", "content": self.doctor_persona}
        ])

    def _reasoning_kwargs(self):
        """为OpenRouter请求附加reasoning开关配置。"""
        if self.reasoning_extra_body:
            return {"extra_body": self.reasoning_extra_body}
        return {}

    def _extract_answer_and_reasoning(self, chat_response):
        """
        从API响应中分离模型输出与reasoning，兼容多种形式：
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
            # reasoning_content 是实际内容，去除 <think> 标签后返回
            clean_content, _ = self._strip_think_tags(reasoning)
            return clean_content, ""  # 这种情况下没有独立的 reasoning

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

    def doctor_response_gen(self, patient_response, dialogue_history):
        """
        生成医生的回复 - 基础版本
        
        该方法是核心的对话生成逻辑，基于模型自主判断，
        根据对话历史和患者回复生成合适的医生回复或诊断结论。
        
        主要流程：
        1. 判断是否为对话开始，如果是则初始化医生
        2. 让模型自主判断是否继续问诊或给出诊断
        3. 生成相应的问题或诊断结果
        
        Args:
            patient_response (str): 患者的回复内容
            dialogue_history (list): 对话历史记录
            
        Returns:
            tuple: (医生回复, 当前话题, 成本信息) 或 (诊断结果, None, 成本信息)
        """
        if self.use_api:
            # === API模式处理逻辑 ===
            if self.dialbegin == True:
                # 对话开始：初始化医生并生成开场问候
                self.doctorbot_init()
                self.dialbegin = False
                
                # 调用模型生成个性化的开场问候
                greeting_prompt = """现在是问诊的开始，请根据你的医生角色和风格，向患者说一句开场问候语，询问患者的情况。

要求：
1. 友好亲切的一个问候语，并询问患者的详细病情，不要询问例如睡眠、食欲之类的具体症状。
2. 简短自然，不超过30字
3. 直接输出问候语，不要包含任何其他内容
4. 不要使用markdown格式

开场问候："""
                
                self.messages.append({"role": "user", "content": greeting_prompt})
                print(f"[BASE] 生成个性化开场问候...")
                print(f"messages: {self.messages}")
                
                chat_response = self.client.chat.completions.create(
                    model=self.api_model_name,
                    messages=self.messages,
                    top_p=0.93,
                    temperature=0.8,
                    max_tokens=llm_tools_api.get_max_tokens(),
                    **self._reasoning_kwargs()
                )
                
                # 获取响应内容和reasoning（兼容reasoning_content与<think>）
                doctor_greeting, greeting_reasoning = self._extract_answer_and_reasoning(chat_response)
                
                # 记录API调用成本
                prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                    chat_response, 
                    messages=self.messages,
                    response_text=doctor_greeting
                )
                super().money_cost(prompt_tokens, completion_tokens)
                
                print(f"[BASE] 医生个性化问候: {doctor_greeting}")
                
                # 移除prompt，将医生问候添加到对话历史
                self.messages.pop()
                self.messages.append({"role": "assistant", "content": doctor_greeting})
                
                # 存储问候语的reasoning
                self._greeting_reasoning = greeting_reasoning
                
                # 返回医生问候，等待患者主诉
                self.expecting_chief_complaint = False
                # 返回格式：(response, topic, reasoning)
                return doctor_greeting, None, greeting_reasoning
            else:   
                # 对话进行中：让模型自主判断是否继续问诊
                print("dialogue_history: ", dialogue_history)
                # 构建继续问诊的提示，让模型自主判断
                continuation_prompt = self._build_continuation_prompt(patient_response, dialogue_history)
                
                # 将构建的提示添加到消息列表并调用API
                self.messages.append({"role": "user", "content": continuation_prompt})
                print(f"[BASE] 问题 {self.question_count + 1}，模型自主判断问诊方向：")
                
                chat_response = self.client.chat.completions.create(
                    model=self.api_model_name,
                    messages=self.messages,
                    top_p=0.93,
                    temperature=0.7,
                    max_tokens=llm_tools_api.get_max_tokens(),
                    **self._reasoning_kwargs()
                )
                
                # 检查响应有效性
                if chat_response is None:
                    print(f"[BASE] 错误: API返回None，跳过此患者")
                    raise Exception("API返回无效响应")
                
                # 安全地获取响应内容和reasoning
                try:
                    doctor_response, doctor_reasoning = self._extract_answer_and_reasoning(chat_response)
                except (AttributeError, IndexError) as e:
                    print(f"[BASE] 错误: 无法提取响应内容: {e}")
                    raise Exception(f"API返回格式异常: {e}")
                
                # 安全地记录API调用成本（如果usage为None则估算）
                prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                    chat_response, 
                    messages=self.messages,
                    response_text=doctor_response
                )
                super().money_cost(prompt_tokens, completion_tokens)
                
                # 检查是否为诊断结束
                if self._is_diagnosis_response(doctor_response):
                    # 如果是诊断标记，调用专门的诊断方法
                    print(f"[BASE] 模型判断信息充分，开始生成诊断")
                    self.messages.pop()
                    diagnosis_result, diagnosis_reasoning = self._generate_final_diagnosis(dialogue_history)
                    print(f"Doctor_reasoning: {diagnosis_reasoning}")
                    print(f"Doctor_diagnosis: {diagnosis_result}")
                    # 存储诊断的reasoning
                    self._diagnosis_reasoning = diagnosis_reasoning
                    # 返回格式：(response, topic, reasoning) - 诊断时topic为None
                    return diagnosis_result, None, diagnosis_reasoning
                else:
                    # 检查是否达到最大轮数
                    if self.question_count >= self.MAX_QUESTIONS:
                        print(f"[BASE] 达到最大轮数{self.MAX_QUESTIONS}，强制进入诊断")
                        self.messages.pop()
                        diagnosis_result, diagnosis_reasoning = self._generate_final_diagnosis(dialogue_history)
                        print(f"Doctor_reasoning: {diagnosis_reasoning}")
                        print(f"Doctor_diagnosis: {diagnosis_result}")
                        # 存储诊断的reasoning
                        self._diagnosis_reasoning = diagnosis_reasoning
                        # 返回格式：(response, topic, reasoning) - 诊断时topic为None
                        return diagnosis_result, None, diagnosis_reasoning
                    else:
                        # 如果是继续问诊，输出问题并更新计数
                        print(f"Doctor_reasoning: {doctor_reasoning}")
                        print(f"Doctor_ask: {doctor_response}")
                        self.messages.pop()
                        self.question_count += 1
                        # 存储问题的reasoning
                        self._last_question_reasoning = doctor_reasoning
                        # 返回格式：(response, topic, reasoning)
                        return doctor_response, f"问题{self.question_count}", doctor_reasoning
        else:
            # === 本地模型处理逻辑 ===
            if self.dialbegin == True:
                # 对话开始：初始化医生并生成开场问候
                self.doctorbot_init()
                self.dialbegin = False
                
                # 调用模型生成个性化的开场问候
                greeting_prompt = """现在是问诊的开始，请根据你的医生角色和风格，向患者说一句开场问候语，询问患者的情况。

要求：
1. 体现你的问诊风格（专业特长、问诊节奏、交流方式）
2. 简短自然，不超过30字
3. 直接输出问候语，不要包含任何其他内容
4. 不要使用markdown格式

开场问候："""
                
                self.messages.append({"role": "user", "content": greeting_prompt})
                print(f"[BASE] 生成个性化开场问候...")
                
                # 格式化消息并生成回复
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=128,
                    temperature=0.8,
                    top_p=0.93
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                raw_greeting = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                doctor_greeting, greeting_reasoning = self._strip_think_tags(raw_greeting)
                
                print(f"[BASE] 医生个性化问候: {doctor_greeting}")
                
                # 移除prompt，将医生问候添加到对话历史
                self.messages.pop()
                self.messages.append({"role": "assistant", "content": doctor_greeting})
                
                # 返回医生问候，等待患者主诉
                self.expecting_chief_complaint = False
                # 返回格式：(response, topic, reasoning) - 本地模型没有reasoning
                return doctor_greeting, None, greeting_reasoning
            else:
                # 对话进行中：让模型自主判断是否继续问诊（与API模式保持一致）
                # 构建继续问诊的提示，让模型自主判断
                continuation_prompt = self._build_continuation_prompt(patient_response, dialogue_history)
                
                # 将构建的提示添加到消息列表
                self.messages.append({"role": "user", "content": continuation_prompt})
                print(f"[BASE] 问题 {self.question_count + 1}，模型自主判断问诊方向：")
                
                # 格式化消息并生成回复
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.93
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                raw_response = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                doctor_response, doctor_reasoning = self._strip_think_tags(raw_response)
                
                # 检查是否为诊断结束
                if self._is_diagnosis_response(doctor_response):
                    print(f"[BASE] 模型判断信息充分，开始生成诊断")
                    self.messages.pop()  # 移除临时添加的continuation_prompt
                    diagnosis_result, diagnosis_reasoning = self._generate_final_diagnosis(dialogue_history)
                    print(f"Doctor_diagnosis: {diagnosis_result}")
                    # 返回格式：(response, topic, reasoning)
                    return diagnosis_result, None, diagnosis_reasoning
                else:
                    # 检查是否达到最大轮数
                    if self.question_count >= self.MAX_QUESTIONS:
                        print(f"[BASE] 达到最大轮数{self.MAX_QUESTIONS}，强制进入诊断")
                        self.messages.pop()  # 移除临时添加的continuation_prompt
                        diagnosis_result, diagnosis_reasoning = self._generate_final_diagnosis(dialogue_history)
                        print(f"Doctor_diagnosis: {diagnosis_result}")
                        # 返回格式：(response, topic, reasoning)
                        return diagnosis_result, None, diagnosis_reasoning
                    else:
                        # 输出医生问题
                        print(f"Doctor_reasoning: {doctor_reasoning}")
                        print(f"Doctor_ask: {doctor_response}")
                        # 移除临时添加的continuation_prompt，将医生回复添加到对话历史
                        self.messages.pop()
                        self.messages.append({"role": "assistant", "content": doctor_response})
                        self.question_count += 1
                        # 返回格式：(response, topic, reasoning) - 本地模型没有reasoning
                        return doctor_response, f"问题{self.question_count}", doctor_reasoning


    def _build_continuation_prompt(self, patient_response, dialogue_history):
        """
        构建继续问诊的提示词，让模型自主判断
        
        Args:
            patient_response (str): 患者最新回复
            dialogue_history (list): 对话历史
            
        Returns:
            str: 构建的提示词
        """
        # 计算剩余可问诊轮数
        remaining_turns = self.MAX_QUESTIONS - self.question_count
        urgency_hint = ""
        if remaining_turns <= 5:
            urgency_hint = f"\n【注意】只剩 {remaining_turns} 轮问诊机会，请尽快收集关键信息并准备诊断。"
        elif remaining_turns <= 10:
            urgency_hint = f"\n【提示】剩余 {remaining_turns} 轮问诊机会，请高效问诊。"
        
        return f"""
你是一名精神科医生，正在对患者{self.patient_persona}进行精神科问诊。
当前是第 {self.question_count + 1} 轮问诊（最多 {self.MAX_QUESTIONS} 轮）。{urgency_hint}

## 对话历史
{dialogue_history[-8:] if len(dialogue_history) > 8 else dialogue_history}

## 患者最新回复
{patient_response}

## 请选择下一步行动：

### 选项1：继续问诊
如果你认为还需要更多信息才能做出准确诊断，请提出新的问题。
- 不要重复询问已经问过的问题或患者已经回答过的内容
- 每次只问1-2个具体问题
- 问题要有针对性，帮助确定诊断

### 选项2：准备诊断
如果你认为已经收集到足够信息，可以按照ICD-10标准做出诊断时，请**严格**输出以下固定格式：
可以诊断：信息已充分

## 输出要求
- 直接输出问题内容或诊断指令，不要加"医生："前缀
- 不要重复对话历史中已经问过的问题
- 不要输出思考过程

请给出你的回复："""

    def get_last_reasoning(self):
        """
        获取最近一次交互的reasoning内容
        
        Returns:
            str: reasoning内容，如果没有则返回空字符串
        """
        if hasattr(self, '_last_question_reasoning'):
            return self._last_question_reasoning
        elif hasattr(self, '_greeting_reasoning'):
            return self._greeting_reasoning
        elif hasattr(self, '_diagnosis_reasoning'):
            return self._diagnosis_reasoning
        return ""
    
    def _is_diagnosis_response(self, response):
        """
        判断回复是否为可以做出诊断的标记
        
        Args:
            response (str): 医生回复
            
        Returns:
            bool: 是否为准备诊断标记
        """
        diagnosis_indicators = [
            "可以诊断：信息已充分",
            "可以诊断",
            "信息已充分"
        ]
        return any(indicator in response for indicator in diagnosis_indicators)

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
        
        # f"基于当前的对话历史：{dialogue_history} 以及之前的对话信息：{self.patient_template.get('cleaned_text', '')}，"
        #     "请输出本次会诊的最终精神科主诊断（仅从以下 ICD-10 代码中选择其一："
        #     "F20、F31、F32、F39、F41、F42、F43、F45、F51、F98）。\n\n"
        #     "你是一名资深精神科主任医师。请严格参考《国际疾病分类 ICD-10 精神与行为障碍诊断标准（中文版）》的临床原始描述，结合患者的症状表现、病程特点和功能受损情况来进行判断，而不是仅仅依赖要点式总结。\n\n"
        #     "## 诊断范围（含要点提示）\n"
        #     "- F20 精神分裂症：持续 ≥1 月的妄想、幻觉、思维形式障碍或明显阴性症状，排除器质性和物质原因。\n"
        #     "- F31 躁郁症（双相情感障碍）：至少一次躁狂或轻躁狂发作，常与抑郁发作交替。\n"
        #     "- F32 抑郁发作：以情绪低落、兴趣缺乏、快感缺失为核心，伴疲乏、睡眠食欲改变、自罪或无价值感。\n"
        #     "- F39 情感障碍未特指：临床明确存在情感障碍，但无法明确归入抑郁或双相等具体亚型。\n"
        #     "- F41 焦虑障碍：包括恐慌发作或广泛性焦虑，表现为持续紧张、担忧及自主神经症状。\n"
        #     "- F42 强迫障碍：反复出现的强迫思维或强迫行为，个体自知过度，但难以控制，显著影响功能。\n"
        #     "- F43 应激相关障碍：在重大应激或创伤后出现急性应激反应、创伤后应激障碍或适应障碍。\n"
        #     "- F45 躯体形式障碍：以难以用医学解释的多种躯体主诉为核心，伴健康焦虑及反复求医。\n"
        #     "- F51 非器质性睡眠障碍：失眠、过度嗜睡、噩梦、睡眠节律紊乱等，排除器质性原因。\n"
        #     "- F98 儿童期起病的其他行为与情绪障碍：如遗尿、遗粪、拔毛、习惯性抽动、儿童期睡眠障碍等。\n\n"
        #     "## 输出要求\n"
        #     "严格按照以下格式输出唯一诊断结果：icd_code{F20|F31|F32|F39|F41|F42|F43|F45|F51|F98}。\n"
        #     "除上述花括号内的唯一 ICD 代码外，不要输出任何其他文字、标点或解释。\n"
        
        
        # diagnosis_prompt = (
        #     f"基于当前的对话历史：{dialogue_history}以及之前的对话信息：{self.patient_template['cleaned_text']}，"
        #     "输出最终的精神科诊断结果。\n\n"
        #     "你是一名资深精神科主任医师。请严格参考《国际疾病分类 ICD-10 精神与行为障碍诊断标准（中文版）》的临床原始描述，结合患者的症状表现、病程特点和功能受损情况来进行判断，而不是仅仅依赖要点式总结。\n\n"
        #     "## 诊断范围（含要点提示）\n"
        #     "F32 忧郁发作：在典型的轻度、中度或重度忧郁症发作中，患者通常有忧郁情绪、体力下降、活动减少，快乐感、兴趣、注意力均减低，稍微活动即可感到疲倦。常伴有睡眠障碍、食欲减低、自我评价与自信降低，有罪恶感与无用感的意念（即便在轻度时亦可存在）。情绪持续低潮，对生活情境反应减弱。部分患者出现身体性症状，如失去兴趣或快乐感、清晨早醒、晨间症状加重、精神运动迟滞或激动、食欲减退、体重下降、性欲降低。根据症状数量与严重度可分轻度、中度、重度，重度者可伴精神病性症状。\n"
        #     "F41 焦虑障碍：包括恐慌症和广泛性焦虑症。恐慌症的基本特征是反复发作的强烈焦虑（恐慌），发生与情境无关、不可预期，伴心悸、胸痛、窒息感、头晕、不真实感，以及害怕死亡、失控或发狂等。广泛性焦虑症表现为广泛且持续的焦虑，与特定情境无关，常伴紧张、颤抖、肌肉紧张、出汗、头轻飘感、心悸、头晕、上腹不适等，并可能担心自己或亲人的健康或安全。症状可造成明显困扰或功能受损。\n"
        #     "F32,F41 焦虑抑郁混合障碍：抑郁与焦虑症状同时存在且都达到ICD-10诊断标准。若能够判断主次，应优先选择单一类别（F32 或 F41）。\n"
        #     "Others：除 F32、F41、F32,F41 之外的其他精神障碍，或当前信息不足以明确归入前三类，也包括无明显精神症状的情形。常见范围（ICD-10）：F00–F09 器质性（含症状性）精神障碍；F10–F19 物质使用所致障碍；F20–F29 精神分裂谱系与妄想性障碍；F30–F39 情感性障碍；F40–F48 神经症性/应激性/躯体形式障碍；F50–F59 伴生理障碍与躯体因素的行为综合征；F60–F69 成人人格与行为障碍；F70–F79 精神发育迟缓；F80–F89 心理发育障碍；F90–F98 儿童青少年期行为与精神障碍；F99 精神障碍NOS。\n"
        #     "## 判别参考（内部使用，不在输出中展示）\n"
        #     "1) 优先判断单一类别（F32 或 F41），如一方症状明显占优则直接选该类。\n"
        #     "2) 仅在两类症状与严重程度接近、抑郁与焦虑症状同时存在且都达到ICD-10诊断标准时，才选 F32,F41。\n"
        #     "3) 当抑郁和焦虑症状均不足、或症状特征更符合上述其他精神障碍，或呈现无明显精神症状时，选 Others。\n"
        #     "## 输出要求\n"
        #     "严格按照以下格式输出诊断结果：icd_code{F32}、icd_code{F41}、icd_code{F32,F41} 或 icd_code{Others}（四选一），不要输出其他任何文字。\n"
        # )
        
        # diagnosis_prompt = (
        #     f"基于当前的对话历史：{dialogue_history}以及之前的对话信息：{self.patient_template.get('cleaned_text', '')}，"
        #     "输出最终的精神科诊断结果 \n\n"
        #     "你是一名资深精神科主任医师。请严格参考《国际疾病分类 ICD-10 精神与行为障碍诊断标准（中文版）》的临床原始描述，结合患者的症状表现、病程特点和功能受损情况来进行判断，而不是仅仅依赖要点式总结。\n\n"
        #     "## 诊断范围（含要点提示）"
        #     "F32 忧郁发作：在典型的轻度、中度或重度忧郁症发作中，患者通常有忧郁情绪、体力下降、活动减少，快乐感、兴趣、注意力均减低，稍微活动即可感到疲倦。常伴有睡眠障碍、食欲减低、自我评价与自信降低，有罪恶感与无用感的意念（即便在轻度时亦可存在）。情绪持续低潮，对生活情境反应减弱。部分患者出现身体性症状，如失去兴趣或快乐感、清晨早醒、晨间症状加重、精神运动迟滞或激动、食欲减退、体重下降、性欲降低。根据症状数量与严重度可分轻度、中度、重度，重度者可伴精神病性症状。\n"
        #     "F41 焦虑障碍：包括恐慌症和广泛性焦虑症。恐慌症的基本特征是反复发作的强烈焦虑（恐慌），发生与情境无关、不可预期，伴心悸、胸痛、窒息感、头晕、不真实感，以及害怕死亡、失控或发狂等。广泛性焦虑症表现为广泛且持续的焦虑，与特定情境无关，常伴紧张、颤抖、肌肉紧张、出汗、头轻飘感、心悸、头晕、上腹不适等，并可能担心自己或亲人的健康或安全。症状可造成明显困扰或功能受损。\n"
        #     "## 判别参考（内部使用，不在输出中展示）\n"
        #     "1) 当抑郁症状为主时，选 F32。\n"
        #     "2) 当焦虑症状为主时，选 F41。\n"
        #     "3) 如两类症状均存在但一方更为突出，则选该方对应的 ICD 代码。\n"
        #     "## 输出要求\n"
        #     "严格按照以下格式输出诊断结果：icd_code{F32} 或 icd_code{F41}（二选一），不要输出其他任何文字。\n"
        # )
        

        diagnosis_messages = [
            {"role": "system", "content": self.doctor_persona},
            {"role": "user", "content": diagnosis_prompt}
        ]
        
        if self.use_api:
            chat_response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=diagnosis_messages,
                temperature=0.1,
                max_tokens=llm_tools_api.get_max_tokens(),
                **self._reasoning_kwargs()
            )
            
            # 检查响应有效性
            if chat_response is None:
                print(f"[BASE] 错误: 诊断API返回None")
                raise Exception("诊断API返回无效响应")
            
            # 安全地获取诊断结果和reasoning（兼容 reasoning_content 与 <think>）
            try:
                diag_result, diag_reasoning = self._extract_answer_and_reasoning(chat_response)
            except (AttributeError, IndexError) as e:
                print(f"[BASE] 错误: 无法提取诊断内容: {e}")
                raise Exception(f"诊断API返回格式异常: {e}")
            
            # 安全地记录API调用成本（如果usage为None则估算）
            prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                chat_response,
                messages=diagnosis_messages,
                response_text=diag_result
            )
            super().money_cost(prompt_tokens, completion_tokens)
        else:
            # 本地模型诊断逻辑
            text = self.doctor_tokenizer.apply_chat_template(
                diagnosis_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
            
            generated_ids = self.doctor_model.generate(
                doctor_model_inputs.input_ids,
                max_new_tokens=128,
                temperature=0.3,
                top_p=0.93
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
            ]
            
            raw_diag_result = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            diag_result, diag_reasoning = self._strip_think_tags(raw_diag_result)
        
        # 移除可能的 xxx 占位符
        diag_result = re.sub(r'^\s*xxx\s*', '', diag_result).strip()
        
        # 添加诊断结束前缀（如果还没有）
        if not diag_result.startswith("诊断结束"):
            diag_result = "诊断结束，你的诊断结果为：" + diag_result
        
        return diag_result, diag_reasoning

