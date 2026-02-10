import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import llm_tools_api
from src.patient.name_generator import generate_patient_name_with_id


def api_call_with_retry(client, model_name, messages, max_retries=2, default_response="默认响应", context="API调用", **kwargs):
    """
    带重试机制的API调用辅助函数
    
    Args:
        client: API客户端
        model_name: 模型名称
        messages: 消息列表
        max_retries: 最大重试次数
        default_response: 失败时的默认响应
        context: 上下文描述（用于日志）
        **kwargs: 其他API参数
        
    Returns:
        tuple: (response_content, reasoning_content, chat_response) 或 (default_response, "", None)
    """
    for retry in range(max_retries):
        try:
            # 为OpenRouter API添加reasoning配置
            api_params = kwargs.copy()
            
            # if 'extra_body' not in api_params:
            #     api_params['extra_body'] = llm_tools_api.get_reasoning_config(enabled=False)
            
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **api_params
            )
            
            # 检查响应有效性
            if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
                if retry < max_retries - 1:
                    print(f"[{context}] 警告: API返回无效响应，重试第 {retry + 1} 次...")
                    continue
                else:
                    print(f"[{context}] 错误: API返回无效响应，已达最大重试次数")
                    return default_response, "", None
            
            # 使用统一函数分离content和reasoning
            try:
                content, reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
                if content is None or content.strip() == "":
                    if retry < max_retries - 1:
                        print(f"[{context}] 警告: API返回空内容，重试第 {retry + 1} 次...")
                        continue
                    else:
                        print(f"[{context}] 错误: API返回空内容，已达最大重试次数")
                        return default_response, "", None
                
                return content, reasoning, chat_response  # 成功
                
            except (AttributeError, IndexError) as e:
                if retry < max_retries - 1:
                    print(f"[{context}] 警告: 无法提取响应内容: {e}，重试第 {retry + 1} 次...")
                    continue
                else:
                    print(f"[{context}] 错误: 无法提取响应内容: {e}，已达最大重试次数")
                    return default_response, "", None
                    
        except Exception as e:
            if retry < max_retries - 1:
                print(f"[{context}] 警告: API调用异常: {e}，重试第 {retry + 1} 次...")
                continue
            else:
                print(f"[{context}] 错误: API调用异常: {e}，已达最大重试次数")
                return default_response, "", None
    
    return default_response, "", None


class Patient(llm_tools_api.PatientCost):
    def __init__(self, patient_template, model_path, use_api, enable_chief_complaint=True) -> None:
        super().__init__(model_path)
        self.model_path = model_path
        self.model_name = model_path
        # 提取基础模型名称用于API调用中的model参数（去掉@host:port部分）
        if '@' in model_path:
            self.api_model_name = model_path.split('@')[0]
        else:
            self.api_model_name = model_path.split(':')[0]
        self.patient_model = None
        self.patient_tokenizer = None

        self.patient_template = patient_template
        self.use_api = use_api
        # 判断是否使用OpenRouter（检查模型名称格式）
        self.use_openrouter = llm_tools_api.is_openrouter_model(model_path)
        self.reasoning_enabled = llm_tools_api.is_reasoning_enabled("patient", default=True)
        self.reasoning_extra_body = llm_tools_api.build_reasoning_extra_body(
            self.reasoning_enabled, self.use_openrouter, self.model_name
        )

        self.system_prompt = self._generate_system_prompt()
        self.system_prompt = llm_tools_api.apply_reasoning_prompt_prefix(
            self.system_prompt, self.model_name, self.use_openrouter, self.reasoning_enabled
        )
        self.messages = []
        self.client = None
        self.dialbegin = True
        
        # 患者主诉生成开关
        self.enable_chief_complaint = enable_chief_complaint
        print(f"[PATIENT] 患者主诉生成功能: {'启用' if enable_chief_complaint else '禁用'}")
        
        # OpenRouter站点信息配置（可选）
        import os
        self.openrouter_site_url = os.getenv('OPENROUTER_SITE_URL')
        self.openrouter_site_name = os.getenv('OPENROUTER_SITE_NAME')

    def _generate_system_prompt(self):
        """根据患者模板数据生成系统提示，处理空值情况"""
        age = self.patient_template['Age']
        gender = self.patient_template['Gender']
        chief_complaint = self.patient_template.get('ChiefComplaint')
        department = self.patient_template['Department']
        
        # 生成患者姓名
        patient_id = self.patient_template.get('patient_id', self.patient_template.get('患者', 0))
        patient_name = generate_patient_name_with_id(patient_id, gender, age)
        
        # 基础身份描述（包含姓名）
        identity = f"你是一名{age}岁的{gender}性患者，你的名字叫{patient_name}"
        
        # 处理主诉：检查是否为None或空字符串
        if chief_complaint and chief_complaint.strip():
            # 去除"主诉："前缀，只保留核心内容
            clean_complaint = chief_complaint.replace('主诉：', '').strip()
            reason = f"，你因为{clean_complaint}，来到医院{department}就诊"
        else:
            reason = f"，来到医院{department}就诊"
        
        # 组合完整的系统提示
        system_prompt = f"{identity}{reason}。使用口语化的表达，如果医生的问题可以用是/否来回答，你的回复要简短精确。不要有括号的使用。"
        
        return system_prompt

    def patientbot_init(self):
        if self.use_api:
            self.client = llm_tools_api.patient_client_init(self.model_name)
        else:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.patient_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.messages.append({"role": "system", "content": self.system_prompt})

    def _reasoning_kwargs(self):
        """为OpenRouter请求附加reasoning配置。"""
        if self.reasoning_extra_body:
            return {"extra_body": self.reasoning_extra_body}
        return {}


    def _classify_question(self, current_doctor_question, patient_template, dialogue_history):
        """第一步：对医生问题进行分类"""
        classification_prompt = get_classification_prompt(
            current_doctor_question=current_doctor_question,
            patient_info=patient_template['Patient info'],
            cleaned_text=patient_template['cleaned_text'],
            dialogue_history=dialogue_history
        )
        
        # 临时添加分类prompt到消息中
        temp_messages = self.messages + [{"role": "user", "content": classification_prompt}]
        
        # 使用重试机制调用API（现在返回reasoning）
        classification_result, classification_reasoning, chat_response = api_call_with_retry(
            client=self.client,
            model_name=self.api_model_name,
            messages=temp_messages,
            max_retries=2,
            default_response="1",  # 默认分类为1
            context="Patient分类",
            top_p=0.1,
            temperature=0.1,
            **self._reasoning_kwargs()
        )
        
        classification_result = classification_result.strip()
        
        # 记录API调用成本
        if chat_response is not None:
            prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                chat_response,
                messages=temp_messages,
                response_text=classification_result
            )
            super().money_cost(prompt_tokens, completion_tokens)
        
        print(f"\n=== 问题分类调试信息 ===")
        print(f"[分类] 当前医生问题: {current_doctor_question}")
        print(f"[分类] 分类结果: {classification_result}")
        if classification_reasoning:
            print(f"[分类] 分类推理: {classification_reasoning[:200]}...")  # 只显示前200字符
        
        # 提取分类数字
        try:
            category = int(classification_result)
            if category not in [1, 2, 3, 4]:
                print(f"[分类] 警告: 分类结果 {category} 不在有效范围内，使用默认值 1")
                category = 1  # 默认为第一类
            else:
                print(f"[分类] 最终分类类别: {category}")
        except ValueError:
            print(f"[分类] 错误: 无法解析分类结果 '{classification_result}'，使用默认值 1")
            category = 1  # 解析失败时默认为第一类
            
        print(f"=== 分类完成，使用类别 {category} 的 prompt ===\n")
        return category, classification_reasoning
    
    def _generate_response_by_category(self, category, current_doctor_question, patient_template, dialogue_history):
        """第二步：根据分类生成回复"""
        print(f"\n=== 生成回复 (类别 {category}) ===")
        # 使用prompt工厂函数创建prompt
        patient_prompt = create_response_prompt(
            category=category,
            current_doctor_question=current_doctor_question,
            patient_info=patient_template.get('Patient info', ''),
            cleaned_text=patient_template.get('cleaned_text', ''),
            dialogue_history=dialogue_history
        )
        
        # 发送请求
        self.messages.append({"role": "user", "content": patient_prompt})
        
        # 打印prompt的前200个字符，帮助调试
        prompt_preview = patient_prompt[:200].replace('\n', ' ')
        print(f"[回复生成] Prompt预览: {prompt_preview}...")
        # print(f"Patient messages: {self.messages}")  # 可选：完整的messages太长，暂时注释
        
        # 准备API调用参数
        api_kwargs = {"top_p": 0.85, "frequency_penalty": 0.8}
        
        if self.use_openrouter:
            # OpenRouter API 调用（需要extra_headers）
            extra_headers = {}
            if self.openrouter_site_url:
                extra_headers["HTTP-Referer"] = self.openrouter_site_url
            if self.openrouter_site_name:
                extra_headers["X-Title"] = self.openrouter_site_name
            api_kwargs["extra_headers"] = extra_headers
        
        # 使用重试机制调用API（现在返回reasoning）
        patient_response, response_reasoning, chat_response = api_call_with_retry(
            client=self.client,
            model_name=self.api_model_name,
            messages=self.messages,
            max_retries=2,
            default_response="我不太明白您的问题，您能再说一遍吗？",
            context="Patient回复生成",
            **api_kwargs,
            **self._reasoning_kwargs()
        )
        
        # 清除回答的非法token和换行符
        patient_response = patient_response.replace('<think>', '').replace('</think>', '').replace("\n", "")
        
        # 记录API调用成本
        if chat_response is not None:
            prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                chat_response,
                messages=self.messages,
                response_text=patient_response
            )
            super().money_cost(prompt_tokens, completion_tokens)
        
        print(f"[回复生成] 患者回复: {patient_response}")
        if response_reasoning:
            print(f"[回复生成] 回复推理: {response_reasoning[:200]}...")  # 只显示前200字符
        print(f"=== 回复生成完成 ===\n")
        
        self.messages.pop()  # 移除临时添加的prompt
        
        return patient_response, response_reasoning

    def patient_response_gen(self, current_topic, dialogue_history, current_doctor_question):
        """外部CoT：两步生成患者回复"""
        classification_info = None
        patient_reasoning = ""  # 患者响应的reasoning
        
        if self.use_api:
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
                
            patient_template = {key: val for key, val in self.patient_template.items() if key != '处理意见'}

            # 第一步：问题分类
            category, classification_reasoning = self._classify_question(current_doctor_question, patient_template, dialogue_history)
            
            # 保存分类信息（包含reasoning）
            classification_info = {
                "category": category,
                "classification_reasoning": classification_reasoning  # 分类步骤的reasoning
            }
            
            # 第二步：根据分类生成回复
            patient_response, response_reasoning = self._generate_response_by_category(
                category, current_doctor_question, patient_template, dialogue_history
            )
            
            # 保存响应生成步骤的reasoning
            patient_reasoning = response_reasoning
            
        else:
            # TODO: 实现本地模型的CoT逻辑
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            text = self.patient_tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            patient_model_inputs = self.patient_tokenizer([text], return_tensors="pt").to(self.patient_model.device)
            generated_ids = self.patient_model.generate(
                patient_model_inputs.input_ids,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(patient_model_inputs.input_ids, generated_ids)
            ]
            patient_response = self.patient_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.messages.append({"role": "assistant", "content": patient_response})
        
        return patient_response, super().get_cost(), classification_info, patient_reasoning

    def api_generate_chief_complaint(self):
        """
        生成患者主诉回复 - 移至患者模块的架构重构版本
        
        按照优先级从patient_template中提取信息，生成患者的简短主诉：
        1. 优先使用ChiefComplaint
        2. 其次使用PresentIllnessHistory  
        3. 再次总结cleaned_text
        4. 最后提供兜底回复
        
        Returns:
            str: 患者的简短主诉回复
        """
        if not self.enable_chief_complaint:
            # 如果禁用主诉生成，返回通用回复
            return "医生，我最近感觉不太舒服，心情不好，想来看看。"
            
        patient_id = self.patient_template.get('patient_id', 'unknown')
        
        # 按优先级获取患者信息
        source_content = ""
        content_type = ""
        
        # 1. 优先使用ChiefComplaint
        chief_complaint = self.patient_template.get('ChiefComplaint', '')
        if chief_complaint and chief_complaint.strip():
            # 去除"主诉："前缀，只保留核心内容
            source_content = chief_complaint.replace('主诉：', '').replace('主诉:', '').strip()
            content_type = "主诉"
        # 2. 其次使用PresentIllnessHistory
        elif self.patient_template.get('PresentIllnessHistory') and self.patient_template['PresentIllnessHistory'].strip():
            source_content = self.patient_template['PresentIllnessHistory']
            content_type = "现病史"
        # 3. 再次使用cleaned_text总结
        elif self.patient_template.get('cleaned_text') and self.patient_template['cleaned_text'].strip():
            source_content = self.patient_template['cleaned_text']
            content_type = "病历文本"
        # 4. 兜底方案
        else:
            return "医生，我最近感觉不太舒服，心情不好，想来看看。"
        
        # 改进提示词，确保模型能理解并回复
        patient_prompt = f"""你是一名患者，正在向精神科医生描述你的情况。

医生询问："你的情况怎么样？有什么不舒服的地方吗？"

你的实际症状和困扰：{source_content}

请注意：
1. 用自然口语化的方式表达，就像和医生正常对话一样
2. 不要直接复述上述描述性文字，要用第一人称"我"来表达
3. 用1-2句简短的话说明主要的不适症状和持续时间
4. 语气要自然，像日常说话一样

患者回答："""

        # 根据使用的模型类型调用LLM
        if self.use_api:
            # 确保客户端已初始化
            if not self.client:
                self.client = llm_tools_api.patient_client_init(self.model_name)
                
            print(f"[PATIENT] 患者 {patient_id}: 生成主诉中... (基于{content_type})")
            print(f"[PATIENT] 主诉内容: {source_content[:150]}{'...' if len(source_content) > 150 else ''}")
            
            # 确保客户端和消息历史已初始化
            if not hasattr(self, 'messages') or not self.messages:
                self.patientbot_init()
            
            # 临时添加到messages中
            self.messages.append({"role": "user", "content": patient_prompt})
            
            # 使用重试机制调用API（现在返回reasoning）
            patient_complaint, complaint_reasoning, chat_response = api_call_with_retry(
                client=self.client,
                model_name=self.api_model_name,
                messages=self.messages,
                max_retries=2,
                default_response="医生，我最近感觉不太舒服，心情不好，想来看看。",
                context=f"Patient主诉生成(患者{patient_id})",
                top_p=0.85,
                frequency_penalty=0.8,
                **self._reasoning_kwargs()
            )
            
            # 移除临时添加的prompt
            self.messages.pop()
            
            # 记录API调用成本
            if chat_response is not None:
                temp_messages = self.messages.copy()
                temp_messages.append({"role": "user", "content": patient_prompt})
                prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                    chat_response,
                    messages=temp_messages,
                    response_text=patient_complaint
                )
                super().money_cost(prompt_tokens, completion_tokens)
            
            patient_complaint = patient_complaint.strip()
            # 存储reasoning（如果需要的话）
            self._chief_complaint_reasoning = complaint_reasoning
            
        else:
            # 本地模型方式生成患者主诉
            # 确保模型已初始化
            if not self.patient_model:
                self.patient_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                self.patient_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                
            complaint_messages = [
                {"role": "user", "content": patient_prompt}
            ]
            
            text = self.patient_tokenizer.apply_chat_template(
                complaint_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.patient_tokenizer([text], return_tensors="pt").to(self.patient_model.device)
            
            generated_ids = self.patient_model.generate(
                model_inputs.input_ids,
                max_new_tokens=100,
                temperature=0.1,
                top_p=0.9
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            # 添加空值检查
            decoded_response = self.patient_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if decoded_response is None or decoded_response.strip() == "":
                patient_complaint = "医生，我最近感觉不太舒服，心情不好，想来看看。"
            else:
                patient_complaint = decoded_response.strip()
        
        print(f"[PATIENT] 基于{content_type}生成患者主诉: {patient_complaint}")
        return patient_complaint
    

# ==================== Prompt Templates ====================

def get_common_rules():
    """获取通用硬性规则"""
    return """
回答时遵循以下硬性规则：
1. 回答时直接用第一人称作答，不要复述医生的问题，也不要出现"医生问/我回答"等措辞，语气要自然，可以口语化。避免机械重复句式。不要有括号的使用。
2. 只回答与【当前医生问题】直接相关的内容，不夹带无关信息，不重复在【最近几轮对话】中已经回答过的事实（若医生重复提问，可以简短确认）
3. 如果医生的问题包含诱导性细节，只有在证据中出现过相同或等价表述时才能复述；否则使用"未有此体验/不清楚/没注意频率"来回答问题。
4. 可适当在已知病情范围内延伸，但不能给出与已知事实矛盾的回答。
5. 若有【最近几轮对话】，回答时不要与【最近几轮对话】中的已陈述内容相矛盾；若发现前后表述不一，使用“记不清/我刚才说得不准确”进行自然澄清。
6. 补充说明的内容不要放在括号里，自然回答。
"""

def get_classification_prompt(current_doctor_question, patient_info, cleaned_text, dialogue_history):
    """获取问题分类prompt（4类，单字符输出）"""
    return f"""
你是一位精神科患者，正在和医生交流。

【定义】
- 【病例】：你的既往病历与基本信息。
- 【既往对话】：你与医生以往已经确认过的对话内容（可作为回答依据）。
- 【最近几轮对话】：本次会话的近期上下文（仅用于理解提问语境与避免重复，不作为新增事实依据）。
-  当【病例】、【既往对话】和【最近几轮对话】出现矛盾时，以【既往对话】为准；

【病例】：{patient_info}
【既往对话】：{cleaned_text}
【最近几轮对话】：{dialogue_history}
【当前医生问题】：{current_doctor_question}

请把医生的问题归入以下4类之一，只输出对应的数字（且仅输出这一位数字）：

1. 目前信息可以回答医生关于问诊的问题
- 问题具体（如涉及来这里的主要目的/大致情况/症状/部位/时间/频率/程度等）
- 且在【病例】或【既往对话】中能找到可直接对应的信息（同义/等价表述亦可）

2. 目前信息不足以回答医生问题或者无法判断
- 问题具体，但【病例】与【既往对话】都没有对应信息，或属于无法提供的数据
- 不允许凭想象补全

3. 医生的问题过于模糊，需要医生更具体
- 问题语义不通，和上下文没有连接，或者突兀，根据上下文无法确定到底医生询问哪一方面，需要询问医生想问什么

4. 医生的问题是日常闲聊
- 与病情无直接关系的社交/家常话题等

【判定顺序（冲突时按此优先级执行）】
先判 4（家常）→ 再判 3（模糊）→ 再判 1（具体且有依据）→ 否则 2（具体但无依据/无法提供/无法判断）

【多子问题处理】
- 若问题包含多个子问题：只要其中任一子问题满足“具体且有依据”，判为1；
- 若都很具体但都无依据，判为2；
- 若整体仍不具体，判为3。

【输出要求】
- 只输出一个数字：1 或 2 或 3 或 4
- 不要输出任何其他字符、标点、解释或换行。
""".strip()


def get_response_prompt_type1(current_doctor_question, patient_info, cleaned_text, dialogue_history):
    """类型1：可回答的病情问题"""
    return f"""
你正在和一位精神科医生进行交流，基于你的【病例】、【既往对话】和【最近几轮对话】中的已知事实，回答医生的问题。

【当前医生问题】：{current_doctor_question}
【病例】：{patient_info}
【既往对话】：{cleaned_text}
【最近几轮对话】：{dialogue_history}

{get_common_rules()}

【额外规则】
1) 先给直接结论（"是/不是/有/没有/简短事实"其一），如需详细说明，最多再加1–2句自然描述；所有内容必须能在上述证据中找到原句或等价表述（允许同义，不要求逐字），要以连贯自然的语句表达。
2) 若问题包含诱导性细节（具体数字/比喻），只有当证据中出现过相同或等价表述时才能复述；否则不用这些细节。
3) 频率：若证据无频率可以在病症后补充“没特别注意次数/不清楚”；若仅有模糊词（“偶尔/经常/时不时”）→可用模糊词，禁止编数字。
4) 冲突时以【既往对话】>【病例】为准，与【最近几轮对话】保持不矛盾；无法判断时说“记不清/不太确定”。
5) 回答要符合患者的认知水平和表达习惯，避免过于专业的医学术语，需要有标点符号，但是不用换行。

"""

def get_response_prompt_type2(current_doctor_question, patient_info, cleaned_text, dialogue_history):
    """类型2：信息不足的问题"""
    return f"""
你正在和一位精神科医生进行交流，医生询问了你病例和对话历史中没有涉及的信息。

【当前医生问题】：{current_doctor_question}
【病例】：{patient_info}
【既往对话】：{cleaned_text}
【最近几轮对话】：{dialogue_history}

{get_common_rules()}
【额外规则】
1) 只用“没有/不清楚/没注意到/记不太清”这类自然否定或不确定表达（可任选其一即可）。
2) 如需缓冲，可加时间窗限定（如“最近没有注意到”/“这段时间没留意过”）；不得给出数字、频率、严重度等新细节。
3) 不得编造或猜测；不复述医生诱导性细节（数字/比喻）作为自己的体验。如果是证据中没有的“频率”问题，则回答“没特别注意次数/不清楚”。
4) 不需要重复病史；一句话即可。

输出格式：
- 一句否定/不确定表达，可选再加一句自然解释，不要放在括号里，自然回答。
"""

def get_response_prompt_type3(current_doctor_question, patient_info, cleaned_text, dialogue_history):
    """类型3：模糊问题（尝试猜测后向医生确认）"""
    return f"""
你正在和一位精神科医生交流。医生的问题比较模糊，需要你结合病例和既往对话，推测医生可能想了解的内容，然后以疑问句的方式向医生确认你的理解是否正确。

【当前医生问题】：{current_doctor_question}
【病例】：{patient_info}
【既往对话】：{cleaned_text}
【最近几轮对话】：{dialogue_history}

{get_common_rules()}

【回答策略】
1) 综合【对话】【病例】【最近几轮对话】内容，合理猜测医生可能想了解的具体点，只能基于已有信息，不能编造。
2) 用一句简短自然的话疑问地表达你的猜测，例如：“医生，您是想了解我最近晚上睡觉的情况吗？”、“您是在问我最近心情怎么样吗？”（猜测点不得超出前述证据）。
3) 如果无法合理推测，可以直接礼貌询问医生是否能具体说明想问的内容，如：“医生，您方便具体说说想了解哪方面吗？”
4) 语言自然、不复述医生原话、不用书面语，不主动新增病例中没有的内容。

输出：只写你的问题或确认性表达。
"""

def get_response_prompt_type4(current_doctor_question, patient_info, cleaned_text, dialogue_history):
    """类型4：日常对话"""
    return f"""
你正在和一位精神科医生进行日常寒暄（与病情无直接关系）。

【当前医生问题】：{current_doctor_question}

{get_common_rules()}

1) 自然、简短地回应即可，保持礼貌与适当距离感。符合患者身份。
2) 不主动引入医疗细节或新症状；不延展话题，不长篇大论。
"""

# ==================== Prompt Factory ====================

def create_response_prompt(category, current_doctor_question, patient_info="", cleaned_text="", dialogue_history=""):
    """根据分类创建对应的回复prompt"""
    prompt_functions = {
        1: get_response_prompt_type1,
        2: get_response_prompt_type2,
        3: get_response_prompt_type3,
        4: get_response_prompt_type4
    }
    
    # 所有类型都使用完整的参数
    if category in prompt_functions:
        return prompt_functions[category](current_doctor_question, patient_info, cleaned_text, dialogue_history)
    else:
        # 默认使用类型1
        return prompt_functions[1](current_doctor_question, patient_info, cleaned_text, dialogue_history)
