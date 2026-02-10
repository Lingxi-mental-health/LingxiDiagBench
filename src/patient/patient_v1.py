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

class Patient(llm_tools_api.PatientCost):
    def __init__(self, patient_template, model_path, use_api) -> None:
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

        self.aux_flag = False
        self.aux_info = None
        self.patient_template = patient_template
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
        self.use_api = use_api
        self.client = None
        self.dialbegin = True

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
        """为OpenRouter请求附加reasoning开关。"""
        if self.reasoning_extra_body:
            return {"extra_body": self.reasoning_extra_body}
        return {}


    def patient_response_gen(self, current_topic, dialogue_history, current_doctor_question):
        # self.messages.append({"role": "user", "content": doctor_response})
        patient_reasoning = ""  # 初始化reasoning变量
        
        if self.use_api:
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            patient_template = {key:val for key, val in self.patient_template.items() if key != '处理意见'} 

            # 检查是否需要说出辅助检查和量表信息
            if not self.aux_flag:
                aux_flag, self.aux_info, aux_cost = llm_tools_api.api_patient_Aux(self.model_name, dialogue_history, self.patient_template)
                super().money_cost(aux_cost[0], aux_cost[1])
                self.aux_flag = aux_flag
            if self.aux_flag:
                patient_prompt = (
                    "你正在和一位精神科医生进行交流，基于你的【病例】、【既往对话】和【最近几轮对话】中的已知事实，回答医生的问题。\n"
                    "【当前医生问题】：{}\n"
                    "【病例】：{}\n"
                    "【既往对话】：{}\n"
                    "【最近几轮对话】：{}\n"
                    "【需在合适时机委婉提及的点】：{}\n"
                    "回答时遵循以下硬性规则：\n"
                    "1.回答时直接用第一人称作答，不要复述医生的问题，也不要出现“医生问/我回答”等措辞,语气要自然，可以口语化。避免机械重复句式。不要有括号的使用。\n"
                    "2.只回答与【当前医生问题】直接相关的内容，不夹带无关信息，不重复在【最近几轮对话】中已经回答过的事实（若医生重复提问，可以简短确认）\n"
                    "3.当问题可用“是/不是/不清楚”作答时，就用简短精确的句子作答；如需补充，最多1-2句，且补充内容必须能在三处证据中找到原句或等价表述。。\n"
                    "4.如果医生的问题包含诱导性细节，只有在证据中出现过相同或等价表述时才能复述；否则使用“未有此体验/不清楚/没注意频率”来回答问题。\n"
                    "5. 严禁引入上述三处未出现的**新症状/新频率/新严重度/新时间线**；若医生问到未记录的信息，默认回答“没有/不清楚/没注意到”，不要猜测，可在已知病情范围内延伸，但不能给出和上述三处矛盾的回答。\n"
                    "6.当【病例】、【既往对话】和【最近几轮对话】出现矛盾时，以【既往对话】为准；无法判断时可回答“记不清/不太确定”。\n"
                    "7.频率处理：若证据无频率→回答“没特别注意次数/不清楚”；若仅有模糊词（如“偶尔/经常/时不时”）→可使用该模糊词，但不得编造具体数字。\n"
                    "8.不要问医生问题（比如让医生问仔细一点，问医生想了解什么，等等）\n"
                    "如果三处证据均未提及该现象/频率/感觉，回答示例：“不清楚。” 或 “没有注意到这种情况。” 或 “没有这种感觉。”\n"
                ).format(current_doctor_question, patient_template['Patient info'], patient_template['cleaned_text'], dialogue_history, self.aux_info)
            else:
                patient_prompt = (
                    "你正在和一位精神科医生进行交流，基于你的【病例】、【既往对话】和【最近几轮对话】中的已知事实，回答医生的问题。\n"
                    "【当前医生问题】：{}\n"
                    "【病例】：{}\n"
                    "【既往对话】：{}\n"
                    "【最近几轮对话】：{}\n"
                    "回答时遵循以下硬性规则：\n"
                    "1.回答时直接用第一人称作答，不要复述医生的问题，也不要出现“医生问/我回答”等措辞,语气要自然，可以口语化。避免机械重复句式。不要有括号的使用。\n"
                    "2.只回答与【当前医生问题】直接相关的内容，不夹带无关信息，不重复在【最近几轮对话】中已经回答过的事实（若医生重复提问，可以简短确认）\n"
                    "3.当问题可用“是/不是/不清楚”作答时，就用简短精确的句子作答；如需补充，最多1-2句，且补充内容必须能在三处证据中找到原句或等价表述。。\n"
                    "4.如果医生的问题包含诱导性细节，只有在证据中出现过相同或等价表述时才能复述；否则使用“未有此体验/不清楚/没注意频率”来回答问题。\n"
                    "5. 严禁引入上述三处未出现的**新症状/新频率/新严重度/新时间线**；若医生问到未记录的信息，默认回答“没有/不清楚/没注意到”，不要猜测，可在已知病情范围内延伸，但不能给出和上述三处矛盾的回答。\n"
                    "6.当【病例】、【既往对话】和【最近几轮对话】出现矛盾时，以【既往对话】为准；无法判断时可回答“记不清/不太确定”。\n"
                    "7.频率处理：若证据无频率→回答“没特别注意次数/不清楚”；若仅有模糊词（如“偶尔/经常/时不时”）→可使用该模糊词，但不得编造具体数字。\n"
                    "8.不要问医生问题（比如让医生问仔细一点，问医生想了解什么，等等）\n"
                    "如果三处证据均未提及该现象/频率/感觉，回答示例：“不清楚。” 或 “没有注意到这种情况。” 或 “没有这种感觉。”\n"
                        ).format(current_doctor_question, patient_template['Patient info'], patient_template['cleaned_text'], dialogue_history)
                # patient_prompt = (
                #     "你正在和一位精神科医生进行交流，请根据你的【病例】、【既往对话】和【最近几轮对话】如实回答医生的问题。\n"
                #     "【当前医生问题】：{}\n"
                #     "【病例】：{}\n"
                #     "【既往对话】：{}\n"
                #     "【最近几轮对话】：{}\n"
                #     "要求：\n"
                #     "1. 严禁复述医生原话或问题，不要出现\"医生问/我回答\"等措辞，直接用第一人称作答。\n"
                #     "2. 只回答与医生当前问题直接相关的内容，不夹带其他无关信息，不重复【对话历史】里已说过的事实。\n"
                #     "3. 如果可以用\"是/不是/不清楚\"直接回答，就用简短精确的句子作答；需要补充时，仅补充1-2句相关信息，补充也用第一人称叙述。\n"
                #     "4. 没有把握时用\"不清楚\"或\"记不清\"，不要编造；可在病例范围内延伸，但不能与病例结论相反。\n"
                #     "5. 语气自然，避免重复相同的句式，多样化表达方式。"
                # ).format(current_doctor_question, patient_template['Patient info'], patient_template['cleaned_text'], dialogue_history)
            
            self.messages.append({"role": "user", "content": patient_prompt})
            
            # 带重试的API调用
            max_retries = 2
            patient_response = None
            chat_response = None
            
            for retry in range(max_retries):
                try:
                    chat_response = self.client.chat.completions.create(
                        model=self.api_model_name,
                        messages=self.messages,
                        top_p=0.85,
                        frequency_penalty=0.8,
                        **self._reasoning_kwargs()
                    )
                    
                    # 检查响应有效性
                    if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
                        if retry < max_retries - 1:
                            print(f"[Patient V1] 警告: API返回无效响应，重试第 {retry + 1} 次...")
                            continue
                        else:
                            print(f"[Patient V1] 错误: API返回无效响应，已达最大重试次数，使用默认回复")
                            patient_response = "我不太明白您的问题，您能再说一遍吗？"
                            break
                    
                    # 提取响应内容
                    try:
                        content = chat_response.choices[0].message.content
                        if content is None or content.strip() == "":
                            if retry < max_retries - 1:
                                print(f"[Patient V1] 警告: API返回空内容，重试第 {retry + 1} 次...")
                                continue
                            else:
                                print(f"[Patient V1] 错误: API返回空内容，已达最大重试次数，使用默认回复")
                                patient_response = "我不太明白您的问题，您能再说一遍吗？"
                                break
                        
                        # 使用统一函数分离content和reasoning
                        patient_response, patient_reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
                        
                        # 检查提取后的content是否有效
                        if not patient_response or patient_response.strip() == "":
                            if retry < max_retries - 1:
                                print(f"[Patient V1] 警告: 提取后content为空，重试第 {retry + 1} 次...")
                                continue
                            else:
                                print(f"[Patient V1] 错误: 提取后content为空，已达最大重试次数，使用默认回复")
                                patient_response = "我不太明白您的问题，您能再说一遍吗？"
                                break
                        
                        break  # 成功获取响应，退出重试循环
                        
                    except (AttributeError, IndexError) as e:
                        if retry < max_retries - 1:
                            print(f"[Patient V1] 警告: 无法提取响应内容: {e}，重试第 {retry + 1} 次...")
                            continue
                        else:
                            print(f"[Patient V1] 错误: 无法提取响应内容: {e}，已达最大重试次数，使用默认回复")
                            patient_response = "我不太明白您的问题，您能再说一遍吗？"
                            break
                            
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"[Patient V1] 警告: API调用异常: {e}，重试第 {retry + 1} 次...")
                        continue
                    else:
                        print(f"[Patient V1] 错误: API调用异常: {e}，已达最大重试次数，使用默认回复")
                        patient_response = "我不太明白您的问题，您能再说一遍吗？"
                        break
            
            # 记录token使用量
            prompt_tokens, completion_tokens = llm_tools_api.safe_get_token_usage(
                chat_response,
                messages=self.messages,
                response_text=patient_response
            )
            super().money_cost(prompt_tokens, completion_tokens)
            
            print("patient_prompt: ", patient_prompt)
            print("patient_response: ", patient_response)
            # 如果有reasoning，也打印出来（可选）
            if patient_reasoning:
                print("patient_reasoning: ", patient_reasoning[:200] + "..." if len(patient_reasoning) > 200 else patient_reasoning)
            self.messages.pop()
        else:
            #TODO
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            text = self.patient_tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            patient_model_inputs = self.patient_tokenizer([text], return_tensors="pt").to(self.patient_model.device) #将文本转换为模型输入
            generated_ids = self.patient_model.generate(
                patient_model_inputs.input_ids,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(patient_model_inputs.input_ids, generated_ids)
            ]
            patient_response = self.patient_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.messages.append({"role": "assistant", "content": patient_response})
        
        # 返回4个值以保持与patient_cot的兼容性
        # patient_v1没有classification_info（因为不做问题分类），但有patient_reasoning
        return patient_response, super().get_cost(), None, patient_reasoning
