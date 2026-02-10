import json
import random
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import llm_tools_api


# ============================================================
# 客户端缓存（线程安全，避免重复初始化）
# ============================================================
_client_cache = {}
_client_cache_lock = threading.Lock()


def get_cached_client(model_name: str):
    """
    获取缓存的客户端，如果不存在则创建并缓存
    
    Args:
        model_name: 模型名称（完整路径，包含@host:port）
    
    Returns:
        OpenAI 客户端实例
    """
    with _client_cache_lock:
        if model_name not in _client_cache:
            _client_cache[model_name] = llm_tools_api.patient_client_init(model_name)
        return _client_cache[model_name]


def get_field(template: dict, *keys, default=""):
    """
    从模板中获取字段值，支持多个候选键名（中英文兼容）
    
    Args:
        template: 患者模板字典
        *keys: 候选键名列表，按优先级排列
        default: 默认值
    
    Returns:
        找到的第一个存在的键对应的值，或默认值
    """
    for key in keys:
        if key in template and template[key] is not None:
            return template[key]
    return default


class Patient(llm_tools_api.PatientCost):
    def __init__(self, patient_template, model_path, use_api) -> None:
        super().__init__(model_path)
        self.model_path = model_path
        self.model_name = model_path  # 保留完整路径用于客户端初始化
        # 提取基础模型名称用于API调用中的model参数（去掉@host:port部分）
        if '@' in model_path:
            self.api_model_name = model_path.split('@')[0]
        else:
            self.api_model_name = model_path.split(':')[0]
        self.patient_model = None
        self.patient_tokenizer = None
        self.patient_template = patient_template
        
        # 兼容中英文字段名
        age = get_field(patient_template, '年龄', 'Age', default='未知')
        gender = get_field(patient_template, '性别', 'Gender', default='未知')
        diagnosis = get_field(patient_template, '诊断结果', 'Diagnosis', 'diagnosis', default='精神科')
        
        self.system_prompt = "你是一名{}岁的{}性{}患者，正在和一位精神科医生交流，使用口语化的表达，输出一整段没有空行的内容。如果医生的问题可以用是/否来回答，你的回复要简短精确。".format(age, gender, diagnosis)
        self.messages = []
        self.use_api = use_api
        self.client = None
        self.dialbegin = True

    def patientbot_init(self):
        if self.use_api:
            self.client = get_cached_client(self.model_name)  # 使用缓存的客户端
        else:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.patient_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.messages.append({"role": "system", "content": self.system_prompt})

    def patient_response_gen(self, current_topic, dialogue_history, current_doctor_question=None):
        # self.messages.append({"role": "user", "content": doctor_response})
        if self.use_api:
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            
            # 过滤掉处理意见字段
            patient_template = {key:val for key, val in self.patient_template.items() if key not in ('处理意见', 'Treatment')}
            
            # 兼容中英文字段名获取诊断结果
            diagnosis = get_field(self.patient_template, '诊断结果', 'Diagnosis', 'diagnosis', default='精神科')
            
            patient_prompt = (
                "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。"
                "如果医生的问题可以用是/否来回答，你的回复要简短精确。\n"
                "你的病例为\"{}\"，\n"
                "你和医生的对话历史为{}，\n"
                "现在请根据下面要求生成:\n"
                "1.使用第一人称口语化的回答，如果不是必要情况，不要生成疑问句，不要总是以\"医生，\"开头。\n"
                "2.回答围绕{}展开，如果医生的问题可以用是/否来回答，你的回复要简短精确。在对话历史中提到过的内容不要重复再提起。\n"
                "3.回复内容必须根据病例内容，对话历史。如果出现不在病例内容中的问题，发挥想象力虚构回答。"
            ).format(diagnosis, patient_template, dialogue_history[-8:], current_topic)
            self.messages.append({"role": "user", "content": patient_prompt})
            chat_response = self.client.chat.completions.create(
                    model=self.api_model_name,  # 使用纯模型名，不包含@host:port
                    messages=self.messages,
                    top_p=0.85,
                    frequency_penalty=0.8
                )
            super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
            
            # 使用统一函数分离content和reasoning
            patient_response, patient_reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
            self.messages.pop()
        else:
            #TODO
            patient_reasoning = ""
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
            raw_response = self.patient_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # 使用统一函数处理可能的<think>标签
            patient_response, patient_reasoning = llm_tools_api.strip_think_tags(raw_response)
            self.messages.append({"role": "assistant", "content": patient_response})
        
        # 返回4个值以保持与其他Patient类的兼容性
        return patient_response, super().get_cost(), None, patient_reasoning
