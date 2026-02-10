import json
import random
import re
import threading
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm import llm_tools_api


# ============================================================
# 分布映射缓存（全局单例）
# ============================================================
_distribution_mapping = None
_distribution_mapping_lock = threading.Lock()


def load_distribution_mapping():
    """加载分布映射（懒加载，线程安全）"""
    global _distribution_mapping
    if _distribution_mapping is None:
        with _distribution_mapping_lock:
            if _distribution_mapping is None:
                mapping_file = Path(__file__).parent.parent.parent / "emr_generation" / "mapping" / "distribution_mapping.json"
                if mapping_file.exists():
                    with open(mapping_file, 'r', encoding='utf-8') as f:
                        _distribution_mapping = json.load(f)
                else:
                    _distribution_mapping = {}
    return _distribution_mapping


def parse_diagnosis_codes(diagnosis_code: str) -> list:
    """
    解析诊断编码为大类列表
    
    例如:
    - "F28.x02,F39.x00,F42.000x011" -> ["F28", "F39", "F42.0"]
    - "F32.100" -> ["F32.1"]
    """
    if not diagnosis_code:
        return []
    
    codes = []
    raw_codes = [c.strip() for c in diagnosis_code.split(",")]
    
    for raw_code in raw_codes:
        if not raw_code:
            continue
        
        match = re.match(r'^([A-Z]\d+)(?:\.(\d))?', raw_code)
        
        if match:
            main_code = match.group(1)
            sub_code = match.group(2)
            
            if sub_code:
                parsed = f"{main_code}.{sub_code}"
            else:
                parsed = main_code
            
            if parsed not in codes:
                codes.append(parsed)
    
    return codes


def sample_patient_avg_chars(icd_codes: list) -> tuple:
    """
    根据 ICD 编码列表采样患者平均回复字数区间
    
    Args:
        icd_codes: ICD 编码列表，如 ["F32.9"] 或 ["F28", "F39"]
    
    Returns:
        (length_bin, target_length): 区间字符串和目标长度
        如果未找到匹配，返回 (None, None)
    """
    mapping = load_distribution_mapping()
    icd_codes_dist = mapping.get("icd_codes_length_distribution", {})
    
    # 将 list 转换为 JSON 字符串作为 key
    codes_key = json.dumps(icd_codes, ensure_ascii=False)
    
    if codes_key in icd_codes_dist:
        patient_avg_dist = icd_codes_dist[codes_key].get("patient_avg_chars", {})
        if patient_avg_dist:
            # 从分布中采样
            bins = list(patient_avg_dist.keys())
            probs = list(patient_avg_dist.values())
            sampled_bin = random.choices(bins, weights=probs, k=1)[0]
            
            # 将区间转换为目标长度
            target_length = bin_to_target_length(sampled_bin)
            return sampled_bin, target_length
    
    # 如果精确匹配失败，尝试只使用第一个编码
    if len(icd_codes) > 0:
        single_key = json.dumps([icd_codes[0]], ensure_ascii=False)
        if single_key in icd_codes_dist:
            patient_avg_dist = icd_codes_dist[single_key].get("patient_avg_chars", {})
            if patient_avg_dist:
                bins = list(patient_avg_dist.keys())
                probs = list(patient_avg_dist.values())
                sampled_bin = random.choices(bins, weights=probs, k=1)[0]
                target_length = bin_to_target_length(sampled_bin)
                return sampled_bin, target_length
    
    return None, None


def bin_to_target_length(length_bin: str) -> int:
    """
    将长度区间转换为目标长度（在区间内随机取值）
    
    Args:
        length_bin: 长度区间字符串（如 "10-20", "100+"）
    
    Returns:
        目标长度（整数）
    """
    if length_bin == "0":
        return 5  # 最小默认长度
    
    if "+" in length_bin:
        base = int(length_bin.replace("+", ""))
        return random.randint(base, int(base * 1.3))
    
    if "-" in length_bin:
        parts = length_bin.split("-")
        low = int(parts[0])
        high = int(parts[1])
        return random.randint(low, high)
    
    try:
        return int(length_bin)
    except:
        return 20  # 默认值


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
        
        # OpenRouter reasoning 相关配置
        self.use_openrouter = llm_tools_api.is_openrouter_model(model_path)
        self.reasoning_enabled = llm_tools_api.is_reasoning_enabled("patient", default=True)
        self.reasoning_extra_body = llm_tools_api.build_reasoning_extra_body(
            self.reasoning_enabled, self.use_openrouter, self.model_name
        )
        
        # 兼容中英文字段名
        age = get_field(patient_template, '年龄', 'Age', default='未知')
        gender = get_field(patient_template, '性别', 'Gender', default='未知')
        diagnosis = get_field(patient_template, '诊断结果', 'Diagnosis', 'diagnosis', default='精神疾病')
        self.chat_history = get_field(patient_template, '对话历史', 'cleaned_text', default='')
        
        # 解析 ICD 编码用于回复长度采样
        diagnosis_code = get_field(patient_template, 'DiagnosisCode', '诊断编码', default='')
        self.icd_codes = parse_diagnosis_codes(diagnosis_code)

        # 合并患者的主诉，现病史，个人史，家族史，如果没有则返回空字符串
        chief_complaint = get_field(patient_template, '主诉', 'ChiefComplaint', default='')
        present_illness_history = get_field(patient_template, '现病史', 'PresentIllnessHistory', default='')
        personal_history = get_field(patient_template, '个人史', 'PersonalHistory', default='')
        family_history = get_field(patient_template, '家族史', 'FamilyHistory', default='')
        
        self.patient_info = ""
        if chief_complaint:
            self.patient_info += "主诉：" + chief_complaint + "\n"
        if present_illness_history:
            self.patient_info += "现病史：" + present_illness_history + "\n"
        if personal_history:
            self.patient_info += "个人史：" + personal_history + "\n"
        if family_history:
            self.patient_info += "家族史：" + family_history + "\n"

        self.system_prompt = """## 角色
你是一名{}岁的{}性{}患者。

## 任务
你正在医院看病，和一位精神科医生面对面进行对话问诊。你需要参考病例内容和对话历史，以纯文本内容。

## 核心规则
1. 回复简短：禁止写"小作文"，禁止一次性说很多事情，只在询问的问题上结合背景和历史对话合理展开背景故事，不要主动补充未被问到的信息。
2. 语言自然：使用第一人称，口语化，用词简单粗糙（"烦"、"累"、"不想说"），不要使用优美的比喻或修辞
3. 情感含蓄：更侧重回忆性情感描述，禁止过度描述情感（如"心里像被压住"、"失控的感觉"等）
4. 诚实回答：如果医生问的内容在病例中不存在，要诚实表达不是很清楚
5. 基于事实：回复内容必须基于病例内容和对话历史，可以基于诊断背景进行简短的合理演绎，但不能与病例矛盾

### 其他要求
- 使用第一人称自然的对话式回复，如果不是必要情况不要生成疑问句
- 每句话都需要有逗号和句号。
- 不包含括号，省略号等语气说明。
- 不要重复生成和历史对话相同的句子。""".format(age, gender, diagnosis)
        
        # 对 system_prompt 应用 reasoning 前缀
        self.system_prompt = llm_tools_api.apply_reasoning_prompt_prefix(
            self.system_prompt, self.model_name, self.use_openrouter, self.reasoning_enabled
        )
        
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

    def _reasoning_kwargs(self):
        """为OpenRouter请求附加reasoning开关。"""
        if self.reasoning_extra_body:
            return {"extra_body": self.reasoning_extra_body}
        return {}

    def _postprocess_response(self, raw_response: str) -> str:
        """
        后处理患者回复：修正标点符号并去除括号
        
        通过大模型进行以下处理：
        1. 添加缺失的标点符号（逗号、句号）
        2. 将空格替换为适当的标点
        3. 去除所有括号及括号内的内容（包括中英文括号）
        
        Args:
            raw_response: 原始患者回复
        
        Returns:
            处理后的患者回复
        """
        # 使用字符串拼接避免 {} 被误当作格式化占位符
        postprocess_prompt = f"""/nothink 请对以下患者回复进行标点修正和格式处理，严格遵循以下规则：

## 处理规则
1. **标点修正**：
   - 如果句子缺少标点，添加适当的标点。
   - 如果用空格代替标点，将空格替换为适合的标点符号
   - 确保每句话结尾有标点。
   
2. **去除括号**：
   - 删除所有括号及括号内的内容，包括：（）、()、【】、[]、大括号、花括号
   - 删除省略号（……、...）

3. **保持原意**：
   - 不要改变句子的原意和内容
   - 不要添加新的信息
   - 不要删除括号以外的内容

## 原始回复
{raw_response}

## 要求
直接输出处理后的回复，不要有任何解释或说明。"""

        messages = [
            {"role": "user", "content": postprocess_prompt}
        ]
        
        try:
            chat_response = self.client.chat.completions.create(
                model=self.api_model_name,
                messages=messages,
                temperature=0.1,  # 低温度，确保输出稳定
                max_tokens=2048,  # 减少max_tokens以适应vLLM服务器的max_model_len限制
                **self._reasoning_kwargs()
            )
            super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
            processed_response, processed_reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
            
            # 基本的后处理检查：如果处理后的回复为空或异常长，返回原始回复
            if not processed_response or len(processed_response) > len(raw_response) * 2:
                print("[Patient V3] 后处理结果异常，使用原始回复")
                return raw_response
                
            return processed_response
            
        except Exception as e:
            print(f"[Patient V3] 后处理失败: {e}，使用原始回复")
            return raw_response

    def patient_response_gen(self, current_topic, dialogue_history, current_doctor_question=None):
        patient_reasoning = ""  # 初始化reasoning变量
        
        if self.dialbegin:
            self.patientbot_init()
            self.dialbegin = False
        
        # 根据 ICD 编码采样回复长度
        length_bin, target_length = sample_patient_avg_chars(self.icd_codes)
        
        # 构建长度提示
        if target_length is not None:
            length_hint = f"\n**回复长度要求**：请控制回复在{target_length}字左右（区间：{length_bin}字）"
        else:
            length_hint = ""
            
        print("[Patient V3] 长度提示：", length_hint)
        
        patient_prompt = (
        "你的病例相关信息：{}\n"
        "你需要参考的对话历史：{}\n"
        "**当前对话历史**：{}\n"
        "**当前医生问题**：{}{}\n"
        "按核心规则要求, 生成你对当前医生问题的回答："
        ).format(self.patient_info, self.chat_history, dialogue_history, current_doctor_question, length_hint)
        
        self.messages.append({"role": "user", "content": patient_prompt})
        chat_response = self.client.chat.completions.create(
                model=self.api_model_name,  # 使用纯模型名，不包含@host:port
                messages=self.messages,
                top_p=0.85,
                frequency_penalty=0.8,
                max_tokens=2048,  # 减少max_tokens以适应vLLM服务器的max_model_len限制
                **self._reasoning_kwargs()
            )
        super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
        
        # 使用统一函数分离content和reasoning
        patient_response, patient_reasoning = llm_tools_api.extract_answer_and_reasoning(chat_response)
        self.messages.pop()
        
        print("[Patient V3] 患者原始回复：", patient_response)
        if patient_reasoning:
            print("[Patient V3] 患者reasoning：", patient_reasoning[:200] + "..." if len(patient_reasoning) > 200 else patient_reasoning)
        
        # 后处理：标点修正和去除括号（只处理纯content，不包含reasoning）
        patient_response = self._postprocess_response(patient_response)
        print("[Patient V3] 患者处理后回复：", patient_response)

        return patient_response, super().get_cost(), None, patient_reasoning
