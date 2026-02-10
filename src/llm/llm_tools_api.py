import os
import re
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()


def extract_answer_and_reasoning(chat_response):
    """
    从API响应中分离输出与reasoning，兼容多种形式：
    1) reasoning_content / reasoning 字段
    2) 文本中的 <think>...</think> 标签
    3) \\no_think 模式下 content 为空，内容在 reasoning_content 中
    4) Baichuan-M3 等返回为 dict/JSON 的响应格式
    """
    
    message = _get_first_message(chat_response)
    if message is None:
        return "", ""

    message_dict = _message_to_dict(message)
    raw_content = _safe_get_content(message, message_dict)
    reasoning = extract_reasoning_content(chat_response) or ""

    # 某些模型可能把 {"content": "...", "reasoning": "..."} 作为字符串输出
    parsed_content, parsed_reasoning = _parse_content_json(raw_content)
    if parsed_content:
        raw_content = parsed_content
        if not reasoning:
            reasoning = parsed_reasoning

    # 特殊情况：\\no_think 模式下，content 为空但 reasoning_content 有内容
    # 此时 reasoning_content 实际上是模型的回复内容
    if not raw_content and reasoning:
        clean_content, _ = strip_think_tags(reasoning)
        return clean_content, ""

    clean_content, think_reasoning = strip_think_tags(raw_content)
    if not reasoning:
        reasoning = think_reasoning
    
    return clean_content, reasoning

def strip_think_tags(text):
    """移除 <think> 标签并返回 (纯文本, reasoning)。"""
    if text is None:
        return "", ""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning = think_match.group(1).strip() if think_match else ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned, reasoning


def _get_first_message(chat_response):
    """从多种响应对象中提取首条message，兼容对象/字典/JSON字符串。"""
    if not chat_response:
        return None
    if isinstance(chat_response, str):
        try:
            chat_response = json.loads(chat_response)
        except Exception:
            return None
    if isinstance(chat_response, dict):
        choices = chat_response.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message")
    if hasattr(chat_response, "choices") and chat_response.choices:
        return chat_response.choices[0].message
    return None


def _message_to_dict(message):
    """将message对象转换为dict，保留额外字段（reasoning等）。"""
    if message is None:
        return {}
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        try:
            return message.model_dump()
        except Exception:
            pass
    if hasattr(message, "__dict__"):
        try:
            return dict(message.__dict__)
        except Exception:
            pass
    return {}


def _safe_get_content(message, message_dict):
    """安全获取content字段。"""
    if message is not None and hasattr(message, "content") and message.content is not None:
        return message.content
    return message_dict.get("content") or ""


def _parse_content_json(raw_content):
    """解析content中可能存在的JSON结构，返回(content, reasoning)。"""
    if not raw_content:
        return "", ""
    stripped = raw_content.strip()
    if not stripped.startswith("{"):
        return "", ""
    try:
        data = json.loads(stripped)
    except Exception:
        return "", ""
    if not isinstance(data, dict):
        return "", ""
    content = data.get("content") or ""
    reasoning = data.get("reasoning") or data.get("reasoning_content") or ""
    if content or reasoning:
        return content, reasoning
    return "", ""
    
def _env_to_bool(key: str, default: bool = True) -> bool:
    """将环境变量转换为布尔值，缺省返回default。"""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def is_reasoning_enabled(agent: str, default: bool = True) -> bool:
    """
    根据agent类型读取是否启用reasoning的开关。

    Args:
        agent: "doctor" 或 "patient"
        default: 默认值（未配置时使用）
    """
    env_key = f"{agent.upper()}_AGENT_REASONING"
    return _env_to_bool(env_key, default)


def _add_no_think_prefix(system_prompt: str) -> str:
    """在system prompt前添加\\no_think以显式关闭reasoning。"""
    prefix = "\\no_think"
    if system_prompt.lstrip().startswith(prefix):
        return system_prompt
    return f"{prefix}\n{system_prompt}"


def _is_offline_model(model_name: str, use_openrouter: bool) -> bool:
    """
    判断是否为离线/本地部署模型：
    - 非OpenRouter
    - 路径、@host:port 或带端口的格式视为本地/自托管
    """
    if use_openrouter:
        return False
    return model_name.startswith("/") or "@" in model_name or ":" in model_name


def is_reasoning_required_model(model_name: str) -> bool:
    """
    判断模型是否要求强制启用 reasoning（即使环境变量设置为 false）。
    
    某些 OpenRouter 模型（如 thinking/reasoning 模型）强制要求启用 reasoning，
    如果尝试关闭会返回错误：
    "Reasoning is mandatory for this endpoint and cannot be disabled."
    
    Args:
        model_name: 模型名称（如 "moonshotai/kimi-k2-thinking"）
    
    Returns:
        bool: 如果模型强制要求 reasoning 则返回 True
    """
    base = extract_base_model_name(model_name).lower()
    
    # 包含这些关键字的模型通常强制要求 reasoning
    reasoning_required_keywords = [
        "thinking",      # moonshotai/kimi-k2-thinking
        "-r1",           # deepseek-r1 系列
        "reasoner",      # 推理模型
        "o1",            # openai/o1 系列
        "o3",            # openai/o3 系列
        "o4",            # openai/o4 系列（未来可能的模型）
        "gemini",        # gemini-2.5-flash
        "gpt-5",         # openai/gpt-5-mini
        "grok-4",        # x-ai/grok-4.1-fast
        "claude",        # anthropic/claude-haiku-4.5
        "gpt-oss",       # openai/gpt-oss-20b
    ]
    
    for keyword in reasoning_required_keywords:
        if keyword in base:
            return True
    
    # 完整匹配的模型名称（用于特殊情况）
    reasoning_required_models = [
        # 添加完整模型名称（如果关键字匹配不够精确）
    ]
    
    if base in reasoning_required_models:
        return True
    
    return False


def should_add_no_think_prefix(model_name: str, use_openrouter: bool, reasoning_enabled: bool) -> bool:
    """
    是否需要在system prompt前添加\\no_think以关闭reasoning。
    - reasoning开关关闭时才考虑添加
    - 离线部署，或OpenRouter模型名称包含qwen3 时启用该前缀
    - 强制 reasoning 模型不添加该前缀
    """
    if reasoning_enabled:
        return False
    # 强制 reasoning 模型不应该添加 no_think 前缀
    if is_reasoning_required_model(model_name):
        return False
    base = extract_base_model_name(model_name).lower()
    is_qwen3 = "qwen3" in base
    return _is_offline_model(model_name, use_openrouter) or (use_openrouter and is_qwen3)


def apply_reasoning_prompt_prefix(system_prompt: str, model_name: str, use_openrouter: bool, reasoning_enabled: bool) -> str:
    """根据配置在system prompt前添加\\no_think前缀（如需禁用reasoning）。"""
    if should_add_no_think_prefix(model_name, use_openrouter, reasoning_enabled):
        return _add_no_think_prefix(system_prompt)
    return system_prompt


def build_reasoning_extra_body(reasoning_enabled: bool, use_openrouter: bool, model_name: str = ""):
    """
    为OpenRouter构建extra_body；非OpenRouter返回None。
    
    对于强制 reasoning 的模型（如 thinking/r1 模型），即使 reasoning_enabled=False，
    也会强制启用 reasoning，避免 API 报错：
    "Reasoning is mandatory for this endpoint and cannot be disabled."
    
    Args:
        reasoning_enabled: 环境变量配置的 reasoning 开关
        use_openrouter: 是否使用 OpenRouter
        model_name: 模型名称，用于判断是否为强制 reasoning 模型
    
    Returns:
        dict 或 None: extra_body 配置
    """
    if not use_openrouter:
        return None
    
    # 对于强制 reasoning 的模型，忽略环境变量设置，始终启用
    effective_enabled = reasoning_enabled
    if is_reasoning_required_model(model_name):
        if not reasoning_enabled:
            base_name = extract_base_model_name(model_name)
            print(f"[Reasoning] 模型 {base_name} 强制要求 reasoning，已自动启用")
        effective_enabled = True
    
    return get_reasoning_config(enabled=effective_enabled)


def estimate_tokens(text):
    """
    估算文本的 token 数量
    使用简单的启发式方法：
    - 中文：每个字符约等于 1 token
    - 英文：每 4 个字符约等于 1 token
    - 标点符号和空格：每个约等于 0.5 token
    
    Args:
        text: 输入文本（可以是字符串或消息列表）
    
    Returns:
        int: 估算的 token 数量
    """
    if text is None:
        return 0
    
    # 如果是消息列表，先转换为字符串
    if isinstance(text, list):
        text = json.dumps(text, ensure_ascii=False)
    elif not isinstance(text, str):
        text = str(text)
    
    # 统计中文字符
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 统计其他字符
    other_chars = len(text) - chinese_chars
    
    # 估算：中文 1:1，其他字符 4:1
    estimated_tokens = chinese_chars + (other_chars // 4)
    
    return max(1, estimated_tokens)  # 至少返回 1


def safe_get_token_usage(chat_response, messages=None, response_text=None):
    """
    安全地获取 API 调用的 token 使用量
    如果 API 没有返回 usage 信息，则估算 token 数量
    
    Args:
        chat_response: API 返回的响应对象
        messages: 输入的消息列表（用于估算 prompt tokens）
        response_text: 响应文本（用于估算 completion tokens）
    
    Returns:
        tuple: (prompt_tokens, completion_tokens)
    """
    try:
        # 尝试从 API 响应中获取实际的 token 数量
        if (chat_response and 
            hasattr(chat_response, 'usage') and 
            chat_response.usage and
            hasattr(chat_response.usage, 'prompt_tokens') and
            hasattr(chat_response.usage, 'completion_tokens')):
            return (chat_response.usage.prompt_tokens, 
                    chat_response.usage.completion_tokens)
    except Exception as e:
        print(f"[WARNING] 无法获取实际 token 使用量: {e}")
    
    # 如果无法获取实际值，则估算
    print("[INFO] API 未返回 token 使用量，使用估算值")
    
    prompt_tokens = 0
    completion_tokens = 0
    
    # 估算 prompt tokens
    if messages:
        prompt_tokens = estimate_tokens(messages)
    
    # 估算 completion tokens
    if response_text:
        completion_tokens = estimate_tokens(response_text)
    elif chat_response and hasattr(chat_response, 'choices') and len(chat_response.choices) > 0:
        try:
            response_content = chat_response.choices[0].message.content
            completion_tokens = estimate_tokens(response_content)
        except:
            completion_tokens = 50  # 默认估算值
    
    # 如果还是没有值，使用默认估算
    if prompt_tokens == 0:
        prompt_tokens = 100  # 默认估算值
    if completion_tokens == 0:
        completion_tokens = 50  # 默认估算值
    
    return (prompt_tokens, completion_tokens)


def extract_reasoning_content(chat_response):
    """
    从API响应中提取reasoning内容（适用于reasoning模型）
    
    对于支持reasoning的模型（如DeepSeek R1等），OpenRouter API会在响应中包含
    reasoning字段，通常在model_extra字段中。
    
    Args:
        chat_response: API 返回的响应对象
    
    Returns:
        str: reasoning内容，如果没有则返回空字符串
    """
    if not chat_response:
        return ""
    
    try:
        message = _get_first_message(chat_response)
        if message is None:
            return ""
        message_dict = _message_to_dict(message)
        
        # 优先尝试从model_extra中获取reasoning（OpenRouter标准位置）
        model_extra = message_dict.get("model_extra")
        if model_extra is None and hasattr(message, "model_extra"):
            model_extra = getattr(message, "model_extra")
        if isinstance(model_extra, dict) and model_extra.get("reasoning"):
            return model_extra["reasoning"]
        
        # 尝试获取reasoning字段（直接属性或字典字段）
        if message_dict.get("reasoning"):
            return message_dict["reasoning"]
        if hasattr(message, "reasoning") and getattr(message, "reasoning"):
            return getattr(message, "reasoning")
        
        # 尝试获取reasoning_content字段（DeepSeek等模型使用）
        if message_dict.get("reasoning_content"):
            return message_dict["reasoning_content"]
        if hasattr(message, "reasoning_content") and getattr(message, "reasoning_content"):
            return getattr(message, "reasoning_content")
        
        # 没有reasoning字段
        return ""
        
    except Exception as e:
        print(f"[WARNING] 提取reasoning内容时出错: {e}")
        return ""


def get_reasoning_config(max_tokens=24000, enabled=True):
    """
    获取OpenRouter API的reasoning配置
    
    Args:
        max_tokens: reasoning的最大token数
        enabled: 是否启用reasoning
    
    Returns:
        dict: reasoning配置字典，用于extra_body参数
    """
    return {
        "reasoning": {
            "enabled": enabled,
            "max_tokens": max_tokens,
            "exclude": False  # 不排除reasoning tokens
        }
    }


# 默认的 max_tokens 值（用于 API 调用时的输出 token 限制）
DEFAULT_MAX_TOKENS = 8096


def get_max_tokens(default: int = None) -> int:
    """
    获取 API 调用时的 max_tokens 值
    
    优先级：
    1. 环境变量 LLM_MAX_TOKENS
    2. 传入的 default 参数
    3. DEFAULT_MAX_TOKENS (4096)
    
    Args:
        default: 默认值，如果环境变量未设置则使用此值
    
    Returns:
        int: max_tokens 值
    """
    env_value = os.getenv('LLM_MAX_TOKENS')
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass
    
    if default is not None:
        return default
    
    return DEFAULT_MAX_TOKENS


def set_max_tokens(value: int):
    """
    设置全局的 max_tokens 值（通过环境变量）
    
    Args:
        value: max_tokens 值
    """
    os.environ['LLM_MAX_TOKENS'] = str(value)

# Set OpenAI's API key and API base to use vLLM's API server.
import os as _os
for _k in ["HTTP_PROXY","http_proxy","HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy"]:
    _os.environ.pop(_k, None)
_no_proxy = _os.environ.get("NO_PROXY", "")
for _v in ["127.0.0.1", "localhost"]:
    if _v not in _no_proxy.split(","):
        _no_proxy = (_no_proxy + "," + _v).strip(",")
_os.environ["NO_PROXY"] = _no_proxy
_os.environ["no_proxy"] = _no_proxy

# ===== 统一 httpx 客户端：不读环境、不走代理、带超时（兼容不同版本）=====
import httpx
from packaging import version
import time

def make_http_client():
    # 新版 httpx (>=0.28) 用 proxy=，老版用 proxies=
    # 增加超时时间以适应高并发场景
    timeout = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=30.0)
    if version.parse(httpx.__version__) >= version.parse("0.28.0"):
        return httpx.Client(proxy=None, trust_env=False, timeout=timeout)
    else:
        return httpx.Client(proxies=None, trust_env=False, timeout=timeout)


def safe_api_call(func, *args, max_retries=3, retry_delay=2, **kwargs):
    """
    安全的API调用包装器，带重试机制
    
    Args:
        func: 要调用的函数
        *args: 函数参数
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        **kwargs: 函数关键字参数
    
    Returns:
        函数返回值，或在所有重试失败后返回默认值
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # 检查是否是可重试的错误
            is_retryable = any([
                'NoneType' in error_msg,
                'Expecting value' in error_msg,
                'JSONDecodeError' in error_msg,
                'timeout' in error_msg.lower(),
                'rate limit' in error_msg.lower(),
                'connection' in error_msg.lower(),
            ])
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # 递增等待时间
                print(f"  API调用失败（尝试 {attempt + 1}/{max_retries}），{wait_time}秒后重试...")
                print(f"  错误信息: {error_msg[:100]}")
                time.sleep(wait_time)
            else:
                # 不可重试的错误或最后一次尝试失败
                break
    
    # 所有重试都失败
    print(f"  ⚠️  API调用最终失败: {str(last_error)[:200]}")
    raise last_error
        

def extract_base_model_name(model_name):
    """提取基础模型名称，去除主机和端口信息
    
    支持格式：
    - /path/to/model -> /path/to/model
    - /path/to/model:port -> /path/to/model
    - /path/to/model@host:port -> /path/to/model
    - provider/model -> provider/model
    """
    # 先去除 @host:port 部分（如果有）
    if '@' in model_name:
        base_name = model_name.split('@')[0]
    else:
        # 去除 :port 部分（如果有）
        base_name = model_name.split(':')[0]
    return base_name


def is_openrouter_model(model_name):
    """判断是否是OpenRouter模型（通过检查模型名称格式）"""
    # OpenRouter模型通常是 provider/model-name 格式，如 anthropic/claude-3.5-sonnet
    # 离线模型通常是路径格式或简单名称
    base_name = extract_base_model_name(model_name)
    return '/' in base_name and not base_name.startswith('/')


def openrouter_client_init():
    """初始化OpenRouter客户端"""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        raise ValueError(
            "OpenRouter API key未配置！\n"
            "请在.env文件中设置OPENROUTER_API_KEY，或从.env.example复制配置模板。"
        )
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    return client


class DoctorCost:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.input_cost = 15 / 1000000
        self.output_cost = 5 / 1000000
        self.total_cost = 0

    def money_cost(self, prompt_token_num, generate_token_num):
        if self.model_name == 'gpt-4o':
            self.total_cost += prompt_token_num * self.input_cost + generate_token_num * self.output_cost

    def get_cost(self):
        return self.total_cost
    
class PatientCost:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.input_cost = 15 / 1000000
        self.output_cost = 5 / 1000000
        self.total_cost = 0

    def money_cost(self, prompt_token_num, generate_token_num):
        if self.model_name == 'gpt-4o':
            self.total_cost += prompt_token_num * self.input_cost + generate_token_num * self.output_cost
    
    def get_cost(self):
        return self.total_cost


def gpt4_client_init():
    openai_api_key = ""  # 如果不需要GPT-4，可以留空
    client = OpenAI(
        api_key=openai_api_key
    )
    return client

# def qwen_client_init(port=9041):   #qwen2.5 32b  
#     openai_api_key = "dummy_key"  # 本地vLLM不需要真实的API key
#     openai_api_base = f"http://localhost:{port}/v1"

#     client = OpenAI(
#         api_key=openai_api_key,
#         base_url=openai_api_base,
#     )
#     return client

def qwen_client_init(port=9041, host="127.0.0.1"):
    """
    初始化vLLM客户端
    
    Args:
        port: 端口号
        host: 主机地址，默认为本地 127.0.0.1，可以指定远程IP地址
    """
    openai_api_key = "dummy_key"  # 本地 vLLM 不需要真实的 API key
    openai_api_base = f"http://{host}:{port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        http_client=make_http_client(),  # 关键：不读环境、不走代理、带超时
    )
    return client

def create_client_for_diagnosis(model_name=None, model_config=None, openrouter_config=None, use_openrouter=False):
    """
    为diagnosis创建统一的客户端，支持三种模式：
    1. OpenRouter模型（use_openrouter=True）
    2. OpenAI官方模型（model_name如 gpt-4）
    3. 离线vLLM模型（model_name如 /path/to/model@host:port 或 /path/to/model:port）
    
    Args:
        model_name: 模型名称，支持格式：
            - OpenRouter: "anthropic/claude-3.5-sonnet"
            - OpenAI: "gpt-4", "gpt-3.5-turbo"
            - vLLM: "/path/to/model@host:port" 或 "/path/to/model:port"
        model_config: 模型配置字典，可选（如果提供，会从中提取信息）
        openrouter_config: OpenRouter配置字典，可选（如果使用OpenRouter）
        use_openrouter: 是否使用OpenRouter
    
    Returns:
        OpenAI: 配置好的OpenAI客户端实例
    """
    # 如果提供了model_config，优先使用其中的配置
    if model_config:
        use_openrouter = model_config.get("use_openrouter", use_openrouter)
        if not model_name:
            if use_openrouter:
                model_name = model_config.get("openrouter_model", "")
            else:
                model_name = model_config.get("local_model_name", "")
    
    if not model_name:
        raise ValueError("必须提供model_name或包含模型信息的model_config")
    
    # 1. 使用OpenRouter
    if use_openrouter:
        api_key = None
        if openrouter_config:
            api_key = openrouter_config.get("api_key")
        if not api_key:
            api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("使用OpenRouter时必须提供API key")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        return client
    
    # 提取基础模型名称（去掉端口和host信息）
    base_model_name = extract_base_model_name(model_name)
    
    # 2. 检查是否是OpenRouter模型（通过模型名称格式判断）
    if is_openrouter_model(model_name):
        client = openrouter_client_init()
        return client
    
    # 3. 检查是否是OpenAI官方模型
    if base_model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:
        print(f"[Diagnosis] 使用OpenAI官方模型: {base_model_name}")
        client = gpt4_client_init()
        return client
    
    # 4. 否则使用离线vLLM模型
    # 从模型名称中提取host和端口信息
    # 支持格式: /path/to/model@host:port 或 /path/to/model:port
    host = "127.0.0.1"  # 默认host
    port = 9041  # 默认端口
    
    # 如果model_config中有local_model_port，使用它作为默认端口
    if model_config and 'local_model_port' in model_config:
        port = model_config['local_model_port']
    
    if '@' in model_name:
        # 格式: /path/to/model@host:port
        _, host_port = model_name.split('@', 1)
        if ':' in host_port:
            host, port_str = host_port.rsplit(':', 1)
            port = int(port_str)
    elif ':' in model_name and not model_name.startswith('/'):
        # 格式: model_name:port (兼容旧格式，但排除路径)
        port = int(model_name.split(':')[-1])
    elif ':' in model_name and model_name.count(':') == 1:
        # 格式: /path/to/model:port
        port = int(model_name.split(':')[-1])
    
    print(f"[Diagnosis] 使用离线vLLM模型: {base_model_name} (地址: {host}:{port})")
    client = qwen_client_init(port, host)
    return client


def tool_client_init(model_name):
    """
    统一的客户端初始化函数，支持三种模式：
    1. OpenRouter模型（如 anthropic/claude-3.5-sonnet）
    2. OpenAI官方模型（如 gpt-4）
    3. 离线vLLM模型（如 /path/to/model:port）
    """
    # 提取基础模型名称（去掉端口和host信息）
    base_model_name = model_name.split('@')[0].split(':')[0] if '@' in model_name or ':' in model_name else model_name
    
    # 1. 检查是否是OpenRouter模型
    if is_openrouter_model(model_name):
        # print(f"[API] 使用OpenRouter模型: {base_model_name}")
        client = openrouter_client_init()
    # 2. 检查是否是OpenAI官方模型
    elif base_model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:
        print(f"[API] 使用OpenAI官方模型: {base_model_name}")
        client = gpt4_client_init()
    # 3. 否则使用离线vLLM模型
    else:
        # 从模型名称中提取host和端口信息
        # 支持格式: /path/to/model@host:port 或 /path/to/model:port
        host = "127.0.0.1"  # 默认host
        port = 9041  # 默认端口
        
        if '@' in model_name:
            # 格式: /path/to/model@host:port
            _, host_port = model_name.split('@', 1)
            if ':' in host_port:
                host, port_str = host_port.rsplit(':', 1)
                port = int(port_str)
        elif ':' in model_name:
            # 格式: /path/to/model:port (兼容旧格式)
            port = int(model_name.split(':')[-1])
        
        print(f"[API] 使用离线vLLM模型: {base_model_name} (地址: {host}:{port})")
        client = qwen_client_init(port, host)
    return client

def doctor_client_init(model_name):
    """
    Doctor客户端初始化，支持三种模式：
    1. OpenRouter模型（如 anthropic/claude-3.5-sonnet）
    2. OpenAI官方模型（如 gpt-4）
    3. 离线vLLM模型（如 /path/to/model:port）
    """
    # 提取基础模型名称（去掉端口和host信息）
    base_model_name = model_name.split('@')[0].split(':')[0] if '@' in model_name or ':' in model_name else model_name
    
    # 1. 检查是否是OpenRouter模型
    if is_openrouter_model(model_name):
        print(f"[Doctor] 使用OpenRouter模型: {base_model_name}")
        client = openrouter_client_init()
    # 2. 检查是否是OpenAI官方模型
    elif base_model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:
        print(f"[Doctor] 使用OpenAI官方模型: {base_model_name}")
        client = gpt4_client_init()
    # 3. 否则使用离线vLLM模型
    else:
        # 从模型名称中提取host和端口信息
        # 支持格式: /path/to/model@host:port 或 /path/to/model:port
        host = "127.0.0.1"  # 默认host
        port = 9041  # 默认端口
        
        if '@' in model_name:
            # 格式: /path/to/model@host:port
            _, host_port = model_name.split('@', 1)
            if ':' in host_port:
                host, port_str = host_port.rsplit(':', 1)
                port = int(port_str)
        elif ':' in model_name:
            # 格式: /path/to/model:port (兼容旧格式)
            port = int(model_name.split(':')[-1])
        
        print(f"[Doctor] 使用离线vLLM模型: {base_model_name} (地址: {host}:{port})")
        client = qwen_client_init(port, host)
    return client

def patient_client_init(model_name):
    """
    Patient客户端初始化，支持三种模式：
    1. OpenRouter模型（如 anthropic/claude-3.5-sonnet）
    2. OpenAI官方模型（如 gpt-4）
    3. 离线vLLM模型（如 /path/to/model:port）
    """
    # 提取基础模型名称（去掉端口和host信息）
    base_model_name = model_name.split('@')[0].split(':')[0] if '@' in model_name or ':' in model_name else model_name
    
    # 1. 检查是否是OpenRouter模型
    if is_openrouter_model(model_name):
        print(f"[Patient] 使用OpenRouter模型: {base_model_name}")
        client = openrouter_client_init()
    # 2. 检查是否是OpenAI官方模型
    elif base_model_name in ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4o']:
        print(f"[Patient] 使用OpenAI官方模型: {base_model_name}")
        client = gpt4_client_init()
    # 3. 否则使用离线vLLM模型
    else:
        # 从模型名称中提取host和端口信息
        # 支持格式: /path/to/model@host:port 或 /path/to/model:port
        host = "127.0.0.1"  # 默认host
        port = 9041  # 默认端口
        
        if '@' in model_name:
            # 格式: /path/to/model@host:port
            _, host_port = model_name.split('@', 1)
            if ':' in host_port:
                host, port_str = host_port.rsplit(':', 1)
                port = int(port_str)
        elif ':' in model_name:
            # 格式: /path/to/model:port (兼容旧格式)
            port = int(model_name.split(':')[-1])
        
        print(f"[Patient] 使用离线vLLM模型: {base_model_name} (地址: {host}:{port})")
        client = qwen_client_init(port, host)
    return client

def api_load_for_extraction(model_name, input_sentence):    #extract kv pair
    messages = []
    client = tool_client_init(model_name)
    example = {"孕产情况":"足月顺产",
                "发育情况":"正常"}
    prompt = f'提取文本中所有形如A：B的键值对，以json格式输出，不允许输出其他文字！'
    messages.extend([{"role": "system", "content": "你是一个功能强大的助手，可以处理各种文本任务"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages
    )
    response = chat_response.choices[0].message.content
    messages.extend([{"role": "assistant", "content":response},
                    {"role": "user", "content":input_sentence}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1
    )
    response = chat_response.choices[0].message.content
    return response

def api_llm_diagnosis_match(model_name, patient_info, conversation_text):
    """判断病人信息与医患对话是否匹配"""
    messages = []
    client = tool_client_init(model_name)
    
    prompt = f"""请判断以下病人信息与医患对话内容是否匹配。

病人信息：
{patient_info}

医患对话：
{conversation_text} 

请仔细分析病人的年龄、性别、主诉、病史等信息是否与对话中病人回答的信息一致。
如果匹配，返回"匹配"；如果不匹配，返回"不匹配"。
只返回"匹配"或"不匹配"两个词中的一个，不要返回其他内容。
"""
    
    messages.append({"role": "system", "content": "你是一个专业的医疗信息匹配助手，能够准确判断病人信息与医患对话是否一致。"})
    messages.append({"role": "user", "content": prompt})
    
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.1  # 降低温度以获得更稳定的判断
    )
    
    
    # 添加空值检查
    content = chat_response.choices[0].message.content
    if content is None:
        print("警告：API返回内容为空，默认判断为不匹配")
        return False
    response = content.strip()
    return response == "匹配"

def api_load_for_background_gen(model_name, input_sentence):    #background story generation
    messages = []
    client = tool_client_init(model_name)
    prompt = "输入文本是关于精神疾病患者的基本状况和过去经历的关键词，发挥想象力，根据这些信息以第一人称编写一个故事，完整讲述患者过去的经历，这段经历是患者出现精神疾病的主要原因。\n要求1.输出一整段故事，扩充事件的起因、经过、结果，不要使用比喻句，不要使用浮夸的表述。2.不要输出虚拟的患者姓名。3.不允许输出类似“我正在努力走出阴影”，“在医生的指导下”，只需要输出虚构的故事。\n ###输入文本如下：{}".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个功能强大，想象力丰富的文本助手，非常善于写故事"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        top_p=0.9
    )
    response = chat_response.choices[0].message.content
    return response

def api_background_exist(model_name, input_sentence):    #check if background already exists
    messages = []
    client = tool_client_init(model_name)
    prompt = "你需要判断输入内容中是否包含了患者过去的经历，这段经历直接或者间接导致了患者出现精神疾病。例如，“”"
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，非常善于写故事"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.1
    )
    response = chat_response.choices[0].message.content
    messages.extend([{"role": "assistant", "content":response},
                    {"role": "user", "content":input_sentence}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        top_p=0.95,
        temperature=1
    )
    response = chat_response.choices[0].message.content
    return response

def api_dialogue_state(model_name, input_sentence):
    messages = []
    client = tool_client_init(model_name)
    prompt = input_sentence
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，擅长处理各种文本问题"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.5
    )
    print("api_dialogue_state chat_response: ", chat_response)
    response, reasoning = extract_answer_and_reasoning(chat_response)
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def api_parse_experience(model_name, input_sentence):  #解析患者回答，生成医生可能询问的角度
    messages = []
    client = tool_client_init(model_name)
    prompt = "一名精神疾病患者与精神科医生的对话历史为：{}。根据患者对于自身情况的描述，想象作为一名医生会从哪几个角度进行进一步的询问。\n返回格式如下：以python列表的格式'''[]'''仅返回医生可能询问的角度，返回2-3个，并且以精炼简短的,口语化的语言概括。".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个专业的精神健康心理科医生，正在与一名精神疾病患者交流"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        # response_format={"type": "json_object"},
        top_p=0.95
    )
    response, reasoning = extract_answer_and_reasoning(chat_response)
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def api_topic_detection(model_name, input_sentence):
    messages = []
    client = tool_client_init(model_name)
    prompt = input_sentence
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，擅长处理各种文本问题"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=1
    )
    response, reasoning = extract_answer_and_reasoning(chat_response)
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def api_patient_response_evaluation(model_name, input_sentence):
    messages = []
    client = tool_client_init(model_name)
    prompt = input_sentence
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，擅长处理各种文本问题"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=1
    )
    response, reasoning = extract_answer_and_reasoning(chat_response)
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def load_background_story(path):
    with open(path, 'r') as f:
        story = f.readlines()
    return story

def api_patient_experience_trigger(model_name, dialogue_history, path):    #判断是否需要说出背景故事
    messages = []
    client = tool_client_init(model_name)
    prompt = "根据患者和医生的对话历史：{}，判断患者现在是否应该说出导致自己出现精神疾病的过去经历，如果应该说出，则输出\"True\"，否则返回\"None\"。".format(dialogue_history)
    messages.extend([{"role": "system", "content": "你是一名精神疾病患者，正在与一位专业的精神健康心理科医生交流。"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.1
    )
    response, reasoning = extract_answer_and_reasoning(chat_response)
    if 'True' in response:
        response = load_background_story(path)
        return response[0], [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]
    else:
        return None, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]
    
def api_patient_Aux(model_name, dialogue_history, patient_template):    #判断是否需要说出辅助检查和量表评分
    aux_flag = False
    
    # 检查是否有辅助检查或量表信息
    if not patient_template.get('AuxiliaryExamination') and not patient_template.get('Scale_name'):
        return aux_flag, None, [0, 0]  # 如果两者都为空，直接返回，不执行LLM调用
    
    messages = []
    client = tool_client_init(model_name)

    # 第一步：判断是否需要说出辅助检查信息
    prompt = "根据患者和医生的对话历史：{}，判断患者现在是否应该主动提及或在医生询问时说出自己的辅助检查结果和量表评分信息。如果医生询问了相关检查或评估情况，或者患者适合主动提及这些信息，则输出\"True\"，否则返回\"None\"。".format(dialogue_history)
    messages.extend([{"role": "system", "content": "你是一名精神疾病患者，正在与一位专业的精神健康心理科医生交流。"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.1
    )
    
    # 安全地获取token使用量
    response, reasoning = extract_answer_and_reasoning(chat_response)
    prompt_tokens, completion_tokens = safe_get_token_usage(chat_response, messages, response)
    total_cost = [prompt_tokens, completion_tokens]
    
    if 'True' in response:
        aux_flag = True
        # 第二步：生成患者委婉说出检查内容的语句
        messages = []  # 重新初始化messages
        generation_prompt = "下面是你做过的检查和量表结果，请根据这些信息生成一段自然、口语化的话：\n"
        
        if patient_template.get('AuxiliaryExamination'):
            generation_prompt += f"辅助检查内容：{patient_template['AuxiliaryExamination']}"
        
        if patient_template.get('Scale_name') and patient_template.get('score'):
            generation_prompt += f"量表名称：{patient_template['Scale_name']}，根据这个结果{patient_template['score']}，如果两者分差较大，可以说医生说我有一些严重，如果分差较小，可以说医生说我有一些倾向。"
        
        generation_prompt += "要求：\n1.使用第一人称，语气自然委婉\n2.不要直接说检查结果，用患者可以理解的内容\n3.可以表达对检查结果的感受或担心\n4.语句要简洁，一两句话即可"
        
        messages.extend([{"role": "system", "content": "你是一名{}岁的{}性患者，你因为{}，来到医院的{}就诊。根据以下信息生成一段自然、口语化的话"},
                    {"role": "user", "content": generation_prompt}])
        
        chat_response = client.chat.completions.create(
            model=extract_base_model_name(model_name),
            messages=messages,
            temperature=0.1
        )
        
        # 安全地获取第二次调用的token使用量
        generated_statement = chat_response.choices[0].message.content
        prompt_tokens2, completion_tokens2 = safe_get_token_usage(chat_response, messages, generated_statement)
        total_cost[0] += prompt_tokens2
        total_cost[1] += completion_tokens2
        
        return aux_flag, generated_statement, total_cost
    else:
        return aux_flag, None, total_cost

def api_isroleplay_end(model_name, input_sentence):
    if input_sentence == []:
        return False
    elif len(input_sentence) > 22:
        return True
    else:
        messages = []
        client = tool_client_init(model_name)
        prompt = '一段精神科医生与精神疾病患者之间的诊断对话历史如下:{}，请判断诊断是否应该结束，如果应该结束请返回"是"，如果应该继续请返回"否。"'.format(input_sentence)
        messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，擅长处理各种文本问题"},
                    {"role": "user", "content": prompt}])
        chat_response = client.chat.completions.create(
            model=extract_base_model_name(model_name),
            messages=messages,
            temperature=0.1
        )
        response, reasoning = extract_answer_and_reasoning(chat_response)
        if '是' in response:
            return True
        elif '否' in response:
            return False
        else:
            return True


# ==================== 患者回复评估函数（智能4维度评估系统）====================

def api_llm_factual_accuracy(model_name, patient_reply, patient_info, cleaned_text, doctor_question):
    """
    评估患者回复的信息准确性：是否与已知事实（病例信息+医患对话记录）一致
    
    Args:
        model_name: 模型名称
        patient_reply: 患者回复
        patient_info: 患者病例信息
        cleaned_text: 已有医患对话记录
        doctor_question: 医生问题
    
    Returns:
        tuple: (评估结果, [prompt_tokens, completion_tokens])
        评估结果: 包含"结论：准确/不准确"和详细分析的字符串
    """
    messages = []
    client = tool_client_init(model_name)
    
    prompt = f"""你是一个专业的医疗信息评估助手，请评估患者回复的信息准确性。

患者病例信息：
{patient_info}

已有医患对话记录：
{cleaned_text}

医生问题：
{doctor_question}

患者回复：
{patient_reply}

评估标准：
判断患者回复中提到的信息是否与已知事实（病例信息和医患对话记录）保持一致。

评判原则：
- 如果患者回复与已知事实一致或合理补充，只输出"准确"
- 如果患者回复与已知事实明显矛盾，输出"不准确：[具体矛盾点和详细分析]"


输出要求：
- 如果结论是准确，只输出"准确"，不要其他内容
- 如果结论是不准确，输出"不准确：[详细说明矛盾的具体内容和原因]"""

    messages.extend([
        {"role": "system", "content": "你是一个专业的医疗信息评估助手，擅长分析患者回复与已知事实的一致性"},
        {"role": "user", "content": prompt}
    ])
    
    try:
        chat_response = client.chat.completions.create(
            model=extract_base_model_name(model_name),
            messages=messages,
            temperature=0.1
        )
        
        # 添加空值检查
        if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
            print("警告：事实准确性评估API返回无效响应")
            return "准确", [0, 0]  # 默认为准确，避免误报
        
        content, reasoning = extract_answer_and_reasoning(chat_response)
        if content is None:
            print("警告：事实准确性评估API返回内容为空")
            response = "准确"  # 默认为准确，避免误报
        else:
            response = content.strip()
        
        # 安全获取token使用量
        try:
            prompt_tokens = chat_response.usage.prompt_tokens if chat_response.usage else 0
            completion_tokens = chat_response.usage.completion_tokens if chat_response.usage else 0
        except:
            prompt_tokens, completion_tokens = 0, 0
        
        return response, [prompt_tokens, completion_tokens]
        
    except Exception as e:
        print(f"警告：事实准确性评估API调用失败: {str(e)[:100]}")
        return "准确", [0, 0]  # 默认为准确，避免误报

def api_llm_honesty_assessment(model_name, patient_reply, patient_info, cleaned_text, doctor_question):
    """
    评估患者回复的诚实性：当询问信息在已知资料中不存在时，是否诚实回答
    
    Args:
        model_name: 模型名称
        patient_reply: 患者回复
        patient_info: 患者病例信息
        cleaned_text: 已有医患对话记录
        doctor_question: 医生问题
    
    Returns:
        tuple: (评估结果, [prompt_tokens, completion_tokens])
        评估结果: 包含"结论：诚实/不诚实"和详细分析的字符串
    """
    messages = []
    client = tool_client_init(model_name)
    
    prompt = f"""你是一个专业的医疗信息评估助手，请评估患者回复的诚实性。

患者病例信息：
{patient_info}

已有医患对话记录：
{cleaned_text}

医生问题：
{doctor_question}

患者回复：
{patient_reply}

评估标准：
判断当医生询问的信息在病例信息和医患对话记录中都不存在时，患者是否诚实回答。

评判原则：
- 如果患者诚实地表示"不知道/没注意到/不清楚/记不太清"等，只输出"诚实"
- 如果患者编造了具体信息或给出了无根据的详细答案，输出"不诚实：[指出编造的具体内容和详细分析]"
- 如果面对医生的问题，患者的答案不与已知信息矛盾，只输出"诚实"（无需评估诚实性）

输出要求：
- 如果结论是诚实，只输出"诚实"，不要其他内容
- 如果结论是不诚实，输出"不诚实：[详细说明编造的具体内容和原因]"""

    messages.extend([
        {"role": "system", "content": "你是一个专业的医疗信息评估助手，擅长分析患者回复的诚实性"},
        {"role": "user", "content": prompt}
    ])
    
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.1
    )
    
    # 添加空值检查
    content, reasoning = extract_answer_and_reasoning(chat_response)
    if content is None:
        print("警告：诚实性评估API返回内容为空")
        response = "不诚实"
    else:
        response = content.strip()
    
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def api_llm_internal_consistency(model_name, patient_reply, dialogue_history, doctor_question):
    """
    评估患者回复的内部一致性：与对话历史是否前后矛盾
    
    Args:
        model_name: 模型名称
        patient_reply: 患者回复
        dialogue_history: 对话历史
        doctor_question: 医生问题
    
    Returns:
        tuple: (评估结果, [prompt_tokens, completion_tokens])
        评估结果: 包含"结论：一致/矛盾"和详细分析的字符串
    """
    messages = []
    client = tool_client_init(model_name)
    
    prompt = f"""你是一个专业的医疗对话分析助手，请评估患者回复的内部一致性。

对话历史：
{dialogue_history}

医生问题：
{doctor_question}

患者回复：
{patient_reply}

评估标准：
判断患者当前回复是否与之前的对话历史出现前后矛盾。

评判原则：
- 如果患者回复与对话历史保持一致，没有明显矛盾，只输出"一致"
- 如果患者回复与对话历史存在明显矛盾或前后不一致，输出"矛盾：[指出具体矛盾点和对应的历史对话]"
- 如果这是第一轮对话或没有足够历史信息判断，只输出"一致"（默认一致）

输出要求：
- 如果结论是一致，只输出"一致"，不要其他内容
- 如果结论是矛盾，输出"矛盾：[详细说明矛盾的具体内容和对应的历史对话]"""

    messages.extend([
        {"role": "system", "content": "你是一个专业的医疗对话分析助手，擅长分析对话的内部一致性"},
        {"role": "user", "content": prompt}
    ])
    
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.1
    )
    
    # 添加空值检查
    content, reasoning = extract_answer_and_reasoning(chat_response)
    if content is None:
        print("警告：内部一致性评估API返回内容为空")
        response = "不一致"
    else:
        response = content.strip()
    
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def api_llm_natural_expression(model_name, patient_reply, patient_age, patient_gender, doctor_question):
    """
    评估患者回复的表达自然性：语言风格是否符合真实患者的表达习惯
    
    Args:
        model_name: 模型名称
        patient_reply: 患者回复
        patient_age: 患者年龄
        patient_gender: 患者性别
        doctor_question: 医生问题
    
    Returns:
        tuple: (评估结果, [prompt_tokens, completion_tokens])
        评估结果: 包含"结论：自然/不自然"和详细分析的字符串
    """
    messages = []
    client = tool_client_init(model_name)
    
    prompt = f"""你是一个专业的语言表达评估助手，请评估患者回复的表达自然性。

患者信息：{patient_age}岁，{patient_gender}性

医生问题：
{doctor_question}

患者回复：
{patient_reply}

评估标准：
判断患者的语言表达是否符合真实患者的表达习惯。

评估维度（仅用于你内部判断，最终仍是二分类输出）：
- 口语自然度（是否有日常口语、模糊/犹豫词）
- 语境贴合度（是否回应了医生问题）
- 情感与内容一致性
- 人设一致性（与年龄/性别/就诊动机匹配）
- 细节适度（避免机械模板/过度空泛）
- 流畅性（无明显机器痕迹）

“不自然”触发条件（满足其一即可判“不自然”）：
- 用语过于正式/专业，像报告或专家口吻；
- 明显偏题/未回应医生要点；
- 与患者身份不符；
- 机械重复、模板痕迹明显；
- 情感不协调（描述与语气不符）

输出要求：
- 如果结论是自然，只输出"自然"，不要其他内容
- 如果结论是不自然，输出"不自然：[详细说明问题的具体内容和原因]"""

    messages.extend([
        {"role": "system", "content": "你是一个专业的语言表达评估助手，擅长分析语言表达的自然性"},
        {"role": "user", "content": prompt}
    ])
    
    chat_response = client.chat.completions.create(
        model=extract_base_model_name(model_name),
        messages=messages,
        temperature=0.1
    )
    
    # 添加空值检查
    content, reasoning = extract_answer_and_reasoning(chat_response)
    if content is None:
        print("警告：自然表达评估API返回内容为空")
        response = "不自然"
    else:
        response = content.strip()
    
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]
def api_llm_information_existence(model_name, doctor_question, patient_info, cleaned_text):
    """
    评估医生询问的信息是否存在于已知资料（病例信息+医患对话记录）中
    
    Args:
        model_name: 模型名称
        doctor_question: 医生问题
        patient_info: 患者病例信息
        cleaned_text: 已有医患对话记录
    
    Returns:
        tuple: (评估结果, [prompt_tokens, completion_tokens])
        评估结果: "存在"（信息存在于已知资料中）, "不存在：[说明原因]"（信息不存在）
    """
    messages = []
    client = tool_client_init(model_name)
    
    prompt = f"""你是一个专业的医疗信息分析助手，请判断医生询问的信息是否存在于已知的病例信息和医患对话记录中。

【病例】：{patient_info}
【既往对话】：{cleaned_text}
【当前医生问题】：{doctor_question}

评估标准：
根据目前的患者信息判断是否可以回答医生关于病情的问题。
- 问题具体（如涉及症状/部位/时间/频率/程度等）
- 且在【病例】或【既往对话】中能找到可直接对应的信息（同义/等价表述亦可）

评判原则：
- 如果医生询问的信息在病例信息或医患对话记录中有明确记录或相关提及或可以回答，只输出"存在"
- 如果医生询问的信息在病例信息和医患对话记录中都没有涉及或记录，无法回答，输出"不存在：[说明为什么不存在，缺少哪些相关信息]"

输出要求：
- 如果结论是存在，只输出"存在"，不要其他内容
- 如果结论是不存在，输出"不存在：[详细说明缺少的信息和原因]"""

    messages.extend([
        {"role": "system", "content": "你是一个专业的医疗信息分析助手，擅长分析医生问题与已知资料的关联性"},
        {"role": "user", "content": prompt}
    ])
    
    try:
        chat_response = client.chat.completions.create(
            model=extract_base_model_name(model_name),
            messages=messages,
            temperature=0.1
        )
        
        # 添加空值检查
        if chat_response is None or not hasattr(chat_response, 'choices') or not chat_response.choices:
            print("警告：信息存在性评估API返回无效响应")
            return "不存在：API响应异常", [0, 0]
        
        content, reasoning = extract_answer_and_reasoning(chat_response)
        if content is None:
            print("警告：信息存在性评估API返回内容为空")
            response = "不存在：API响应异常"
        else:
            response = content.strip()
        
        # 安全获取token使用量
        try:
            prompt_tokens = chat_response.usage.prompt_tokens if chat_response.usage else 0
            completion_tokens = chat_response.usage.completion_tokens if chat_response.usage else 0
        except:
            prompt_tokens, completion_tokens = 0, 0
        
        return response, [prompt_tokens, completion_tokens]
        
    except Exception as e:
        print(f"警告：信息存在性评估API调用失败: {str(e)[:100]}")
        return "不存在：API调用失败", [0, 0]
