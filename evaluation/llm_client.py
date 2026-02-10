"""
LLM 客户端模块 - 支持 vLLM 本地部署模型和 OpenRouter API
"""

import os
import json
import time
import re
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal


# ============================================================
# Pydantic 模型定义
# ============================================================

class Evaluation(BaseModel):
    """单维度评估结果"""
    score: int = Field(..., ge=0, le=5, description="评分 0-5")
    reason: Optional[str] = Field(None, description="评分理由")


class MetricScore(BaseModel):
    """单个指标的评分"""
    score: int = Field(..., ge=1, le=5, description="评分 1-5")
    reasoning: str = Field(..., description="简要分析")


class RealnessMultiMetrics(BaseModel):
    """多维度真实性评估的指标集合"""
    Response_Brevity: MetricScore = Field(..., description="回复简洁度")
    Information_Proactivity: MetricScore = Field(..., description="信息主动性")
    Emotional_Restraint: MetricScore = Field(..., description="情感表达度")
    Language_Polish: MetricScore = Field(..., description="语言修饰度")
    Conversational_Cooperation: MetricScore = Field(..., description="对话配合度")


class RealnessMultiEvaluation(BaseModel):
    """多维度真实性评估结果"""
    metrics: RealnessMultiMetrics = Field(..., description="五个维度的评分")
    overall_impression: Literal["Real Patient", "AI Agent"] = Field(
        ..., description="整体印象判断"
    )
    average_score: float = Field(..., ge=1.0, le=5.0, description="五个维度的平均分")


# ============================================================
# 辅助函数
# ============================================================

def make_http_client():
    """创建 HTTP 客户端，关闭代理"""
    timeout = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=30.0)
    return httpx.Client(proxy=None, trust_env=False, timeout=timeout)


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
        # 检查响应是否有choices
        if not hasattr(chat_response, 'choices') or not chat_response.choices:
            return ""
        
        message = chat_response.choices[0].message
        
        # 优先尝试从model_extra中获取reasoning（OpenRouter标准位置）
        if hasattr(message, 'model_extra') and message.model_extra:
            if isinstance(message.model_extra, dict) and 'reasoning' in message.model_extra:
                reasoning = message.model_extra['reasoning']
                if reasoning:
                    return reasoning
        
        # 尝试获取reasoning字段（直接属性）
        if hasattr(message, 'reasoning') and message.reasoning:
            return message.reasoning
        
        # 尝试获取reasoning_content字段（DeepSeek等模型使用）
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            return message.reasoning_content
        
        # 没有reasoning字段
        return ""
        
    except Exception as e:
        print(f"[WARNING] 提取reasoning内容时出错: {e}")
        return ""


def extract_message_text(message: Any) -> str:
    """
    从 OpenAI 兼容 message 中提取文本内容。
    
    注意：部分 vLLM 部署在开启 reasoning 后可能返回 `content=null`，
    实际文本在 `reasoning_content`（或 `reasoning` / `model_extra.reasoning`）里。
    """
    if message is None:
        return ""
    
    # dict-like
    if isinstance(message, dict):
        content = message.get("content")
        if content:
            return str(content)
        for k in ("reasoning_content", "reasoning"):
            v = message.get(k)
            if v:
                return str(v)
        me = message.get("model_extra")
        if isinstance(me, dict):
            for k in ("reasoning_content", "reasoning"):
                v = me.get(k)
                if v:
                    return str(v)
        return ""
    
    # SDK object
    content = getattr(message, "content", None)
    if content:
        return str(content)
    for k in ("reasoning_content", "reasoning"):
        v = getattr(message, k, None)
        if v:
            return str(v)
    me = getattr(message, "model_extra", None)
    if isinstance(me, dict):
        for k in ("reasoning_content", "reasoning"):
            v = me.get(k)
            if v:
                return str(v)
    return ""


def get_pydantic_response_format(model_class: type[BaseModel]) -> dict:
    """
    从 Pydantic 模型生成 OpenAI 兼容的 response_format
    """
    schema = copy.deepcopy(model_class.model_json_schema())
    if "$defs" in schema:
        defs = schema.pop("$defs")
        schema["$defs"] = defs
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_class.__name__,
            "schema": schema,
            "strict": True
        }
    }


def parse_model_spec(model_spec: str) -> dict:
    """
    解析模型规格字符串
    
    支持的格式：
    - OpenRouter: "provider/model-name" (如 "qwen/qwen3-30b-a3b-instruct-2507")
    - vLLM 简化格式: "model_name:port" (如 "qwen3-30b:9041")
    - vLLM 完整格式: "model_name@host:port" (如 "qwen3-30b@{host}:9041")
    """
    result = {
        "type": "vllm",
        "model_name": model_spec,
        "host": None,
        "port": None,
        "base_url": None,
    }
    
    # 检查是否是 OpenRouter 模型
    if '/' in model_spec and '@' not in model_spec:
        base_name = model_spec.split(':')[0] if ':' in model_spec else model_spec
        if not base_name.startswith('/'):
            result["type"] = "openrouter"
            result["model_name"] = model_spec
            return result
    
    # vLLM 完整格式: model_name@host:port
    if '@' in model_spec:
        at_idx = model_spec.index('@')
        result["model_name"] = model_spec[:at_idx]
        host_port = model_spec[at_idx + 1:]
        
        if ':' in host_port:
            host, port_str = host_port.rsplit(':', 1)
            result["host"] = host
            try:
                result["port"] = int(port_str)
            except ValueError:
                result["port"] = 9041
        else:
            result["host"] = host_port
            result["port"] = 9041
        
        result["base_url"] = f"http://{result['host']}:{result['port']}/v1"
        return result
    
    # vLLM 简化格式: model_name:port
    if ':' in model_spec:
        parts = model_spec.rsplit(':', 1)
        result["model_name"] = parts[0]
        try:
            result["port"] = int(parts[1])
        except ValueError:
            result["port"] = 9041
        result["host"] = "127.0.0.1"
        result["base_url"] = f"http://{result['host']}:{result['port']}/v1"
        return result
    
    # 仅模型名称，使用默认端口和 host
    result["host"] = "127.0.0.1"
    result["port"] = 9041
    result["base_url"] = f"http://{result['host']}:{result['port']}/v1"
    return result


class OpenRouterClient:
    """OpenRouter API 客户端"""
    
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.model_id = model_name
        
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.api_key:
            raise ValueError("OpenRouter API key 未配置！")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        
        print(f"[OpenRouter] 使用模型: {self.model_id}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_model: Optional[type[BaseModel]] = None,
        use_json_format: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        return_raw_response: bool = False,
    ) -> Tuple[Union[str, BaseModel], Tuple[int, int]]:
        last_error = None
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                if response_model is not None or use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**kwargs)
                
                content = extract_message_text(response.choices[0].message)
                if content is None:
                    content = ""
                
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
                
                if response_model is not None:
                    try:
                        parsed = response_model.model_validate_json(content.strip())
                        if return_raw_response:
                            return parsed, (prompt_tokens, completion_tokens), response
                        return parsed, (prompt_tokens, completion_tokens)
                    except Exception:
                        if return_raw_response:
                            return content.strip(), (prompt_tokens, completion_tokens), response
                        return content.strip(), (prompt_tokens, completion_tokens)
                
                if return_raw_response:
                    return content.strip(), (prompt_tokens, completion_tokens), response
                return content.strip(), (prompt_tokens, completion_tokens)
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"  OpenRouter API 调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(wait_time)
        
        raise last_error


class VLLMClient:
    """vLLM 客户端"""
    
    def __init__(self, model_name: str, base_url: str = None):
        self.model_name = model_name
        
        parsed = parse_model_spec(model_name)
        self.model_id = parsed["model_name"]
        self.port = parsed["port"] or 9041
        self.host = parsed["host"] or "127.0.0.1"
        
        if base_url:
            self.base_url = base_url
        elif parsed["base_url"]:
            self.base_url = parsed["base_url"]
        else:
            self.base_url = f"http://{self.host}:{self.port}/v1"
        
        self.client = OpenAI(
            api_key="dummy_key",
            base_url=self.base_url,
            http_client=make_http_client(),
        )
        
        print(f"[vLLM] 连接到: {self.base_url}, 模型: {self.model_id}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_model: Optional[type[BaseModel]] = None,
        use_json_format: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        return_raw_response: bool = False,
    ) -> Tuple[Union[str, BaseModel], Tuple[int, int]]:
        last_error = None
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                if response_model is not None:
                    kwargs["response_format"] = get_pydantic_response_format(response_model)
                elif use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}
                
                response = self.client.chat.completions.create(**kwargs)
                
                content = extract_message_text(response.choices[0].message)
                if content is None:
                    content = ""
         
                prompt_tokens = 0
                completion_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                
                if response_model is not None:
                    try:
                        parsed = response_model.model_validate_json(content.strip())
                        if return_raw_response:
                            return parsed, (prompt_tokens, completion_tokens), response
                        return parsed, (prompt_tokens, completion_tokens)
                    except Exception:
                        if return_raw_response:
                            return content.strip(), (prompt_tokens, completion_tokens), response
                        return content.strip(), (prompt_tokens, completion_tokens)
                
                if return_raw_response:
                    return content.strip(), (prompt_tokens, completion_tokens), response
                return content.strip(), (prompt_tokens, completion_tokens)
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    print(f"  API 调用失败 (尝试 {attempt + 1}/{max_retries}): {str(e)[:100]}")
                    time.sleep(wait_time)
        
        raise last_error


class PromptLoader:
    """Prompt 加载器"""
    
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "prompts" / "evaluation"
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}
    
    def load_prompt(self, dimension: str) -> str:
        if dimension in self._cache:
            return self._cache[dimension]
        
        prompt_file = self.prompts_dir / f"{dimension}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt 文件不存在: {prompt_file}")
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._cache[dimension] = content
        return content
    
    def format_prompt(self, dimension: str, **kwargs) -> str:
        template = self.load_prompt(dimension)
        try:
            return template.format(**kwargs)
        except KeyError:
            result = template
            for key, value in kwargs.items():
                result = result.replace("{" + key + "}", str(value))
            return result


def parse_evaluation_result(result_text: Any) -> Dict[str, Any]:
    """解析评估结果"""
    result = {"score": None, "reason": "评估结果为空"}
    
    if result_text is None:
        return result
    
    if isinstance(result_text, dict):
        score_val = result_text.get("score")
        final_score = None
        if score_val is not None:
            try:
                s_float = float(score_val)
                final_score = int(round(s_float))
                final_score = max(0, min(5, final_score))
            except (ValueError, TypeError):
                match = re.search(r"(-?\d+)", str(score_val))
                if match:
                    final_score = int(match.group(1))
                    final_score = max(0, min(5, final_score))
        
        result["score"] = final_score
        result["reason"] = str(result_text.get("reason") or "不做多余解释")
        return result
    
    text = str(result_text).strip()
    if not text:
        return result
    
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
        try:
            data = json.loads(json_str)
            return parse_evaluation_result(data)
        except json.JSONDecodeError:
            pass
    
    try:
        data = json.loads(text)
        return parse_evaluation_result(data)
    except json.JSONDecodeError:
        pass
    
    score_match = re.search(r"(?:score|分数)\s*[:=：]\s*(-?\d+)", text, re.IGNORECASE)
    if score_match:
        try:
            score = int(score_match.group(1))
            result["score"] = max(0, min(5, score))
        except ValueError:
            pass
    
    reason_match = re.search(r"(?:reason|原因|理由)\s*[:=：]\s*(.+)", text, re.IGNORECASE)
    if reason_match:
        result["reason"] = reason_match.group(1).strip()
    else:
        result["reason"] = text[:200]
    
    return result


def _strip_json_fence(text: str) -> str:
    """移除 Markdown JSON 代码块包裹"""
    if not text:
        return ""
    match = re.search(r"```json\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _trim_to_balanced_json(text: str) -> str:
    """尝试截断到最后一个完整的 JSON 对象"""
    if not text:
        return ""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    last_good = None
    in_string = False
    escape = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue
        if ch == "\"":
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_good = idx
    if last_good is not None:
        return text[start:last_good + 1]
    return text[start:]


def _try_load_json(text: str) -> Optional[Dict[str, Any]]:
    """尽力解析 JSON（容忍截断/多余文本）"""
    if not text:
        return None
    candidates = []
    cleaned = _strip_json_fence(text)
    candidates.append(cleaned)
    # 优先取首个 { ... } 范围
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        candidates.append(cleaned[start:end + 1])
    # 截断到最后一个完整对象
    candidates.append(_trim_to_balanced_json(cleaned))

    for cand in candidates:
        if not cand:
            continue
        normalized = re.sub(r",\s*([}\]])", r"\1", cand)
        try:
            data = json.loads(normalized)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            # 尝试补齐大括号
            depth = normalized.count("{") - normalized.count("}")
            if depth > 0:
                try:
                    data = json.loads(normalized + ("}" * depth))
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
    return None


def parse_realness_multi_result(result_text: Any) -> Dict[str, Any]:
    """解析多维度真实性评估结果"""
    default_metric = {"score": None, "reasoning": "解析失败"}
    result = {
        "metrics": {
            "Response_Brevity": default_metric.copy(),
            "Information_Proactivity": default_metric.copy(),
            "Emotional_Restraint": default_metric.copy(),
            "Language_Polish": default_metric.copy(),
            "Conversational_Cooperation": default_metric.copy(),
        },
        "overall_impression": "Unknown",
        "average_score": None,
        "raw_response": "",
    }
    
    if result_text is None:
        return result
    
    if isinstance(result_text, RealnessMultiEvaluation):
        metrics = result_text.metrics
        result["metrics"] = {
            "Response_Brevity": {
                "score": metrics.Response_Brevity.score,
                "reasoning": metrics.Response_Brevity.reasoning
            },
            "Information_Proactivity": {
                "score": metrics.Information_Proactivity.score,
                "reasoning": metrics.Information_Proactivity.reasoning
            },
            "Emotional_Restraint": {
                "score": metrics.Emotional_Restraint.score,
                "reasoning": metrics.Emotional_Restraint.reasoning
            },
            "Language_Polish": {
                "score": metrics.Language_Polish.score,
                "reasoning": metrics.Language_Polish.reasoning
            },
            "Conversational_Cooperation": {
                "score": metrics.Conversational_Cooperation.score,
                "reasoning": metrics.Conversational_Cooperation.reasoning
            },
        }
        result["overall_impression"] = result_text.overall_impression
        result["average_score"] = result_text.average_score
        result["raw_response"] = result_text.model_dump_json()
        return result
    
    if isinstance(result_text, BaseModel):
        return parse_realness_multi_result(result_text.model_dump())
    
    if isinstance(result_text, dict):
        if "metrics" in result_text:
            for key in result["metrics"]:
                if key in result_text["metrics"]:
                    metric_data = result_text["metrics"][key]
                    score = metric_data.get("score")
                    if score is not None:
                        try:
                            score = int(score)
                            score = max(1, min(5, score))
                        except (ValueError, TypeError):
                            score = None
                    result["metrics"][key] = {
                        "score": score,
                        "reasoning": str(metric_data.get("reasoning", ""))
                    }
            result["overall_impression"] = result_text.get("overall_impression", "Unknown")
            avg = result_text.get("average_score")
            if avg is not None:
                try:
                    result["average_score"] = float(avg)
                except (ValueError, TypeError):
                    pass
        result["raw_response"] = json.dumps(result_text, ensure_ascii=False)
        return result
    
    text = str(result_text).strip()
    result["raw_response"] = text
    if not text:
        return result

    data = _try_load_json(text)
    if data is not None:
        parsed = parse_realness_multi_result(data)
        parsed["raw_response"] = text
        return parsed

    return result


def create_llm_client(model_name: str, base_url: str = None, api_key: str = None):
    """
    工厂函数：根据模型名称自动选择客户端类型
    """
    parsed = parse_model_spec(model_name)
    
    if parsed["type"] == "openrouter":
        return OpenRouterClient(model_name, api_key=api_key)
    else:
        return VLLMClient(model_name, base_url=base_url)


