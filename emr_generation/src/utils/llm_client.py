"""
LLM 客户端 - 支持本地 vLLM 部署的模型调用
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from ..config import Config


T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """LLM 客户端，支持本地 vLLM 部署"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        model: str = None,
        api_key: str = "EMPTY",  # vLLM 不需要真实的 API key
    ):
        """
        初始化 LLM 客户端
        
        Args:
            host: LLM 服务地址
            port: LLM 服务端口
            model: 模型名称
            api_key: API密钥（vLLM通常不需要）
        """
        if OpenAI is None:
            raise ImportError("请安装 openai 库: pip install openai")
        
        self.host = host or Config.LLM_DEFAULT_HOST
        self.port = port or Config.LLM_DEFAULT_PORT
        self.model = model or Config.LLM_DEFAULT_MODEL
        self.base_url = Config.get_llm_base_url(self.host, self.port)
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
        )
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            模型响应文本
        """
        temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        max_tokens = max_tokens or Config.LLM_MAX_TOKENS
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        content = response.choices[0].message.content
        return content if content else ""
    
    def extract_structured(
        self,
        text: str,
        output_schema: Type[T],
        instruction: str = None,
        temperature: float = 0.3,
    ) -> Optional[T]:
        """
        从文本中提取结构化信息
        
        Args:
            text: 输入文本
            output_schema: Pydantic 模型类
            instruction: 额外的指令
            temperature: 温度参数（结构化提取建议使用较低温度）
            
        Returns:
            解析后的 Pydantic 模型实例
        """
        schema_json = json.dumps(
            output_schema.model_json_schema(),
            ensure_ascii=False,
            indent=2
        )
        
        system_prompt = f"""你是一个专业的医疗信息提取助手。请从给定的文本中提取结构化信息。

输出必须是严格的 JSON 格式，符合以下 schema：
{schema_json}

注意：
1. 只输出 JSON，不要有其他文字
2. 如果某个字段无法从文本中提取，设为 null
3. 确保 JSON 格式正确"""

        user_prompt = f"""请从以下文本中提取信息：

{text}

{instruction if instruction else ''}

请输出 JSON："""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.chat(messages, temperature=temperature)
            
            # 尝试解析 JSON
            # 处理可能的 markdown 代码块
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response.strip())
            return output_schema.model_validate(data)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"解析结构化输出失败: {e}")
            print(f"原始响应: {response if 'response' in locals() else 'N/A'}")
            return None
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            生成的文本
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)
    
    def generate_with_context(
        self,
        template: str,
        context: Dict[str, Any],
        system_prompt: str = None,
        temperature: float = None,
    ) -> str:
        """
        使用模板和上下文生成文本
        
        Args:
            template: 提示模板（使用 {key} 格式）
            context: 上下文字典
            system_prompt: 系统提示
            temperature: 温度参数
            
        Returns:
            生成的文本
        """
        prompt = template.format(**context)
        return self.generate_text(prompt, system_prompt, temperature)
