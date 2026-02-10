"""
Prompt加载器

提供统一的prompt模板加载和格式化功能
"""

import os
from pathlib import Path
from typing import Dict, Optional


class PromptLoader:
    """
    Prompt模板加载器
    
    支持从文件加载prompt模板并进行格式化
    """
    
    def __init__(self, prompts_dir: str = None):
        """
        初始化加载器
        
        Args:
            prompts_dir: prompt模板目录路径，默认为当前模块所在目录
        """
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
    
    def load(self, name: str, use_cache: bool = True) -> str:
        """
        加载prompt模板
        
        Args:
            name: 模板名称（不含.txt后缀）
            use_cache: 是否使用缓存
            
        Returns:
            模板内容
        """
        if use_cache and name in self._cache:
            return self._cache[name]
        
        # 尝试多种文件扩展名
        for ext in ['.txt', '.md', '.prompt', '']:
            prompt_file = self.prompts_dir / f"{name}{ext}"
            if prompt_file.exists():
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if use_cache:
                    self._cache[name] = content
                return content
        
        raise FileNotFoundError(f"Prompt模板不存在: {name}")
    
    def format(self, name: str, **kwargs) -> str:
        """
        加载并格式化prompt模板
        
        Args:
            name: 模板名称
            **kwargs: 格式化参数
            
        Returns:
            格式化后的内容
        """
        template = self.load(name)
        
        try:
            return template.format(**kwargs)
        except KeyError:
            # 如果format失败，尝试手动替换
            result = template
            for key, value in kwargs.items():
                result = result.replace("{" + key + "}", str(value))
            return result
    
    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
    
    def list_prompts(self) -> list:
        """列出所有可用的prompt模板"""
        prompts = []
        for ext in ['.txt', '.md', '.prompt']:
            prompts.extend([
                f.stem for f in self.prompts_dir.glob(f"*{ext}")
            ])
        return sorted(set(prompts))


# 全局加载器实例
_default_loader: Optional[PromptLoader] = None


def get_loader() -> PromptLoader:
    """获取默认加载器"""
    global _default_loader
    if _default_loader is None:
        _default_loader = PromptLoader()
    return _default_loader


def load_prompt(name: str) -> str:
    """
    加载prompt模板（快捷函数）
    
    Args:
        name: 模板名称
        
    Returns:
        模板内容
    """
    return get_loader().load(name)


def format_prompt(name: str, **kwargs) -> str:
    """
    加载并格式化prompt模板（快捷函数）
    
    Args:
        name: 模板名称
        **kwargs: 格式化参数
        
    Returns:
        格式化后的内容
    """
    return get_loader().format(name, **kwargs)

