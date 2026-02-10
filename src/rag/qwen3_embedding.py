# -*- coding: utf-8 -*-
"""
Qwen3 Embedding 模型包装器

支持使用 transformers 库加载 Qwen3-Embedding-8B 模型
"""

import os
import torch
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)


class Qwen3Embedding:
    """Qwen3 Embedding 模型包装器"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化 Qwen3 Embedding 模型
        
        Args:
            model_path: 模型路径
            device: 设备（'cuda' 或 'cpu'），如果为None则自动选择
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("需要安装 transformers 库: pip install transformers")
        
        self.model_path = model_path
        
        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"正在加载 Qwen3 Embedding 模型: {model_path}")
        logger.info(f"使用设备: {self.device}")
        
        try:
            # 加载 tokenizer 和 model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # 确定数据类型
            dtype = torch.float16 if self.device == 'cuda' else torch.float32
            
            # 检查是否可以使用 device_map（需要 accelerate 库）
            use_device_map = False
            if self.device == 'cuda':
                try:
                    import accelerate
                    use_device_map = True
                except ImportError:
                    logger.warning("accelerate 库未安装，将手动将模型移动到 GPU")
                    logger.info("提示: 安装 accelerate 可以自动管理模型设备: pip install accelerate")
                    use_device_map = False
            
            # 加载模型
            if use_device_map:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map='auto'
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype
                )
                # 手动移动到设备
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # 获取嵌入维度
            # Qwen3-Embedding-8B 的维度通常是 8192
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt").to(self.device)
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state.shape[-1]
            
            logger.info(f"✓ Qwen3 Embedding 模型加载完成，维度: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"加载 Qwen3 Embedding 模型失败: {e}")
            raise
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本或文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
            convert_to_numpy: 是否转换为 numpy 数组
            normalize_embeddings: 是否归一化向量
        
        Returns:
            嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 批处理
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用 mean pooling
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            
            # 归一化
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu())
        
        # 合并所有批次
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            return all_embeddings.numpy()
        else:
            return all_embeddings
    
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling
        
        Args:
            last_hidden_state: 最后一层的隐藏状态
            attention_mask: 注意力掩码
        
        Returns:
            池化后的向量
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def get_sentence_embedding_dimension(self) -> int:
        """获取嵌入维度"""
        return self.embedding_dim

