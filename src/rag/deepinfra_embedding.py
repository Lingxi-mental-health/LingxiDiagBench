# -*- coding: utf-8 -*-
"""
DeepInfra Embedding API 包装器

使用 DeepInfra API 进行文本嵌入
"""

import os
import requests
import numpy as np
from typing import List, Union
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

load_dotenv()

logger = logging.getLogger(__name__)


class DeepInfraEmbedding:
    """DeepInfra Embedding API 包装器"""
    
    def __init__(self, api_key: str = None, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        初始化 DeepInfra Embedding
        
        Args:
            api_key: DeepInfra API key（如果为None，从环境变量读取）
            model_name: 模型名称（DeepInfra 支持的 embedding 模型）
        """
        self.api_key = api_key or os.getenv('DEEPINFRA_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DeepInfra API key 未设置\n"
                "请在 .env 文件中设置 DEEPINFRA_API_KEY"
            )
        
        self.model_name = model_name
        self.api_base = "https://api.deepinfra.com/v1"
        self.embedding_dim = None  # 将在首次调用时确定
        
        # 创建 session 以复用连接
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
        
        logger.info(f"初始化 DeepInfra Embedding: {self.model_name}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 50,  # 增加默认批次大小
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        max_workers: int = 5  # 并发请求数
    ) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本或文本列表
            batch_size: 批处理大小（DeepInfra API 支持批量请求）
            show_progress_bar: 是否显示进度条
            convert_to_numpy: 是否转换为 numpy 数组
            normalize_embeddings: 是否归一化向量
            max_workers: 最大并发请求数
        
        Returns:
            嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
        
        total_texts = len(texts)
        if total_texts == 0:
            return np.array([])
        
        # 将文本分成批次
        batches = []
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append((i // batch_size, batch_texts))
        
        total_batches = len(batches)
        logger.info(f"开始生成嵌入向量: {total_texts} 个文本，{total_batches} 个批次，批次大小: {batch_size}，并发数: {max_workers}")
        
        all_embeddings = [None] * total_batches
        completed = 0
        start_time = time.time()
        
        # 使用线程池并发处理批次
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_batch = {
                executor.submit(self._call_api, batch_texts): (batch_idx, batch_texts)
                for batch_idx, batch_texts in batches
            }
            
            # 收集结果
            for future in as_completed(future_to_batch):
                batch_idx, batch_texts = future_to_batch[future]
                try:
                    embeddings = future.result()
                    
                    # 归一化
                    if normalize_embeddings:
                        embeddings = self._normalize(embeddings)
                    
                    all_embeddings[batch_idx] = embeddings
                    completed += 1
                    
                    # 显示进度
                    if show_progress_bar or completed % max(1, total_batches // 10) == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        remaining = (total_batches - completed) / rate if rate > 0 else 0
                        logger.info(
                            f"进度: {completed}/{total_batches} 批次 "
                            f"({completed * batch_size}/{total_texts} 文本) "
                            f"速度: {rate:.2f} 批次/秒 "
                            f"预计剩余: {remaining:.1f} 秒"
                        )
                        
                except Exception as e:
                    logger.error(f"批次 {batch_idx} 处理失败: {e}")
                    # 使用空向量作为占位符
                    batch_size_actual = len(batch_texts)
                    all_embeddings[batch_idx] = np.zeros((batch_size_actual, self.embedding_dim or 4096))
        
        # 合并所有批次（按顺序）
        result = np.vstack(all_embeddings)
        
        elapsed_total = time.time() - start_time
        logger.info(f"✓ 嵌入向量生成完成: {total_texts} 个文本，耗时 {elapsed_total:.2f} 秒，平均速度 {total_texts/elapsed_total:.2f} 文本/秒")
        
        if not convert_to_numpy:
            return result.tolist()
        
        return result
    
    def _call_api(self, texts: List[str], retry_count: int = 3) -> np.ndarray:
        """
        调用 DeepInfra Embedding API
        
        Args:
            texts: 文本列表
            retry_count: 重试次数
        
        Returns:
            嵌入向量数组
        """
        url = f"{self.api_base}/embeddings"
        
        # DeepInfra 使用 OpenAI 兼容的 API 格式
        # 对于多个文本，DeepInfra 支持批量处理
        data = {
            "model": self.model_name,
            "input": texts  # 直接传递列表，DeepInfra 支持批量
        }
        
        last_error = None
        for attempt in range(retry_count):
            try:
                response = self.session.post(url, json=data, timeout=1200)  # 进一步大幅增加超时时间到1200秒（20分钟），允许代码很慢
                response.raise_for_status()
                
                result = response.json()
                
                # 解析响应（OpenAI 兼容格式）
                if isinstance(result, dict) and 'data' in result:
                    # OpenAI 兼容格式: {"data": [{"embedding": [...], "index": 0}, ...]}
                    embeddings_list = []
                    for item in result['data']:
                        if isinstance(item, dict) and 'embedding' in item:
                            embeddings_list.append(item['embedding'])
                        elif isinstance(item, list):
                            # 如果直接是嵌入向量
                            embeddings_list.append(item)
                    embeddings = np.array(embeddings_list)
                elif isinstance(result, list):
                    # 直接返回嵌入列表
                    embeddings = np.array(result)
                else:
                    raise ValueError(f"无法解析 API 响应: {result}")
                
                # 记录维度（首次调用时）
                if self.embedding_dim is None:
                    self.embedding_dim = embeddings.shape[-1]
                    logger.info(f"✓ DeepInfra Embedding 维度: {self.embedding_dim}")
                
                return embeddings
                
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2  # 指数退避
                    logger.warning(f"API 调用失败 (尝试 {attempt + 1}/{retry_count})，{wait_time} 秒后重试: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"DeepInfra Embedding API 调用失败: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                        logger.error(f"响应内容: {e.response.text}")
                    raise
    
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        归一化向量
        
        Args:
            embeddings: 嵌入向量数组
        
        Returns:
            归一化后的向量数组
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        return embeddings / norms
    
    def get_sentence_embedding_dimension(self) -> int:
        """获取嵌入维度"""
        if self.embedding_dim is None:
            # 如果还没有调用过，先调用一次测试
            test_embedding = self.encode("test")
            return self.embedding_dim
        return self.embedding_dim

