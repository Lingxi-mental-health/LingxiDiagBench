# -*- coding: utf-8 -*-
"""
DeepInfra Reranker API 包装器

使用 DeepInfra API 进行文档重排序
"""

import os
import requests
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DeepInfraReranker:
    """DeepInfra Reranker API 包装器"""
    
    def __init__(self, api_key: str = None, model_name: str = "Qwen/Qwen3-Reranker-8B"):
        """
        初始化 DeepInfra Reranker
        
        Args:
            api_key: DeepInfra API key（如果为None，从环境变量读取）
            model_name: 模型名称（DeepInfra 支持的 reranker 模型，例如 Qwen/Qwen3-Reranker-8B）
        """
        self.api_key = api_key or os.getenv('DEEPINFRA_API_KEY')
        if not self.api_key:
            raise ValueError(
                "DeepInfra API key 未设置\n"
                "请在 .env 文件中设置 DEEPINFRA_API_KEY"
            )
        
        self.model_name = model_name
        self.api_base = "https://api.deepinfra.com/v1"
        # DeepInfra Reranker API 端点格式: /v1/inference/{model_name}
        self.api_url = f"{self.api_base}/inference/{self.model_name}"
        
        logger.info(f"初始化 DeepInfra Reranker: {self.model_name}")
        logger.info(f"API URL: {self.api_url}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果（如果为None，返回所有结果）
            batch_size: 批处理大小
        
        Returns:
            重排序后的结果列表，每个结果包含 'text', 'score', 'rank'
        """
        if not documents:
            return []
        
        if top_k is None:
            top_k = len(documents)
        top_k = min(top_k, len(documents))
        
        # 计算相关性分数
        scores = []
        
        # 批处理
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # 调用 DeepInfra API
            batch_scores = self._call_api(query, batch_docs)
            scores.extend(batch_scores)
        
        # 创建结果列表
        results = [
            {
                'text': doc,
                'score': score,
                'rank': i + 1
            }
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        
        # 按分数降序排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回 top_k 个结果
        return results[:top_k]
    
    def _call_api(self, query: str, documents: List[str]) -> List[float]:
        """
        调用 DeepInfra Reranker API
        
        使用 DeepInfra 的 inference API 格式：
        POST https://api.deepinfra.com/v1/inference/{model_name}
        Body: {"queries": [...], "documents": [...]}
        
        Args:
            query: 查询文本
            documents: 文档列表
        
        Returns:
            相关性分数列表
        """
        headers = {
            "Authorization": f"bearer {self.api_key}",  # 注意：使用小写 "bearer"
            "Content-Type": "application/json"
        }
        
        # DeepInfra Reranker API 格式
        # 注意：使用 "queries"（复数）和 "documents"（复数）
        data = {
            "queries": [query],  # 查询列表（即使只有一个查询也要用列表）
            "documents": documents  # 文档列表
        }
        
        try:
            logger.debug(f"调用 DeepInfra Reranker API: {self.api_url}")
            logger.debug(f"查询: {query[:100]}...")
            logger.debug(f"文档数量: {len(documents)}")
            
            response = requests.post(self.api_url, json=data, headers=headers, timeout=1200)  # 进一步增加超时时间到1200秒（20分钟），允许代码很慢
            
            # 检查响应状态
            if response.status_code == 404:
                logger.error(f"端点不存在: {self.api_url}")
                logger.error("请检查模型名称是否正确")
                raise ValueError(f"API 端点不存在: {self.api_url}")
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"API 响应: {result}")
            
            # 解析响应
            # DeepInfra Reranker API 返回格式可能是：
            # 1. {"results": [[score1, score2, ...]]} - 每个查询对应一个分数列表
            # 2. 直接返回分数列表
            scores = None
            
            if isinstance(result, dict):
                if 'results' in result:
                    # 格式: {"results": [[score1, score2, ...]]}
                    results_list = result['results']
                    if isinstance(results_list, list) and len(results_list) > 0:
                        # 取第一个查询的结果（因为我们只发送了一个查询）
                        scores = results_list[0]
                        if isinstance(scores, list):
                            scores = [float(s) for s in scores]
                elif 'scores' in result:
                    # 格式: {"scores": [score1, score2, ...]}
                    scores = [float(s) for s in result['scores']]
                elif 'output' in result:
                    # 可能的格式: {"output": [[score1, score2, ...]]}
                    output = result['output']
                    if isinstance(output, list) and len(output) > 0:
                        scores = output[0] if isinstance(output[0], list) else output
                        scores = [float(s) for s in scores] if isinstance(scores, list) else [float(scores)]
            elif isinstance(result, list):
                # 直接返回分数列表
                if len(result) > 0 and isinstance(result[0], list):
                    # 嵌套列表格式: [[score1, score2, ...]]
                    scores = [float(s) for s in result[0]]
                else:
                    # 直接列表格式: [score1, score2, ...]
                    scores = [float(s) for s in result]
            
            if scores is None:
                logger.error(f"无法解析 API 响应格式: {result}")
                raise ValueError(f"无法解析 API 响应: {result}")
            
            # 确保分数列表长度与文档列表一致
            if len(scores) != len(documents):
                logger.warning(f"分数列表长度 ({len(scores)}) 与文档列表长度 ({len(documents)}) 不一致")
                # 补齐或截断
                if len(scores) < len(documents):
                    scores.extend([0.0] * (len(documents) - len(scores)))
                else:
                    scores = scores[:len(documents)]
            
            logger.info(f"✓ DeepInfra Reranker API 调用成功，返回 {len(scores)} 个分数")
            logger.debug(f"分数范围: {min(scores):.4f} - {max(scores):.4f}")
            
            return scores
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"DeepInfra Reranker API HTTP 错误: {e}")
            if e.response is not None:
                try:
                    error_detail = e.response.json()
                    logger.error(f"错误详情: {error_detail}")
                except:
                    logger.error(f"响应内容: {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepInfra Reranker API 请求异常: {e}")
            raise
        except Exception as e:
            logger.error(f"DeepInfra Reranker API 调用失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def rerank_search_results(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            search_results: 检索结果列表（包含 'text', 'score', 'chunk' 等字段）
            top_k: 返回前k个结果（如果为None，返回所有结果）
        
        Returns:
            重排序后的结果列表
        """
        if not search_results:
            return []
        
        # 提取文档文本
        documents = [result['text'] for result in search_results]
        
        # 重排序
        reranked = self.rerank(query, documents, top_k=top_k)
        
        # 更新原始结果
        reranked_results = []
        for rerank_item in reranked:
            # 找到对应的原始结果
            original_result = next(
                (r for r in search_results if r['text'] == rerank_item['text']),
                None
            )
            
            if original_result:
                # 更新分数和排名
                updated_result = original_result.copy()
                updated_result['score'] = rerank_item['score']
                updated_result['rank'] = rerank_item['rank']
                updated_result['rerank_score'] = rerank_item['score']  # 保留重排序分数
                updated_result['original_score'] = original_result.get('score', 0)  # 保留原始分数
                reranked_results.append(updated_result)
        
        return reranked_results

