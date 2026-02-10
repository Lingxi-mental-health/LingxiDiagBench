# -*- coding: utf-8 -*-
"""
Reranker 模块

使用 Qwen3-Reranker-8B 对检索结果进行重排序
"""

import os
import torch
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class Reranker:
    """Reranker 模型包装器"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化 Reranker 模型
        
        Args:
            model_path: 模型路径（如果为None，从环境变量读取）
            device: 设备（'cuda' 或 'cpu'），如果为None则自动选择
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError("需要安装 transformers 库: pip install transformers")
        
        # 从环境变量或参数获取模型路径
        if model_path is None:
            model_path = os.getenv('LOCAL_RERANKER_MODEL_PATH')
        
        if not model_path or not os.path.exists(model_path):
            raise ValueError(
                f"Reranker 模型路径不存在: {model_path}\n"
                "请在 .env 文件中设置 LOCAL_RERANKER_MODEL_PATH"
            )
        
        self.model_path = model_path
        
        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"正在加载 Reranker 模型: {model_path}")
        logger.info(f"使用设备: {self.device}")
        
        try:
            # 加载 tokenizer 和 model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            
            # 确保 tokenizer 有 padding token
            if self.tokenizer.pad_token is None:
                # 如果没有 pad_token，使用 eos_token 或 unk_token
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                elif self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                    self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
                else:
                    # 如果都没有，添加一个特殊的 pad_token
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.info(f"设置 padding token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
            
            # 确保 pad_token_id 有效
            if self.tokenizer.pad_token_id is None:
                logger.warning("pad_token_id 为 None，尝试从 pad_token 获取")
                if self.tokenizer.pad_token:
                    # 尝试通过编码获取 ID
                    try:
                        pad_ids = self.tokenizer.encode(self.tokenizer.pad_token, add_special_tokens=False)
                        if pad_ids:
                            self.tokenizer.pad_token_id = pad_ids[0]
                            logger.info(f"从 pad_token 获取 pad_token_id: {self.tokenizer.pad_token_id}")
                    except:
                        pass
            
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
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map='auto'
                )
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype
                )
                # 手动移动到设备
                self.model = self.model.to(self.device)
            
            # 如果添加了新的 pad_token，需要调整模型嵌入层大小
            if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is None:
                # 设置模型的 pad_token_id
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            # 如果 tokenizer 的 pad_token_id 与模型不一致，调整模型嵌入层
            if self.tokenizer.pad_token_id is not None:
                if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id != self.tokenizer.pad_token_id:
                    # 检查是否需要调整嵌入层大小
                    if len(self.tokenizer) > self.model.config.vocab_size:
                        logger.info(f"调整模型嵌入层大小: {self.model.config.vocab_size} -> {len(self.tokenizer)}")
                        self.model.resize_token_embeddings(len(self.tokenizer))
                    # 更新模型的 pad_token_id
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
            self.model.eval()
            
            logger.info(f"✓ Reranker 模型加载完成")
            
        except Exception as e:
            logger.error(f"加载 Reranker 模型失败: {e}")
            raise
    
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
        
        # 检查是否有 padding token，如果没有则使用 batch_size=1
        has_pad_token = (self.tokenizer.pad_token is not None and 
                        self.tokenizer.pad_token_id is not None)
        
        if not has_pad_token:
            logger.warning("Tokenizer 没有有效的 padding token，将逐个处理文档")
            batch_size = 1
        
        # 批处理
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # 构建输入对 (query, document)
            pairs = [[query, doc] for doc in batch_docs]
            
            # 尝试批处理，如果失败则降级到逐个处理
            try:
                # Tokenize
                if has_pad_token and len(pairs) > 1:
                    # 批处理模式
                    inputs = self.tokenizer(
                        pairs,
                        truncation=True,
                        max_length=512,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # 计算分数
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        batch_scores = outputs.logits[:, 0].cpu().tolist()
                    
                    scores.extend(batch_scores)
                else:
                    # 逐个处理模式
                    for pair in pairs:
                        inputs = self.tokenizer(
                            pair,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt"
                        ).to(self.device)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            score = outputs.logits[0, 0].item()
                            scores.append(score)
            
            except (ValueError, RuntimeError) as e:
                # 如果批处理失败（例如 padding token 问题），降级到逐个处理
                if "padding token" in str(e).lower() or "batch" in str(e).lower():
                    logger.warning(f"批处理失败 ({e})，降级到逐个处理模式")
                    # 逐个处理当前批次
                    for pair in pairs:
                        try:
                            inputs = self.tokenizer(
                                pair,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt"
                            ).to(self.device)
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                                score = outputs.logits[0, 0].item()
                                scores.append(score)
                        except Exception as e2:
                            logger.error(f"处理单个文档失败: {e2}，跳过该文档")
                            scores.append(0.0)  # 使用默认分数
                    # 后续批次也使用逐个处理
                    batch_size = 1
                    has_pad_token = False
                else:
                    # 其他错误，重新抛出
                    raise
        
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

