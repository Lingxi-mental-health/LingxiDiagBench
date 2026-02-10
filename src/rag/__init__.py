# -*- coding: utf-8 -*-
"""
RAG (Retrieval-Augmented Generation) 模块

提供基于向量检索的知识增强功能，用于辅助医生AI进行诊断
"""

from .vector_store import VectorStore
from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .rag_config import RAGConfig

# 可选导入 Qwen3 相关模块
try:
    from .qwen3_embedding import Qwen3Embedding
    __all__ = ['VectorStore', 'DocumentLoader', 'TextProcessor', 'RAGConfig', 'Qwen3Embedding']
except ImportError:
    __all__ = ['VectorStore', 'DocumentLoader', 'TextProcessor', 'RAGConfig']

try:
    from .reranker import Reranker
    __all__.append('Reranker')
except ImportError:
    pass

# 可选导入 DeepInfra 相关模块
try:
    from .deepinfra_embedding import DeepInfraEmbedding
    __all__.append('DeepInfraEmbedding')
except ImportError:
    pass

try:
    from .deepinfra_reranker import DeepInfraReranker
    __all__.append('DeepInfraReranker')
except ImportError:
    pass

