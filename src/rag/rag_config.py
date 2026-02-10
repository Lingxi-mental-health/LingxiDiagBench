# -*- coding: utf-8 -*-
"""
RAG配置模块
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 向量化配置
# 支持本地路径或Hugging Face模型名称
# 如果是本地路径，直接使用；如果是模型名称，会尝试从Hugging Face下载
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# 本地嵌入模型路径（如果设置了，优先使用本地模型）
LOCAL_EMBEDDING_MODEL_PATH = os.getenv('LOCAL_EMBEDDING_MODEL_PATH', None)
CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '500'))
CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '50'))

# FAISS配置
FAISS_INDEX_TYPE = os.getenv('FAISS_INDEX_TYPE', 'Flat')
FAISS_METRIC = os.getenv('FAISS_METRIC', 'L2')

# 检索配置
RAG_TOP_K_RESULTS = int(os.getenv('RAG_TOP_K_RESULTS', '5'))
RAG_SCORE_THRESHOLD = float(os.getenv('RAG_SCORE_THRESHOLD', '0.6'))

# Reranker 配置
LOCAL_RERANKER_MODEL_PATH = os.getenv('LOCAL_RERANKER_MODEL_PATH', None)
ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'false').lower() == 'true'
RERANKER_TOP_K = int(os.getenv('RERANKER_TOP_K', '3'))

# DeepInfra 配置
DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY', None)
DEEPINFRA_EMBEDDING_MODEL = os.getenv('DEEPINFRA_EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-8B')
DEEPINFRA_RERANKER_MODEL = os.getenv('DEEPINFRA_RERANKER_MODEL', 'Qwen/Qwen3-Reranker-8B')
USE_DEEPINFRA_EMBEDDING = os.getenv('USE_DEEPINFRA_EMBEDDING', 'true').lower() == 'true'
USE_DEEPINFRA_RERANKER = os.getenv('USE_DEEPINFRA_RERANKER', 'false').lower() == 'true'

# 知识库目录
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / 'knowledge_base'
KNOWLEDGE_BASE_DOC_DIR = KNOWLEDGE_BASE_DIR / 'doc'
KNOWLEDGE_BASE_INDEX_DIR = KNOWLEDGE_BASE_DIR / 'indices'

# 确保目录存在
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_DOC_DIR.mkdir(exist_ok=True)
KNOWLEDGE_BASE_INDEX_DIR.mkdir(exist_ok=True)

# Hugging Face镜像配置（强制设置，确保使用镜像）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class RAGConfig:
    """RAG配置类，统一管理所有RAG相关配置"""
    
    # 向量化配置
    EMBEDDING_MODEL = EMBEDDING_MODEL
    LOCAL_EMBEDDING_MODEL_PATH = LOCAL_EMBEDDING_MODEL_PATH
    CHUNK_SIZE = CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP
    
    # FAISS配置
    FAISS_INDEX_TYPE = FAISS_INDEX_TYPE
    FAISS_METRIC = FAISS_METRIC
    
    # 检索配置
    TOP_K_RESULTS = RAG_TOP_K_RESULTS
    SCORE_THRESHOLD = RAG_SCORE_THRESHOLD
    
    # Reranker 配置
    LOCAL_RERANKER_MODEL_PATH = LOCAL_RERANKER_MODEL_PATH
    ENABLE_RERANKING = ENABLE_RERANKING
    RERANKER_TOP_K = RERANKER_TOP_K
    
    # DeepInfra 配置
    DEEPINFRA_API_KEY = DEEPINFRA_API_KEY
    DEEPINFRA_EMBEDDING_MODEL = DEEPINFRA_EMBEDDING_MODEL
    DEEPINFRA_RERANKER_MODEL = DEEPINFRA_RERANKER_MODEL
    USE_DEEPINFRA_EMBEDDING = USE_DEEPINFRA_EMBEDDING
    USE_DEEPINFRA_RERANKER = USE_DEEPINFRA_RERANKER
    
    # 知识库目录
    KNOWLEDGE_BASE_DIR = KNOWLEDGE_BASE_DIR
    KNOWLEDGE_BASE_DOC_DIR = KNOWLEDGE_BASE_DOC_DIR
    KNOWLEDGE_BASE_INDEX_DIR = KNOWLEDGE_BASE_INDEX_DIR

