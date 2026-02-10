# -*- coding: utf-8 -*-
"""
向量存储模块
使用FAISS进行向量索引和检索
"""

import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# 强制设置 Hugging Face 镜像（必须在导入任何 huggingface 相关库之前）
# 设置多个环境变量以确保镜像生效
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.expanduser('~/.cache/huggingface')
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# 尝试设置 huggingface_hub 的镜像端点（在导入之前）
try:
    # 如果已经导入过，直接设置
    import huggingface_hub
    if hasattr(huggingface_hub, 'constants'):
        huggingface_hub.constants.ENDPOINT = 'https://hf-mirror.com'
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
except ImportError:
    pass

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    # 在导入 sentence_transformers 之前再次确保镜像设置
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .rag_config import EMBEDDING_MODEL, LOCAL_EMBEDDING_MODEL_PATH, RAG_TOP_K_RESULTS, KNOWLEDGE_BASE_INDEX_DIR, RAGConfig

# 尝试导入 Qwen3 Embedding
try:
    from .qwen3_embedding import Qwen3Embedding
    Qwen3EmbeddingAvailable = True
except ImportError:
    Qwen3EmbeddingAvailable = False

# 尝试导入 DeepInfra Embedding
try:
    from .deepinfra_embedding import DeepInfraEmbedding
    DeepInfraEmbeddingAvailable = True
except ImportError:
    DeepInfraEmbeddingAvailable = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS向量存储"""
    
    def __init__(self, embedding_model: str = None, index_path: Optional[str] = None):
        """
        初始化向量存储
        
        Args:
            embedding_model: 嵌入模型路径（本地路径）或模型名称（Hugging Face）
            index_path: 索引保存路径
        """
        if faiss is None:
            raise ImportError("需要安装faiss-cpu库: pip install faiss-cpu")
        
        # 检查是否使用 DeepInfra
        use_deepinfra = RAGConfig.USE_DEEPINFRA_EMBEDDING
        
        if use_deepinfra:
            # 使用 DeepInfra Embedding
            self.embedding_model_name = RAGConfig.DEEPINFRA_EMBEDDING_MODEL
            is_local_path = False
            is_qwen3 = False
            is_deepinfra = True
        else:
            # 优先使用本地模型路径，如果没有则使用配置的模型
            if embedding_model:
                self.embedding_model_name = embedding_model
            elif LOCAL_EMBEDDING_MODEL_PATH and os.path.exists(LOCAL_EMBEDDING_MODEL_PATH):
                self.embedding_model_name = LOCAL_EMBEDDING_MODEL_PATH
            else:
                self.embedding_model_name = EMBEDDING_MODEL
            
            # 判断是本地路径还是模型名称
            is_local_path = os.path.exists(self.embedding_model_name) or os.path.isdir(self.embedding_model_name)
            is_deepinfra = False
        
        self.index_path = index_path or str(KNOWLEDGE_BASE_INDEX_DIR / "faiss_index")
        
        # 检测是否是 Qwen3 Embedding 模型（仅在非 DeepInfra 模式下）
        if not is_deepinfra:
            is_qwen3 = False
            if is_local_path:
                # 检查是否是 Qwen3 模型（通过检查 config.json 或目录名）
                model_path = Path(self.embedding_model_name)
                config_file = model_path / "config.json"
                if config_file.exists():
                    try:
                        import json
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                            # 检查模型类型
                            model_type = config.get('model_type', '').lower()
                            arch = config.get('architectures', [])
                            if 'qwen' in model_type or any('qwen' in str(a).lower() for a in arch):
                                is_qwen3 = True
                    except:
                        pass
                # 也可以通过路径名判断
                if 'qwen' in str(model_path).lower() and 'embedding' in str(model_path).lower():
                    is_qwen3 = True
            else:
                # 检查模型名称
                if 'qwen' in self.embedding_model_name.lower() and 'embedding' in self.embedding_model_name.lower():
                    is_qwen3 = True
        else:
            is_qwen3 = False
        
        # 初始化嵌入模型
        logger.info(f"正在加载嵌入模型: {self.embedding_model_name}")
        if is_deepinfra:
            logger.info("使用 DeepInfra API 嵌入模型（需要网络连接和 API key）")
        elif is_local_path:
            logger.info("使用本地部署的嵌入模型（无需网络连接）")
        else:
            logger.info(f"使用Hugging Face模型（需要网络连接）")
            # 只有在使用Hugging Face模型时才设置镜像
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        try:
            if is_deepinfra:
                # 使用 DeepInfra Embedding
                if not DeepInfraEmbeddingAvailable:
                    raise ImportError("DeepInfra Embedding 模块不可用，请检查依赖")
                logger.info("使用 DeepInfra Embedding API")
                self.embedding_model = DeepInfraEmbedding(
                    api_key=RAGConfig.DEEPINFRA_API_KEY,
                    model_name=self.embedding_model_name
                )
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            elif is_qwen3:
                # 使用 Qwen3 Embedding
                if not Qwen3EmbeddingAvailable:
                    raise ImportError("需要安装 transformers 库以使用 Qwen3 Embedding: pip install transformers")
                logger.info("检测到 Qwen3 Embedding 模型，使用 Qwen3Embedding 加载")
                self.embedding_model = Qwen3Embedding(self.embedding_model_name)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            else:
                # 使用 SentenceTransformer
                if SentenceTransformer is None:
                    raise ImportError("需要安装sentence-transformers库: pip install sentence-transformers")
                if is_local_path:
                    # 从本地路径加载模型，明确指定设备避免 accelerate 的 device_map 问题
                    logger.info(f"从本地路径加载模型: {self.embedding_model_name}")
                    self.embedding_model = SentenceTransformer(
                        self.embedding_model_name,
                        device="cpu",
                        trust_remote_code=True
                    )
                else:
                    # 从Hugging Face加载模型
                    cache_dir = os.path.expanduser('~/.cache/huggingface')
                    self.embedding_model = SentenceTransformer(
                        self.embedding_model_name,
                        cache_folder=cache_dir,
                        device="cpu",
                        trust_remote_code=True
                    )
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            logger.info(f"✓ 嵌入模型加载完成，维度: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            if not is_local_path:
                logger.info("提示: 如果网络无法访问，请设置本地模型路径:")
                logger.info("  export LOCAL_EMBEDDING_MODEL_PATH=/path/to/local/model")
            else:
                logger.info(f"提示: 请检查本地模型路径是否正确: {self.embedding_model_name}")
            raise
        
        # 初始化FAISS索引
        self.index = None
        self.chunks = []  # 存储原始文本块
        self.chunk_embeddings = None  # 存储嵌入向量
    
    def build_index(self, chunks: List[Dict[str, Any]]):
        """
        构建向量索引
        
        Args:
            chunks: 文本块列表
        """
        if not chunks:
            raise ValueError("文本块列表不能为空")
        
        logger.info(f"正在为 {len(chunks)} 个文本块构建索引...")
        
        # 提取文本
        texts = [chunk['text'] for chunk in chunks]
        
        # 生成嵌入向量
        logger.info("正在生成嵌入向量...")
        # 对于 DeepInfra API，启用进度显示和并发处理
        encode_kwargs = {
            'show_progress_bar': True,
            'convert_to_numpy': True,
            'normalize_embeddings': True  # 归一化，用于余弦相似度
        }
        # 如果是 DeepInfra Embedding，增加并发数
        if hasattr(self.embedding_model, 'session'):  # DeepInfra 有 session 属性
            encode_kwargs['max_workers'] = 5  # 并发请求数
            encode_kwargs['batch_size'] = 50  # 更大的批次大小
        embeddings = self.embedding_model.encode(texts, **encode_kwargs)
        
        # 创建FAISS索引（使用内积，因为向量已归一化，等价于余弦相似度）
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # 添加向量到索引
        self.index.add(embeddings.astype('float32'))
        
        # 保存文本块和嵌入向量
        self.chunks = chunks
        self.chunk_embeddings = embeddings
        
        logger.info(f"✓ 索引构建完成，共 {self.index.ntotal} 个向量")
    
    def search(
        self,
        query: str,
        top_k: int = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        搜索相关文本块
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            return_scores: 是否返回相似度分数
            
        Returns:
            搜索结果列表
        """
        if self.index is None or not self.chunks:
            raise ValueError("索引尚未构建，请先调用build_index()或load_index()")
        
        top_k = top_k or RAG_TOP_K_RESULTS
        top_k = min(top_k, len(self.chunks))  # 不能超过总数
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 搜索
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # 构建结果
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            result = {
                'rank': i + 1,
                'chunk': self.chunks[idx],
                'text': self.chunks[idx]['text'],
                'source': self.chunks[idx]['source_file'],
            }
            
            if return_scores:
                result['score'] = float(score)
            
            results.append(result)
        
        return results
    
    def save_index(self, path: Optional[str] = None):
        """
        保存索引到磁盘
        
        Args:
            path: 保存路径（不包含扩展名）
        """
        if self.index is None:
            raise ValueError("索引尚未构建")
        
        path = path or self.index_path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, str(path) + ".index")
        
        # 保存文本块和元数据
        with open(str(path) + ".chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # 保存嵌入向量
        with open(str(path) + ".embeddings.pkl", 'wb') as f:
            pickle.dump(self.chunk_embeddings, f)
        
        # 保存配置
        config = {
            'embedding_model': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.chunks)
        }
        with open(str(path) + ".config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ 索引已保存到: {path}")
    
    def load_index(self, path: Optional[str] = None):
        """
        从磁盘加载索引
        
        Args:
            path: 加载路径（不包含扩展名）
        """
        path = path or self.index_path
        path = Path(path)
        
        if not (path.parent / (path.name + ".index")).exists():
            raise FileNotFoundError(f"索引文件不存在: {path}.index")
        
        # 加载配置
        with open(str(path) + ".config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 验证配置
        # 检查模型是否一致（考虑路径和模型名称两种情况）
        config_model = config['embedding_model']
        current_model = self.embedding_model_name
        
        # 如果配置中的模型名称是 Hugging Face 格式，当前是本地路径，检查是否是同一个模型
        # 通过检查嵌入维度是否一致来判断
        models_match = False
        if config_model != current_model:
            # 检查嵌入维度是否一致（如果维度一致，很可能是同一个模型）
            if config.get('embedding_dim') == self.embedding_dim:
                # 进一步检查：如果配置中的模型名称包含当前路径的模型名称，或者反之
                if (config_model in current_model or 
                    current_model in config_model or
                    Path(config_model).name in current_model or
                    Path(current_model).name in config_model):
                    models_match = True
                    logger.info(
                        f"索引使用的嵌入模型 ({config_model}) "
                        f"与当前模型 ({current_model}) 路径不同，但嵌入维度一致，视为兼容"
                    )
                else:
                    logger.warning(
                        f"索引使用的嵌入模型 ({config_model}) "
                        f"与当前模型 ({current_model}) 不一致，但嵌入维度相同 ({self.embedding_dim})，"
                        f"可以继续使用，但建议使用相同模型重新构建索引以获得最佳效果"
                    )
            else:
                logger.error(
                    f"索引使用的嵌入模型 ({config_model}, 维度: {config.get('embedding_dim')}) "
                    f"与当前模型 ({current_model}, 维度: {self.embedding_dim}) 不一致且维度不同！"
                    f"这可能导致检索结果不准确，请使用相同模型重新构建索引。"
                )
        
        # 加载FAISS索引
        self.index = faiss.read_index(str(path) + ".index")
        
        # 加载文本块
        with open(str(path) + ".chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        # 加载嵌入向量
        with open(str(path) + ".embeddings.pkl", 'rb') as f:
            self.chunk_embeddings = pickle.load(f)
        
        logger.info(f"✓ 索引已加载: {config['num_chunks']} 个文本块")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        向现有索引添加新的文本块
        
        Args:
            chunks: 新的文本块列表
        """
        if not chunks:
            return
        
        logger.info(f"正在添加 {len(chunks)} 个新文本块...")
        
        # 提取文本
        texts = [chunk['text'] for chunk in chunks]
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # 添加到索引
        self.index.add(embeddings.astype('float32'))
        
        # 更新文本块和嵌入向量
        self.chunks.extend(chunks)
        if self.chunk_embeddings is None:
            self.chunk_embeddings = embeddings
        else:
            self.chunk_embeddings = np.vstack([self.chunk_embeddings, embeddings])
        
        logger.info(f"✓ 已添加 {len(chunks)} 个文本块，索引总数: {self.index.ntotal}")

