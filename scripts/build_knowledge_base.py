#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建RAG知识库索引脚本

从PDF文档中读取内容，处理成文本块，生成向量索引。

使用示例:
    python scripts/build_knowledge_base.py \
        --pdf ../../project/Lingxi_annotation_0111/knowledge_base/doc/疾病诊断指南.pdf \
        --output ../../project/Lingxi_annotation_0111/knowledge_base/indices/faiss_index

环境变量配置（可选）:
    # 使用本地SentenceTransformer模型（默认）
    export EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    
    # 使用本地Qwen3 Embedding模型
    export LOCAL_EMBEDDING_MODEL_PATH=/path/to/qwen3-embedding
    
    # 使用DeepInfra API
    export USE_DEEPINFRA_EMBEDDING=true
    export DEEPINFRA_API_KEY=your_api_key
    export DEEPINFRA_EMBEDDING_MODEL=Qwen/Qwen2.5-7B-Instruct
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pdf(pdf_path: str) -> str:
    """
    加载PDF文件并提取文本
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        提取的文本内容
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("需要安装PyMuPDF库: pip install PyMuPDF")
        raise
    
    logger.info(f"正在读取PDF文件: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    text_parts = []
    total_pages = len(doc)  # 在关闭文档前保存页数
    
    for page_num, page in enumerate(doc, 1):
        text = page.get_text()
        if text.strip():
            text_parts.append(f"[第{page_num}页]\n{text}")
        
        if page_num % 10 == 0:
            logger.info(f"已处理 {page_num}/{total_pages} 页")
    
    doc.close()
    
    full_text = "\n\n".join(text_parts)
    logger.info(f"✓ PDF读取完成，共 {total_pages} 页，{len(full_text)} 字符")
    
    return full_text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_file: str = "unknown"
) -> List[Dict[str, Any]]:
    """
    将文本分割成块
    
    Args:
        text: 待分割的文本
        chunk_size: 每个块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        source_file: 来源文件名
        
    Returns:
        文本块列表
    """
    logger.info(f"正在分割文本（块大小: {chunk_size}, 重叠: {chunk_overlap}）...")
    
    # 按段落分割
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # 如果当前段落加上已有内容超过chunk_size，保存当前块
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'source_file': source_file,
                'char_count': len(current_chunk.strip())
            })
            chunk_id += 1
            
            # 保留重叠部分
            if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # 保存最后一个块
    if current_chunk.strip():
        chunks.append({
            'chunk_id': chunk_id,
            'text': current_chunk.strip(),
            'source_file': source_file,
            'char_count': len(current_chunk.strip())
        })
    
    logger.info(f"✓ 文本分割完成，共 {len(chunks)} 个块")
    
    # 显示统计信息
    char_counts = [c['char_count'] for c in chunks]
    logger.info(f"   块字符数统计: 最小={min(char_counts)}, 最大={max(char_counts)}, 平均={sum(char_counts)/len(char_counts):.1f}")
    
    return chunks


def build_index(chunks: List[Dict[str, Any]], output_path: str):
    """
    构建向量索引
    
    Args:
        chunks: 文本块列表
        output_path: 索引输出路径
    """
    from src.rag.vector_store import VectorStore
    from src.rag.rag_config import RAGConfig
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("RAG配置信息:")
    logger.info(f"  Embedding模型: {RAGConfig.EMBEDDING_MODEL}")
    logger.info(f"  本地模型路径: {RAGConfig.LOCAL_EMBEDDING_MODEL_PATH}")
    logger.info(f"  使用DeepInfra: {RAGConfig.USE_DEEPINFRA_EMBEDDING}")
    if RAGConfig.USE_DEEPINFRA_EMBEDDING:
        logger.info(f"  DeepInfra模型: {RAGConfig.DEEPINFRA_EMBEDDING_MODEL}")
    logger.info(f"  块大小: {RAGConfig.CHUNK_SIZE}")
    logger.info(f"  块重叠: {RAGConfig.CHUNK_OVERLAP}")
    logger.info("=" * 60)
    
    # 初始化向量存储
    logger.info("正在初始化向量存储...")
    vector_store = VectorStore()
    
    # 构建索引
    logger.info(f"正在为 {len(chunks)} 个文本块构建索引...")
    vector_store.build_index(chunks)
    
    # 保存索引
    logger.info(f"正在保存索引到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vector_store.save_index(output_path)
    
    logger.info("✓ 索引构建完成!")
    
    # 测试检索
    logger.info("\n测试检索...")
    test_query = "抑郁症的诊断标准"
    results = vector_store.search(test_query, top_k=3)
    logger.info(f"查询: '{test_query}'")
    for i, result in enumerate(results, 1):
        logger.info(f"  [{i}] 相似度: {result.get('score', 0):.4f}")
        logger.info(f"      内容: {result['text'][:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description='构建RAG知识库索引',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--pdf',
        type=str,
        default='../../project/Lingxi_annotation_0111/knowledge_base/doc/疾病诊断指南.pdf',
        help='PDF文件路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../../project/Lingxi_annotation_0111/knowledge_base/indices/faiss_index',
        help='索引输出路径（不包含扩展名）'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='文本块大小（默认: 500）'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=50,
        help='文本块重叠大小（默认: 50）'
    )
    
    parser.add_argument(
        '--use-deepinfra',
        action='store_true',
        help='使用DeepInfra API进行embedding（需要设置DEEPINFRA_API_KEY）'
    )
    
    args = parser.parse_args()
    
    # 如果指定使用DeepInfra，设置环境变量
    if args.use_deepinfra:
        os.environ['USE_DEEPINFRA_EMBEDDING'] = 'true'
        if not os.getenv('DEEPINFRA_API_KEY'):
            logger.error("使用DeepInfra需要设置DEEPINFRA_API_KEY环境变量")
            sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("开始构建RAG知识库索引")
    logger.info("=" * 60)
    
    # 1. 读取PDF
    pdf_text = load_pdf(args.pdf)
    
    # 2. 分割文本
    source_file = os.path.basename(args.pdf)
    chunks = split_text_into_chunks(
        pdf_text,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        source_file=source_file
    )
    
    # 3. 构建索引
    build_index(chunks, args.output)
    
    logger.info("=" * 60)
    logger.info("✓ 知识库索引构建完成!")
    logger.info(f"  索引文件: {args.output}.index")
    logger.info(f"  块数据文件: {args.output}.chunks.pkl")
    logger.info(f"  配置文件: {args.output}.config.json")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

