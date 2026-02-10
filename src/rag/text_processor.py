# -*- coding: utf-8 -*-
"""
文本处理模块
负责文本切分、清理和预处理
"""

import re
from typing import List, Dict, Any
import logging

from .rag_config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """文本处理器"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        初始化文本处理器
        
        Args:
            chunk_size: 文本块大小（字符数）
            chunk_overlap: 文本块重叠大小
        """
        self.chunk_size = chunk_size or CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    
    def process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理文档，将其切分为文本块
        
        Args:
            document: 文档字典（来自DocumentLoader）
            
        Returns:
            文本块列表
        """
        content = document['content']
        
        # 清理文本
        cleaned_content = self.clean_text(content)
        
        # 切分文本
        chunks = self.split_text(cleaned_content)
        
        # 为每个chunk添加元数据
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk = {
                'text': chunk_text,
                'chunk_id': i,
                'source_file': document['file_name'],
                'source_path': document['file_path'],
                'file_type': document['file_type'],
                'metadata': document['metadata']
            }
            processed_chunks.append(chunk)
        
        logger.info(f"✓ 文档 {document['file_name']} 切分为 {len(processed_chunks)} 个文本块")
        return processed_chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量处理文档
        
        Args:
            documents: 文档列表
            
        Returns:
            所有文本块的列表
        """
        all_chunks = []
        
        for document in documents:
            chunks = self.process_document(document)
            all_chunks.extend(chunks)
        
        logger.info(f"✓ 总共处理 {len(documents)} 个文档，生成 {len(all_chunks)} 个文本块")
        return all_chunks
    
    def clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 去除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 去除特殊字符（保留中英文、数字、常用标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：、,.!?;:\-\(\)\[\]（）【】]', '', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def split_text(self, text: str) -> List[str]:
        """
        切分文本为固定大小的块（带重叠）
        
        Args:
            text: 待切分的文本
            
        Returns:
            文本块列表
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # 确定结束位置
            end = start + self.chunk_size
            
            # 如果不是最后一块，尝试在句子边界处切分
            if end < text_length:
                # 在附近寻找句子结束符
                search_start = max(start, end - 100)
                search_end = min(text_length, end + 100)
                search_text = text[search_start:search_end]
                
                # 寻找句子结束符
                sentence_ends = [m.end() for m in re.finditer(r'[。！？\n]', search_text)]
                
                if sentence_ends:
                    # 找到最接近chunk_size的句子结束位置
                    target_pos = end - search_start
                    closest_end = min(sentence_ends, key=lambda x: abs(x - target_pos))
                    end = search_start + closest_end
            
            # 提取文本块
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # 移动到下一个位置（考虑重叠）
            start = end - self.chunk_overlap
            
            # 避免无限循环
            if start <= end - self.chunk_size:
                start = end
        
        return chunks

