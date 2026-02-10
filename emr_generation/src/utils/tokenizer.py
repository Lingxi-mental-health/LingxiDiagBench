"""
中文分词器 - 基于 jieba
"""

import re
from typing import List, Set, Dict, Counter as TypingCounter
from collections import Counter

try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

from ..config import Config


class ChineseTokenizer:
    """中文分词器"""
    
    def __init__(self, custom_dict: List[str] = None, stopwords: Set[str] = None):
        """
        初始化分词器
        
        Args:
            custom_dict: 自定义词典列表
            stopwords: 停用词集合
        """
        if not JIEBA_AVAILABLE:
            raise ImportError("请安装 jieba 库: pip install jieba")
        
        self.stopwords = stopwords or Config.STOPWORDS
        
        # 添加医学领域自定义词
        self.medical_terms = [
            # 症状
            "入睡困难", "睡眠障碍", "情绪低落", "兴趣减退", "注意力不集中",
            "焦虑烦躁", "紧张不安", "心慌", "胸闷", "躯体不适",
            "自伤行为", "自残行为", "消极想法", "自杀观念",
            "幻听", "幻觉", "妄想", "被害妄想", "被监视感",
            "强迫行为", "强迫思维", "反复洗手", "反复检查",
            "眠浅多梦", "早醒", "日夜颠倒",
            # 时间
            "无明显诱因", "逐渐出现", "加重",
            # 病程
            "病程", "总病程", "发病", "起病",
            # 治疗
            "服药治疗", "药物治疗", "心理治疗", "住院治疗",
            # 诊断
            "抑郁发作", "焦虑障碍", "双相情感障碍", "精神分裂症",
            "抑郁状态", "焦虑抑郁状态",
            # 个人史
            "足月顺产", "发育正常", "已婚", "未婚", "离异",
            "吸烟史", "饮酒史", "无特殊嗜好",
        ]
        
        # 添加自定义词到jieba
        for term in self.medical_terms:
            jieba.add_word(term)
        
        if custom_dict:
            for word in custom_dict:
                jieba.add_word(word)
    
    def tokenize(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        分词
        
        Args:
            text: 输入文本
            remove_stopwords: 是否移除停用词
            
        Returns:
            分词结果列表
        """
        # 预处理：移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 分词
        words = jieba.lcut(text)
        
        # 过滤
        if remove_stopwords:
            words = [w for w in words if w.strip() and w not in self.stopwords]
        
        # 过滤单字符（除了特定有意义的单字）
        meaningful_single = {"痛", "哭", "怕", "累", "烦", "呕"}
        words = [w for w in words if len(w) > 1 or w in meaningful_single]
        
        return words
    
    def tokenize_with_pos(self, text: str) -> List[tuple]:
        """
        带词性标注的分词
        
        Args:
            text: 输入文本
            
        Returns:
            (词, 词性) 元组列表
        """
        return list(pseg.cut(text))
    
    def extract_keywords(
        self,
        text: str,
        top_k: int = 10,
        pos_filter: Set[str] = None
    ) -> List[tuple]:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回前K个关键词
            pos_filter: 词性过滤集合（如 {'n', 'v', 'a'} 只保留名词、动词、形容词）
            
        Returns:
            (词, 频次) 元组列表
        """
        if pos_filter:
            words_with_pos = self.tokenize_with_pos(text)
            words = [
                w for w, flag in words_with_pos 
                if flag in pos_filter and w not in self.stopwords and len(w) > 1
            ]
        else:
            words = self.tokenize(text)
        
        counter = Counter(words)
        return counter.most_common(top_k)
    
    def extract_symptom_keywords(self, text: str) -> List[str]:
        """
        提取症状相关关键词
        
        Args:
            text: 输入文本
            
        Returns:
            症状关键词列表
        """
        # 症状相关的模式
        symptom_patterns = [
            r'(入睡困难|睡眠[差障碍不好]|失眠|眠[浅差]|多梦|早醒)',
            r'(情绪[低落不好不稳定]|心情[不好差低落])',
            r'(焦虑|紧张|烦躁|不安|恐惧|害怕)',
            r'(抑郁|闷闷不乐|不愉快|不开心)',
            r'(兴趣[减退下降缺乏]|无精打采)',
            r'(自[伤残杀]|消极[想法念头])',
            r'(幻[听觉]|妄想|被[害监视跟踪])',
            r'(强迫[行为思想]|反复[洗检查])',
            r'(注意力[不集中难以集中]|记忆力[差下降])',
            r'(心[慌悸]|胸闷|气[短促])',
            r'(头[痛晕]|恶心|呕吐)',
            r'(食欲[差下降减退]|胃口[差不好])',
            r'(乏力|疲[劳惫乏]|精力[下降不足])',
        ]
        
        symptoms = []
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text)
            symptoms.extend(matches)
        
        return list(set(symptoms))
    
    def extract_time_expressions(self, text: str) -> List[str]:
        """
        提取时间表达式
        
        Args:
            text: 输入文本
            
        Returns:
            时间表达式列表
        """
        time_patterns = [
            r'\d+年前',
            r'\d+个?月前',
            r'\d+周前',
            r'\d+天前',
            r'近\d+[年月周天]',
            r'最近\d+[年月周天]',
            r'\d+年\d+月',
            r'20\d{2}年\d{1,2}月',
        ]
        
        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text)
            times.extend(matches)
        
        return list(set(times))
    
    def get_word_frequency(self, texts: List[str]) -> Counter:
        """
        统计多个文本中的词频
        
        Args:
            texts: 文本列表
            
        Returns:
            词频 Counter
        """
        word_freq = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_freq.update(words)
        return word_freq
    
    def build_keyword_mapping(
        self,
        texts: List[str],
        labels: List[str],
    ) -> Dict[str, Counter]:
        """
        构建关键词与标签的映射关系
        
        Args:
            texts: 文本列表
            labels: 对应的标签列表
            
        Returns:
            {标签: 关键词Counter} 的映射
        """
        mapping = {}
        
        for text, label in zip(texts, labels):
            if label not in mapping:
                mapping[label] = Counter()
            
            words = self.tokenize(text)
            mapping[label].update(words)
        
        return mapping
