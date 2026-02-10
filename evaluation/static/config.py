"""
Benchmark配置文件

定义所有常量、路径和配置参数
"""

import os
from typing import List, Dict

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "evaluation", "static", "LingxiDiag-16K_outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "evaluation", "static", "models")

# 数据文件
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "LingxiDiag-16K_train_data.json")
TEST_DATA_FILE = os.path.join(DATA_DIR, "LingxiDiag-16K_validation_data.json")
TEST_DATA_100_FILE = os.path.join(DATA_DIR, "LingxiDiag-16K_validation_data.json")
TEST_DATA_500_FILE = os.path.join(DATA_DIR, "LingxiDiag-16K_validation_data.json")

# ICD代码分类配置（大类）
VALID_ICD_CODES: List[str] = [
    "F20", "F31", "F32", "F39", "F41", "F42", "F43", "F45", "F51", "F98", "Z71", "Others"
]

# ICD-10 小类代码列表（用于12class_detailed分类）
# 根据category_12class.txt定义的所有细分代码
VALID_ICD_SUBCODES: List[str] = [
    # F20 精神分裂症
    "F20.0", "F20.1", "F20.2", "F20.3", "F20.4", "F20.5", "F20.6", "F20.8", "F20.9",
    # F31 双相情感障碍
    "F31.0", "F31.1", "F31.2", "F31.3", "F31.4", "F31.5", "F31.6", "F31.9",
    # F32 抑郁发作
    "F32.0", "F32.1", "F32.2", "F32.3", "F32.8", "F32.9",
    # F39 心境障碍（未特指）
    "F39",
    # F41 焦虑障碍
    "F41.0", "F41.1", "F41.2", "F41.3", "F41.9",
    # F42 强迫障碍
    "F42.0", "F42.1", "F42.2", "F42.9",
    # F43 应激相关障碍
    "F43.0", "F43.1", "F43.2", "F43.8", "F43.9",
    # F45 躯体形式障碍
    "F45.0", "F45.1", "F45.2", "F45.3", "F45.4", "F45.8", "F45.9",
    # F51 睡眠障碍
    "F51.0", "F51.1", "F51.2", "F51.3", "F51.4", "F51.5", "F51.9",
    # F98 儿童青少年行为障碍
    "F98.0", "F98.1", "F98.2", "F98.3", "F98.4", "F98.5", "F98.6", "F98.8", "F98.9",
    # Z71 咨询
    "Z71.9",
    # Others
    "Others"
]

# ICD代码到中文名称映射（大类）
ICD_CODE_NAMES: Dict[str, str] = {
    "F20": "精神分裂症",
    "F31": "双相情感障碍",
    "F32": "抑郁发作",
    "F39": "心境障碍",
    "F41": "焦虑障碍",
    "F42": "强迫性障碍",
    "F43": "应激相关障碍",
    "F45": "躯体形式障碍",
    "F51": "睡眠障碍",
    "F98": "儿童青少年行为障碍",
    "Z71": "咨询",
    "Others": "其他"
}

# ICD-10 小类代码到中文名称映射
ICD_SUBCODE_NAMES: Dict[str, str] = {
    # F20 精神分裂症
    "F20.0": "偏执型精神分裂症",
    "F20.1": "紊乱型精神分裂症",
    "F20.2": "紧张型精神分裂症",
    "F20.3": "未分化型精神分裂症",
    "F20.4": "精神分裂症残留状态",
    "F20.5": "精神分裂症后抑郁",
    "F20.6": "单纯型精神分裂症",
    "F20.8": "其他类型精神分裂症",
    "F20.9": "精神分裂症，未特指",
    # F31 双相情感障碍
    "F31.0": "双相情感障碍，躁狂期，无精神病性症状",
    "F31.1": "双相情感障碍，躁狂期，有精神病性症状",
    "F31.2": "双相情感障碍，抑郁期，无精神病性症状",
    "F31.3": "双相情感障碍，抑郁期，有精神病性症状",
    "F31.4": "双相情感障碍，混合状态",
    "F31.5": "双相情感障碍，缓解期",
    "F31.6": "双相情感障碍，其他状态",
    "F31.9": "双相情感障碍，未特指",
    # F32 抑郁发作
    "F32.0": "轻度抑郁发作",
    "F32.1": "中度抑郁发作",
    "F32.2": "重度抑郁发作，无精神病性症状",
    "F32.3": "重度抑郁发作，有精神病性症状",
    "F32.8": "其他抑郁发作",
    "F32.9": "抑郁发作，未特指",
    # F39 心境障碍
    "F39": "未特指的心境障碍",
    # F41 焦虑障碍
    "F41.0": "惊恐障碍",
    "F41.1": "广泛性焦虑障碍",
    "F41.2": "混合性焦虑与抑郁障碍",
    "F41.3": "其他混合性焦虑障碍",
    "F41.9": "焦虑障碍，未特指",
    # F42 强迫障碍
    "F42.0": "以强迫观念为主的强迫障碍",
    "F42.1": "以强迫行为为主的强迫障碍",
    "F42.2": "强迫观念与强迫行为混合",
    "F42.9": "强迫障碍，未特指",
    # F43 应激相关障碍
    "F43.0": "急性应激反应",
    "F43.1": "创伤后应激障碍",
    "F43.2": "适应障碍",
    "F43.8": "其他反应性障碍",
    "F43.9": "应激相关障碍，未特指",
    # F45 躯体形式障碍
    "F45.0": "躯体化障碍",
    "F45.1": "未分化的躯体形式障碍",
    "F45.2": "疑病障碍",
    "F45.3": "自主神经功能紊乱型",
    "F45.4": "持续性躯体疼痛障碍",
    "F45.8": "其他躯体形式障碍",
    "F45.9": "躯体形式障碍，未特指",
    # F51 睡眠障碍
    "F51.0": "非器质性失眠",
    "F51.1": "非器质性嗜睡",
    "F51.2": "非器质性睡眠-觉醒节律障碍",
    "F51.3": "梦魇障碍",
    "F51.4": "睡眠惊恐（夜惊）",
    "F51.5": "梦游症",
    "F51.9": "非器质性睡眠障碍，未特指",
    # F98 儿童青少年行为障碍
    "F98.0": "非器质性遗尿症",
    "F98.1": "非器质性遗粪症",
    "F98.2": "婴儿期或儿童期进食障碍",
    "F98.3": "异食癖",
    "F98.4": "刻板性运动障碍",
    "F98.5": "口吃",
    "F98.6": "习惯性动作障碍",
    "F98.8": "其他特指的儿童行为和情绪障碍",
    "F98.9": "未特指的儿童行为和情绪障碍",
    # Z71 咨询
    "Z71.9": "未特指的咨询",
    # Others
    "Others": "其他"
}

# 2分类标签
TWO_CLASS_LABELS: List[str] = ["Depression", "Anxiety"]

# 4分类标签
FOUR_CLASS_LABELS: List[str] = ["Depression", "Anxiety", "Mixed", "Others"]

# 12分类标签（大类，即VALID_ICD_CODES）
TWELVE_CLASS_LABELS: List[str] = VALID_ICD_CODES

# 12分类标签（小类，支持ICD-10细分代码）
TWELVE_CLASS_DETAILED_LABELS: List[str] = VALID_ICD_SUBCODES

# BERT模型配置
BERT_CONFIG = {
    "max_samples": 10000,  # 随机采样的最大切分样本数，None表示不限制
    "model_name": "bert-base-chinese",
    "max_length": 512,
    "chunk_overlap": 16,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "warmup_ratio": 0.1,
}

# RoBERTa模型配置（与BERT保持一致的训练参数）
ROBERTA_CONFIG = {
    "max_samples": 10000,  # 随机采样的最大切分样本数，None表示不限制
    "model_name": "hfl/chinese-roberta-wwm-ext",  # 中文RoBERTa模型
    "max_length": 512,
    "chunk_overlap": 16,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "warmup_ratio": 0.1,
}

# TF-IDF配置
TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
}

# LLM配置
# LLM 配置
# 支持的模型格式:
#   - vLLM简化格式: "model:port" (如 "qwen3-32b:9041")
#   - vLLM完整格式: "model@host:port" (如 "qwen3-30b@10.119.28.185:9041")
#   - OpenRouter格式: "provider/model-name" (如 "qwen/qwen3-30b-a3b-instruct-2507")
LLM_CONFIG = {
    "model_name": "Qwen3-1.7B@10.119.28.185:9047",
    "temperature": 0.6,
    "max_tokens": 4096,
}

# 随机种子
RANDOM_SEED = 42

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

