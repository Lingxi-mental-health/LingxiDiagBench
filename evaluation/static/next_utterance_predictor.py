"""
医生提问下一句预测模块

评估医生提问预测的质量，计算BLEU、RougeL、BertScore等指标
支持使用大模型（OpenRouter/vLLM）生成预测
"""

import json
import os
import re
import time
import threading
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import math

import numpy as np

# 尝试导入可选依赖
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("警告: transformers未安装，BertScore将使用简化版本")

try:
    from .config import OUTPUT_DIR, RANDOM_SEED, LLM_CONFIG
    from .data_utils import load_and_process_data
except ImportError:
    from config import OUTPUT_DIR, RANDOM_SEED, LLM_CONFIG
    from data_utils import load_and_process_data

# 导入LLM客户端
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from llm_client import create_llm_client
except ImportError:
    # 如果在static目录下运行，尝试从上级目录导入
    try:
        from evaluation.llm_client import create_llm_client
    except ImportError:
        create_llm_client = None
        print("警告: llm_client未找到，LLM预测功能将不可用")


def tokenize_chinese(text: str) -> List[str]:
    """
    中文分词（按字符分割）
    
    Args:
        text: 输入文本
        
    Returns:
        分词后的字符列表
    """
    # 去除空格和特殊字符，按字符分割
    text = re.sub(r'\s+', '', text)
    return list(text)


def tokenize_with_jieba(text: str) -> List[str]:
    """
    使用jieba进行中文分词
    
    Args:
        text: 输入文本
        
    Returns:
        分词后的词列表
    """
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        # 如果没有jieba，使用字符级分词
        return tokenize_chinese(text)


class NextUtteranceEvaluator:
    """
    医生提问下一句预测评估器
    
    评估预测的医生提问与真实提问之间的相似度
    """
    
    def __init__(
        self,
        use_char_tokenize: bool = True,
        use_bert_score: bool = True,
        bert_model: str = "bert-base-chinese"
    ):
        """
        初始化评估器
        
        Args:
            use_char_tokenize: 是否使用字符级分词
            use_bert_score: 是否计算BertScore
            bert_model: BertScore使用的模型
        """
        self.use_char_tokenize = use_char_tokenize
        self.use_bert_score = use_bert_score and BERT_AVAILABLE
        self.bert_model = bert_model
        
        # 初始化BERT模型（如果需要）
        if self.use_bert_score:
            self._init_bert_model()
    
    def _init_bert_model(self):
        """初始化BERT模型用于计算语义相似度"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
            self.model = AutoModel.from_pretrained(self.bert_model)
            self.model.eval()
            
            # 如果有GPU，使用GPU
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            print(f"BertScore模型已加载: {self.bert_model}, 设备: {self.device}")
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            self.use_bert_score = False
    
    def _tokenize(self, text: str) -> List[str]:
        """分词"""
        if self.use_char_tokenize:
            return tokenize_chinese(text)
        else:
            return tokenize_with_jieba(text)
    
    def calculate_bleu(
        self,
        prediction: str,
        reference: str,
        n: int = 4
    ) -> Dict[str, float]:
        """
        计算BLEU分数
        
        Args:
            prediction: 预测文本
            reference: 参考文本
            n: 最大n-gram
            
        Returns:
            包含BLEU-1到BLEU-n和BLEU的字典
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) == 0:
            return {f'bleu-{i}': 0.0 for i in range(1, n+1)} | {'bleu': 0.0}
        
        # 计算各阶n-gram精度
        precisions = []
        bleu_scores = {}
        
        for i in range(1, n + 1):
            if len(pred_tokens) < i or len(ref_tokens) < i:
                precision = 0.0
            else:
                # 获取n-gram
                pred_ngrams = Counter(
                    tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)
                )
                ref_ngrams = Counter(
                    tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)
                )
                
                # 计算重叠
                overlap = sum((pred_ngrams & ref_ngrams).values())
                total = sum(pred_ngrams.values())
                
                precision = overlap / total if total > 0 else 0.0
            
            precisions.append(precision)
            bleu_scores[f'bleu-{i}'] = precision
        
        # 计算BLEU（几何平均 + 短句惩罚）
        if all(p == 0 for p in precisions):
            bleu_scores['bleu'] = 0.0
        else:
            # 避免log(0)
            log_precisions = [
                math.log(p) if p > 0 else -float('inf')
                for p in precisions
            ]
            
            if all(lp == -float('inf') for lp in log_precisions):
                geo_mean = 0.0
            else:
                # 只计算非零精度的几何平均
                valid_log_prec = [lp for lp in log_precisions if lp != -float('inf')]
                if valid_log_prec:
                    geo_mean = math.exp(sum(valid_log_prec) / len(log_precisions))
                else:
                    geo_mean = 0.0
            
            # 短句惩罚
            if len(pred_tokens) >= len(ref_tokens):
                bp = 1.0
            else:
                bp = math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
            
            bleu_scores['bleu'] = bp * geo_mean
        
        return bleu_scores
    
    def calculate_rouge_l(
        self,
        prediction: str,
        reference: str
    ) -> Dict[str, float]:
        """
        计算Rouge-L分数
        
        Args:
            prediction: 预测文本
            reference: 参考文本
            
        Returns:
            包含precision, recall, f1的字典
        """
        pred_tokens = self._tokenize(prediction)
        ref_tokens = self._tokenize(reference)
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return {'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}
        
        # 计算LCS长度（动态规划）
        m, n = len(ref_tokens), len(pred_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_tokens[i-1] == pred_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(ref_tokens)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return {
            'rouge_l_precision': precision,
            'rouge_l_recall': recall,
            'rouge_l_f1': f1
        }
    
    def calculate_bert_score(
        self,
        predictions: List[str],
        references: List[str],
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        计算BertScore（基于BERT的语义相似度）
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            batch_size: 批处理大小
            
        Returns:
            包含precision, recall, f1的字典
        """
        if not self.use_bert_score:
            return {
                'bert_score_precision': None,
                'bert_score_recall': None,
                'bert_score_f1': None,
                'note': 'BERT模型未加载'
            }
        
        def get_embeddings(texts: List[str]) -> torch.Tensor:
            """获取文本的BERT嵌入"""
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用[CLS]token的嵌入
                    embeddings = outputs.last_hidden_state[:, 0, :]
                    all_embeddings.append(embeddings.cpu())
            
            return torch.cat(all_embeddings, dim=0)
        
        try:
            pred_embeddings = get_embeddings(predictions)
            ref_embeddings = get_embeddings(references)
            
            # 计算余弦相似度
            pred_norm = pred_embeddings / pred_embeddings.norm(dim=1, keepdim=True)
            ref_norm = ref_embeddings / ref_embeddings.norm(dim=1, keepdim=True)
            
            similarities = (pred_norm * ref_norm).sum(dim=1)
            
            avg_similarity = similarities.mean().item()
            
            return {
                'bert_score_precision': avg_similarity,
                'bert_score_recall': avg_similarity,
                'bert_score_f1': avg_similarity
            }
        except Exception as e:
            print(f"计算BertScore失败: {e}")
            return {
                'bert_score_precision': None,
                'bert_score_recall': None,
                'bert_score_f1': None,
                'note': str(e)
            }
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, Any]:
        """
        评估预测结果
        
        Args:
            predictions: 预测文本列表
            references: 参考文本列表
            
        Returns:
            评估指标字典
        """
        assert len(predictions) == len(references), "预测和参考数量必须相同"
        
        print(f"正在评估 {len(predictions)} 个样本...")
        start_time = time.time()
        
        # 计算每个样本的BLEU和Rouge-L
        all_bleu = []
        all_rouge_l = []
        
        for pred, ref in zip(predictions, references):
            bleu_scores = self.calculate_bleu(pred, ref)
            rouge_l_scores = self.calculate_rouge_l(pred, ref)
            
            all_bleu.append(bleu_scores)
            all_rouge_l.append(rouge_l_scores)
        
        # 聚合BLEU分数
        bleu_metrics = {}
        for key in all_bleu[0].keys():
            values = [b[key] for b in all_bleu]
            bleu_metrics[key] = np.mean(values)
        
        # 聚合Rouge-L分数
        rouge_l_metrics = {}
        for key in all_rouge_l[0].keys():
            values = [r[key] for r in all_rouge_l]
            rouge_l_metrics[key] = np.mean(values)
        
        # 计算BertScore
        bert_score_metrics = self.calculate_bert_score(predictions, references)
        
        elapsed = time.time() - start_time
        
        result = {
            'total_samples': len(predictions),
            'evaluation_time': elapsed,
            **bleu_metrics,
            **rouge_l_metrics,
            **bert_score_metrics
        }
        
        return result


def extract_doctor_utterances(
    data: List[Dict],
    eval_interval: int = 1
) -> List[Tuple[str, str]]:
    """
    从数据中提取医生提问对（上文，医生下一句）
    
    只采样医生的对话作为预测目标，确保每次采样的都是医生的回复。
    
    Args:
        data: 数据列表
        eval_interval: 采样间隔，每隔多少个医生轮次采样一次（默认1表示每个医生轮次都采样）
        
    Returns:
        (上文, 医生下一句) 的列表
    """
    pairs = []
    
    for item in data:
        conversation = item.get('cleaned_text', '')
        if not conversation:
            continue
        
        # 分割对话轮次
        turns = conversation.split('\n')
        turns = [t.strip() for t in turns if t.strip()]
        
        if len(turns) < 2:
            continue
        
        # 找出所有医生的对话轮次索引
        doctor_indices = []
        for i, turn in enumerate(turns):
            if turn.startswith('医生：') or turn.startswith('医生:'):
                doctor_indices.append(i)
        
        if not doctor_indices:
            continue
        
        # 跳过第一个医生轮次（索引0通常是医生开场白，没有足够的上下文）
        # 从第二个医生轮次开始采样
        valid_doctor_indices = [idx for idx in doctor_indices if idx > 0]
        
        if not valid_doctor_indices:
            continue
        
        # 根据 eval_interval 从医生轮次中采样
        if eval_interval <= 1:
            # 每个医生轮次都采样
            sampled_doctor_indices = valid_doctor_indices
        else:
            # 间隔采样：从第 eval_interval 个医生轮次开始
            sampled_doctor_indices = valid_doctor_indices[eval_interval - 1::eval_interval]
            # 如果没有采样到任何轮次，至少采样最后一个医生轮次
            if not sampled_doctor_indices and valid_doctor_indices:
                sampled_doctor_indices = [valid_doctor_indices[-1]]
        
        # 提取采样轮次的对话对
        for i in sampled_doctor_indices:
            context = '\n'.join(turns[:i])
            next_turn = turns[i]
            
            # 去掉医生前缀，保持与预测结果格式一致
            if next_turn.startswith('医生：'):
                next_turn = next_turn[3:].strip()
            elif next_turn.startswith('医生:'):
                next_turn = next_turn[3:].strip()
            
            pairs.append((context, next_turn))
    
    return pairs


# ============================================================
# LLM 预测器类
# ============================================================

class NextUtterancePredictor:
    """
    使用大模型预测医生下一句的预测器
    
    支持 OpenRouter 和 vLLM 部署的模型
    """
    
    # 系统提示词
    SYSTEM_PROMPT = """你是一位经验丰富的精神科医生。你正在进行一次门诊问诊对话。
根据当前的对话历史，你需要作为医生继续问诊，提出下一个问题或给出下一句回复。

注意：
1. 你的回复应该像真实的精神科医生一样，简洁、专业、有针对性
2. 根据患者之前的回答，提出合适的问诊问题
3. 只输出医生的下一句话，不要有任何其他内容
4. 不要使用任何标签或前缀（如"医生："），直接输出对话内容"""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = 8096,
        api_key: str = None
    ):
        """
        初始化预测器
        
        Args:
            model_name: 模型名称（支持 vLLM 和 OpenRouter 格式）
            temperature: 生成温度
            max_tokens: 最大生成 token 数
            api_key: API 密钥（仅用于 OpenRouter）
        """
        if create_llm_client is None:
            raise RuntimeError("LLM 客户端未安装，无法使用预测功能")
        
        self.model_name = model_name or LLM_CONFIG.get('model_name', 'qwen3-32b:9041')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        # 初始化 LLM 客户端
        self.client = create_llm_client(self.model_name, api_key=api_key)
        
        print(f"[NextUtterancePredictor] 已初始化，模型: {self.model_name}")
    
    def _build_messages(self, context: str) -> List[Dict[str, str]]:
        """
        构建多轮对话格式的消息列表
        
        Args:
            context: 对话上下文
            
        Returns:
            消息列表
        """
        # 解析对话历史，构建多轮对话格式
        lines = context.strip().split('\n')
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        
        # 尝试解析对话历史
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否有明确的角色标记
            if line.startswith("医生：") or line.startswith("医生:"):
                messages.append({
                    "role": "assistant",
                    "content": line[3:].strip()
                })
            elif line.startswith("患者：") or line.startswith("患者:"):
                messages.append({
                    "role": "user",
                    "content": line[3:].strip()
                })
            elif line.startswith("未知发言人：") or line.startswith("未知发言人:"):
                # 交替处理：假设第一个是医生，第二个是患者
                content = line[6:].strip()
                # 根据当前 messages 的最后一条消息判断角色
                if not messages or messages[-1]["role"] == "user" or messages[-1]["role"] == "system":
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "user", "content": content})
            else:
                # 没有明确标记的内容，追加到最后一条消息
                if messages and messages[-1]["role"] != "system":
                    messages[-1]["content"] += " " + line
                else:
                    # 作为用户消息（患者）
                    messages.append({"role": "user", "content": line})
        
        # 确保最后一条是用户消息（患者），这样模型会生成医生回复
        if messages and messages[-1]["role"] == "assistant":
            # 如果最后是医生，添加一个空的患者响应提示
            messages.append({
                "role": "user",
                "content": "（患者等待医生继续问诊）"
            })
        
        return messages
    
    def predict_single(self, context: str) -> Tuple[str, Optional[str]]:
        """
        预测单个样本的医生下一句
        
        Args:
            context: 对话上下文
            
        Returns:
            Tuple[prediction, error]: 预测结果和可能的错误信息
        """
        messages = self._build_messages(context)
        
        try:
            response, tokens = self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                use_json_format=False
            )
            
            # 清理响应：去除可能的 <think> 标签和前缀
            prediction = response.strip()
            
            # 去除 <think>...</think> 标签
            prediction = re.sub(r'<think>.*?</think>', '', prediction, flags=re.DOTALL).strip()
            
            # 去除可能的角色前缀
            if prediction.startswith("医生：") or prediction.startswith("医生:"):
                prediction = prediction[3:].strip()
            
            return prediction, None
            
        except Exception as e:
            return "", str(e)
    
    def predict_batch(
        self,
        contexts: List[str],
        max_workers: int = 16,
        show_progress: bool = True
    ) -> List[str]:
        """
        批量预测医生下一句
        
        Args:
            contexts: 对话上下文列表
            max_workers: 并行工作线程数
            show_progress: 是否显示进度
            
        Returns:
            预测结果列表
        """
        predictions = [None] * len(contexts)
        errors = []
        
        print(f"\n[NextUtterancePredictor] 开始批量预测...")
        print(f"  模型: {self.model_name}")
        print(f"  样本数: {len(contexts)}")
        print(f"  并行线程数: {max_workers}")
        
        start_time = time.time()
        completed = 0
        lock = threading.Lock()
        
        def predict_task(idx: int, context: str) -> Tuple[int, str, Optional[str]]:
            prediction, error = self.predict_single(context)
            return idx, prediction, error
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(predict_task, idx, ctx): idx
                for idx, ctx in enumerate(contexts)
            }
            
            for future in as_completed(futures):
                try:
                    idx, prediction, error = future.result()
                    predictions[idx] = prediction
                    
                    if error:
                        errors.append((idx, error))
                    
                    with lock:
                        completed += 1
                        if show_progress and completed % 10 == 0:
                            elapsed = time.time() - start_time
                            eta = (elapsed / completed) * (len(contexts) - completed)
                            print(f"  进度: {completed}/{len(contexts)} "
                                  f"({100*completed/len(contexts):.1f}%), "
                                  f"已用时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s")
                
                except Exception as e:
                    idx = futures[future]
                    predictions[idx] = ""
                    errors.append((idx, str(e)))
        
        elapsed = time.time() - start_time
        print(f"[NextUtterancePredictor] 预测完成，总耗时: {elapsed:.2f}秒")
        
        if errors:
            print(f"  失败样本数: {len(errors)}")
        
        # 将 None 替换为空字符串
        predictions = [p if p is not None else "" for p in predictions]
        
        return predictions


def generate_predictions_with_llm(
    test_file: str,
    model_name: str = None,
    eval_interval: int = 5,
    max_workers: int = 16,
    output_file: str = None,
    output_dir: str = None,
    api_key: str = None,
    limit: int = 5000,
    random_seed: int = None
) -> Tuple[List[str], List[str], str]:
    """
    使用 LLM 生成医生下一句的预测
    
    Args:
        test_file: 测试数据文件路径
        model_name: 模型名称
        eval_interval: 采样间隔
        max_workers: 并行工作线程数
        output_file: 输出文件路径（如果为 None，自动生成）
        output_dir: 输出目录（如果为 None，使用默认 OUTPUT_DIR）
        api_key: API 密钥（仅用于 OpenRouter）
        limit: 最大样本数限制（默认 5000）
        random_seed: 随机种子（默认使用 config 中的 RANDOM_SEED）
        
    Returns:
        Tuple[predictions, references, output_file]: 预测列表、参考列表、输出文件路径
    """
    import random
    
    print(f"\n{'='*60}")
    print("LLM 生成医生下一句预测")
    print("="*60)
    
    # 加载数据
    print("正在加载数据...")
    test_data = load_and_process_data(test_file)
    
    # 提取对话对
    pairs = extract_doctor_utterances(test_data, eval_interval=eval_interval)
    print(f"采样间隔: 每 {eval_interval} 轮采样一次")
    print(f"提取到 {len(pairs)} 个对话对")
    
    # 随机采样限制样本数
    if limit and len(pairs) > limit:
        seed = random_seed if random_seed is not None else RANDOM_SEED
        random.seed(seed)
        pairs = random.sample(pairs, limit)
        print(f"随机采样 {limit} 个样本 (seed={seed})")
    
    contexts = [p[0] for p in pairs]
    references = [p[1] for p in pairs]
    
    # 创建预测器并生成预测
    predictor = NextUtterancePredictor(
        model_name=model_name,
        api_key=api_key
    )
    
    predictions = predictor.predict_batch(
        contexts,
        max_workers=max_workers,
        show_progress=True
    )
    
    # 保存预测结果
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', model_name or 'default')
        # 使用指定的 output_dir，如果没有则使用默认的 OUTPUT_DIR
        save_dir = output_dir or OUTPUT_DIR
        os.makedirs(save_dir, exist_ok=True)
        output_file = os.path.join(
            save_dir,
            f"next_utterance_predictions_{model_safe_name}_{timestamp}.json"
        )
    
    # 构建输出数据
    seed = random_seed if random_seed is not None else RANDOM_SEED
    output_data = {
        "metadata": {
            "model_name": model_name,
            "test_file": test_file,
            "eval_interval": eval_interval,
            "limit": limit,
            "random_seed": seed,
            "total_samples": len(predictions),
            "timestamp": datetime.now().isoformat()
        },
        "samples": [
            {
                "index": idx,
                "context": contexts[idx],
                "reference": references[idx],
                "prediction": predictions[idx]
            }
            for idx in range(len(predictions))
        ]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n预测结果已保存到: {output_file}")
    
    return predictions, references, output_file


def load_predictions_from_file(predictions_file: str) -> Tuple[List[str], List[str]]:
    """
    从文件加载预测结果
    
    Args:
        predictions_file: 预测结果文件路径
        
    Returns:
        Tuple[predictions, references]: 预测列表和参考列表
    """
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data.get("samples", [])
    predictions = [s.get("prediction", "") for s in samples]
    references = [s.get("reference", "") for s in samples]
    
    print(f"从文件加载了 {len(predictions)} 个预测结果")
    
    return predictions, references


def format_generation_metrics(metrics: Dict[str, Any], task_name: str = "") -> str:
    """格式化生成任务指标用于打印"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append(f"{task_name} 评估结果")
    lines.append("="*80)
    
    lines.append(f"\n总样本数: {metrics['total_samples']}")
    lines.append(f"评估耗时: {metrics.get('evaluation_time', 0):.2f}秒")
    
    lines.append("\n" + "-"*80)
    lines.append("BLEU分数:")
    lines.append("-"*80)
    for i in range(1, 5):
        key = f'bleu-{i}'
        if key in metrics:
            lines.append(f"  BLEU-{i}: {metrics[key]:.4f} ({metrics[key]*100:.2f}%)")
    if 'bleu' in metrics:
        lines.append(f"  BLEU:   {metrics['bleu']:.4f} ({metrics['bleu']*100:.2f}%)")
    
    lines.append("\n" + "-"*80)
    lines.append("Rouge-L分数:")
    lines.append("-"*80)
    lines.append(f"  Precision: {metrics.get('rouge_l_precision', 0):.4f} "
                f"({metrics.get('rouge_l_precision', 0)*100:.2f}%)")
    lines.append(f"  Recall:    {metrics.get('rouge_l_recall', 0):.4f} "
                f"({metrics.get('rouge_l_recall', 0)*100:.2f}%)")
    lines.append(f"  F1:        {metrics.get('rouge_l_f1', 0):.4f} "
                f"({metrics.get('rouge_l_f1', 0)*100:.2f}%)")
    
    if metrics.get('bert_score_f1') is not None:
        lines.append("\n" + "-"*80)
        lines.append("BertScore (语义相似度):")
        lines.append("-"*80)
        lines.append(f"  Precision: {metrics.get('bert_score_precision', 0):.4f} "
                    f"({metrics.get('bert_score_precision', 0)*100:.2f}%)")
        lines.append(f"  Recall:    {metrics.get('bert_score_recall', 0):.4f} "
                    f"({metrics.get('bert_score_recall', 0)*100:.2f}%)")
        lines.append(f"  F1:        {metrics.get('bert_score_f1', 0):.4f} "
                    f"({metrics.get('bert_score_f1', 0)*100:.2f}%)")
    elif 'note' in metrics:
        lines.append(f"\nBertScore: {metrics.get('note', '未计算')}")
    
    lines.append("="*80)
    
    return '\n'.join(lines)


def run_next_utterance_benchmark(
    test_file: str,
    predictions: List[str] = None,
    predictions_file: str = None,
    use_bert_score: bool = True,
    max_samples: int = None,
    eval_interval: int = 5,
    model_name: str = None,
    max_workers: int = 16,
    api_key: str = None,
    save_predictions: bool = True,
    limit: int = 5000,
    random_seed: int = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    运行医生提问下一句预测benchmark
    
    支持三种模式：
    1. 提供 predictions 列表：直接使用提供的预测进行评估
    2. 提供 predictions_file：从文件加载预测进行评估
    3. 提供 model_name：使用 LLM 生成预测后评估
    
    Args:
        test_file: 测试数据文件路径
        predictions: 预测结果列表（优先级最高）
        predictions_file: 预测结果文件路径（优先级次之）
        use_bert_score: 是否计算BertScore
        max_samples: 最大样本数（用于快速测试，已废弃，请使用 limit）
        eval_interval: 采样间隔，每隔多少轮采样一次（默认5）
        model_name: LLM模型名称（如果提供，将使用LLM生成预测）
        max_workers: 并行工作线程数（用于LLM预测）
        api_key: API密钥（仅用于OpenRouter）
        save_predictions: 是否保存预测结果到文件
        limit: 最大样本数限制，固定 seed 随机采样（默认 5000）
        random_seed: 随机种子（默认使用 config 中的 RANDOM_SEED）
        output_dir: 输出目录（如果为 None，使用默认 OUTPUT_DIR）
        
    Returns:
        评估结果字典
    """
    import random
    print(f"\n{'='*60}")
    print("医生提问下一句预测评估")
    print("="*60)
    
    references = None
    output_file = None
    
    # 模式1：直接使用提供的预测列表
    if predictions is not None:
        print("模式: 使用提供的预测列表")
        # 加载数据获取参考答案
        print("正在加载数据...")
        test_data = load_and_process_data(test_file)
        pairs = extract_doctor_utterances(test_data, eval_interval=eval_interval)
        
        if max_samples and len(pairs) > max_samples:
            pairs = pairs[:max_samples]
        
        references = [p[1] for p in pairs]
        
        # 确保预测数量与参考数量一致
        if len(predictions) != len(references):
            print(f"警告: 预测数量 ({len(predictions)}) 与参考数量 ({len(references)}) 不一致")
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
    
    # 模式2：从文件加载预测
    elif predictions_file is not None:
        print(f"模式: 从文件加载预测")
        print(f"  文件: {predictions_file}")
        predictions, references = load_predictions_from_file(predictions_file)
        
        if max_samples and len(predictions) > max_samples:
            predictions = predictions[:max_samples]
            references = references[:max_samples]
    
    # 模式3：使用 LLM 生成预测
    elif model_name is not None:
        print(f"模式: 使用 LLM 生成预测")
        print(f"  模型: {model_name}")
        
        # 生成预测（传递 limit、random_seed 和 output_dir 参数）
        predictions, references, output_file = generate_predictions_with_llm(
            test_file=test_file,
            model_name=model_name,
            eval_interval=eval_interval,
            max_workers=max_workers,
            output_dir=output_dir,
            api_key=api_key,
            limit=limit,
            random_seed=random_seed
        )
    
    # 模式4：自测试模式（使用参考答案作为预测）
    else:
        print("模式: 自测试（使用参考文本作为预测）")
        print("警告: 未提供预测结果或模型，使用参考文本进行自测试")
        
        # 加载数据
        print("正在加载数据...")
        test_data = load_and_process_data(test_file)
        pairs = extract_doctor_utterances(test_data, eval_interval=eval_interval)
        
        print(f"采样间隔: 每 {eval_interval} 轮采样一次")
        print(f"提取到 {len(pairs)} 个医生提问对")
        
        # 随机采样限制样本数
        if limit and len(pairs) > limit:
            seed = random_seed if random_seed is not None else RANDOM_SEED
            random.seed(seed)
            pairs = random.sample(pairs, limit)
            print(f"随机采样 {limit} 个样本 (seed={seed})")
        
        references = [p[1] for p in pairs]
        predictions = references  # 自测试：使用参考答案作为预测
    
    print(f"\n评估样本数: {len(predictions)}")
    
    # 创建评估器并评估
    evaluator = NextUtteranceEvaluator(use_bert_score=use_bert_score)
    metrics = evaluator.evaluate(predictions, references)
    
    # 打印结果
    print(format_generation_metrics(metrics, "医生提问下一句预测"))
    
    result = {
        'task': 'next_utterance_prediction',
        'eval_interval': eval_interval,
        'model_name': model_name,
        'metrics': metrics
    }
    
    if output_file:
        result['predictions_file'] = output_file
    
    return result


if __name__ == "__main__":
    from .config import TEST_DATA_100_FILE
    
    # 运行benchmark
    result = run_next_utterance_benchmark(
        TEST_DATA_100_FILE,
        use_bert_score=BERT_AVAILABLE,
        max_samples=100
    )
    
    # 保存结果
    output_file = os.path.join(OUTPUT_DIR, "next_utterance_benchmark_results.json")
    
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    result_serializable = convert_to_serializable(result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_serializable, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

