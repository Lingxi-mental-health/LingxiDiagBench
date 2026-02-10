"""
评估指标计算模块

使用sklearn实现多分类评估指标的计算
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer


def calculate_classification_metrics(
    y_true: List[Any],
    y_pred: List[Any],
    labels: List[str],
    classification_type: str = "single"
) -> Dict[str, Any]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        labels: 所有可能的标签列表
        classification_type: 分类类型 ("single" 单标签, "multi" 多标签)
        
    Returns:
        包含各种指标的字典
    """
    if classification_type == "multi":
        return calculate_multilabel_metrics(y_true, y_pred, labels)
    else:
        return calculate_singlelabel_metrics(y_true, y_pred, labels)


def calculate_singlelabel_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> Dict[str, Any]:
    """
    计算单标签分类指标（使用sklearn）
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
        labels: 所有可能的标签列表
        
    Returns:
        包含各种指标的字典
    """
    total_samples = len(y_true)
    
    # 过滤出实际出现的标签
    present_labels = [l for l in labels if l in set(y_true) or l in set(y_pred)]
    
    # 使用sklearn计算指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # Macro指标
    macro_precision = precision_score(y_true, y_pred, labels=present_labels, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=present_labels, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=present_labels, average='macro', zero_division=0)
    
    # Weighted指标
    weighted_precision = precision_score(y_true, y_pred, labels=present_labels, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, labels=present_labels, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=present_labels, average='weighted', zero_division=0)
    
    # 每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, labels=present_labels, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=present_labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=present_labels, average=None, zero_division=0)
    
    # 计算每个类别的support
    support_dict = defaultdict(int)
    for label in y_true:
        support_dict[label] += 1
    
    # 计算混淆矩阵来获取TP, FP, FN
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    
    class_metrics = {}
    for i, label in enumerate(present_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        class_metrics[label] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': support_dict[label],
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # 添加不在结果中的标签
    for label in labels:
        if label not in class_metrics:
            class_metrics[label] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': 0,
                'tp': 0,
                'fp': 0,
                'fn': 0
            }
    
    correct_count = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    
    return {
        'total_samples': total_samples,
        'accuracy': float(accuracy),
        'correct_count': correct_count,
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'class_metrics': class_metrics
    }


def calculate_multilabel_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    labels: List[str]
) -> Dict[str, Any]:
    """
    计算多标签分类指标（使用sklearn）
    
    Args:
        y_true: 真实标签列表（每个元素是标签列表）
        y_pred: 预测标签列表（每个元素是标签列表）
        labels: 所有可能的标签列表
        
    Returns:
        包含各种指标的字典
    """
    total_samples = len(y_true)
    
    # 使用MultiLabelBinarizer转换为二进制矩阵
    mlb = MultiLabelBinarizer(classes=labels)
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    
    # 计算Exact Match (Subset Accuracy)
    exact_match_count = sum(
        1 for t, p in zip(y_true, y_pred) if set(t) == set(p)
    )
    exact_match_acc = exact_match_count / total_samples if total_samples > 0 else 0
    
    # 计算Top-1 Accuracy: 第一个预测是否在真实标签中
    top1_correct = sum(
        1 for t, p in zip(y_true, y_pred) 
        if p and p[0] in set(t)
    )
    top1_acc = top1_correct / total_samples if total_samples > 0 else 0
    
    # 计算Top-3 Accuracy: 前三个预测中至少有一个在真实标签中
    top3_correct = sum(
        1 for t, p in zip(y_true, y_pred)
        if len(set(p[:3]) & set(t)) > 0
    )
    top3_acc = top3_correct / total_samples if total_samples > 0 else 0
    
    # 计算Hamming Score (标签级别的准确率)
    hamming_scores = []
    for t, p in zip(y_true, y_pred):
        t_set, p_set = set(t), set(p)
        union = t_set | p_set
        if union:
            intersection = t_set & p_set
            hamming_scores.append(len(intersection) / len(union))
    hamming_acc = np.mean(hamming_scores) if hamming_scores else 0
    
    # 使用sklearn计算Macro、Weighted、Micro指标
    # Macro指标
    macro_precision = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    
    # Weighted指标
    weighted_precision = precision_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    weighted_recall = recall_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    
    # Micro指标
    micro_precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    micro_recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    micro_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
    
    # 每个类别的指标
    precision_per_class = precision_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    recall_per_class = recall_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    
    # 使用multilabel_confusion_matrix获取每个类别的TP, FP, FN, TN
    mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
    
    class_metrics = {}
    for i, label in enumerate(labels):
        tn, fp, fn, tp = mcm[i].ravel()
        support = int(y_true_bin[:, i].sum())
        
        class_metrics[label] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': support,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # 计算Sample Accuracy (每个样本的标签预测正确率的平均值)
    # 定义：对于每个样本，计算 |预测正确的标签| / |真实标签|
    sample_acc_scores = []
    for t, p in zip(y_true, y_pred):
        t_set, p_set = set(t), set(p)
        if len(t_set) > 0:
            correct_labels = len(t_set & p_set)
            sample_acc_scores.append(correct_labels / len(t_set))
    sample_accuracy = np.mean(sample_acc_scores) if sample_acc_scores else 0
    
    # 计算Label-level Accuracy (所有样本-标签对的正确率)
    # 使用sklearn的accuracy_score在展平的二进制矩阵上计算
    label_accuracy = accuracy_score(y_true_bin.flatten(), y_pred_bin.flatten())
    
    return {
        'total_samples': total_samples,
        # 主要accuracy指标：使用exact_match作为默认accuracy（与单标签分类对齐）
        'accuracy': float(exact_match_acc),
        'exact_match_accuracy': float(exact_match_acc),
        'exact_match_count': exact_match_count,
        # 样本级别的正确率：预测标签与真实标签的交集/真实标签数
        'sample_accuracy': float(sample_accuracy),
        # 标签级别的正确率：所有标签预测的正确率
        'label_accuracy': float(label_accuracy),
        'top1_accuracy': float(top1_acc),
        'top1_correct': top1_correct,
        'top3_accuracy': float(top3_acc),
        'top3_correct': top3_correct,
        'hamming_accuracy': float(hamming_acc),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'weighted_precision': float(weighted_precision),
        'weighted_recall': float(weighted_recall),
        'weighted_f1': float(weighted_f1),
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'class_metrics': class_metrics
    }


def format_metrics_for_print(
    metrics: Dict[str, Any],
    task_name: str = "",
    classification_type: str = "single"
) -> str:
    """
    格式化指标用于打印
    
    Args:
        metrics: 指标字典
        task_name: 任务名称
        classification_type: 分类类型
        
    Returns:
        格式化的字符串
    """
    lines = []
    lines.append("\n" + "="*80)
    lines.append(f"{task_name} 评估结果")
    lines.append("="*80)
    
    lines.append(f"\n总样本数: {metrics['total_samples']}")
    
    if classification_type == "multi":
        lines.append(f"\n准确率(Accuracy): {metrics.get('accuracy', metrics.get('exact_match_accuracy', 0)):.4f} "
                    f"({metrics.get('accuracy', metrics.get('exact_match_accuracy', 0))*100:.2f}%)")
        lines.append(f"Exact Match准确率: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
        if 'sample_accuracy' in metrics:
            lines.append(f"样本级准确率(Sample Acc): {metrics['sample_accuracy']:.4f} ({metrics['sample_accuracy']*100:.2f}%)")
        if 'label_accuracy' in metrics:
            lines.append(f"标签级准确率(Label Acc): {metrics['label_accuracy']:.4f} ({metrics['label_accuracy']*100:.2f}%)")
        lines.append(f"Top-1准确率: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
        lines.append(f"Top-3准确率: {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)")
        lines.append(f"Hamming准确率: {metrics['hamming_accuracy']:.4f} ({metrics['hamming_accuracy']*100:.2f}%)")
    else:
        lines.append(f"\n准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        lines.append(f"正确预测数: {metrics['correct_count']}/{metrics['total_samples']}")
    
    lines.append("\n" + "-"*80)
    lines.append("Macro-averaged指标:")
    lines.append("-"*80)
    lines.append(f"Macro Precision: {metrics['macro_precision']:.4f} ({metrics['macro_precision']*100:.2f}%)")
    lines.append(f"Macro Recall:    {metrics['macro_recall']:.4f} ({metrics['macro_recall']*100:.2f}%)")
    lines.append(f"Macro F1:        {metrics['macro_f1']:.4f} ({metrics['macro_f1']*100:.2f}%)")
    
    lines.append("\n" + "-"*80)
    lines.append("Weighted-averaged指标:")
    lines.append("-"*80)
    lines.append(f"Weighted Precision: {metrics['weighted_precision']:.4f} ({metrics['weighted_precision']*100:.2f}%)")
    lines.append(f"Weighted Recall:    {metrics['weighted_recall']:.4f} ({metrics['weighted_recall']*100:.2f}%)")
    lines.append(f"Weighted F1:        {metrics['weighted_f1']:.4f} ({metrics['weighted_f1']*100:.2f}%)")
    
    if classification_type == "multi":
        lines.append("\n" + "-"*80)
        lines.append("Micro-averaged指标:")
        lines.append("-"*80)
        lines.append(f"Micro Precision: {metrics['micro_precision']:.4f} ({metrics['micro_precision']*100:.2f}%)")
        lines.append(f"Micro Recall:    {metrics['micro_recall']:.4f} ({metrics['micro_recall']*100:.2f}%)")
        lines.append(f"Micro F1:        {metrics['micro_f1']:.4f} ({metrics['micro_f1']*100:.2f}%)")
    
    lines.append("\n" + "-"*80)
    lines.append("各类别详细指标:")
    lines.append("-"*80)
    lines.append(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    lines.append("-"*80)
    
    for cls in sorted(metrics['class_metrics'].keys()):
        m = metrics['class_metrics'][cls]
        if m['support'] > 0:
            lines.append(f"{cls:<15} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}")
    
    lines.append("="*80)
    
    return '\n'.join(lines)


def calculate_generation_metrics(
    predictions: List[str],
    references: List[str],
    use_bert_score: bool = True
) -> Dict[str, float]:
    """
    计算生成任务的评估指标（BLEU, RougeL, BertScore）
    
    使用标准库计算，尝试使用nltk和rouge-score
    
    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        use_bert_score: 是否计算BertScore
        
    Returns:
        包含各种指标的字典
    """
    result = {}
    
    # 尝试使用nltk计算BLEU
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        smoothie = SmoothingFunction().method1
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            # 中文按字符分词
            pred_tokens = list(pred.replace(' ', ''))
            ref_tokens = list(ref.replace(' ', ''))
            
            if pred_tokens and ref_tokens:
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
                bleu_scores.append(score)
            else:
                bleu_scores.append(0.0)
        
        result['bleu'] = float(np.mean(bleu_scores))
    except ImportError:
        # 如果没有nltk，使用简单实现
        result['bleu'] = _simple_bleu(predictions, references)
        result['bleu_note'] = "使用简化版BLEU（nltk未安装）"
    
    # 尝试使用rouge-score计算Rouge-L
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        rouge_l_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge_l_scores.append(scores['rougeL'].fmeasure)
        
        result['rouge_l'] = float(np.mean(rouge_l_scores))
    except ImportError:
        # 如果没有rouge-score，使用简单实现
        result['rouge_l'] = _simple_rouge_l(predictions, references)
        result['rouge_l_note'] = "使用简化版Rouge-L（rouge-score未安装）"
    
    # BertScore
    if use_bert_score:
        try:
            from bert_score import score as bert_score_func
            
            P, R, F1 = bert_score_func(predictions, references, lang='zh', verbose=False)
            result['bert_score_precision'] = float(P.mean())
            result['bert_score_recall'] = float(R.mean())
            result['bert_score_f1'] = float(F1.mean())
        except ImportError:
            result['bert_score'] = None
            result['bert_score_note'] = "bert-score库未安装"
    
    return result


def _simple_bleu(predictions: List[str], references: List[str], n: int = 4) -> float:
    """简单BLEU实现（备用）"""
    from collections import Counter
    import math
    
    bleu_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred.replace(' ', ''))
        ref_tokens = list(ref.replace(' ', ''))
        
        if len(pred_tokens) == 0:
            bleu_scores.append(0.0)
            continue
        
        precisions = []
        for i in range(1, min(n + 1, len(pred_tokens) + 1)):
            pred_ngrams = Counter(tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1))
            ref_ngrams = Counter(tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1))
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            precisions.append(overlap / total if total > 0 else 0)
        
        if not precisions or all(p == 0 for p in precisions):
            bleu_scores.append(0.0)
            continue
        
        # 几何平均
        log_prec = [math.log(p) if p > 0 else -float('inf') for p in precisions]
        if all(lp == -float('inf') for lp in log_prec):
            geo_mean = 0.0
        else:
            valid = [lp for lp in log_prec if lp != -float('inf')]
            geo_mean = math.exp(sum(valid) / len(log_prec)) if valid else 0.0
        
        # 短句惩罚
        bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
        
        bleu_scores.append(bp * geo_mean)
    
    return float(np.mean(bleu_scores))


def _simple_rouge_l(predictions: List[str], references: List[str]) -> float:
    """简单Rouge-L实现（备用）"""
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = list(pred.replace(' ', ''))
        ref_tokens = list(ref.replace(' ', ''))
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            rouge_l_scores.append(0.0)
            continue
        
        # 计算LCS长度
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
        
        rouge_l_scores.append(f1)
    
    return float(np.mean(rouge_l_scores))


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("单标签分类测试")
    print("=" * 60)
    
    y_true = ["A", "B", "A", "C", "B", "A"]
    y_pred = ["A", "B", "B", "C", "A", "A"]
    labels = ["A", "B", "C"]
    
    metrics = calculate_singlelabel_metrics(y_true, y_pred, labels)
    print(format_metrics_for_print(metrics, "单标签测试", "single"))
    
    print("\n" + "=" * 60)
    print("多标签分类测试")
    print("=" * 60)
    
    y_true_multi = [["A"], ["B", "C"], ["A", "B"], ["C"]]
    y_pred_multi = [["A"], ["B"], ["A", "B"], ["A", "C"]]
    
    metrics_multi = calculate_multilabel_metrics(y_true_multi, y_pred_multi, labels)
    print(format_metrics_for_print(metrics_multi, "多标签测试", "multi"))
    
    print("\n" + "=" * 60)
    print("生成任务指标测试")
    print("=" * 60)
    
    predictions = ["这是一个测试", "你好世界"]
    references = ["这是一个测试句子", "你好世界"]
    
    gen_metrics = calculate_generation_metrics(predictions, references, use_bert_score=False)
    print(f"BLEU: {gen_metrics.get('bleu', 'N/A'):.4f}")
    print(f"Rouge-L: {gen_metrics.get('rouge_l', 'N/A'):.4f}")
