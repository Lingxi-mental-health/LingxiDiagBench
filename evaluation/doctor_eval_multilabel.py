"""
医生诊断多标签评估脚本

该脚本用于评估医生诊断结果与标准标签的准确率（支持多标签）
支持从DiagnosisCode字段提取Fxx代码进行比较
计算Exactly Match, Top-1, Top-3准确率以及Macro-F1等指标
"""

import json
import re
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


def extract_icd_code_from_conversation(conversation: List[Dict]) -> Optional[str]:
    """
    从对话列表中提取ICD代码（支持细分代码和多代码）
    
    Args:
        conversation: 对话列表
        
    Returns:
        提取的ICD代码（可能包含细分代码，用分号分隔），如果没有找到则返回None
    """
    try:
        # 查找最后一个医生的回复（包含诊断结果）
        last_doctor_response = None
        for turn in conversation:
            if 'doctor' in turn:
                last_doctor_response = turn['doctor']
        
        if not last_doctor_response:
            return None
        
        # 新格式：查找<box>标签中的内容
        box_pattern = r'<box>([^<]+)</box>'
        box_match = re.search(box_pattern, last_doctor_response)
        if box_match:
            return box_match.group(1).strip()
        
        # 旧格式兼容：处理各种icd_code格式
        if 'icd_code' in last_doctor_response:
            # 策略1：查找icd_code后面花括号中的内容 icd_code{F41}
            pattern1 = r'icd_code\{([^}]+)\}'
            match1 = re.search(pattern1, last_doctor_response)
            if match1:
                return match1.group(1).strip()
            
            # 策略2：查找icd_code后面直接跟的内容（支持细分代码）
            pattern2 = r'icd_code([A-Z]\d{2}(?:\.\d+)?(?:[;,]\s*[A-Z]\d{2}(?:\.\d+)?)*)'
            match2 = re.search(pattern2, last_doctor_response)
            if match2:
                return match2.group(1).strip()
            
            # 策略3：查找icd_code后面空格分隔的内容
            pattern3 = r'icd_code\s+([A-Z]\d{2}(?:\.\d+)?(?:[;,]\s*[A-Z]\d{2}(?:\.\d+)?)*)'
            match3 = re.search(pattern3, last_doctor_response)
            if match3:
                return match3.group(1).strip()
            
            # 策略4：查找icd_code后面冒号分隔的内容
            pattern4 = r'icd_code:\s*([A-Z]\d{2}(?:\.\d+)?(?:[;,]\s*[A-Z]\d{2}(?:\.\d+)?)*)'
            match4 = re.search(pattern4, last_doctor_response)
            if match4:
                return match4.group(1).strip()
        
        return None
        
    except Exception as e:
        print(f"处理对话时出错: {e}")
        return None


def extract_f_codes_from_diagnosis_code(diagnosis_code: str) -> List[str]:
    """
    从DiagnosisCode中提取所有Fxx或Z71代码（大类）
    支持细分代码和多个代码，从F32.000;F41.100中提取["F32", "F41"]
    
    Args:
        diagnosis_code: 诊断代码，如 "F41.900," 或 "F32.000;F41.100" 或 "Z71"
        
    Returns:
        提取的大类代码列表，如 ["F41"] 或 ["F32", "F41"] 或 ["Z71"]
    """
    if not diagnosis_code:
        return ["Others"]
    
    # 清理字符串，去除多余空格
    code = diagnosis_code.strip().upper()
    
    # 分割多个代码（支持分号、逗号分隔）
    code_parts = re.split(r'[;,]', code)
    
    extracted_codes = []
    valid_codes = ["F20", "F31", "F32", "F39", "F41", "F42", "F43", "F45", "F51", "F98"]
    
    for part in code_parts:
        part = part.strip().rstrip(',')
        
        # 检查Z71
        if 'Z71' in part:
            if "Z71" not in extracted_codes:
                extracted_codes.append("Z71")
            continue
        
        # 使用正则表达式提取F开头的代码（支持细分代码）
        pattern = r'F(\d{2})(?:\.\d+)?'
        match = re.search(pattern, part)
        if match:
            f_code = f"F{match.group(1)}"
            # 检查是否在允许的分类中且未重复
            if f_code in valid_codes and f_code not in extracted_codes:
                extracted_codes.append(f_code)
    
    # 如果没有找到任何有效代码，返回Others
    if not extracted_codes:
        return ["Others"]
    
    return extracted_codes


def normalize_diagnosis_codes(raw_code: str) -> List[str]:
    """
    标准化诊断代码，处理LLM输出的不标准格式（支持细分代码和多代码）
    从细分代码（如F32.0;F41.0）中提取所有大类（["F32", "F41"]）用于12分类
    
    Args:
        raw_code: 原始诊断代码（可能包含细分代码和多个代码，如"F32.0;F41.0"）
        
    Returns:
        标准化后的诊断代码列表（大类，如["F32", "F41"]）
    """
    if not raw_code:
        return ["Others"]
    
    # 清理字符串
    code = raw_code.strip().upper()
    
    # 分割多个代码
    code_parts = re.split(r'[;,]', code)
    
    extracted_codes = []
    valid_codes = ["F20", "F31", "F32", "F39", "F41", "F42", "F43", "F45", "F51", "F98"]
    
    for part in code_parts:
        part = part.strip().rstrip(',')
        
        # 检查Z71
        if 'Z71' in part:
            if "Z71" not in extracted_codes:
                extracted_codes.append("Z71")
            continue
        
        # 使用正则表达式提取F开头的代码（支持细分代码如F32.0）
        pattern = r'F(\d{2})(?:\.\d+)?'
        match = re.search(pattern, part)
        if match:
            f_code = f"F{match.group(1)}"
            # 检查是否在允许的分类中且未重复
            if f_code in valid_codes and f_code not in extracted_codes:
                extracted_codes.append(f_code)
    
    # 如果没有找到任何有效代码，返回Others
    if not extracted_codes:
        return ["Others"]
    
    return extracted_codes


def extract_detailed_codes(diagnosis_code: str) -> Dict[str, any]:
    """
    提取详细的诊断代码信息，用于2分类和4分类
    支持多个代码（用分号或逗号分隔）
    
    Args:
        diagnosis_code: 诊断代码字符串（可能包含多个代码，如"F32.0;F41.0"）
        
    Returns:
        包含详细代码信息的字典
    """
    if not diagnosis_code:
        return {
            'has_f32': False, 
            'has_f41': False, 
            'has_f41_0': False,
            'has_f41_1': False, 
            'has_f41_2': False,
            'raw': diagnosis_code,
            'codes': []
        }
    
    code = diagnosis_code.strip().upper()
    
    # 分割多个代码（支持分号和逗号）
    codes = re.split(r'[;,]', code)
    codes = [c.strip() for c in codes if c.strip()]
    
    # 检查是否有F32（任何细分）
    has_f32 = any(re.search(r'F32', c) for c in codes)
    
    # 检查是否有F41（任何细分）
    has_f41 = any(re.search(r'F41', c) for c in codes)
    
    # 检查是否有F41.0（惊恐障碍）
    has_f41_0 = any(re.search(r'F41\.0', c) for c in codes)
    
    # 检查是否有F41.1（广泛性焦虑障碍）
    has_f41_1 = any(re.search(r'F41\.1', c) for c in codes)
    
    # 检查是否有F41.2（混合性焦虑与抑郁障碍）
    has_f41_2 = any(re.search(r'F41\.2', c) for c in codes)
    
    return {
        'has_f32': has_f32,
        'has_f41': has_f41,
        'has_f41_0': has_f41_0,
        'has_f41_1': has_f41_1,
        'has_f41_2': has_f41_2,
        'raw': code,
        'codes': codes
    }


def classify_2class(detailed_info: Dict) -> Optional[str]:
    """
    2分类：抑郁 vs. 焦虑（仅在无共病样本中）
    
    Args:
        detailed_info: 详细代码信息
        
    Returns:
        "Depression" (F32), "Anxiety" (F41), 或 None（有共病或其他情况）
    """
    has_f32 = detailed_info['has_f32']
    has_f41 = detailed_info['has_f41']
    has_f41_2 = detailed_info['has_f41_2']
    
    # 如果有F41.2（混合性焦虑与抑郁障碍）或者同时有F32和F41，则不参与2分类
    if has_f41_2 or (has_f32 and has_f41):
        return None
    
    # 纯抑郁（只有F32，没有F41）
    if has_f32 and not has_f41:
        return "Depression"
    
    # 纯焦虑（只有F41，没有F32）
    if has_f41 and not has_f32:
        return "Anxiety"
    
    return None


def classify_4class(detailed_info: Dict) -> str:
    """
    4分类：抑郁 vs. 焦虑 vs. 抑郁焦虑混合 vs. 其他
    
    Args:
        detailed_info: 详细代码信息
        
    Returns:
        "Depression", "Anxiety", "Mixed", 或 "Others"
    """
    has_f32 = detailed_info['has_f32']
    has_f41 = detailed_info['has_f41']
    has_f41_2 = detailed_info['has_f41_2']
    
    # 混合：F41.2（混合性焦虑与抑郁障碍）或者 F32和F41共存
    if has_f41_2 or (has_f32 and has_f41):
        return "Mixed"
    
    # 纯抑郁（只有F32，没有F41）
    if has_f32 and not has_f41:
        return "Depression"
    
    # 纯焦虑（只有F41，没有F32）
    if has_f41 and not has_f32:
        return "Anxiety"
    
    return "Others"


def load_standard_labels(labels_file_path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    加载标准标签数据，从DiagnosisCode字段提取所有Fxx代码
    
    Args:
        labels_file_path: 标签文件路径
        
    Returns:
        (patient_id到Fxx代码列表的映射字典, patient_id到原始DiagnosisCode的映射字典)
    """
    patient_labels = {}
    raw_diagnosis_codes = {}
    
    try:
        with open(labels_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计每个patient_id的记录数
        patient_count = defaultdict(int)
        
        for patient_data in data:
            # 直接使用patient_id字段进行匹配
            patient_id = patient_data.get('patient_id')
            diagnosis_code = patient_data.get('DiagnosisCode')
            
            if patient_id is not None and diagnosis_code:
                f_codes = extract_f_codes_from_diagnosis_code(diagnosis_code)
                # 如果有多个记录，使用最后一个（假设数据按时间排序）
                patient_labels[str(patient_id)] = f_codes  # 转换为字符串格式
                raw_diagnosis_codes[str(patient_id)] = diagnosis_code  # 保存原始代码
                patient_count[patient_id] += 1
        
        # 打印统计信息
        print(f"标签文件中包含 {len(set(patient_count.keys()))} 个唯一的patient_id")
        multi_label_patients = [pid for pid, count in patient_count.items() if count > 1]
        if multi_label_patients:
            print(f"有 {len(multi_label_patients)} 个患者有多条标签记录，使用最后一条记录")
        
        # 统计各类别分布（包括多标签统计）
        class_distribution = defaultdict(int)
        multi_label_count = 0
        for f_codes in patient_labels.values():
            if len(f_codes) > 1:
                multi_label_count += 1
            for f_code in f_codes:
                class_distribution[f_code] += 1
        
        print("标准标签类别分布:")
        for f_code, count in sorted(class_distribution.items()):
            print(f"  {f_code}: {count}")
        print(f"多标签样本数: {multi_label_count}/{len(patient_labels)} ({multi_label_count/len(patient_labels)*100:.1f}%)")
        
        return patient_labels, raw_diagnosis_codes
    except Exception as e:
        print(f"加载标准标签文件时出错: {e}")
        return {}, {}


def calculate_multilabel_metrics(all_predictions: List[Tuple[List[str], List[str]]]) -> Dict:
    """
    计算多标签分类指标
    
    Args:
        all_predictions: 列表，每个元素是 (predicted_list, true_list) 元组
        
    Returns:
        包含各种指标的字典
    """
    # 定义所有类别
    all_classes = ["F20", "F31", "F32", "F39", "F41", "F42", "F43", "F45", "F51", "F98", "Z71", "Others"]
    
    # 统计变量
    total_samples = len(all_predictions)
    exact_match_count = 0
    top1_correct = 0
    top3_correct = 0
    hamming_scores = []
    
    # 每个类别的TP, FP, FN
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    
    for pred_list, true_list in all_predictions:
        pred_set = set(pred_list)
        true_set = set(true_list)
        
        # 1. Exactly Match (Subset Accuracy)
        if pred_set == true_set:
            exact_match_count += 1
        
        # 2. Top-1 Accuracy: 第一个预测是否在真实标签中
        if pred_list and pred_list[0] in true_set:
            top1_correct += 1
        
        # 3. Top-3 Accuracy: 前三个预测中至少有一个在真实标签中
        top3_pred = set(pred_list[:3])
        if len(top3_pred & true_set) > 0:
            top3_correct += 1
        
        # 4. Hamming Score: 标签级别的准确率
        all_labels = pred_set | true_set
        if all_labels:
            correct_labels = len(pred_set & true_set)
            hamming_score = correct_labels / len(all_labels)
            hamming_scores.append(hamming_score)
        
        # 5. 为每个类别计算TP, FP, FN
        for cls in all_classes:
            if cls in true_set and cls in pred_set:
                class_tp[cls] += 1
            elif cls not in true_set and cls in pred_set:
                class_fp[cls] += 1
            elif cls in true_set and cls not in pred_set:
                class_fn[cls] += 1
    
    # 计算每个类别的 Precision, Recall, F1
    class_metrics = {}
    precision_list = []
    recall_list = []
    f1_list = []
    
    for cls in all_classes:
        tp = class_tp[cls]
        fp = class_fp[cls]
        fn = class_fn[cls]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        support = tp + fn
        
        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        # 只统计有支持数的类别
        if support > 0:
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
    
    # Macro-averaged metrics
    macro_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    macro_recall = sum(recall_list) / len(recall_list) if recall_list else 0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0
    
    # Micro-averaged metrics
    total_tp = sum(class_tp.values())
    total_fp = sum(class_fp.values())
    total_fn = sum(class_fn.values())
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # 计算准确率
    exact_match_acc = exact_match_count / total_samples if total_samples > 0 else 0
    top1_acc = top1_correct / total_samples if total_samples > 0 else 0
    top3_acc = top3_correct / total_samples if total_samples > 0 else 0
    hamming_acc = sum(hamming_scores) / len(hamming_scores) if hamming_scores else 0
    
    return {
        'total_samples': total_samples,
        'exact_match_accuracy': exact_match_acc,
        'exact_match_count': exact_match_count,
        'top1_accuracy': top1_acc,
        'top1_correct': top1_correct,
        'top3_accuracy': top3_acc,
        'top3_correct': top3_correct,
        'hamming_accuracy': hamming_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'class_metrics': class_metrics
    }


def api_llm_diagnosis_evaluation_multilabel(dialogue_path: str, labels_file_path: str) -> Dict:
    """
    评估LLM诊断结果的准确率（多标签版本）
    
    Args:
        dialogue_path: 对话数据路径，支持两种格式：
            1. 目录路径：包含多个patient_*.json文件（旧格式）
            2. JSON文件路径：包含所有患者对话的单个文件（新格式）
        labels_file_path: 标准标签文件路径
        
    Returns:
        包含评估结果的字典
    """
    # 加载标准标签
    print("正在加载标准标签...")
    standard_labels, raw_diagnosis_codes = load_standard_labels(labels_file_path)
    print(f"已加载 {len(standard_labels)} 个标准标签")
    
    # 检测输入路径类型并加载对话数据
    print("正在处理对话数据...")
    patient_dialogues = {}  # {patient_id: conversation}
    
    if os.path.isfile(dialogue_path):
        # 新格式：单个JSON文件包含所有对话
        print(f"检测到单个JSON文件格式: {dialogue_path}")
        try:
            with open(dialogue_path, 'r', encoding='utf-8') as f:
                all_conversations = json.load(f)
            
            for item in all_conversations:
                patient_id = str(item.get('patient_id'))
                conversation = item.get('conversation', [])
                patient_dialogues[patient_id] = conversation
            
            print(f"从文件中加载了 {len(patient_dialogues)} 个患者的对话")
        except Exception as e:
            print(f"读取对话文件出错: {e}")
            return {}
    
    elif os.path.isdir(dialogue_path):
        # 旧格式：目录包含多个patient_*.json文件
        print(f"检测到目录格式: {dialogue_path}")
        for filename in os.listdir(dialogue_path):
            if filename.endswith('.json'):
                # 提取patient_id
                pattern = r'patient_([^.]+)\.json'
                match = re.search(pattern, filename)
                if not match:
                    continue
                patient_id = match.group(1)
                
                try:
                    dialogue_file_path = os.path.join(dialogue_path, filename)
                    with open(dialogue_file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    conversation = data[0]['conversation']
                    patient_dialogues[patient_id] = conversation
                except Exception as e:
                    print(f"读取文件 {filename} 出错: {e}")
                    continue
        
        print(f"从目录中加载了 {len(patient_dialogues)} 个患者的对话")
    else:
        print(f"错误: 路径 {dialogue_path} 既不是文件也不是目录")
        return {}
    
    # 收集所有预测结果
    predictions = {}
    all_predictions = []  # [(pred_list, true_list), ...]
    dialogue_turns = []
    
    # 2分类和4分类统计
    two_class_total = 0
    two_class_correct = 0
    two_class_confusion = defaultdict(lambda: defaultdict(int))
    two_class_predictions = {}
    
    four_class_total = 0
    four_class_correct = 0
    four_class_confusion = defaultdict(lambda: defaultdict(int))
    four_class_predictions = {}
    
    # 处理每个患者的对话数据
    for patient_id, conversation in patient_dialogues.items():
        # 检查是否有对应的标准标签
        if patient_id not in standard_labels:
            print(f"警告: 患者 {patient_id} 没有找到标准标签")
            continue
        
        # 提取预测结果
        raw_prediction = extract_icd_code_from_conversation(conversation)
        if raw_prediction is None:
            print(f"警告: 患者 {patient_id} 的对话中没有找到诊断代码")
            continue
        
        # 统计对话轮数
        turn_count = len(conversation)
        dialogue_turns.append(turn_count)
        
        # 标准化预测结果（返回列表）
        predicted_codes = normalize_diagnosis_codes(raw_prediction)
        true_codes = standard_labels[patient_id]
        
        # 记录结果
        predictions[str(patient_id)] = {
            'predicted': predicted_codes,  # 列表
            'true': true_codes,  # 列表
            'raw_prediction': raw_prediction,
            'patient_id': patient_id
        }
        
        # 添加到统计列表
        all_predictions.append((predicted_codes, true_codes))
        
        # ===== 2分类和4分类评估 =====
        # 获取标准答案和预测的详细代码信息
        true_diagnosis_code = raw_diagnosis_codes.get(patient_id, '')
        true_detailed = extract_detailed_codes(true_diagnosis_code)
        pred_detailed = extract_detailed_codes(raw_prediction)
        
        # 2分类评估（仅在无共病样本中）
        true_2class = classify_2class(true_detailed)
        pred_2class = classify_2class(pred_detailed)
        
        if true_2class is not None:  # 只在无共病的样本中评估
            two_class_total += 1
            
            # 如果预测不是无共病的，将其视为预测错误
            if pred_2class is None:
                # 预测有共病或其他情况，视为预测成另一个类别（作为最坏情况）
                if true_2class == "Depression":
                    pred_2class = "Anxiety"
                else:
                    pred_2class = "Depression"
            
            two_class_predictions[patient_id] = {
                'predicted': pred_2class,
                'true': true_2class
            }
            
            if pred_2class == true_2class:
                two_class_correct += 1
            
            two_class_confusion[true_2class][pred_2class] += 1
        
        # 4分类评估（所有样本）
        true_4class = classify_4class(true_detailed)
        pred_4class = classify_4class(pred_detailed)
        
        four_class_total += 1
        four_class_predictions[patient_id] = {
            'predicted': pred_4class,
            'true': true_4class
        }
        
        if pred_4class == true_4class:
            four_class_correct += 1
        
        four_class_confusion[true_4class][pred_4class] += 1
    
    # 计算多标签指标
    print(f"\n正在计算多标签评估指标（总样本数: {len(all_predictions)}）...")
    metrics = calculate_multilabel_metrics(all_predictions)
    
    # ===== 计算2分类指标 =====
    two_class_accuracy = two_class_correct / two_class_total if two_class_total > 0 else 0
    two_class_labels = ["Depression", "Anxiety"]
    two_class_metrics = {}
    
    for cls in two_class_labels:
        tp = two_class_confusion[cls][cls]
        fp = sum(two_class_confusion[other_cls][cls] for other_cls in two_class_labels if other_cls != cls)
        fn = sum(two_class_confusion[cls][other_cls] for other_cls in two_class_labels if other_cls != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        two_class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    # 2分类宏平均F1
    two_class_macro_f1 = sum(m['f1'] for m in two_class_metrics.values() if m['support'] > 0) / len([m for m in two_class_metrics.values() if m['support'] > 0]) if any(m['support'] > 0 for m in two_class_metrics.values()) else 0
    
    # ===== 计算4分类指标 =====
    four_class_accuracy = four_class_correct / four_class_total if four_class_total > 0 else 0
    four_class_labels = ["Depression", "Anxiety", "Mixed", "Others"]
    four_class_metrics = {}
    
    for cls in four_class_labels:
        tp = four_class_confusion[cls][cls]
        fp = sum(four_class_confusion[other_cls][cls] for other_cls in four_class_labels if other_cls != cls)
        fn = sum(four_class_confusion[cls][other_cls] for other_cls in four_class_labels if other_cls != cls)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = tp + fn
        
        four_class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    # 4分类宏平均F1
    four_class_macro_f1 = sum(m['f1'] for m in four_class_metrics.values() if m['support'] > 0) / len([m for m in four_class_metrics.values() if m['support'] > 0]) if any(m['support'] > 0 for m in four_class_metrics.values()) else 0
    
    # 计算对话轮数统计
    dialogue_stats = {}
    if dialogue_turns:
        avg_turns = sum(dialogue_turns) / len(dialogue_turns)
        min_turns = min(dialogue_turns)
        max_turns = max(dialogue_turns)
        
        # 计算不同轮数区间的百分比
        under_10 = sum(1 for t in dialogue_turns if t < 10)
        between_10_20 = sum(1 for t in dialogue_turns if 10 <= t <= 20)
        above_30 = sum(1 for t in dialogue_turns if t > 30)
        
        total_dialogues = len(dialogue_turns)
        dialogue_stats = {
            'avg_turns': avg_turns,
            'min_turns': min_turns,
            'max_turns': max_turns,
            'total_dialogues': total_dialogues,
            'under_10_percent': (under_10 / total_dialogues) * 100,
            'between_10_20_percent': (between_10_20 / total_dialogues) * 100,
            'above_30_percent': (above_30 / total_dialogues) * 100
        }
    
    # 返回完整结果
    return {
        'predictions': predictions,
        'metrics': metrics,
        'dialogue_statistics': dialogue_stats,
        # 2分类结果
        'two_class': {
            'total_cases': two_class_total,
            'correct_predictions': two_class_correct,
            'accuracy': two_class_accuracy,
            'macro_f1': two_class_macro_f1,
            'class_metrics': two_class_metrics,
            'predictions': two_class_predictions,
            'confusion_matrix': dict(two_class_confusion)
        },
        # 4分类结果
        'four_class': {
            'total_cases': four_class_total,
            'correct_predictions': four_class_correct,
            'accuracy': four_class_accuracy,
            'macro_f1': four_class_macro_f1,
            'class_metrics': four_class_metrics,
            'predictions': four_class_predictions,
            'confusion_matrix': dict(four_class_confusion)
        }
    }


def print_evaluation_results(results: Dict):
    """
    打印多标签评估结果
    
    Args:
        results: 评估结果字典
    """
    metrics = results['metrics']
    
    print("\n" + "="*80)
    print("医生诊断多标签评估结果")
    print("="*80)
    
    print(f"\n总样本数: {metrics['total_samples']}")
    
    print("\n" + "-"*80)
    print("多标签评估指标:")
    print("-"*80)
    print(f"Exactly Match准确率: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"  - 精确匹配数: {metrics['exact_match_count']}/{metrics['total_samples']}")
    print(f"\nTop-1准确率: {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
    print(f"  - Top-1正确数: {metrics['top1_correct']}/{metrics['total_samples']}")
    print(f"\nTop-3准确率: {metrics['top3_accuracy']:.4f} ({metrics['top3_accuracy']*100:.2f}%)")
    print(f"  - Top-3正确数: {metrics['top3_correct']}/{metrics['total_samples']}")
    print(f"\nHamming准确率: {metrics['hamming_accuracy']:.4f} ({metrics['hamming_accuracy']*100:.2f}%)")
    
    print("\n" + "-"*80)
    print("Macro-averaged指标:")
    print("-"*80)
    print(f"Macro Precision: {metrics['macro_precision']:.4f} ({metrics['macro_precision']*100:.2f}%)")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f} ({metrics['macro_recall']*100:.2f}%)")
    print(f"Macro F1:        {metrics['macro_f1']:.4f} ({metrics['macro_f1']*100:.2f}%)")
    
    print("\n" + "-"*80)
    print("Micro-averaged指标:")
    print("-"*80)
    print(f"Micro Precision: {metrics['micro_precision']:.4f} ({metrics['micro_precision']*100:.2f}%)")
    print(f"Micro Recall:    {metrics['micro_recall']:.4f} ({metrics['micro_recall']*100:.2f}%)")
    print(f"Micro F1:        {metrics['micro_f1']:.4f} ({metrics['micro_f1']*100:.2f}%)")
    
    print("\n" + "-"*80)
    print("各类别详细指标:")
    print("-"*80)
    print(f"{'类别':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-"*80)
    
    for cls in sorted(metrics['class_metrics'].keys()):
        m = metrics['class_metrics'][cls]
        if m['support'] > 0:  # 只显示有支持数的类别
            print(f"{cls:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f} {m['support']:<10}")
    
    # 打印对话轮数统计
    if 'dialogue_statistics' in results and results['dialogue_statistics']:
        stats = results['dialogue_statistics']
        print("\n" + "-"*80)
        print("对话轮数统计:")
        print("-"*80)
        print(f"平均轮数: {stats['avg_turns']:.2f}")
        print(f"最高轮数: {stats['max_turns']}")
        print(f"最低轮数: {stats['min_turns']}")
        print(f"10轮以下: {stats['under_10_percent']:.1f}%")
        print(f"10-20轮: {stats['between_10_20_percent']:.1f}%")
        print(f"30轮以上: {stats['above_30_percent']:.1f}%")
    
    # 打印2分类结果
    if 'two_class' in results and results['two_class']['total_cases'] > 0:
        two_class_results = results['two_class']
        print("\n" + "-"*80)
        print("2分类评估结果（抑郁 vs. 焦虑，仅无共病样本）")
        print("-"*80)
        print(f"总样本数: {two_class_results['total_cases']}")
        print(f"正确预测数: {two_class_results['correct_predictions']}")
        print(f"2分类准确率: {two_class_results['accuracy']:.4f} ({two_class_results['accuracy']*100:.2f}%)")
        print(f"2分类宏平均F1: {two_class_results['macro_f1']:.4f} ({two_class_results['macro_f1']*100:.2f}%)")
        
        print("\n各类别详细指标:")
        print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-"*80)
        for cls, metrics in two_class_results['class_metrics'].items():
            if metrics['support'] > 0:
                print(f"{cls:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['support']:<10}")
    
    # 打印4分类结果
    if 'four_class' in results and results['four_class']['total_cases'] > 0:
        four_class_results = results['four_class']
        print("\n" + "-"*80)
        print("4分类评估结果（抑郁 vs. 焦虑 vs. 混合 vs. 其他）")
        print("-"*80)
        print(f"总样本数: {four_class_results['total_cases']}")
        print(f"正确预测数: {four_class_results['correct_predictions']}")
        print(f"4分类准确率: {four_class_results['accuracy']:.4f} ({four_class_results['accuracy']*100:.2f}%)")
        print(f"4分类宏平均F1: {four_class_results['macro_f1']:.4f} ({four_class_results['macro_f1']*100:.2f}%)")
        
        print("\n各类别详细指标:")
        print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
        print("-"*80)
        for cls, metrics in four_class_results['class_metrics'].items():
            if metrics['support'] > 0:
                print(f"{cls:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['support']:<10}")
    
    print("\n" + "="*80)
    print("说明:")
    print("  - Exactly Match: 预测的所有标签与真实标签完全相同")
    print("  - Top-1: 第一个预测的标签在真实标签中")
    print("  - Top-3: 前三个预测的标签中至少有一个在真实标签中")
    print("  - Hamming: 每个样本的标签级别准确率的平均值")
    print("  - Macro: 每个类别指标的平均值（不考虑样本数）")
    print("  - Micro: 所有样本合并后计算的指标（考虑样本数）")
    print("  - 2分类: 仅评估无共病样本（纯抑郁 vs. 纯焦虑）")
    print("  - 4分类: 评估所有样本（抑郁 vs. 焦虑 vs. 混合 vs. 其他）")
    print("="*80)


if __name__ == "__main__":
    import sys
    start_time = time.perf_counter()
    
    # 获取当前脚本的父目录（项目根目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 设置文件路径
    dialogue_path = os.path.join(project_root, "output_conv", "patient_eval", "conversations_doctor_v2_qwen3-32b_patient_v1_qwen3-32b.json")
    labels_file_path = os.path.join(project_root, "raw_data", "SMHC_LingxiDiag-16K_validation_data_100samples.json")
    
    # 执行评估
    results = api_llm_diagnosis_evaluation_multilabel(dialogue_path, labels_file_path)
    
    # 打印结果
    print_evaluation_results(results)
    
    # 保存详细结果到文件
    output_file = os.path.join(project_root, "output_conv", "patient_eval", "evaluation_results", "evaluation_multilabel_doctor_v2_qwen3-32b_patient_v1_qwen3-32b.json")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    elapsed_seconds = time.perf_counter() - start_time
    print(f"总耗时: {elapsed_seconds:.6f} 秒")

