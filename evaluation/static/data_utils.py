"""
数据工具模块

包含数据加载、预处理和ICD代码提取功能
"""

import json
import re
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

try:
    from .config import VALID_ICD_CODES, VALID_ICD_SUBCODES, TWO_CLASS_LABELS, FOUR_CLASS_LABELS
except ImportError:
    from config import VALID_ICD_CODES, VALID_ICD_SUBCODES, TWO_CLASS_LABELS, FOUR_CLASS_LABELS


def extract_f_codes_from_diagnosis_code(diagnosis_code: str) -> List[str]:
    """
    从DiagnosisCode中提取所有Fxx或Z71代码（大类）
    支持细分代码和多个代码，从F32.000;F41.100中提取["F32", "F41"]
    
    参考自: doctor_eval_multilabel.py
    
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


def extract_detailed_icd_codes(diagnosis_code: str) -> List[str]:
    """
    从DiagnosisCode中提取所有ICD-10小类代码
    支持细分代码，从F32.100;F41.000中提取["F32.1", "F41.0"]
    
    Args:
        diagnosis_code: 诊断代码，如 "F41.900," 或 "F32.100;F41.100" 或 "Z71.900"
        
    Returns:
        提取的小类代码列表，如 ["F41.9"] 或 ["F32.1", "F41.1"] 或 ["Z71.9"]
    """
    if not diagnosis_code:
        return ["Others"]
    
    # 清理字符串，去除多余空格
    code = diagnosis_code.strip().upper()
    
    # 分割多个代码（支持分号、逗号分隔）
    code_parts = re.split(r'[;,]', code)
    
    extracted_codes = []
    
    for part in code_parts:
        part = part.strip().rstrip(',')
        if not part:
            continue
        
        # 检查Z71.x格式
        z71_match = re.search(r'Z71(?:\.(\d))?', part)
        if z71_match:
            subcode = z71_match.group(1)
            if subcode:
                code_normalized = f"Z71.{subcode}"
            else:
                code_normalized = "Z71.9"  # 默认使用Z71.9
            if code_normalized not in extracted_codes and code_normalized in VALID_ICD_SUBCODES:
                extracted_codes.append(code_normalized)
            continue
        
        # 检查F39格式（特殊处理，F39没有小数点形式）
        if re.search(r'F39', part):
            if "F39" not in extracted_codes:
                extracted_codes.append("F39")
            continue
        
        # 使用正则表达式提取F开头的代码（支持Fxx.x或Fxx.xxx格式）
        # 匹配格式: F32.100 -> F32.1, F41.200 -> F41.2
        pattern = r'F(\d{2})\.(\d)'
        match = re.search(pattern, part)
        if match:
            f_major = match.group(1)
            f_minor = match.group(2)
            code_normalized = f"F{f_major}.{f_minor}"
            
            # 检查是否在允许的小类代码中且未重复
            if code_normalized not in extracted_codes:
                if code_normalized in VALID_ICD_SUBCODES:
                    extracted_codes.append(code_normalized)
                else:
                    # 如果小类代码不在列表中，尝试回退到大类
                    major_code = f"F{f_major}"
                    # 查找该大类的未特指代码（通常是.9结尾）
                    unspecified_code = f"F{f_major}.9"
                    if unspecified_code in VALID_ICD_SUBCODES and unspecified_code not in extracted_codes:
                        extracted_codes.append(unspecified_code)
            continue
        
        # 如果只有大类代码（如F32），使用未特指的小类代码
        major_pattern = r'F(\d{2})$'
        major_match = re.search(major_pattern, part)
        if major_match:
            f_major = major_match.group(1)
            unspecified_code = f"F{f_major}.9"
            if unspecified_code in VALID_ICD_SUBCODES and unspecified_code not in extracted_codes:
                extracted_codes.append(unspecified_code)
    
    # 如果没有找到任何有效代码，返回Others
    if not extracted_codes:
        return ["Others"]
    
    return extracted_codes


def extract_detailed_codes(diagnosis_code: str) -> Dict[str, Any]:
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


def add_icd_labels_to_data(data: List[Dict]) -> List[Dict]:
    """
    为数据添加icd_clf_label字段（大类）和icd_clf_label_detailed字段（小类）
    
    Args:
        data: 原始数据列表
        
    Returns:
        添加了icd_clf_label和icd_clf_label_detailed字段的数据列表
    """
    for item in data:
        diagnosis_code = item.get('DiagnosisCode', '')
        
        # 如果没有icd_clf_label，则提取12分类标签（大类）
        if 'icd_clf_label' not in item:
            icd_labels = extract_f_codes_from_diagnosis_code(diagnosis_code)
            item['icd_clf_label'] = icd_labels
        
        # 提取12分类标签（小类）
        if 'icd_clf_label_detailed' not in item:
            icd_labels_detailed = extract_detailed_icd_codes(diagnosis_code)
            item['icd_clf_label_detailed'] = icd_labels_detailed
        
        # 提取详细信息用于2分类和4分类
        detailed_info = extract_detailed_codes(diagnosis_code)
        
        # 添加2分类标签
        if 'two_class_label' not in item:
            two_class_label = classify_2class(detailed_info)
            item['two_class_label'] = two_class_label
        
        # 添加4分类标签
        if 'four_class_label' not in item:
            four_class_label = classify_4class(detailed_info)
            item['four_class_label'] = four_class_label
    
    return data


def load_and_process_data(file_path: str, save_processed: bool = False) -> List[Dict]:
    """
    加载并处理数据文件，添加分类标签
    
    Args:
        file_path: 数据文件路径
        save_processed: 是否保存处理后的数据
        
    Returns:
        处理后的数据列表
    """
    print(f"正在加载数据: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据条数: {len(data)}")
    
    # 添加分类标签
    data = add_icd_labels_to_data(data)
    
    # 统计标签分布
    print_label_statistics(data)
    
    # 保存处理后的数据
    if save_processed:
        output_path = file_path.replace('.json', '_processed.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"处理后的数据已保存到: {output_path}")
    
    return data


def print_label_statistics(data: List[Dict]) -> None:
    """
    打印标签分布统计
    
    Args:
        data: 数据列表
    """
    print("\n" + "="*60)
    print("标签分布统计")
    print("="*60)
    
    # 12分类统计
    twelve_class_dist = defaultdict(int)
    multi_label_count = 0
    
    for item in data:
        labels = item.get('icd_clf_label', ['Others'])
        if len(labels) > 1:
            multi_label_count += 1
        for label in labels:
            twelve_class_dist[label] += 1
    
    print("\n12分类分布:")
    for label in sorted(twelve_class_dist.keys()):
        print(f"  {label}: {twelve_class_dist[label]}")
    print(f"  多标签样本数: {multi_label_count}/{len(data)} ({multi_label_count/len(data)*100:.1f}%)")
    
    # 4分类统计
    four_class_dist = defaultdict(int)
    for item in data:
        label = item.get('four_class_label', 'Others')
        four_class_dist[label] += 1
    
    print("\n4分类分布:")
    for label in ["Depression", "Anxiety", "Mixed", "Others"]:
        print(f"  {label}: {four_class_dist[label]}")
    
    # 2分类统计
    two_class_dist = defaultdict(int)
    two_class_valid = 0
    for item in data:
        label = item.get('two_class_label')
        if label is not None:
            two_class_dist[label] += 1
            two_class_valid += 1
    
    print(f"\n2分类分布 (无共病样本数: {two_class_valid}):")
    for label in ["Depression", "Anxiety"]:
        print(f"  {label}: {two_class_dist[label]}")
    
    print("="*60 + "\n")


def get_text_for_classification(item: Dict, use_cleaned_text: bool = True) -> str:
    """
    获取用于分类的文本
    
    Args:
        item: 数据项
        use_cleaned_text: 是否使用cleaned_text字段
        
    Returns:
        用于分类的文本
    """
    if use_cleaned_text and 'cleaned_text' in item:
        return item['cleaned_text']
    
    # 组合多个字段
    text_parts = []
    
    if 'ChiefComplaint' in item:
        text_parts.append(item['ChiefComplaint'])
    if 'PresentIllnessHistory' in item:
        text_parts.append(item['PresentIllnessHistory'])
    if 'PersonalHistory' in item:
        text_parts.append(item['PersonalHistory'])
    
    return ' '.join(text_parts)


def extract_doctor_turns(conversation_text: str) -> List[Tuple[str, str]]:
    """
    从对话文本中提取医生提问对
    用于医生提问下一句预测任务
    
    Args:
        conversation_text: 对话文本
        
    Returns:
        (上文, 医生下一句) 的列表
    """
    # 分割对话轮次
    turns = conversation_text.split('\n')
    turns = [t.strip() for t in turns if t.strip()]
    
    # 提取医生提问对
    doctor_pairs = []
    
    for i, turn in enumerate(turns):
        # 假设偶数索引是医生，奇数索引是患者（或相反）
        # 实际需要根据数据格式调整
        if i > 0:
            context = '\n'.join(turns[:i])
            next_turn = turn
            doctor_pairs.append((context, next_turn))
    
    return doctor_pairs


def prepare_classification_dataset(
    data: List[Dict],
    classification_type: str = "12class",
    use_cleaned_text: bool = True
) -> Tuple[List[str], List[Any]]:
    """
    准备分类数据集
    
    Args:
        data: 数据列表
        classification_type: 分类类型 ("2class", "4class", "12class", "12class_detailed")
        use_cleaned_text: 是否使用cleaned_text
        
    Returns:
        (文本列表, 标签列表)
    """
    try:
        from .config import TWO_CLASS_LABELS, FOUR_CLASS_LABELS
    except ImportError:
        from config import TWO_CLASS_LABELS, FOUR_CLASS_LABELS
    
    texts = []
    labels = []
    skipped_count = 0
    
    for item in data:
        text = get_text_for_classification(item, use_cleaned_text)
        
        if not text or len(text.strip()) == 0:
            continue
        
        if classification_type == "2class":
            label = item.get('two_class_label')
            # 只使用标签在 TWO_CLASS_LABELS 中的样本
            if label is None or label not in TWO_CLASS_LABELS:
                skipped_count += 1
                continue
        elif classification_type == "4class":
            label = item.get('four_class_label', 'Others')
            # 只使用标签在 FOUR_CLASS_LABELS 中的样本
            if label not in FOUR_CLASS_LABELS:
                skipped_count += 1
                continue
        elif classification_type == "12class_detailed":
            # 使用小类标签
            label = item.get('icd_clf_label_detailed', ['Others'])
        else:  # 12class（大类）
            label = item.get('icd_clf_label', ['Others'])
        
        texts.append(text)
        labels.append(label)
    
    if skipped_count > 0:
        print(f"[{classification_type}] 跳过 {skipped_count} 个标签不匹配的样本")
    
    return texts, labels


if __name__ == "__main__":
    # 测试代码
    from .config import TRAIN_DATA_FILE, TEST_DATA_100_FILE
    
    # 加载并处理测试数据
    test_data = load_and_process_data(TEST_DATA_100_FILE, save_processed=False)
    
    # 打印样例
    if test_data:
        sample = test_data[0]
        print("样例数据:")
        print(f"  patient_id: {sample.get('patient_id')}")
        print(f"  DiagnosisCode: {sample.get('DiagnosisCode')}")
        print(f"  icd_clf_label: {sample.get('icd_clf_label')}")
        print(f"  two_class_label: {sample.get('two_class_label')}")
        print(f"  four_class_label: {sample.get('four_class_label')}")

