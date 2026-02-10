"""
数据预处理脚本

为训练和测试数据添加icd_clf_label字段
"""

import json
import os
import argparse
from typing import Dict, List

from data_utils import (
    extract_f_codes_from_diagnosis_code,
    extract_detailed_codes,
    classify_2class,
    classify_4class,
    print_label_statistics
)
from config import (
    TRAIN_DATA_FILE, TEST_DATA_FILE,
    TEST_DATA_100_FILE, TEST_DATA_500_FILE
)


def add_labels_to_file(input_file: str, output_file: str = None, overwrite: bool = False) -> str:
    """
    为数据文件添加分类标签
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（如果为None，则覆盖原文件或添加_labeled后缀）
        overwrite: 是否覆盖原文件
        
    Returns:
        输出文件路径
    """
    print(f"\n正在处理文件: {input_file}")
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始数据条数: {len(data)}")
    
    # 统计已有标签的数量
    existing_labels = sum(1 for item in data if 'icd_clf_label' in item)
    if existing_labels > 0:
        print(f"已有 {existing_labels} 条数据包含icd_clf_label字段")
    
    # 添加标签
    added_count = 0
    for item in data:
        # 如果已经有icd_clf_label则跳过
        if 'icd_clf_label' in item:
            continue
        
        diagnosis_code = item.get('DiagnosisCode', '')
        
        # 提取12分类标签
        icd_labels = extract_f_codes_from_diagnosis_code(diagnosis_code)
        item['icd_clf_label'] = icd_labels
        
        # 提取详细信息用于2分类和4分类
        detailed_info = extract_detailed_codes(diagnosis_code)
        
        # 添加2分类标签
        two_class_label = classify_2class(detailed_info)
        item['two_class_label'] = two_class_label
        
        # 添加4分类标签
        four_class_label = classify_4class(detailed_info)
        item['four_class_label'] = four_class_label
        
        added_count += 1
    
    print(f"新添加标签: {added_count} 条")
    
    # 打印标签统计
    print_label_statistics(data)
    
    # 确定输出路径
    if output_file is None:
        if overwrite:
            output_file = input_file
        else:
            base, ext = os.path.splitext(input_file)
            output_file = f"{base}_labeled{ext}"
    
    # 保存数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存到: {output_file}")
    return output_file


def preprocess_all_data(overwrite: bool = False):
    """
    预处理所有数据文件
    
    Args:
        overwrite: 是否覆盖原文件
    """
    print("="*80)
    print("开始预处理所有数据文件")
    print("="*80)
    
    # 需要处理的文件列表
    files = [
        TRAIN_DATA_FILE,
        TEST_DATA_FILE,
        TEST_DATA_100_FILE,
        TEST_DATA_500_FILE
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            try:
                add_labels_to_file(file_path, overwrite=overwrite)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
        else:
            print(f"文件不存在: {file_path}")
    
    print("\n" + "="*80)
    print("数据预处理完成！")
    print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据预处理脚本')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--overwrite', action='store_true', help='覆盖原文件')
    parser.add_argument('--all', action='store_true', help='处理所有数据文件')
    
    args = parser.parse_args()
    
    if args.all:
        preprocess_all_data(overwrite=args.overwrite)
    elif args.input:
        add_labels_to_file(args.input, args.output, args.overwrite)
    else:
        print("请指定 --input 参数或使用 --all 处理所有文件")
        print("示例:")
        print("  python preprocess_data.py --all")
        print("  python preprocess_data.py --input data.json --overwrite")


if __name__ == "__main__":
    main()

