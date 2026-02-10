#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为男性数据填充"孕产情况"字段
按照女性数据中"孕产情况"的分布进行随机采样填充

输入: LingxiDiag-16K_fixed_v3.json
输出: LingxiDiag-16K_all_data.json
"""

import json
import re
import random
from collections import Counter
from pathlib import Path

# 设置随机种子以保证可复现性
random.seed(42)

# 路径配置
INPUT_FILE = Path("/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/PresentIllnessHistory/LingxiDiag-16K_fixed_v3.json")
OUTPUT_FILE = Path(__file__).parent / "LingxiDiag-16K_all_data.json"


def extract_pregnancy_status(personal_history: str) -> str:
    """从PersonalHistory中提取孕产情况"""
    match = re.search(r'孕产情况：([^，,]+)', personal_history)
    if match:
        return match.group(1)
    return None


def add_pregnancy_to_personal_history(personal_history: str, pregnancy_status: str) -> str:
    """在PersonalHistory开头添加孕产情况"""
    return f"孕产情况：{pregnancy_status}，{personal_history}"


def update_patient_info(patient_info: str, pregnancy_status: str) -> str:
    """同步更新Patient info字段，在开头添加孕产情况"""
    return f"孕产情况：{pregnancy_status}，{patient_info}"


def main():
    # 读取源数据
    print(f"读取数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"总记录数: {len(data)}")

    # 分离男性和女性数据
    female_data = [d for d in data if d.get('Gender') == '女']
    male_data = [d for d in data if d.get('Gender') == '男']

    print(f"女性记录数: {len(female_data)}")
    print(f"男性记录数: {len(male_data)}")

    # 统计女性数据中孕产情况的分布
    pregnancy_statuses = []
    for d in female_data:
        personal_history = d.get('PersonalHistory', '')
        status = extract_pregnancy_status(personal_history)
        if status:
            pregnancy_statuses.append(status)

    # 统计分布
    status_counter = Counter(pregnancy_statuses)
    total_with_status = len(pregnancy_statuses)

    print(f"\n女性数据中孕产情况分布 (共{total_with_status}条有孕产情况):")
    for status, count in status_counter.most_common():
        print(f"  {status}: {count} ({count/total_with_status*100:.2f}%)")

    # 构建用于随机采样的列表
    status_list = list(status_counter.keys())
    status_weights = [status_counter[s] for s in status_list]

    # 为男性数据填充孕产情况
    male_filled_count = 0
    for d in data:
        if d.get('Gender') == '男':
            personal_history = d.get('PersonalHistory', '')
            patient_info = d.get('Patient info', '')

            # 检查是否已有孕产情况
            if '孕产情况' not in personal_history:
                # 按分布随机选择一个孕产情况
                selected_status = random.choices(status_list, weights=status_weights, k=1)[0]

                # 更新PersonalHistory
                d['PersonalHistory'] = add_pregnancy_to_personal_history(personal_history, selected_status)

                # 同步更新Patient info
                if patient_info and '孕产情况' not in patient_info:
                    d['Patient info'] = update_patient_info(patient_info, selected_status)

                male_filled_count += 1

    print(f"\n已为 {male_filled_count} 条男性记录填充孕产情况")

    # 保存输出
    print(f"\n保存数据到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("处理完成!")

    # 验证结果
    print("\n=== 验证结果 ===")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    # 检查男性数据
    male_output = [d for d in output_data if d.get('Gender') == '男']
    male_with_pregnancy = sum(1 for d in male_output if '孕产情况' in d.get('PersonalHistory', ''))
    print(f"输出中男性记录数: {len(male_output)}")
    print(f"输出中有孕产情况的男性记录数: {male_with_pregnancy}")

    # 展示几条男性数据的PersonalHistory
    print("\n前3条男性数据的PersonalHistory示例:")
    for i, d in enumerate(male_output[:3]):
        print(f"  [{i+1}] {d.get('PersonalHistory', '')[:80]}...")


if __name__ == "__main__":
    main()
