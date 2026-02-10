#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复年龄分布的脚本

将已生成数据中的十年区间年龄值（如 20, 30, 40）随机化为指定映射区间内的具体年龄。

年龄映射关系：
    10岁 → 10-18岁
    20岁 → 18-25岁
    30岁 → 25-35岁
    40岁 → 35-45岁
    50岁 → 45-55岁
    60岁 → 55-65岁
    70岁 → 65-75岁
    80岁 → 75-95岁

使用方法：
    python scripts/fix_age_distribution.py --input outputs/generated_emrs_xxx.json
    python scripts/fix_age_distribution.py --input outputs/generated_emrs_xxx.json --output outputs/fixed.json
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def is_decade_age(age: int) -> bool:
    """判断年龄是否为十年区间值（10, 20, 30, 40, ...）"""
    return age % 10 == 0


def randomize_age_in_decade(age: int) -> int:
    """
    将十年区间值随机化为指定区间内的具体年龄
    
    映射关系：
        10岁 → 10-18岁
        20岁 → 18-25岁
        30岁 → 25-35岁
        40岁 → 35-45岁
        50岁 → 45-55岁
        60岁 → 55-65岁
        70岁 → 65-75岁
        80岁 → 75-95岁
    
    Args:
        age: 原始年龄（如 20, 30, 40）
        
    Returns:
        映射区间内的随机年龄
    """
    if not is_decade_age(age):
        return age
    
    # 定义年龄映射区间
    age_mapping = {
        10: (10, 18),
        20: (18, 25),
        30: (25, 35),
        40: (35, 45),
        50: (45, 55),
        60: (55, 65),
        70: (65, 75),
        80: (75, 95),
    }
    
    # 特殊处理边界情况
    if age < 10:
        return random.randint(0, 9)
    elif age in age_mapping:
        min_age, max_age = age_mapping[age]
        return random.randint(min_age, max_age)
    elif age > 80:
        # 90岁及以上也映射到 75-95
        return random.randint(75, 95)
    else:
        # 其他情况（理论上不会发生）
        return random.randint(age, age + 9)


def fix_age_distribution(
    input_file: str,
    output_file: str = None,
    dry_run: bool = False,
):
    """
    修复年龄分布
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（默认覆盖原文件）
        dry_run: 是否只预览不修改
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_file}")
        return
    
    output_path = Path(output_file) if output_file else input_path
    
    # 加载数据
    print(f"加载文件: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print(f"总记录数: {len(records)}")
    
    # 统计原始年龄分布
    original_age_counts = {}
    decade_count = 0
    
    for record in records:
        age_str = record.get("Age", "")
        try:
            age = int(age_str)
            original_age_counts[age] = original_age_counts.get(age, 0) + 1
            if is_decade_age(age):
                decade_count += 1
        except (ValueError, TypeError):
            pass
    
    print(f"\n原始年龄分布:")
    for age in sorted(original_age_counts.keys()):
        count = original_age_counts[age]
        pct = count / len(records) * 100
        marker = " ← 十年区间值" if is_decade_age(age) else ""
        print(f"  {age}: {count} ({pct:.1f}%){marker}")
    
    print(f"\n十年区间值数量: {decade_count}/{len(records)} ({decade_count/len(records)*100:.1f}%)")
    
    if decade_count == 0:
        print("\n没有需要修复的十年区间年龄值")
        return
    
    if dry_run:
        print("\n[预览模式] 不会修改文件")
        # 模拟修复并显示预期分布
        simulated_ages = []
        for record in records:
            age_str = record.get("Age", "")
            try:
                age = int(age_str)
                new_age = randomize_age_in_decade(age)
                simulated_ages.append(new_age)
            except (ValueError, TypeError):
                pass
        
        # 统计模拟后的年龄组分布
        age_groups = {"0-18": 0, "18-30": 0, "30-45": 0, "45-60": 0, "60+": 0}
        for age in simulated_ages:
            if age < 18:
                age_groups["0-18"] += 1
            elif age < 30:
                age_groups["18-30"] += 1
            elif age < 45:
                age_groups["30-45"] += 1
            elif age < 60:
                age_groups["45-60"] += 1
            else:
                age_groups["60+"] += 1
        
        print(f"\n预期修复后的年龄组分布:")
        for group, count in age_groups.items():
            pct = count / len(simulated_ages) * 100 if simulated_ages else 0
            print(f"  {group}: {pct:.1f}%")
        return
    
    # 执行修复
    print("\n开始修复年龄分布...")
    fixed_count = 0
    
    for record in records:
        age_str = record.get("Age", "")
        try:
            age = int(age_str)
            if is_decade_age(age):
                new_age = randomize_age_in_decade(age)
                record["Age"] = str(new_age)
                fixed_count += 1
        except (ValueError, TypeError):
            pass
    
    print(f"修复完成: {fixed_count} 条记录")
    
    # 统计修复后的年龄分布
    new_age_counts = {}
    for record in records:
        age_str = record.get("Age", "")
        try:
            age = int(age_str)
            new_age_counts[age] = new_age_counts.get(age, 0) + 1
        except (ValueError, TypeError):
            pass
    
    # 按年龄组统计
    age_groups = {"0-18": 0, "18-30": 0, "30-45": 0, "45-60": 0, "60+": 0}
    for age, count in new_age_counts.items():
        if age < 18:
            age_groups["0-18"] += count
        elif age < 30:
            age_groups["18-30"] += count
        elif age < 45:
            age_groups["30-45"] += count
        elif age < 60:
            age_groups["45-60"] += count
        else:
            age_groups["60+"] += count
    
    print(f"\n修复后的年龄组分布:")
    for group, count in age_groups.items():
        pct = count / len(records) * 100
        print(f"  {group}: {pct:.1f}%")
    
    # 保存结果
    print(f"\n保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    print("\n修复完成!")
    
    # 显示几个修复示例
    print("\n修复示例:")
    sample_indices = random.sample(range(len(records)), min(5, len(records)))
    for idx in sample_indices:
        record = records[idx]
        print(f"  #{idx}: Age={record.get('Age')}, Gender={record.get('Gender')}")


def main():
    parser = argparse.ArgumentParser(description="修复年龄分布")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认覆盖原文件）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不修改文件"
    )
    
    args = parser.parse_args()
    
    fix_age_distribution(
        input_file=args.input,
        output_file=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
