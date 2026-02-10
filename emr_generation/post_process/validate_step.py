#!/usr/bin/env python3
"""
步骤间数据验证脚本

验证每一步输出的数据质量：
1. 记录数是否正确（默认 16,000）
2. patient_id 是否唯一
3. 必要字段是否存在且非空
4. 锚定字段（对话、诊断）是否与基准一致
5. 字段值分布是否合理

用法：
  python validate_step.py --input v3_output.json
  python validate_step.py --input v7_output.json --baseline v0_raw.json --check-anchors
"""

import argparse
import json
import sys
from collections import Counter


# 必须存在的字段
REQUIRED_FIELDS = [
    "patient_id", "Age", "Gender", "ChiefComplaint",
    "PresentIllnessHistory", "PersonalHistory", "FamilyHistory",
    "ImportantRelevantPhysicalIllnessHistory", "Diagnosis",
    "DiagnosisCode", "cleaned_text", "AccompanyingPerson",
]

# 锚定字段（不应被修改）
ANCHOR_FIELDS = ["cleaned_text", "Diagnosis", "DiagnosisCode"]

# Gender 允许值
VALID_GENDERS = {"男", "女"}


def validate(input_file: str, expected_count: int = 16000,
             baseline_file: str = None, check_anchors: bool = False):
    """执行验证"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = []
    warnings = []

    # 1. 记录数
    if len(data) != expected_count:
        errors.append(f"记录数: {len(data)} (期望 {expected_count})")
    else:
        print(f"[OK] 记录数: {len(data)}")

    # 2. patient_id 唯一性
    pids = [r.get("patient_id") for r in data]
    pid_counts = Counter(pids)
    duplicates = {k: v for k, v in pid_counts.items() if v > 1}
    if duplicates:
        errors.append(f"patient_id 重复: {len(duplicates)} 个 (如 {list(duplicates.keys())[:3]})")
    else:
        print(f"[OK] patient_id 唯一")

    # 3. 必要字段
    missing_fields = {}
    empty_fields = {}
    for r in data:
        for field in REQUIRED_FIELDS:
            if field not in r:
                missing_fields[field] = missing_fields.get(field, 0) + 1
            elif not str(r[field]).strip():
                empty_fields[field] = empty_fields.get(field, 0) + 1

    if missing_fields:
        for f, c in missing_fields.items():
            errors.append(f"字段缺失 {f}: {c} 条")
    else:
        print(f"[OK] 所有必要字段存在")

    if empty_fields:
        for f, c in empty_fields.items():
            if c > len(data) * 0.01:  # >1% 才报错
                warnings.append(f"字段为空 {f}: {c} 条 ({c/len(data)*100:.1f}%)")
    if not empty_fields:
        print(f"[OK] 所有必要字段非空")

    # 4. Gender 分布
    genders = Counter(r.get("Gender", "") for r in data)
    invalid_genders = {k: v for k, v in genders.items() if k not in VALID_GENDERS}
    if invalid_genders:
        errors.append(f"非法 Gender 值: {invalid_genders}")
    print(f"[INFO] Gender 分布: {dict(genders)}")

    # 5. Age 合理性
    age_issues = 0
    for r in data:
        try:
            age = int(str(r.get("Age", "0")).replace("岁", "").strip())
            if age < 1 or age > 120:
                age_issues += 1
        except (ValueError, TypeError):
            age_issues += 1
    if age_issues:
        warnings.append(f"Age 异常: {age_issues} 条")
    else:
        print(f"[OK] Age 值合理")

    # 6. 锚定字段检查
    if check_anchors and baseline_file:
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_map = {r["patient_id"]: r for r in baseline}

        for anchor in ANCHOR_FIELDS:
            changed = 0
            for r in data:
                pid = r["patient_id"]
                if pid in baseline_map:
                    if str(r.get(anchor, "")).strip() != str(baseline_map[pid].get(anchor, "")).strip():
                        changed += 1
            if changed:
                errors.append(f"锚定字段 {anchor} 被修改: {changed} 条")
            else:
                print(f"[OK] 锚定字段 {anchor} 未被修改")

    # 汇总
    print()
    if errors:
        print(f"=== {len(errors)} 个错误 ===")
        for e in errors:
            print(f"  [ERROR] {e}")
    if warnings:
        print(f"=== {len(warnings)} 个警告 ===")
        for w in warnings:
            print(f"  [WARN] {w}")
    if not errors and not warnings:
        print("=== 验证通过 ===")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(description="步骤间数据验证")
    parser.add_argument("--input", required=True, help="待验证的JSON文件")
    parser.add_argument("--expected-count", type=int, default=16000, help="期望记录数")
    parser.add_argument("--baseline", default=None, help="基准数据（用于锚定字段检查）")
    parser.add_argument("--check-anchors", action="store_true", help="检查锚定字段是否被修改")
    args = parser.parse_args()

    ok = validate(args.input, args.expected_count, args.baseline, args.check_anchors)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
