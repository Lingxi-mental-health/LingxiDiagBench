#!/usr/bin/env python3
"""
v6 → v7 合并脚本：将 fix_results.jsonl 的修复结果合并到原始数据集

功能：
1. 加载原始数据（始终从原始数据合并，不累积）
2. 从 fix_results.jsonl 提取修复字段并覆盖
3. 重建 Patient info 字段
4. 数据验证（记录数、patient_id唯一性、锚定字段完整性）

用法：
  python v6tov7_merge_results.py \
    --original v6_data.json \
    --fix-results fix_results.jsonl \
    --output v7_merged.json
"""

import argparse
import json
import sys
from pathlib import Path


# 锚定字段（不应被修改）
ANCHOR_FIELDS = ["cleaned_text", "Diagnosis", "DiagnosisCode"]

# Patient info 重建时使用的字段
PATIENT_INFO_FIELDS = {
    "PersonalHistory": None,  # 直接拼接
    "ChiefComplaint": "主诉",
    "PresentIllnessHistory": "现病史",
    "FamilyHistory": "家族史",
}


def rebuild_patient_info(record: dict) -> str:
    """重建 Patient info 字段"""
    parts = []
    for field, prefix in PATIENT_INFO_FIELDS.items():
        value = record.get(field, "")
        if value:
            if prefix:
                parts.append(f"{prefix}:{value}")
            else:
                parts.append(value)
    return "|".join(parts)


def load_fix_results(fix_results_file: str) -> dict:
    """加载修复结果，返回 {patient_id: fixed_fields}"""
    fixes = {}
    total = 0
    fixed = 0
    errors = 0

    with open(fix_results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            status = r.get("status", "")

            if "_error" in status:
                errors += 1
                continue

            if status == "fixed" and r.get("num_changes", 0) > 0:
                fixes[r["patient_id"]] = r["fixed_fields"]
                fixed += 1

    print(f"fix_results: 总计 {total}, 已修复 {fixed}, 错误 {errors}")
    return fixes


def merge(original_file: str, fix_results_file: str, output_file: str,
          exclude_fields: list = None):
    """
    合并修复结果到原始数据

    Args:
        original_file: 原始数据JSON
        fix_results_file: 修复结果JSONL
        output_file: 输出合并后的JSON
        exclude_fields: 排除不合并的字段列表（如 ["Age"] 可防止年龄被错误修改）
    """
    exclude_fields = set(exclude_fields or [])

    # 加载原始数据
    with open(original_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"原始数据: {len(data)} 条")

    # 加载修复结果
    fixes = load_fix_results(fix_results_file)

    # 应用修复
    n_applied = 0
    n_fields_applied = 0
    n_fields_skipped = 0
    n_anchor_violations = 0

    for record in data:
        pid = record.get("patient_id")
        if pid in fixes:
            for field, value in fixes[pid].items():
                # 跳过锚定字段
                if field in ANCHOR_FIELDS:
                    n_anchor_violations += 1
                    continue
                # 跳过排除字段
                if field in exclude_fields:
                    n_fields_skipped += 1
                    continue
                if value:
                    record[field] = value
                    n_fields_applied += 1
            n_applied += 1

        # 始终重建 Patient info
        record["Patient info"] = rebuild_patient_info(record)

    # 验证
    patient_ids = [r["patient_id"] for r in data]
    assert len(data) == len(set(patient_ids)), "patient_id 不唯一！"

    # 输出
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n合并结果:")
    print(f"  应用修复: {n_applied} 条记录")
    print(f"  字段变更: {n_fields_applied}")
    if n_fields_skipped:
        print(f"  字段跳过: {n_fields_skipped} (排除: {exclude_fields})")
    if n_anchor_violations:
        print(f"  锚定字段拦截: {n_anchor_violations} (不允许修改 {ANCHOR_FIELDS})")
    print(f"  输出: {output_file} ({len(data)} 条)")


def main():
    parser = argparse.ArgumentParser(description="v6→v7: 合并修复结果到原始数据")
    parser.add_argument("--original", required=True, help="原始数据JSON文件")
    parser.add_argument("--fix-results", required=True, help="修复结果JSONL文件")
    parser.add_argument("--output", required=True, help="输出合并后的JSON文件")
    parser.add_argument("--exclude-fields", nargs="*", default=[],
                        help="排除不合并的字段（如 Age）")
    args = parser.parse_args()

    merge(args.original, args.fix_results, args.output, args.exclude_fields)


if __name__ == "__main__":
    main()
