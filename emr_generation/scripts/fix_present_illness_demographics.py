#!/usr/bin/env python3
"""
批量修复现病史中的年龄/性别冲突

用途：
  - 针对已生成的 EMR JSON（list[dict]），如果 PresentIllnessHistory 里出现“患者XX岁(男性/女性)”
    等表述，则强制替换为记录的 Age/Gender，避免 LLM 幻觉导致冲突。

说明：
  - 只做“纠错”，不强制插入年龄/性别（文本里没写就不补）
  - 默认输出到新文件（避免覆盖原始数据）
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple


def enforce_demographics(text: str, age: int, gender: str) -> str:
    if not text:
        return text

    gender_word = "男性" if gender == "男" else "女性"

    pattern_age_then_gender = re.compile(
        r"(患者(?:为|系)?\s*)(\d{1,3})(\s*岁\s*[,，]?\s*)(男性|女性)"
    )
    text = pattern_age_then_gender.sub(lambda m: f"{m.group(1)}{age}{m.group(3)}{gender_word}", text)

    pattern_gender_then_age = re.compile(
        r"(患者(?:为|系)?\s*)(男性|女性)(\s*[,，]?\s*)(\d{1,3})(\s*岁)"
    )
    text = pattern_gender_then_age.sub(lambda m: f"{m.group(1)}{gender_word}{m.group(3)}{age}{m.group(5)}", text)

    pattern_age_only = re.compile(r"(患者(?:为|系)?\s*)(\d{1,3})(\s*岁)")
    text = pattern_age_only.sub(lambda m: f"{m.group(1)}{age}{m.group(3)}", text)

    pattern_gender_only = re.compile(r"(患者(?:为|系)?\s*)(男性|女性)")
    text = pattern_gender_only.sub(lambda m: f"{m.group(1)}{gender_word}", text)

    return text


def count_age_mismatch(records) -> Tuple[int, int, int]:
    """
    Returns: (pi_has_age_mention, age_match, age_mismatch)
    """
    age_re = re.compile(r"患者\s*(\d{1,3})\s*岁")
    has = match = mismatch = 0
    for r in records:
        pi = r.get("PresentIllnessHistory") or ""
        m = age_re.search(pi)
        if not m:
            continue
        has += 1
        try:
            a = int(str(r.get("Age")))
            a2 = int(m.group(1))
        except Exception:
            continue
        if a == a2:
            match += 1
        else:
            mismatch += 1
    return has, match, mismatch


def main():
    ap = argparse.ArgumentParser(description="修复现病史中的年龄/性别冲突")
    ap.add_argument("--input", "-i", required=True, help="输入 EMR JSON 文件（list[dict]）")
    ap.add_argument("--output", "-o", default=None, help="输出文件路径（默认在同目录生成 *_fixed_demographics.json）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_fixed_demographics.json")

    records = json.loads(in_path.read_text(encoding="utf-8"))

    before = count_age_mismatch(records)

    changed = 0
    for r in records:
        pi = r.get("PresentIllnessHistory")
        if not pi:
            continue
        try:
            age = int(str(r.get("Age")))
        except Exception:
            continue
        gender = r.get("Gender")
        if gender not in ("男", "女"):
            continue
        fixed = enforce_demographics(pi, age=age, gender=gender)
        if fixed != pi:
            r["PresentIllnessHistory"] = fixed
            changed += 1

    after = count_age_mismatch(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("input:", in_path)
    print("output:", out_path)
    print("records:", len(records))
    print("changed_present_illness:", changed)
    print("age_mismatch_before: has=%d match=%d mismatch=%d" % before)
    print("age_mismatch_after:  has=%d match=%d mismatch=%d" % after)


if __name__ == "__main__":
    main()


