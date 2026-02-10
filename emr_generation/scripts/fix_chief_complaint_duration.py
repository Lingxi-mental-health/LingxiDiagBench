#!/usr/bin/env python3
"""
批量修复主诉（ChiefComplaint）中的 duration，使其与现病史（PresentIllnessHistory）的起病时间对齐。

动机：
  - 你们的数据里，PI 常写“约1个月/2周/2023年11月…”，但 CC 末尾的时间经常不一致。
  - 这里以 PI 为准，回写 CC 的末尾时间 token（主诉对齐到现病史）。

注意：
  - 只纠错：PI 开头解析不到起病时间 token，则不改 CC
  - 不改“加重X周/月”等逗号后内容
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple


def extract_pi_onset_token(pi: str) -> Optional[str]:
    if not pi:
        return None
    head = str(pi)[:160]
    m = re.search(r"(20\d{2}年\d{1,2}月)", head)
    if m:
        return m.group(1)
    m = re.search(r"(约|近|大约|约莫|约为|约在)?\s*\d{1,3}\s*(年|个月|月|周|天|日)\s*(前|来|以来)", head)
    if m:
        return (m.group(0) or "").strip()
    return None


def rewrite_cc_duration(cc: str, new_token: str) -> str:
    if not cc or not new_token:
        return cc
    cc = str(cc).strip()

    prefix = ""
    if cc.startswith("主诉："):
        prefix = "主诉："
        body = cc[len(prefix):].strip()
    elif cc.startswith("主诉:"):
        prefix = "主诉:"
        body = cc[len(prefix):].strip()
    else:
        body = cc

    # preserve comma tail
    sep = None
    for s in ["，", ","]:
        if s in body:
            sep = s
            break
    if sep:
        before, after = body.split(sep, 1)
        after = sep + after
    else:
        before, after = body, ""

    before = before.strip()
    if " " not in before:
        return cc

    symptom_part = before.rsplit(" ", 1)[0]
    new_before = f"{symptom_part} {new_token.strip()}"
    return (prefix + new_before + after).strip()


def extract_cc_duration(cc: str) -> Optional[str]:
    if not cc:
        return None
    text = str(cc).strip()
    text = text.replace("主诉：", "", 1).replace("主诉:", "", 1).strip()
    for sep in ["，", ","]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    if " " not in text:
        return None
    return text.rsplit(" ", 1)[-1].strip() or None


def mismatch_count(records) -> Tuple[int, int]:
    """
    Returns: (comparable, mismatch)
    comparable: both CC duration and PI onset token exist
    mismatch:   CC duration != PI onset token (string compare)
    """
    comparable = mismatch = 0
    for r in records:
        cc_d = extract_cc_duration(r.get("ChiefComplaint") or "")
        pi_d = extract_pi_onset_token(r.get("PresentIllnessHistory") or "")
        if not cc_d or not pi_d:
            continue
        comparable += 1
        if cc_d != pi_d:
            mismatch += 1
    return comparable, mismatch


def main():
    ap = argparse.ArgumentParser(description="主诉 duration 对齐到现病史起病时间")
    ap.add_argument("--input", "-i", required=True, help="输入 EMR JSON（list[dict]）")
    ap.add_argument("--output", "-o", default=None, help="输出文件（默认 *_fixed_cc_duration.json）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_fixed_cc_duration.json")

    records = json.loads(in_path.read_text(encoding="utf-8"))

    before = mismatch_count(records)

    changed = 0
    for r in records:
        pi_token = extract_pi_onset_token(r.get("PresentIllnessHistory") or "")
        if not pi_token:
            continue
        cc = r.get("ChiefComplaint")
        if not cc:
            continue
        fixed = rewrite_cc_duration(cc, pi_token)
        if fixed != cc:
            r["ChiefComplaint"] = fixed
            changed += 1

    after = mismatch_count(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("input:", in_path)
    print("output:", out_path)
    print("records:", len(records))
    print("changed_chief_complaint:", changed)
    print("cc_vs_pi_before: comparable=%d mismatch=%d" % before)
    print("cc_vs_pi_after:  comparable=%d mismatch=%d" % after)


if __name__ == "__main__":
    main()


