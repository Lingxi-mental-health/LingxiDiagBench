#!/usr/bin/env python3
"""
批量修复现病史中的起病时间/病程时长，使其与主诉的 duration 对齐。

对齐策略：
  - 从 ChiefComplaint 中按模板位置提取 duration token（如“3周前”“2月”“2023年11月”）
  - 若 PresentIllnessHistory 开头已包含相对时间（约X周/月/年前/近X月来等）或日期（2023年11月），
    则替换为目标 duration（必要时将“2月”转为“近2月来”，将日期转为“于YYYY年MM月”）
  - 不强制插入：如果现病史没有写起病时间，则不补

输出：
  - 生成 *_fixed_duration.json
  - 打印修复前后“主诉 duration vs 现病史首个时间表达”不一致的数量（仅统计能解析到的相对时间/日期）
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple


CC_PREFIX = re.compile(r"^\s*主诉[:：]\s*")
DATE_YYYY_MM = re.compile(r"20\d{2}年\d{1,2}月")
REL_TIME = re.compile(r"(?:约|近|大约|约莫|约为|约在)?\s*(\d{1,3})\s*(年|个月|月|周|天|日)\s*(前|来|以来)?")


def extract_cc_duration(cc: str) -> Optional[str]:
    if not cc:
        return None
    text = str(cc).strip()
    text = text.replace("主诉：", "", 1).replace("主诉:", "", 1).strip()
    # 主诉主体（不含加重）
    for sep in ["，", ","]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()
            break
    if " " not in text:
        return None
    dur = text.rsplit(" ", 1)[-1].strip()
    return dur or None


def target_phrase_from_duration(dur: str) -> str:
    dur = (dur or "").strip()
    if not dur:
        return dur
    if DATE_YYYY_MM.fullmatch(dur):
        return f"于{dur}"
    if dur.startswith(("近", "约", "大约", "约莫", "约为", "约在")):
        return dur
    if "前" in dur or dur.endswith("来") or dur.endswith("以来"):
        return dur
    if re.fullmatch(r"\d{1,3}\s*(年|个月|月|周|天|日)", dur):
        return f"近{dur}来"
    return dur


def enforce_pi_onset_duration(pi: str, target_duration: str) -> str:
    if not pi or not target_duration:
        return pi
    target_phrase = target_phrase_from_duration(target_duration)
    prefix_window = r"(现病史[:：][^。！？；\n]{0,80}?)"

    pat_rel = re.compile(
        prefix_window + r"(?:约|近|大约|约莫|约为|约在)?\s*\d{1,3}\s*(?:年|个月|月|周|天|日)\s*(?:前|来|以来)"
    )
    fixed, n = pat_rel.subn(lambda m: m.group(1) + target_phrase, pi, count=1)
    if n:
        return fixed

    pat_date = re.compile(prefix_window + r"(20\d{2}年\d{1,2}月)")
    fixed, n = pat_date.subn(lambda m: m.group(1) + target_phrase, pi, count=1)
    if n:
        return fixed

    return pi


def extract_pi_onset_token(pi: str) -> Optional[str]:
    if not pi:
        return None
    # only look near beginning (avoid “近2周加重”)
    head = pi[:120]
    m = DATE_YYYY_MM.search(head)
    if m:
        return m.group(0)
    m = re.search(r"(约|近|大约|约莫|约为|约在)?\s*\d{1,3}\s*(年|个月|月|周|天|日)\s*(前|来|以来)", head)
    if m:
        return (m.group(0) or "").strip()
    return None


def mismatch_count(records) -> Tuple[int, int]:
    """
    Returns: (comparable_count, mismatch_count)
    comparable_count: both CC and PI have extractable onset tokens
    """
    comparable = mism = 0
    for r in records:
        cc_d = extract_cc_duration(r.get("ChiefComplaint") or "")
        pi_d = extract_pi_onset_token(r.get("PresentIllnessHistory") or "")
        if not cc_d or not pi_d:
            continue
        comparable += 1
        # compare by exact string containment after normalization for PI target phrase
        # (we align PI to CC, so mismatch means PI head doesn't reflect CC duration token)
        if cc_d not in (r.get("PresentIllnessHistory") or "")[:200]:
            mism += 1
    return comparable, mism


def main():
    ap = argparse.ArgumentParser(description="修复现病史起病时间/病程时长，使其与主诉对齐")
    ap.add_argument("--input", "-i", required=True, help="输入 EMR JSON 文件（list[dict]）")
    ap.add_argument("--output", "-o", default=None, help="输出文件（默认 *_fixed_duration.json）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_fixed_duration.json")

    records = json.loads(in_path.read_text(encoding="utf-8"))

    before = mismatch_count(records)

    changed = 0
    for r in records:
        cc_d = extract_cc_duration(r.get("ChiefComplaint") or "")
        if not cc_d:
            continue
        pi = r.get("PresentIllnessHistory")
        if not pi:
            continue
        fixed = enforce_pi_onset_duration(pi, cc_d)
        if fixed != pi:
            r["PresentIllnessHistory"] = fixed
            changed += 1

    after = mismatch_count(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print("input:", in_path)
    print("output:", out_path)
    print("records:", len(records))
    print("changed_present_illness:", changed)
    print("duration_mismatch_before: comparable=%d mismatch=%d" % before)
    print("duration_mismatch_after:  comparable=%d mismatch=%d" % after)


if __name__ == "__main__":
    main()


