import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple


PERCENT_RE = re.compile(r"(?P<num>\d{1,3}(?:\.\d+)?)\s*%")


def _soften_percent_phrases(text: str, *, keep_weight_percent: bool) -> Tuple[str, int, Counter]:
    """
    Replace unnatural percent-based quantification in Chinese clinical free text.

    Returns:
        (new_text, num_replacements, counter_by_rule)
    """
    if not text or "%" not in text:
        return text, 0, Counter()

    total_n = 0
    rule_counts: Counter = Counter()
    s = text

    def subn(rule_name: str, pattern: str, repl: str) -> None:
        nonlocal s, total_n, rule_counts
        s, n = re.subn(pattern, repl, s)
        if n:
            total_n += n
            rule_counts[rule_name] += n

    # 1) 学习/工作/专注等“功能效率”百分比：非常不临床
    subn(
        "efficiency_pct",
        r"(学习效率|工作效率|效率|专注力|注意力).{0,10}?(下降|降低|减少)\s*(约|大约|约为|约)\s*\d{1,3}(?:\.\d+)?\s*%",
        r"\1\2明显",
    )
    subn(
        "efficiency_pct_simple",
        r"(学习效率|工作效率|效率|专注力|注意力)\s*(下降|降低|减少)\s*\d{1,3}(?:\.\d+)?\s*%",
        r"\1\2明显",
    )

    # 2) 食欲/进食/饮食量百分比：常见但“30%/50%”过于机械
    subn(
        "appetite_pct",
        r"(食欲|胃口|饮食量|进食量|食量).{0,10}?(下降|减退|减少)\s*(约|大约|约为|约)\s*\d{1,3}(?:\.\d+)?\s*%",
        r"\1\2",
    )
    subn(
        "appetite_pct_simple",
        r"(食欲|胃口|饮食量|进食量|食量)\s*(下降|减退|减少)\s*\d{1,3}(?:\.\d+)?\s*%",
        r"\1\2",
    )

    # 3) 体重百分比：允许保留（如“下降5%”），也可选择统一模糊化
    if not keep_weight_percent:
        subn(
            "weight_pct",
            r"(体重).{0,10}?(下降|减轻|减少)\s*(约|大约|约为|约)\s*\d{1,3}(?:\.\d+)?\s*%",
            r"\1有所下降",
        )
        subn(
            "weight_pct_simple",
            r"(体重).{0,10}?(下降|减轻|减少)\s*\d{1,3}(?:\.\d+)?\s*%",
            r"\1有所下降",
        )

    # 4) 兜底：任何 “下降/降低/减少 xx%” 改成“下降明显/减少明显”
    subn(
        "generic_drop_pct",
        r"(下降|降低|减少)\s*(约|大约|约为|约)?\s*\d{1,3}(?:\.\d+)?\s*%",
        r"\1明显",
    )

    # 5) 最终兜底：残余的 “xx%” 直接去掉（尽量少触发）
    if "%" in s:
        if keep_weight_percent:
            # 若保留体重百分比，避免误删：仅删除“非体重”上下文的孤立百分号（保守做法）
            # 这里采用一个简单启发式：如果 % 左侧 12 字符内包含“体重”，则保留该百分比。
            def _remove_non_weight_percent(m: re.Match) -> str:
                start = m.start()
                ctx = s[max(0, start - 12) : start]
                if "体重" in ctx:
                    return m.group(0)
                rule_counts["fallback_drop_pct"] += 1
                return ""

            s2 = ""
            last = 0
            for m in PERCENT_RE.finditer(s):
                s2 += s[last : m.start()] + _remove_non_weight_percent(m)
                last = m.end()
            s2 += s[last:]
            if s2 != s:
                total_n += rule_counts["fallback_drop_pct"]
                s = s2
        else:
            s, n = PERCENT_RE.subn("", s)
            if n:
                total_n += n
                rule_counts["fallback_drop_pct"] += n

    # 清理可能遗留的 “约” 等
    s = re.sub(r"约\s*(?=[，。；,.;])", "", s)
    s = re.sub(r"\s{2,}", " ", s)

    return s, total_n, rule_counts


def fix_record(record: Dict[str, Any], *, keep_weight_percent: bool) -> Tuple[int, Counter]:
    fields = [
        "ChiefComplaint",
        "PresentIllnessHistory",
        "PersonalHistory",
        "FamilyHistory",
        "ImportantRelevantPhysicalIllnessHistory",
        "AuxiliaryExamination",
    ]
    n_total = 0
    c_total: Counter = Counter()
    for k in fields:
        v = record.get(k)
        if isinstance(v, str) and "%" in v:
            new_v, n, c = _soften_percent_phrases(v, keep_weight_percent=keep_weight_percent)
            if n:
                record[k] = new_v
                n_total += n
                c_total.update(c)
    return n_total, c_total


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove/soften percent quantification in generated EMR free text.")
    ap.add_argument("--input", required=True, help="input JSON (list[dict])")
    ap.add_argument("--output", required=True, help="output JSON (list[dict])")
    ap.add_argument(
        "--keep_weight_percent",
        action="store_true",
        help="keep weight percent patterns like '体重下降5%' (default: remove/soften all %)",
    )
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("input must be a JSON list")

    total_repl = 0
    per_rule = Counter()
    per_field_hits = defaultdict(int)

    for rec in data:
        if not isinstance(rec, dict):
            continue
        before = {k: rec.get(k) for k in ("PresentIllnessHistory", "ChiefComplaint") if isinstance(rec.get(k), str)}
        n, c = fix_record(rec, keep_weight_percent=bool(args.keep_weight_percent))
        if n:
            total_repl += n
            per_rule.update(c)
            # 粗略统计哪些字段常见
            for k, v in before.items():
                if isinstance(v, str) and "%" in v:
                    per_field_hits[k] += 1

    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[fix_percent_quantification] input={in_path}")
    print(f"[fix_percent_quantification] output={out_path}")
    print(f"[fix_percent_quantification] keep_weight_percent={bool(args.keep_weight_percent)}")
    print(f"[fix_percent_quantification] total_replacements={total_repl}")
    print(f"[fix_percent_quantification] field_hits={dict(per_field_hits)}")
    print(f"[fix_percent_quantification] rule_counts={dict(per_rule)}")


if __name__ == "__main__":
    main()


