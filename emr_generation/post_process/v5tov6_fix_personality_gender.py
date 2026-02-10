#!/usr/bin/env python3
"""
v5 → v6：性格矛盾修复 + 性别-疾病矛盾修复

修复规则（纯规则，无LLM）：
1. PersonalHistory 中"病前性格"同时包含"内向"和"外向" → 统一改为"内向"
2. 男性患者的躯体疾病史中包含女性专属疾病（卵巢、子宫、宫颈、巧克力囊肿等）→ 删除
3. 女性患者的躯体疾病史中包含男性专属疾病（前列腺等）→ 删除
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ==================== 规则定义 ====================

# 女性专属疾病关键词（男性不应有）
# 注意：不包含"宫外孕"和"月经不调"
FEMALE_ONLY_DISEASES = [
    r"卵巢", r"子宫", r"巧克力囊肿", r"多囊",
]

# 男性专属疾病关键词（女性不应有）
MALE_ONLY_DISEASES = [
    r"前列腺",
]


def fix_personality(personal_history: str) -> tuple[str, bool]:
    """
    修复性格矛盾：同时包含"内向"和"外向" → 统一改为"内向"
    返回 (修复后文本, 是否有变化)
    """
    # 匹配"病前性格：XXX"中同时包含内向和外向的情况
    m = re.search(r"(病前性格：)([^，]+)", personal_history)
    if not m:
        return personal_history, False

    personality_value = m.group(2)

    # 检查是否同时包含内向和外向
    has_introvert = "内向" in personality_value
    has_extrovert = "外向" in personality_value

    if not (has_introvert and has_extrovert):
        # 只有"偏外向"这类也要处理
        if re.match(r"^偏?外向$", personality_value.strip()):
            # 纯外向不处理
            return personal_history, False
        return personal_history, False

    # 替换：同时有内向+外向时，统一改为纯"内向"（删除所有其他性格词）
    new_value = "内向"
    new_text = personal_history[:m.start(2)] + new_value + personal_history[m.end(2):]

    return new_text, new_text != personal_history


def fix_physical_history_gender(phys_history: str, gender: str) -> tuple[str, bool]:
    """
    修复性别-疾病矛盾：
    - 男性有女性专属疾病 → 删除该疾病
    - 女性有男性专属疾病 → 删除该疾病
    返回 (修复后文本, 是否有变化)
    """
    if not phys_history:
        return phys_history, False

    # 确定需要删除的疾病关键词
    if gender == "男":
        disease_patterns = FEMALE_ONLY_DISEASES
    elif gender == "女":
        disease_patterns = MALE_ONLY_DISEASES
    else:
        return phys_history, False

    # 检查是否包含冲突疾病
    has_conflict = any(re.search(p, phys_history) for p in disease_patterns)
    if not has_conflict:
        return phys_history, False

    # 格式1：重要或相关躯体疾病史：有（XXX，YYY）
    m = re.search(r"(重要或相关躯体疾病史：)有[（(](.+?)[)）]", phys_history)
    if m:
        prefix = m.group(1)
        diseases_text = m.group(2)

        # 如果整段内容主要是冲突疾病描述，直接置为"无"
        # 检查：去掉冲突关键词后，剩余是否有实质性疾病名称
        test_text = diseases_text
        for pat in disease_patterns:
            test_text = re.sub(pat + r"[^\s，,、）)]*", "", test_text)
        # 清理残留的非疾病文本
        test_clean = re.sub(r"(曾因|双侧|右侧|左侧|行|手术治疗|做过手术|术后|查出|综合征|"
                            r"建议药物治疗|但患者不愿服药|腹腔镜手术|手术一年|"
                            r"具体不详|等病史|妇幼保健院|\d+年?\s*\.?\d*)", "", test_text)
        test_clean = re.sub(r"[,，、\s（()）]+", "", test_clean)

        if not test_clean:
            # 没有其他实质性疾病 → 无
            new_text = f"{prefix}无"
        else:
            # 按顿号和逗号分割，保留非冲突疾病
            parts = re.split(r"[、，,]", diseases_text)
            kept = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if any(re.search(pat, p) for pat in disease_patterns):
                    continue
                kept.append(p)
            # 检查"等病史"是否应追加
            if "等病史" in diseases_text and kept:
                # 如果原文有"等病史"修饰，保留在最后一个元素后
                kept[-1] = kept[-1] + "等病史" if "等病史" not in kept[-1] else kept[-1]
            if kept:
                new_text = f"{prefix}有（{'，'.join(kept)}）"
            else:
                new_text = f"{prefix}无"
        return new_text, True

    # 格式2：重要或相关躯体疾病史：XXX、YYY（无括号，用顿号分隔）
    m2 = re.search(r"(重要或相关躯体疾病史：)(.+)", phys_history)
    if m2:
        prefix = m2.group(1)
        content = m2.group(2).strip()

        # 按顿号和逗号分割
        parts = re.split(r"[、，,]", content)
        kept = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if any(re.search(pat, p) for pat in disease_patterns):
                continue
            kept.append(p)

        if kept:
            new_text = f"{prefix}{'、'.join(kept)}"
        else:
            new_text = f"{prefix}无"
        return new_text, True

    return phys_history, False


def process(input_file: str, output_file: str, dry_run: bool = False):
    """处理数据"""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {
        "total": len(data),
        "personality_fixed": 0,
        "physical_history_fixed": 0,
        "records_changed": 0,
    }

    for record in data:
        changed = False
        gender = record.get("Gender", "")

        # 1. 修复性格矛盾
        ps = record.get("PersonalHistory", "")
        new_ps, ps_changed = fix_personality(ps)
        if ps_changed:
            if not dry_run:
                record["PersonalHistory"] = new_ps
            stats["personality_fixed"] += 1
            changed = True

        # 2. 修复性别-疾病矛盾
        phys = record.get("ImportantRelevantPhysicalIllnessHistory", "")
        new_phys, phys_changed = fix_physical_history_gender(phys, gender)
        if phys_changed:
            if not dry_run:
                record["ImportantRelevantPhysicalIllnessHistory"] = new_phys
            stats["physical_history_fixed"] += 1
            changed = True

        if changed:
            stats["records_changed"] += 1

    print(f"总记录: {stats['total']}")
    print(f"性格矛盾修复: {stats['personality_fixed']}")
    print(f"性别-疾病矛盾修复: {stats['physical_history_fixed']}")
    print(f"变更记录数: {stats['records_changed']}")

    if dry_run:
        print("\n[DRY RUN] 未写入文件")
    else:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n输出: {output_file}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="v5→v6: 性格矛盾+性别疾病矛盾修复")
    parser.add_argument("--input", required=True, help="输入JSON文件")
    parser.add_argument("--output", required=True, help="输出JSON文件")
    parser.add_argument("--dry-run", action="store_true", help="仅检测不修改")
    args = parser.parse_args()

    process(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()
