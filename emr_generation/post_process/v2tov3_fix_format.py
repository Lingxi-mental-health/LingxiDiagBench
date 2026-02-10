#!/usr/bin/env python3
"""
格式修复脚本 - 使用规则修复EMR数据中的格式问题

问题类型:
1. 家族史重复前缀: "家族史：家族史：" → "家族史："
2. PersonalHistory重复词: "内向,内向" → "内向"
3. AccompanyingPerson标准化: "自己" → "本人"
4. Age字段清理: "18岁" → "18"

使用方法:
    python fix_format.py                          # 执行修复
    python fix_format.py --dry-run                # 仅检测，不修改
    python fix_format.py --input xxx.json         # 指定输入文件
"""

import json
import re
import logging
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict


# ============== 配置 ==============
INPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory/LingxiDiag-16K_fixed_v2.json"
OUTPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory/LingxiDiag-16K_fixed_v3.json"
LOG_DIR = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/Format Restoration/log"


def setup_logging(dry_run: bool = False) -> str:
    """设置日志"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "dryrun" if dry_run else "fix"
    log_file = f"{LOG_DIR}/format_{mode}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file


class FormatFixer:
    """格式修复器"""

    def __init__(self):
        # 统计信息
        self.stats = defaultdict(int)
        self.changes = []  # 记录所有修改

    def fix_family_history(self, record: Dict) -> bool:
        """修复家族史重复前缀"""
        field = "FamilyHistory"
        if field not in record:
            return False

        original = record[field]
        if not original:
            return False

        # 修复重复前缀
        fixed = original
        patterns = [
            ("家族史：家族史：", "家族史："),
            ("家族史:家族史:", "家族史:"),
            ("家族史：家族史:", "家族史："),
            ("家族史:家族史：", "家族史："),
        ]

        for pattern, replacement in patterns:
            if pattern in fixed:
                fixed = fixed.replace(pattern, replacement)

        if fixed != original:
            record[field] = fixed
            self.stats["family_history_prefix"] += 1
            self.changes.append({
                "patient_id": record.get("patient_id", "unknown"),
                "field": field,
                "type": "duplicate_prefix",
                "original": original,
                "fixed": fixed
            })
            return True
        return False

    def fix_personal_history(self, record: Dict) -> bool:
        """修复PersonalHistory中的重复词"""
        field = "PersonalHistory"
        if field not in record:
            return False

        original = record[field]
        if not original:
            return False

        fixed = original

        # 修复连续重复的词 (用逗号或顿号分隔)
        # 例如: "内向,内向" → "内向", "内向、内向" → "内向"
        # 使用循环处理可能的多次重复
        prev = None
        while prev != fixed:
            prev = fixed
            # 匹配 "词,词" 或 "词、词" 模式
            fixed = re.sub(r'([^,、\s]{1,10})[,、]\1(?=[,、\s]|$)', r'\1', fixed)

        if fixed != original:
            record[field] = fixed
            self.stats["personal_history_duplicate"] += 1
            self.changes.append({
                "patient_id": record.get("patient_id", "unknown"),
                "field": field,
                "type": "duplicate_word",
                "original": original,
                "fixed": fixed
            })
            return True
        return False

    def fix_accompanying_person(self, record: Dict) -> bool:
        """标准化陪同人字段"""
        field = "AccompanyingPerson"
        if field not in record:
            return False

        original = record[field]
        if not original:
            return False

        fixed = original

        # 标准化 "自己" → "本人"
        if fixed == "自己":
            fixed = "本人"

        if fixed != original:
            record[field] = fixed
            self.stats["accompanying_person"] += 1
            self.changes.append({
                "patient_id": record.get("patient_id", "unknown"),
                "field": field,
                "type": "standardize",
                "original": original,
                "fixed": fixed
            })
            return True
        return False

    def fix_age(self, record: Dict) -> bool:
        """清理Age字段"""
        field = "Age"
        if field not in record:
            return False

        original = record[field]
        if not original:
            return False

        # 提取数字，去除 "岁" 等后缀
        fixed = re.sub(r'[岁\s]', '', str(original))

        if fixed != original:
            record[field] = fixed
            self.stats["age_cleanup"] += 1
            self.changes.append({
                "patient_id": record.get("patient_id", "unknown"),
                "field": field,
                "type": "cleanup",
                "original": original,
                "fixed": fixed
            })
            return True
        return False

    def fix_record(self, record: Dict) -> bool:
        """修复单条记录的所有格式问题"""
        modified = False
        modified |= self.fix_family_history(record)
        modified |= self.fix_personal_history(record)
        modified |= self.fix_accompanying_person(record)
        modified |= self.fix_age(record)
        return modified

    def fix_all(self, data: List[Dict]) -> List[Dict]:
        """修复所有记录"""
        for idx, record in enumerate(data):
            if self.fix_record(record):
                self.stats["records_modified"] += 1

        self.stats["total_records"] = len(data)
        return data

    def get_summary(self) -> str:
        """获取修复摘要"""
        lines = [
            "=" * 60,
            "格式修复摘要",
            "=" * 60,
            f"总记录数: {self.stats['total_records']}",
            f"修改记录数: {self.stats['records_modified']}",
            "-" * 60,
            "按问题类型统计:",
            f"  家族史重复前缀: {self.stats['family_history_prefix']}",
            f"  个人史重复词: {self.stats['personal_history_duplicate']}",
            f"  陪同人标准化: {self.stats['accompanying_person']}",
            f"  年龄字段清理: {self.stats['age_cleanup']}",
            "=" * 60,
        ]
        return "\n".join(lines)


def detect_only(data: List[Dict]) -> Dict[str, List]:
    """仅检测问题，不修改"""
    issues = defaultdict(list)

    for idx, record in enumerate(data):
        patient_id = record.get("patient_id", f"unknown_{idx}")

        # 检测家族史重复前缀
        fh = record.get("FamilyHistory", "")
        if fh and ("家族史：家族史：" in fh or "家族史:家族史:" in fh):
            issues["family_history_prefix"].append({
                "idx": idx,
                "patient_id": patient_id,
                "value": fh[:50]
            })

        # 检测PersonalHistory重复词
        ph = record.get("PersonalHistory", "")
        if ph and re.search(r'([^,、\s]{1,10})[,、]\1(?=[,、\s]|$)', ph):
            issues["personal_history_duplicate"].append({
                "idx": idx,
                "patient_id": patient_id,
                "value": ph[:80]
            })

        # 检测陪同人需要标准化
        ap = record.get("AccompanyingPerson", "")
        if ap == "自己":
            issues["accompanying_person"].append({
                "idx": idx,
                "patient_id": patient_id,
                "value": ap
            })

        # 检测Age需要清理
        age = record.get("Age", "")
        if age and re.search(r'[岁\s]', str(age)):
            issues["age_cleanup"].append({
                "idx": idx,
                "patient_id": patient_id,
                "value": age
            })

    return dict(issues)


def main():
    parser = argparse.ArgumentParser(description="格式修复脚本")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="输入文件路径")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="输出文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅检测，不修改")
    args = parser.parse_args()

    log_file = setup_logging(dry_run=args.dry_run)

    logging.info("=" * 60)
    logging.info("EMR 格式修复工具")
    logging.info(f"模式: {'检测' if args.dry_run else '修复'}")
    logging.info(f"输入: {args.input}")
    logging.info(f"输出: {args.output}")
    logging.info("=" * 60)

    # 加载数据
    logging.info("加载数据...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"总记录数: {len(data)}")

    if args.dry_run:
        # 仅检测模式
        logging.info("-" * 60)
        logging.info("检测格式问题...")
        issues = detect_only(data)

        logging.info("-" * 60)
        logging.info("检测结果:")
        total_issues = 0
        for issue_type, records in issues.items():
            count = len(records)
            total_issues += count
            logging.info(f"  {issue_type}: {count} 条")
            # 显示前3个示例
            for r in records[:3]:
                logging.info(f"    idx={r['idx']} | {r['patient_id']} | {r['value'][:40]}...")

        logging.info("-" * 60)
        logging.info(f"总计发现 {total_issues} 个格式问题")

        # 保存检测报告
        report_file = log_file.replace('.log', '_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(issues, f, ensure_ascii=False, indent=2)
        logging.info(f"检测报告: {report_file}")

    else:
        # 修复模式
        logging.info("-" * 60)
        logging.info("执行格式修复...")

        fixer = FormatFixer()
        data = fixer.fix_all(data)

        # 输出摘要
        logging.info(fixer.get_summary())

        # 保存修复后的数据
        logging.info(f"保存到: {args.output}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 保存修改详情
        if fixer.changes:
            changes_file = log_file.replace('.log', '_changes.json')
            with open(changes_file, 'w', encoding='utf-8') as f:
                json.dump(fixer.changes, f, ensure_ascii=False, indent=2)
            logging.info(f"修改详情: {changes_file}")

    logging.info(f"日志文件: {log_file}")
    logging.info("完成!")


if __name__ == "__main__":
    main()
