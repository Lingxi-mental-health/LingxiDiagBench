#!/usr/bin/env python3
"""
迭代修复 runner：自动运行 fix → merge → detect 循环
始终从原始数据出发修复，不在中间版本上累积
"""

import json
import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FIX_SCRIPT = os.path.join(SCRIPT_DIR, "fix_with_anchors.py")
DETECT_SCRIPT = os.path.join(PROJECT_DIR, "Detect_inconsistencies",
                              "detect_unreasonableness.py")

# 不可修复的问题类型（与 fix_with_anchors.py 保持一致）
UNFIXABLE_SF_TYPES = {
    "诊断编码不匹配", "对话不自然", "对话重复",
    "对话自相矛盾", "对话模板化", "对话格式错误",
}
UNFIXABLE_CF_TYPES = {"诊断编码不匹配"}  # 诊断症状矛盾中有可修复案例(诊断↔现病史)

logger = logging.getLogger("iterative")


def setup_logging(output_dir: str):
    """配置日志"""
    log_file = os.path.join(output_dir, "iterative.log")
    formatter = logging.Formatter("%(asctime)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def rebuild_patient_info(record: dict) -> str:
    """重建 Patient info 字段"""
    parts = []
    ps = record.get("PersonalHistory", "")
    if ps:
        parts.append(ps)
    cc = record.get("ChiefComplaint", "")
    if cc:
        parts.append(f"主诉:{cc}")
    phi = record.get("PresentIllnessHistory", "")
    if phi:
        parts.append(f"现病史:{phi}")
    fh = record.get("FamilyHistory", "")
    if fh:
        parts.append(f"家族史:{fh}")
    return "|".join(parts)


def merge_fix_results(original_file: str, fix_results_file: str,
                       output_file: str) -> dict:
    """
    将修复结果合并到原始数据（始终从原始数据合并，不累积）
    返回合并统计
    """
    with open(original_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 加载修复结果
    fixes = {}
    with open(fix_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("status") == "fixed" and r.get("num_changes", 0) > 0:
                    fixes[r["patient_id"]] = r["fixed_fields"]

    # 应用修复
    n_applied = 0
    for record in data:
        pid = record.get("patient_id")
        if pid in fixes:
            for field, value in fixes[pid].items():
                if value:
                    record[field] = value
            n_applied += 1
        # 始终重建 Patient info（确保一致性）
        record["Patient info"] = rebuild_patient_info(record)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"total": len(data), "applied": n_applied}


def count_fixable_issues(summary_file: str) -> int:
    """统计可修复问题数量"""
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    sf_count = 0
    for t, c in summary.get("single_field", {}).get("type_distribution", {}).items():
        if t not in UNFIXABLE_SF_TYPES:
            sf_count += c

    cf_count = 0
    for t, c in summary.get("cross_field", {}).get("type_distribution", {}).items():
        if t not in UNFIXABLE_CF_TYPES:
            cf_count += c

    return sf_count + cf_count


def find_results_jsonl(detect_dir: str) -> str:
    """在检测输出目录下找到 results.jsonl"""
    # 检测脚本会在 detect_dir 下创建时间戳子目录
    for entry in sorted(os.listdir(detect_dir), reverse=True):
        candidate = os.path.join(detect_dir, entry, "results.jsonl")
        if os.path.exists(candidate):
            return candidate
    # 直接在目录下
    direct = os.path.join(detect_dir, "results.jsonl")
    if os.path.exists(direct):
        return direct
    return None


def find_summary_json(detect_dir: str) -> str:
    """在检测输出目录下找到 summary.json"""
    for entry in sorted(os.listdir(detect_dir), reverse=True):
        candidate = os.path.join(detect_dir, entry, "summary.json")
        if os.path.exists(candidate):
            return candidate
    direct = os.path.join(detect_dir, "summary.json")
    if os.path.exists(direct):
        return direct
    return None


def main():
    parser = argparse.ArgumentParser(description="迭代修复 runner")
    parser.add_argument("--input", type=str, required=True,
                        help="原始数据文件（始终从此出发修复）")
    parser.add_argument("--detection", type=str, required=True,
                        help="初始检测结果文件")
    parser.add_argument("--max-rounds", type=int, default=4,
                        help="最大迭代轮数 (默认 4)")
    parser.add_argument("--concurrent", type=int, default=16,
                        help="并发数 (默认 16)")
    parser.add_argument("--limit", type=int, default=0,
                        help="每轮只修复前 N 条 (默认 0=全量)")
    parser.add_argument("--min-improvement", type=float, default=5.0,
                        help="最小改善百分比，低于此值停止 (默认 5.0)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认自动生成)")
    args = parser.parse_args()

    # 输出目录
    if args.output_dir:
        base_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        base_dir = os.path.join(SCRIPT_DIR, "output", f"iter_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    setup_logging(base_dir)

    logger.info("=" * 60)
    logger.info("迭代修复 Runner")
    logger.info("=" * 60)
    logger.info(f"原始数据: {args.input}")
    logger.info(f"初始检测: {args.detection}")
    logger.info(f"最大轮数: {args.max_rounds}")
    logger.info(f"并发数: {args.concurrent}")
    logger.info(f"输出目录: {base_dir}")
    if args.limit:
        logger.info(f"每轮限制: {args.limit} 条")

    issue_history = []
    prev_fix_file = None
    detection_file = args.detection

    for round_num in range(1, args.max_rounds + 1):
        round_dir = os.path.join(base_dir, f"round_{round_num}")
        fix_dir = os.path.join(round_dir, "fix")
        detect_dir = os.path.join(round_dir, "detect")
        os.makedirs(fix_dir, exist_ok=True)
        os.makedirs(detect_dir, exist_ok=True)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Round {round_num}")
        logger.info("=" * 60)

        # Step 1: Fix
        logger.info(f"[R{round_num}] Step 1: 修复...")
        fix_cmd = [
            sys.executable, FIX_SCRIPT,
            "--input", args.input,
            "--detection", detection_file,
            "--round", str(round_num),
            "--no-think",
            "--concurrent", str(args.concurrent),
            "--output-dir", fix_dir,
        ]
        if prev_fix_file:
            fix_cmd.extend(["--prev-fix", prev_fix_file])
        if args.limit:
            fix_cmd.extend(["--limit", str(args.limit)])

        logger.info(f"  命令: {' '.join(fix_cmd)}")
        result = subprocess.run(fix_cmd, capture_output=False)
        if result.returncode != 0:
            logger.error(f"  修复失败! 返回码: {result.returncode}")
            break

        fix_results_file = os.path.join(fix_dir, "fix_results.jsonl")
        if not os.path.exists(fix_results_file):
            logger.error(f"  修复结果文件不存在: {fix_results_file}")
            break

        # Step 2: Merge（始终从原始数据合并）
        merged_file = os.path.join(round_dir, f"merged_round_{round_num}.json")
        logger.info(f"[R{round_num}] Step 2: 合并到原始数据...")
        merge_stats = merge_fix_results(args.input, fix_results_file, merged_file)
        logger.info(f"  合并: {merge_stats['applied']}/{merge_stats['total']} 条应用修复")

        # Step 3: Detect
        logger.info(f"[R{round_num}] Step 3: 检测修复后数据...")
        detect_cmd = [
            sys.executable, DETECT_SCRIPT,
            "--input", merged_file,
            "--output-dir", detect_dir,
            "--concurrent", str(args.concurrent),
        ]
        logger.info(f"  命令: {' '.join(detect_cmd)}")
        result = subprocess.run(detect_cmd, capture_output=False)
        if result.returncode != 0:
            logger.error(f"  检测失败! 返回码: {result.returncode}")
            break

        # Step 4: 评估收敛
        summary_file = find_summary_json(detect_dir)
        if not summary_file:
            logger.error(f"  找不到检测 summary.json")
            break

        fixable = count_fixable_issues(summary_file)
        issue_history.append(fixable)
        logger.info(f"[R{round_num}] 可修复问题数: {fixable}")
        logger.info(f"  历史: {issue_history}")

        # 收敛判断
        if len(issue_history) >= 2:
            prev = issue_history[-2]
            curr = issue_history[-1]
            if curr >= prev:
                logger.info(f"  停止: 问题未减少 ({prev} -> {curr})")
                break
            improvement = (prev - curr) / prev * 100
            logger.info(f"  改善: {prev} -> {curr} (-{improvement:.1f}%)")
            if improvement < args.min_improvement:
                logger.info(f"  停止: 改善 {improvement:.1f}% < {args.min_improvement}% 阈值")
                break

        # 为下一轮准备
        new_results = find_results_jsonl(detect_dir)
        if new_results:
            detection_file = new_results
        else:
            logger.error(f"  找不到检测 results.jsonl")
            break
        prev_fix_file = fix_results_file

    # 最终报告
    report = {
        "total_rounds": len(issue_history),
        "issue_history": issue_history,
        "original_input": args.input,
        "initial_detection": args.detection,
        "output_dir": base_dir,
    }
    if len(issue_history) >= 2:
        report["total_improvement"] = issue_history[0] - issue_history[-1]
        report["total_improvement_pct"] = round(
            (issue_history[0] - issue_history[-1]) / issue_history[0] * 100, 1
        ) if issue_history[0] > 0 else 0

    report_file = os.path.join(base_dir, "iteration_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("")
    logger.info("=" * 60)
    logger.info("迭代修复完成!")
    logger.info("=" * 60)
    logger.info(f"总轮数: {len(issue_history)}")
    logger.info(f"问题历史: {issue_history}")
    if len(issue_history) >= 2:
        logger.info(f"总改善: {issue_history[0]} -> {issue_history[-1]} "
                    f"(-{report.get('total_improvement_pct', 0)}%)")
    logger.info(f"报告: {report_file}")


if __name__ == "__main__":
    main()
