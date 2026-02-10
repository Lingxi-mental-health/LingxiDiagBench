#!/usr/bin/env python3
"""
LingxiDiag-16K 数据清洗全流程串联脚本

版本链：v0 → v1 → v2 → v3 → v4 → v5 → v6 → v7

用法：
  # 从 v0 跑完整流程
  python run_pipeline.py --v0-input /path/to/v0_raw.json --output-dir ./pipeline_output

  # 从某个中间版本开始
  python run_pipeline.py --start-from v3 --v3-input /path/to/v3.json --output-dir ./pipeline_output

  # 只跑某一步
  python run_pipeline.py --only v5tov6 --input /path/to/v5.json --output-dir ./pipeline_output

注意：
  - v0→v1, v3→v4, v4→v5, v6→v7 需要 vLLM 服务运行（Qwen3-32B at localhost:8000）
  - v2→v3, v5→v6 是纯规则修复，不需要 LLM
  - v1→v2 是统计+随机填充，不需要 LLM
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
VLLM_URL = "http://localhost:8000/v1/models"


def check_vllm_service():
    """检查 vLLM 服务是否可用，返回 True/False"""
    try:
        req = urllib.request.Request(VLLM_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            print(f"[OK] vLLM 服务可用，模型: {', '.join(models)}")
            return True
    except Exception as e:
        return False

# 每一步的脚本和参数
STEPS = {
    "v0_check": {
        "desc": "v0 规则质检",
        "script": "v0_quality_check.py",
        "needs_llm": False,
        "notes": "仅检测，不修改数据",
    },
    "v0tov1": {
        "desc": "v0→v1 补充现病史",
        "scripts": ["v0tov1_complete_history.py", "v0tov1_fix_all_problems.py"],
        "needs_llm": True,
    },
    "v1tov2": {
        "desc": "v1→v2 男性孕产补充",
        "script": "v1tov2_fill_pregnancy_for_male.py",
        "needs_llm": False,
    },
    "v2tov3": {
        "desc": "v2→v3 格式规则修复",
        "script": "v2tov3_fix_format.py",
        "needs_llm": False,
    },
    "v3tov4": {
        "desc": "v3→v4 对话去重 + 躯体疾病补充",
        "scripts": ["v3tov4_fix_dialogue_duplicates.py", "v3tov4_fix_physical_history.py"],
        "needs_llm": True,
    },
    "v4tov5": {
        "desc": "v4→v5 跨字段检测+修复",
        "scripts": ["v4tov5_detect_inconsistency.py", "v4tov5_fix_inconsistency.py"],
        "needs_llm": True,
    },
    "v5tov6": {
        "desc": "v5→v6 性格矛盾+性别疾病修复",
        "script": "v5tov6_fix_personality_gender.py",
        "needs_llm": False,
    },
    "v6tov7": {
        "desc": "v6→v7 检测+锚定修复+合并",
        "scripts": [
            "v6tov7_detect_unreasonableness.py",
            "v6tov7_fix_with_anchors.py",
            "v6tov7_merge_results.py",
        ],
        "needs_llm": True,
    },
}

STEP_ORDER = ["v0_check", "v0tov1", "v1tov2", "v2tov3", "v3tov4", "v4tov5", "v5tov6", "v6tov7"]


def run_step(step_name: str, input_file: str, output_dir: str,
             concurrent: int = 32, validate: bool = True):
    """运行单个步骤"""
    step = STEPS[step_name]
    step_dir = os.path.join(output_dir, step_name)
    os.makedirs(step_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {step['desc']}")
    print(f"  输入: {input_file}")
    print(f"  输出目录: {step_dir}")
    print(f"{'='*60}")

    output_file = os.path.join(step_dir, f"{step_name}_output.json")

    if step_name == "v0_check":
        # 只做检测，不修改
        print("[INFO] v0 质检仅检测，输出报告到步骤目录")
        print("[INFO] 原始数据直接传递到下一步")
        output_file = input_file  # pass through

    elif step_name == "v1tov2":
        cmd = [
            "python3", str(SCRIPT_DIR / step["script"]),
            "--input", input_file,
            "--output", output_file,
        ]
        subprocess.run(cmd, check=True)

    elif step_name == "v2tov3":
        cmd = [
            "python3", str(SCRIPT_DIR / step["script"]),
            "--input", input_file,
            "--output", output_file,
        ]
        subprocess.run(cmd, check=True)

    elif step_name == "v3tov4":
        # 先对话去重
        dialogue_output = os.path.join(step_dir, "dialogue_fixed.json")
        cmd = [
            "python3", str(SCRIPT_DIR / "v3tov4_fix_dialogue_duplicates.py"),
            "--input", input_file,
            "--output", dialogue_output,
            "--workers", str(concurrent),
        ]
        subprocess.run(cmd, check=True)

        # 再躯体疾病补充
        cmd = [
            "python3", str(SCRIPT_DIR / "v3tov4_fix_physical_history.py"),
            "--input", dialogue_output,
            "--output", output_file,
            "--workers", str(concurrent),
        ]
        subprocess.run(cmd, check=True)

    elif step_name == "v4tov5":
        # 先检测
        detect_output = os.path.join(step_dir, "detect_results.jsonl")
        cmd = [
            "python3", str(SCRIPT_DIR / "v4tov5_detect_inconsistency.py"),
            "--input", input_file,
            "--output", detect_output,
        ]
        subprocess.run(cmd, check=True)

        # 再修复
        cmd = [
            "python3", str(SCRIPT_DIR / "v4tov5_fix_inconsistency.py"),
            "--input", input_file,
            "--output", output_file,
        ]
        subprocess.run(cmd, check=True)

    elif step_name == "v5tov6":
        cmd = [
            "python3", str(SCRIPT_DIR / step["script"]),
            "--input", input_file,
            "--output", output_file,
        ]
        subprocess.run(cmd, check=True)

    elif step_name == "v6tov7":
        # 1. 检测
        detect_dir = os.path.join(step_dir, "detect")
        cmd = [
            "python3", str(SCRIPT_DIR / "v6tov7_detect_unreasonableness.py"),
            "--input", input_file,
            "--output-dir", detect_dir,
            "--concurrent", str(concurrent),
        ]
        subprocess.run(cmd, check=True)

        # 找到 results.jsonl
        results_jsonl = os.path.join(detect_dir, "results.jsonl")
        if not os.path.exists(results_jsonl):
            # 可能在时间戳子目录下
            for d in sorted(os.listdir(detect_dir), reverse=True):
                candidate = os.path.join(detect_dir, d, "results.jsonl")
                if os.path.exists(candidate):
                    results_jsonl = candidate
                    break

        # 2. 修复
        fix_dir = os.path.join(step_dir, "fix")
        cmd = [
            "python3", str(SCRIPT_DIR / "v6tov7_fix_with_anchors.py"),
            "--input", input_file,
            "--detection", results_jsonl,
            "--output-dir", fix_dir,
            "--concurrent", str(concurrent),
            "--no-think",
        ]
        subprocess.run(cmd, check=True)

        fix_results = os.path.join(fix_dir, "fix_results.jsonl")
        if not os.path.exists(fix_results):
            for d in sorted(os.listdir(fix_dir), reverse=True):
                candidate = os.path.join(fix_dir, d, "fix_results.jsonl")
                if os.path.exists(candidate):
                    fix_results = candidate
                    break

        # 3. 合并
        cmd = [
            "python3", str(SCRIPT_DIR / "v6tov7_merge_results.py"),
            "--original", input_file,
            "--fix-results", fix_results,
            "--output", output_file,
        ]
        subprocess.run(cmd, check=True)

    else:
        print(f"[WARN] {step_name} 需要手动执行，暂不支持自动化")
        output_file = input_file

    # 验证
    if validate and os.path.exists(output_file) and output_file != input_file:
        print(f"\n--- 验证 {step_name} 输出 ---")
        cmd = [
            "python3", str(SCRIPT_DIR / "validate_step.py"),
            "--input", output_file,
        ]
        subprocess.run(cmd)

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="LingxiDiag-16K 数据清洗全流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 全流程
  python run_pipeline.py --v0-input raw.json --output-dir ./output

  # 从 v5 开始
  python run_pipeline.py --start-from v5tov6 --input v5.json --output-dir ./output

  # 只跑 v6→v7
  python run_pipeline.py --only v6tov7 --input v6.json --output-dir ./output
        """,
    )
    parser.add_argument("--v0-input", help="v0 原始数据文件")
    parser.add_argument("--input", help="指定起始输入文件（配合 --start-from 或 --only）")
    parser.add_argument("--output-dir", required=True, help="输出根目录")
    parser.add_argument("--start-from", choices=STEP_ORDER, help="从某一步开始")
    parser.add_argument("--only", choices=STEP_ORDER, help="只跑某一步")
    parser.add_argument("--concurrent", type=int, default=32, help="LLM 并发数")
    parser.add_argument("--no-validate", action="store_true", help="跳过步骤间验证")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 确定要跑的步骤
    if args.only:
        steps_to_run = [args.only]
        current_input = args.input
        if not current_input:
            print("使用 --only 时必须指定 --input")
            sys.exit(1)
    elif args.start_from:
        idx = STEP_ORDER.index(args.start_from)
        steps_to_run = STEP_ORDER[idx:]
        current_input = args.input
        if not current_input:
            print("使用 --start-from 时必须指定 --input")
            sys.exit(1)
    else:
        steps_to_run = STEP_ORDER
        current_input = args.v0_input
        if not current_input:
            print("全流程运行需要指定 --v0-input")
            sys.exit(1)

    print(f"将运行以下步骤: {' → '.join(steps_to_run)}")
    print(f"起始输入: {current_input}")
    print(f"输出目录: {args.output_dir}")
    print()

    # 检查是否需要 vLLM 服务
    needs_llm = any(STEPS[s].get("needs_llm") for s in steps_to_run)
    if needs_llm:
        llm_steps = [s for s in steps_to_run if STEPS[s].get("needs_llm")]
        print(f"以下步骤需要 vLLM 服务: {', '.join(llm_steps)}")
        if not check_vllm_service():
            print(f"\n[ERROR] vLLM 服务不可用 ({VLLM_URL})")
            print("请先启动 vLLM：")
            print("  vllm serve Qwen/Qwen3-32B \\")
            print("    --tensor-parallel-size 8 \\")
            print("    --max-model-len 16384 \\")
            print("    --port 8000")
            sys.exit(1)
    else:
        print("所有步骤均为纯规则，不需要 vLLM 服务")

    print()

    for step in steps_to_run:
        current_input = run_step(
            step, current_input, args.output_dir,
            concurrent=args.concurrent,
            validate=not args.no_validate,
        )

    print(f"\n{'='*60}")
    print(f"  流程完成！最终输出: {current_input}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
