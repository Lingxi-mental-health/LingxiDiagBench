#!/usr/bin/env python3
"""
现病史完整性检测与补充工具（大模型版 - 32并发）

使用 Qwen32B 检测现病史是否为空或被截断，并从对话记录中补充。

使用方法:
    python complete_history.py                # 处理全部
    python complete_history.py --detect-only  # 仅检测
    python complete_history.py --test 2       # 测试单条
    python complete_history.py --limit 100    # 限制数量
    python complete_history.py --workers 16   # 自定义并发数
    python complete_history.py --retry-errors # 重跑失败的记录
"""

import json
import argparse
import re
import logging
import os
import threading
from datetime import datetime
from typing import Tuple, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


# ============== 配置 ==============
API_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-32b"
MAX_WORKERS = 32  # 并发数

DATA_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory/LingxiDiag-16K_all_data.json"
OUTPUT_DIR = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory"
LOG_DIR = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory/log"

# 线程锁
stats_lock = threading.Lock()
log_lock = threading.Lock()
file_lock = threading.Lock()

# 实时问题记录文件
REALTIME_PROBLEMS_FILE = None


def strip_thinking(text: str) -> str:
    """移除 Qwen3 thinking 模式产生的 <think>...</think> 内容"""
    if not text:
        return text
    # 移除 <think>...</think> 标签及其内容
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 如果只有 <think> 没有闭合标签，也移除
    result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)
    return result.strip()


def setup_logging():
    """设置日志"""
    global REALTIME_PROBLEMS_FILE
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/complete_history_{timestamp}.log"

    # 实时问题记录文件（JSONL格式，每行一个JSON）
    REALTIME_PROBLEMS_FILE = f"{OUTPUT_DIR}/problems_realtime_{timestamp}.jsonl"

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


def write_problem_realtime(problem_record: Dict):
    """实时写入问题记录到 JSONL 文件"""
    if REALTIME_PROBLEMS_FILE is None:
        return
    with file_lock:
        with open(REALTIME_PROBLEMS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(problem_record, ensure_ascii=False) + '\n')


def get_client() -> OpenAI:
    """创建 OpenAI 客户端"""
    return OpenAI(
        base_url=API_BASE_URL,
        api_key="not-needed",
    )


def detect_problem(client: OpenAI, history: str) -> Tuple[bool, str, str]:
    """使用大模型判断现病史是否有问题"""
    if not history or history.strip() in ["", "现病史：", "现病史:"]:
        return True, "empty", "现病史为空"

    system_prompt = """你是医学文本质量检测专家，判断现病史是否完整。

判断标准：
1. empty（空缺）：内容为空或几乎没有实质内容
2. truncated（截断）：句子明显不完整，内容被中途截断，如以"及"、"和"、"，"等结尾
3. ok（完整）：内容完整，有明确的描述

只需要判断是否为空或被截断，不需要判断内容是否详细。"""

    user_prompt = f"""判断以下现病史是否存在问题（空缺或被截断）：

{history}

请仔细分析后，直接回复JSON：{{"has_problem": true/false, "reason": "empty"/"truncated"/"ok", "explanation": "说明"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512,
            temperature=0.1,
        )

        result_text = response.choices[0].message.content.strip()

        # 移除 thinking 内容
        result_text = strip_thinking(result_text)

        # 提取JSON结果
        json_match = re.search(r'\{[^{}]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("has_problem", False), result.get("reason", "unknown"), result.get("explanation", "")
        else:
            if "截断" in result_text or "不完整" in result_text:
                return True, "truncated", result_text[:100]
            elif "空" in result_text:
                return True, "empty", result_text[:100]
            return False, "ok", result_text[:100]

    except Exception as e:
        return False, "error", f"检测失败: {str(e)}"


def complete_history(client: OpenAI, patient_data: Dict) -> str:
    """根据对话记录补充现病史"""
    cleaned_text = patient_data.get("cleaned_text", "")
    if not cleaned_text:
        return patient_data.get("PresentIllnessHistory", "")

    original_history = patient_data.get("PresentIllnessHistory", "")
    chief_complaint = patient_data.get("ChiefComplaint", "")
    diagnosis = patient_data.get("Diagnosis", "")
    age = patient_data.get("Age", "")
    gender = patient_data.get("Gender", "")

    system_prompt = """你是专业的精神科医生，根据医患对话记录撰写规范的现病史。

现病史包含：起病时间/诱因、主要症状、症状演变、伴随症状、既往诊治、目前状态。
要求：语言专业简洁，以"现病史："开头，从对话中提取信息不要编造。"""

    user_prompt = f"""根据医患对话撰写完整现病史：

【患者】{age}岁{gender}
【主诉】{chief_complaint}
【诊断】{diagnosis}
【原现病史】{original_history}

【医患对话】
{cleaned_text}

请输出完整的现病史："""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024,
            temperature=0.3,
        )

        completed = response.choices[0].message.content.strip()
        # 移除 thinking 内容
        completed = strip_thinking(completed)
        if not completed.startswith("现病史"):
            completed = "现病史：" + completed
        return completed

    except Exception as e:
        return original_history


def process_single(args: Tuple[int, Dict, OpenAI, bool]) -> Dict:
    """处理单条记录（供线程池调用）"""
    idx, patient, client, detect_only = args
    patient_id = patient.get("patient_id", f"unknown_{idx}")
    history = patient.get("PresentIllnessHistory", "")

    # 检测
    has_problem, reason, explanation = detect_problem(client, history)

    result = patient.copy()
    result["_idx"] = idx  # 保存原始索引用于排序
    result["detection"] = {"has_problem": has_problem, "reason": reason, "explanation": explanation}

    # 如果有问题
    if has_problem:
        history_preview = history[:100] + "..." if len(history) > 100 else history
        history_preview = history_preview.replace("\n", " ")

        with log_lock:
            logging.info(f"[问题] idx={idx} | patient_id={patient_id} | 类型={reason}")
            logging.info(f"       原因: {explanation}")
            logging.info(f"       内容: {history_preview}")

        # 构建问题记录
        problem_record = {
            "idx": idx,
            "patient_id": patient_id,
            "reason": reason,
            "explanation": explanation,
            "original": history,
            "completed": None
        }

        result["problematic_info"] = {
            "idx": idx,
            "patient_id": patient_id,
            "reason": reason,
            "explanation": explanation,
            "original": history[:200]
        }

        if not detect_only:
            result["PresentIllnessHistory_original"] = history
            completed = complete_history(client, patient)
            result["PresentIllnessHistory"] = completed
            result["history_completed"] = True
            problem_record["completed"] = completed

            completed_preview = completed[:150] + "..." if len(completed) > 150 else completed
            completed_preview = completed_preview.replace("\n", " ")
            with log_lock:
                logging.info(f"       补充: {completed_preview}")
        else:
            result["history_completed"] = False

        # 实时写入问题记录
        write_problem_realtime(problem_record)
    else:
        result["history_completed"] = False

    return result


def process_data(detect_only: bool = False, test_idx: Optional[int] = None,
                 limit: Optional[int] = None, max_workers: int = MAX_WORKERS):
    """处理数据（并发版）"""
    log_file = setup_logging()

    logging.info("=" * 60)
    logging.info("现病史检测与补充工具（大模型版 - 并发）")
    logging.info(f"并发数: {max_workers}")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"实时问题记录: {REALTIME_PROBLEMS_FILE}")
    logging.info("=" * 60)

    # 加载数据
    logging.info(f"加载数据: {DATA_FILE}")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"总记录数: {len(data)}")

    # 测试单条
    if test_idx is not None:
        if test_idx >= len(data):
            logging.error(f"索引超出范围")
            return
        data = [data[test_idx]]
        logging.info(f"测试记录索引: {test_idx}")
        max_workers = 1

    if limit:
        data = data[:limit]
        logging.info(f"限制处理: {limit}条")

    # 创建客户端
    logging.info("连接 Qwen32B API...")
    client = get_client()

    # 统计
    stats = {"total": len(data), "empty": 0, "truncated": 0, "ok": 0, "completed": 0, "error": 0}
    results = [None] * len(data)  # 预分配结果列表保持顺序
    problematic = []

    logging.info("-" * 60)
    logging.info(f"开始并发处理 (workers={max_workers})...")

    # 准备任务参数
    tasks = [(idx, patient, client, detect_only) for idx, patient in enumerate(data)]

    # 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, task): task[0] for task in tasks}

        with tqdm(total=len(data), desc="处理中") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    idx = result.pop("_idx")
                    results[idx] = result

                    # 更新统计
                    reason = result["detection"]["reason"]
                    with stats_lock:
                        if reason in stats:
                            stats[reason] += 1
                        if result.get("history_completed"):
                            stats["completed"] += 1
                        if result.get("problematic_info"):
                            problematic.append(result.pop("problematic_info"))

                except Exception as e:
                    logging.error(f"处理失败: {e}")
                    with stats_lock:
                        stats["error"] += 1

                pbar.update(1)

                # 每500条输出进度
                if pbar.n % 500 == 0:
                    with stats_lock:
                        logging.info(f"--- 进度: {pbar.n}/{len(data)} | 问题: {stats['empty'] + stats['truncated']} ---")

    # 最终统计
    logging.info("=" * 60)
    logging.info("检测完成!")
    logging.info(f"  总记录: {stats['total']}")
    logging.info(f"  空缺: {stats['empty']}")
    logging.info(f"  截断: {stats['truncated']}")
    logging.info(f"  正常: {stats['ok']}")
    logging.info(f"  错误: {stats['error']}")
    if stats['total'] > 0:
        logging.info(f"  问题占比: {(stats['empty'] + stats['truncated']) / stats['total'] * 100:.2f}%")
    if not detect_only:
        logging.info(f"  已补充: {stats['completed']}")

    # 保存
    if test_idx is None:
        # 按索引排序 problematic
        problematic.sort(key=lambda x: x["idx"])

        detection_file = f"{OUTPUT_DIR}/detection_results.json"
        with open(detection_file, 'w', encoding='utf-8') as f:
            json.dump({"stats": stats, "problematic": problematic}, f, ensure_ascii=False, indent=2)
        logging.info(f"检测结果: {detection_file}")

        if not detect_only:
            output_file = f"{OUTPUT_DIR}/completed_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logging.info(f"补充数据: {output_file}")

    logging.info(f"日志已保存: {log_file}")


def retry_errors(max_workers: int = MAX_WORKERS, detect_only: bool = False):
    """重跑失败的记录"""
    log_file = setup_logging()

    logging.info("=" * 60)
    logging.info("重跑失败记录模式")
    logging.info(f"并发数: {max_workers}")
    logging.info("=" * 60)

    # 加载已处理的数据
    completed_file = f"{OUTPUT_DIR}/completed_data.json"
    if not os.path.exists(completed_file):
        logging.error(f"找不到已处理数据文件: {completed_file}")
        return

    logging.info(f"加载已处理数据: {completed_file}")
    with open(completed_file, 'r', encoding='utf-8') as f:
        completed_data = json.load(f)

    # 找出失败的记录
    error_indices = []
    for idx, record in enumerate(completed_data):
        if record and record.get("detection", {}).get("reason") == "error":
            error_indices.append(idx)

    logging.info(f"总记录数: {len(completed_data)}")
    logging.info(f"失败记录数: {len(error_indices)}")

    if not error_indices:
        logging.info("没有需要重跑的失败记录")
        return

    # 加载原始数据获取完整信息
    logging.info(f"加载原始数据: {DATA_FILE}")
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 创建客户端
    logging.info("连接 Qwen32B API...")
    client = get_client()

    # 统计
    stats = {"total": len(error_indices), "empty": 0, "truncated": 0, "ok": 0, "completed": 0, "error": 0}
    problematic = []

    logging.info("-" * 60)
    logging.info(f"开始重跑 {len(error_indices)} 条失败记录...")

    # 准备任务 - 使用原始数据但保留原始索引
    tasks = [(idx, original_data[idx], client, detect_only) for idx in error_indices]

    # 并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single, task): task[0] for task in tasks}

        with tqdm(total=len(error_indices), desc="重跑中") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    idx = result.pop("_idx")
                    # 更新到 completed_data
                    completed_data[idx] = result

                    # 更新统计
                    reason = result["detection"]["reason"]
                    with stats_lock:
                        if reason in stats:
                            stats[reason] += 1
                        if result.get("history_completed"):
                            stats["completed"] += 1
                        if result.get("problematic_info"):
                            problematic.append(result.pop("problematic_info"))

                except Exception as e:
                    logging.error(f"处理失败: {e}")
                    with stats_lock:
                        stats["error"] += 1

                pbar.update(1)

    # 最终统计
    logging.info("=" * 60)
    logging.info("重跑完成!")
    logging.info(f"  处理记录: {stats['total']}")
    logging.info(f"  空缺: {stats['empty']}")
    logging.info(f"  截断: {stats['truncated']}")
    logging.info(f"  正常: {stats['ok']}")
    logging.info(f"  仍失败: {stats['error']}")
    if not detect_only:
        logging.info(f"  已补充: {stats['completed']}")

    # 保存更新后的数据
    logging.info(f"保存更新后的数据: {completed_file}")
    with open(completed_file, 'w', encoding='utf-8') as f:
        json.dump(completed_data, f, ensure_ascii=False, indent=2)

    # 更新检测结果
    detection_file = f"{OUTPUT_DIR}/detection_results.json"
    # 重新统计
    final_stats = {"total": len(completed_data), "empty": 0, "truncated": 0, "ok": 0, "completed": 0, "error": 0}
    all_problematic = []
    for idx, record in enumerate(completed_data):
        if record:
            reason = record.get("detection", {}).get("reason", "unknown")
            if reason in final_stats:
                final_stats[reason] += 1
            if record.get("history_completed"):
                final_stats["completed"] += 1
            if record.get("detection", {}).get("has_problem"):
                all_problematic.append({
                    "idx": idx,
                    "patient_id": record.get("patient_id"),
                    "reason": reason,
                    "explanation": record.get("detection", {}).get("explanation", ""),
                    "original": record.get("PresentIllnessHistory_original", record.get("PresentIllnessHistory", ""))[:200]
                })

    with open(detection_file, 'w', encoding='utf-8') as f:
        json.dump({"stats": final_stats, "problematic": all_problematic}, f, ensure_ascii=False, indent=2)
    logging.info(f"检测结果已更新: {detection_file}")
    logging.info(f"日志已保存: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="现病史检测与补充（大模型版 - 并发）")
    parser.add_argument("--detect-only", action="store_true", help="仅检测，不补充")
    parser.add_argument("--test", type=int, help="测试单条记录索引")
    parser.add_argument("--limit", type=int, help="限制处理数量")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"并发数 (默认: {MAX_WORKERS})")
    parser.add_argument("--retry-errors", action="store_true", help="重跑失败的记录")

    args = parser.parse_args()

    if args.retry_errors:
        retry_errors(max_workers=args.workers, detect_only=args.detect_only)
    else:
        process_data(
            detect_only=args.detect_only,
            test_idx=args.test,
            limit=args.limit,
            max_workers=args.workers
        )


if __name__ == "__main__":
    main()
