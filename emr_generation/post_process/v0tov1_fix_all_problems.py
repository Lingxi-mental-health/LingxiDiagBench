#!/usr/bin/env python3
"""
修复所有问题现病史记录（空/过短/截断）

自动检测并修复 LingxiDiag-16K_fixed_v2.json 中的问题记录

使用方法:
    python fix_all_problems.py                # 修复全部问题记录
    python fix_all_problems.py --workers 16   # 指定并发数
    python fix_all_problems.py --test 88      # 测试单条
    python fix_all_problems.py --limit 100    # 限制处理数量
"""

import json
import re
import logging
import os
import argparse
import threading
from datetime import datetime
from typing import Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


# ============== 配置 ==============
API_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-32b"
MAX_WORKERS = 16
MAX_TOKENS = 2048

INPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory/LingxiDiag-16K_fixed_v2.json"
OUTPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory/LingxiDiag-16K_fixed_v3.json"
LOG_DIR = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B/PresentIllnessHistory/log"

# 截断结尾字符
TRUNCATED_ENDINGS = ['及', '和', '与', '或', '但', '因', '而', '以', '为', '在', '对',
                     '向', '从', '把', '被', '的', '了', '着', '过', '到', '给',
                     '，', '、', '：', '；', '（', '「']

# 线程锁
stats_lock = threading.Lock()
log_lock = threading.Lock()
file_lock = threading.Lock()

# 全局变量
REALTIME_FILE = None


def strip_thinking(text: str) -> str:
    """移除 Qwen3 thinking 模式产生的 <think>...</think> 内容"""
    if not text:
        return text
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)
    return result.strip()


def setup_logging():
    """设置日志"""
    global REALTIME_FILE
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/fix_all_problems_{timestamp}.log"
    REALTIME_FILE = f"{LOG_DIR}/fix_all_problems_{timestamp}.jsonl"

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


def write_realtime(record: Dict):
    """实时写入记录"""
    if REALTIME_FILE is None:
        return
    with file_lock:
        with open(REALTIME_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def get_client() -> OpenAI:
    """创建 OpenAI 客户端"""
    return OpenAI(base_url=API_BASE_URL, api_key="not-needed")


def is_problematic(history: str) -> Tuple[bool, str]:
    """检查现病史是否有问题"""
    if not history or history.strip() in ['', '现病史：', '现病史:']:
        return True, 'empty'

    content = history.replace('现病史：', '').replace('现病史:', '').strip()

    if len(content) < 30:
        return True, 'too_short'

    if content and content[-1] in TRUNCATED_ENDINGS:
        return True, 'truncated'

    return False, 'ok'


def complete_history(client: OpenAI, patient_data: Dict, retry: int = 0) -> str:
    """根据对话记录生成现病史"""
    cleaned_text = patient_data.get("cleaned_text", "")
    if not cleaned_text:
        return patient_data.get("PresentIllnessHistory", "")

    original_history = patient_data.get("PresentIllnessHistory", "")
    chief_complaint = patient_data.get("ChiefComplaint", "")
    diagnosis = patient_data.get("Diagnosis", "")
    age = patient_data.get("Age", "")
    gender = patient_data.get("Gender", "")

    # 根据重试次数调整温度
    temperature = 0.3 + retry * 0.15

    system_prompt = """你是专业的精神科医生，根据医患对话记录撰写规范的现病史。

现病史应包含以下要素（根据对话内容提取）：
1. 起病时间和诱因
2. 主要症状表现
3. 症状的演变过程
4. 伴随症状
5. 既往诊治情况
6. 目前状态

要求：
1. 语言专业简洁，必须以"现病史："开头
2. 只从对话中提取信息，不要编造
3. 必须写成完整的句子，确保结尾是句号
4. 字数控制在200-400字
5. 直接输出现病史内容，不要有任何解释或前言"""

    user_prompt = f"""根据医患对话撰写完整现病史：

【患者】{age}岁{gender}
【主诉】{chief_complaint}
【诊断】{diagnosis}

【医患对话】
{cleaned_text}

请输出完整的现病史（以"现病史："开头，以句号结尾）："""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=temperature,
        )

        completed = response.choices[0].message.content.strip()
        completed = strip_thinking(completed)

        if not completed.startswith("现病史"):
            completed = "现病史：" + completed

        return completed

    except Exception as e:
        logging.error(f"API调用失败: {e}")
        return original_history


def process_single(args: Tuple[int, Dict, OpenAI]) -> Dict:
    """处理单条记录"""
    idx, record, client = args
    patient_id = record.get("patient_id", f"unknown_{idx}")
    history = record.get("PresentIllnessHistory", "")

    has_problem, problem_type = is_problematic(history)

    if not has_problem:
        return {"idx": idx, "record": record, "fixed": False, "type": "ok", "skipped": True}

    # 检查是否有对话数据
    cleaned_text = record.get("cleaned_text", "")
    if not cleaned_text or len(cleaned_text) < 50:
        with log_lock:
            logging.warning(f"[跳过] idx={idx} | patient_id={patient_id} | 无对话数据")
        return {"idx": idx, "record": record, "fixed": False, "type": problem_type, "skipped": True}

    # 最多重试3次
    best_history = history
    best_type = problem_type

    for retry in range(3):
        new_history = complete_history(client, record, retry=retry)
        still_problem, new_type = is_problematic(new_history)

        # 如果没问题了，直接使用
        if not still_problem:
            best_history = new_history
            best_type = new_type
            break

        # 如果比之前的好（更长且内容更丰富），保存
        if len(new_history) > len(best_history) + 30:
            best_history = new_history
            best_type = new_type

    # 更新记录
    updated_record = record.copy()

    if best_type == 'ok' or len(best_history) > len(history) + 50:
        updated_record["PresentIllnessHistory_original"] = history
        updated_record["PresentIllnessHistory"] = best_history
        fixed = True

        with log_lock:
            logging.info(f"[修复] idx={idx} | patient_id={patient_id} | {problem_type} -> {best_type} | {len(history)}字 -> {len(best_history)}字")

        write_realtime({
            "idx": idx,
            "patient_id": patient_id,
            "old_type": problem_type,
            "new_type": best_type,
            "old_len": len(history),
            "new_len": len(best_history),
            "original": history,
            "completed": best_history
        })
    else:
        fixed = False
        with log_lock:
            logging.warning(f"[失败] idx={idx} | patient_id={patient_id} | 仍有问题: {best_type}")

    return {"idx": idx, "record": updated_record, "fixed": fixed, "type": best_type, "skipped": False}


def find_problem_records(data: List[Dict]) -> List[int]:
    """找出所有问题记录的索引"""
    problem_indices = []
    for idx, record in enumerate(data):
        history = record.get("PresentIllnessHistory", "")
        has_problem, _ = is_problematic(history)
        if has_problem:
            # 检查是否有对话数据
            cleaned_text = record.get("cleaned_text", "")
            if cleaned_text and len(cleaned_text) > 50:
                problem_indices.append(idx)
    return problem_indices


def main():
    parser = argparse.ArgumentParser(description="修复所有问题现病史")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"并发数 (默认: {MAX_WORKERS})")
    parser.add_argument("--test", type=int, help="测试单条记录索引")
    parser.add_argument("--limit", type=int, help="限制处理数量")
    args = parser.parse_args()

    log_file = setup_logging()

    logging.info("=" * 60)
    logging.info("修复所有问题现病史")
    logging.info(f"输入文件: {INPUT_FILE}")
    logging.info(f"输出文件: {OUTPUT_FILE}")
    logging.info(f"并发数: {args.workers}")
    logging.info(f"max_tokens: {MAX_TOKENS}")
    logging.info("=" * 60)

    # 加载数据
    logging.info(f"加载数据...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"总记录数: {len(data)}")

    # 找出问题记录
    if args.test is not None:
        problem_indices = [args.test]
        logging.info(f"测试模式: 仅处理 idx={args.test}")
    else:
        problem_indices = find_problem_records(data)
        logging.info(f"问题记录数: {len(problem_indices)}")

    # 统计问题类型
    problem_stats = {"empty": 0, "too_short": 0, "truncated": 0}
    for idx in problem_indices:
        _, ptype = is_problematic(data[idx].get("PresentIllnessHistory", ""))
        problem_stats[ptype] = problem_stats.get(ptype, 0) + 1

    logging.info(f"  空内容: {problem_stats.get('empty', 0)}")
    logging.info(f"  过短: {problem_stats.get('too_short', 0)}")
    logging.info(f"  截断: {problem_stats.get('truncated', 0)}")

    if args.limit:
        problem_indices = problem_indices[:args.limit]
        logging.info(f"限制处理: {args.limit}条")

    if not problem_indices:
        logging.info("没有需要修复的记录")
        return

    # 创建客户端
    logging.info("连接 Qwen32B API...")
    client = get_client()

    # 准备任务
    tasks = [(idx, data[idx], client) for idx in problem_indices]

    # 统计
    stats = {"total": len(problem_indices), "fixed": 0, "failed": 0, "skipped": 0}
    result_types = {"empty": 0, "too_short": 0, "truncated": 0, "ok": 0}

    logging.info("-" * 60)
    logging.info(f"开始处理 {len(problem_indices)} 条记录...")

    # 并发处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single, task): task[0] for task in tasks}

        with tqdm(total=len(problem_indices), desc="修复中") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    idx = result["idx"]
                    data[idx] = result["record"]

                    with stats_lock:
                        if result.get("skipped"):
                            stats["skipped"] += 1
                        elif result["fixed"]:
                            stats["fixed"] += 1
                        else:
                            stats["failed"] += 1

                        rtype = result["type"]
                        if rtype in result_types:
                            result_types[rtype] += 1

                except Exception as e:
                    logging.error(f"处理异常: {e}")
                    with stats_lock:
                        stats["failed"] += 1

                pbar.update(1)

    # 最终统计
    logging.info("=" * 60)
    logging.info("修复完成!")
    logging.info(f"  处理记录: {stats['total']}")
    logging.info(f"  成功修复: {stats['fixed']}")
    logging.info(f"  修复失败: {stats['failed']}")
    logging.info(f"  跳过: {stats['skipped']}")
    logging.info(f"  修复后状态: empty={result_types['empty']}, too_short={result_types['too_short']}, truncated={result_types['truncated']}, ok={result_types['ok']}")

    # 最终检查
    logging.info("-" * 60)
    logging.info("最终数据质量检查:")
    final_problems = {"empty": 0, "too_short": 0, "truncated": 0, "ok": 0}
    for record in data:
        _, ptype = is_problematic(record.get("PresentIllnessHistory", ""))
        final_problems[ptype] = final_problems.get(ptype, 0) + 1

    total = len(data)
    ok_count = final_problems["ok"]
    problem_count = total - ok_count

    logging.info(f"  总记录: {total}")
    logging.info(f"  正常: {ok_count} ({ok_count/total*100:.2f}%)")
    logging.info(f"  问题: {problem_count} ({problem_count/total*100:.2f}%)")
    logging.info(f"    空: {final_problems['empty']}")
    logging.info(f"    过短: {final_problems['too_short']}")
    logging.info(f"    截断: {final_problems['truncated']}")

    # 保存
    logging.info(f"保存到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logging.info(f"日志: {log_file}")
    logging.info(f"实时记录: {REALTIME_FILE}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
