#!/usr/bin/env python3
"""
躯体疾病史补充脚本 - 使用LLM从对话中提取躯体疾病信息

直接让LLM根据躯体疾病历史和对话内容判断是否需要补充，不使用规则预筛选。

使用方法:
    python fix_physical_history.py                    # 修复全部
    python fix_physical_history.py --workers 32      # 指定并发数
    python fix_physical_history.py --test 123        # 测试单条
"""

import json
import re
import logging
import os
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm


# ============== 配置 ==============
API_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-32b"
MAX_WORKERS = 32
MAX_TOKENS = 512

INPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/Single-Field LLM Restoration/LingxiDiag-16K_dialogue_fixed.json"
OUTPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/Single-Field LLM Restoration/LingxiDiag-16K_physical_fixed.json"
LOG_DIR = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/Single-Field LLM Restoration/log"

# 线程锁
stats_lock = threading.Lock()
log_lock = threading.Lock()
file_lock = threading.Lock()

# 实时记录文件
REALTIME_FILE = None


def strip_thinking(text: str) -> str:
    """移除 Qwen3 thinking 模式产生的 <think>...</think> 内容"""
    if not text:
        return text
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)
    return result.strip()


def setup_logging() -> str:
    """设置日志"""
    global REALTIME_FILE
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/physical_fix_{timestamp}.log"
    REALTIME_FILE = f"{LOG_DIR}/physical_fix_{timestamp}.jsonl"

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


def extract_physical_history_with_llm(client: OpenAI, record: Dict) -> Tuple[str, bool]:
    """
    使用LLM根据躯体疾病历史和对话内容判断并补充
    返回: (处理后的文本, 是否有修改)
    """
    cleaned_text = record.get("cleaned_text", "")
    current = record.get("ImportantRelevantPhysicalIllnessHistory", "")

    if not cleaned_text:
        return current, False

    system_prompt = """你是专业的医疗信息提取专家。请根据当前的躯体疾病史记录和医患对话内容，判断是否需要补充或修正躯体疾病史。

任务：
1. 仔细阅读当前躯体疾病史记录和医患对话
2. 从对话中提取患者提及的躯体疾病信息（疾病名称、诊断时间、治疗情况等）
3. 如果对话中有躯体疾病信息但当前记录缺失或不完整，则补充完整
4. 如果当前记录已经完整，或对话中没有提及躯体疾病，则保持原样
5. 绝不编造信息，只提取对话中明确提及的内容

输出格式：重要或相关躯体疾病史：XXX
直接输出结果，不要解释。"""

    user_prompt = f"""【当前躯体疾病史记录】
{current}

【医患对话】
{cleaned_text}

请输出躯体疾病史："""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.2,
        )

        result = response.choices[0].message.content.strip()
        result = strip_thinking(result)

        # 确保格式正确
        if not result.startswith("重要或相关躯体疾病史"):
            result = "重要或相关躯体疾病史：" + result

        # 判断是否有实质性修改
        new_content = result.replace("重要或相关躯体疾病史：", "").replace("重要或相关躯体疾病史:", "").strip()
        old_content = current.replace("重要或相关躯体疾病史：", "").replace("重要或相关躯体疾病史:", "").strip()

        has_change = (new_content != old_content and
                     new_content not in ["无", "无特殊", "不详", "暂无", ""] and
                     len(new_content) > len(old_content))

        return result, has_change

    except Exception as e:
        logging.error(f"LLM调用失败: {e}")
        return current, False


def process_single(args: Tuple[int, Dict, OpenAI]) -> Dict:
    """处理单条记录"""
    idx, record, client = args
    patient_id = record.get("patient_id", f"unknown_{idx}")
    current = record.get("ImportantRelevantPhysicalIllnessHistory", "")

    # 直接调用LLM判断和处理
    new_history, has_change = extract_physical_history_with_llm(client, record)

    # 更新记录
    updated_record = record.copy()

    if has_change:
        updated_record["ImportantRelevantPhysicalIllnessHistory_original"] = current
        updated_record["ImportantRelevantPhysicalIllnessHistory"] = new_history

        with log_lock:
            logging.info(f"[修复] idx={idx} | {patient_id}")
            logging.info(f"       原: {current[:40]}")
            logging.info(f"       新: {new_history[:40]}")

        write_realtime({
            "idx": idx,
            "patient_id": patient_id,
            "original": current,
            "fixed": new_history
        })

    return {"idx": idx, "record": updated_record, "fixed": has_change}


def main():
    parser = argparse.ArgumentParser(description="躯体疾病史补充 - LLM全量处理")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help="输入文件")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="输出文件")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    parser.add_argument("--test", type=int, help="测试单条索引")
    args = parser.parse_args()

    log_file = setup_logging()

    logging.info("=" * 60)
    logging.info("躯体疾病史补充 - LLM全量处理")
    logging.info(f"并发数: {args.workers}")
    logging.info("=" * 60)

    # 加载数据
    logging.info(f"加载数据: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"总记录数: {len(data)}")

    # 创建客户端
    logging.info("连接 Qwen32B API...")
    client = get_client()

    # 确定要处理的记录
    if args.test is not None:
        indices = [args.test]
        logging.info(f"测试模式: idx={args.test}")
    else:
        # 全量处理所有记录
        indices = list(range(len(data)))
        logging.info(f"全量处理: {len(indices)} 条")

    # 准备任务
    tasks = [(idx, data[idx], client) for idx in indices]

    # 统计
    stats = {"total": len(indices), "fixed": 0}

    logging.info("-" * 60)

    # 并行处理
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single, task): task[0] for task in tasks}

        with tqdm(total=len(indices), desc="处理中") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    idx = result["idx"]
                    data[idx] = result["record"]

                    with stats_lock:
                        if result.get("fixed"):
                            stats["fixed"] += 1

                except Exception as e:
                    logging.error(f"处理异常: {e}")

                pbar.update(1)

    # 保存
    logging.info(f"保存到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 统计
    logging.info("=" * 60)
    logging.info("完成!")
    logging.info(f"  处理记录: {stats['total']}")
    logging.info(f"  有修改的: {stats['fixed']}")
    logging.info(f"日志: {log_file}")
    if REALTIME_FILE:
        logging.info(f"实时记录: {REALTIME_FILE}")


if __name__ == "__main__":
    main()
