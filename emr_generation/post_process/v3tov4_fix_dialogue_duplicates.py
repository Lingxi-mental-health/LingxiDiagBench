#!/usr/bin/env python3
"""
对话重复修复脚本 - 最终版
1. 从两个历史jsonl重建已确认的修复（利用deleted_lines）
2. 对剩余记录调LLM处理
3. 所有处理结果（有修改/无修改）都写入jsonl，便于断点续传
4. API调用含重试机制
"""

import json
import re
import logging
import os
import argparse
import threading
import time
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ============== 配置 ==============
API_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-32b"
MAX_WORKERS = 32
MODEL_CONTEXT_LENGTH = 8192
MAX_DELETE_RATIO = 0.30
MIN_SIMILARITY = 0.6
MAX_RETRIES = 3
RETRY_DELAY = 5  # 秒

INPUT_FILE = None  # 通过 --input 指定
OUTPUT_FILE = None  # 通过 --output 指定
LOG_DIR = None  # 通过 --log-dir 指定

# 历史jsonl文件（用于重建已有修复，可选）
# 从零开始跑时不需要，留空即可
PREVIOUS_JSONLS = []

stats_lock = threading.Lock()
log_lock = threading.Lock()
file_lock = threading.Lock()
REALTIME_FILE = None


def strip_thinking(text: str) -> str:
    if not text:
        return text
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)
    return result.strip()


def setup_logging() -> str:
    global REALTIME_FILE
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{LOG_DIR}/dialogue_fix_final_{timestamp}.log"
    REALTIME_FILE = f"{LOG_DIR}/dialogue_fix_final_{timestamp}.jsonl"

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
    if REALTIME_FILE is None:
        return
    with file_lock:
        with open(REALTIME_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def call_llm(messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """使用requests调用LLM API，含重试机制"""
    url = f"{API_BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise last_error


def estimate_tokens(text: str) -> int:
    return int(len(text) * 1.5)


def text_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    t1 = re.sub(r'^(医生|患者)：', '', text1).strip()
    t2 = re.sub(r'^(医生|患者)：', '', text2).strip()
    if not t1 or not t2:
        return 0.0
    if t1 == t2:
        return 1.0
    set1 = set(t1)
    set2 = set(t2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0
    len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2))
    return jaccard * 0.7 + len_ratio * 0.3


def split_dialogue_to_lines(text: str) -> List[str]:
    if not text:
        return []
    lines = []
    pattern = r'(医生：|患者：)'
    parts = re.split(pattern, text)
    current_line = ""
    for part in parts:
        if part in ['医生：', '患者：']:
            if current_line.strip():
                lines.append(current_line.strip())
            current_line = part
        else:
            current_line += part
    if current_line.strip():
        lines.append(current_line.strip())
    return lines


def remove_lines_by_numbers(lines: List[str], line_numbers: List[int]) -> str:
    line_numbers_set = set(line_numbers)
    result = [line for i, line in enumerate(lines, 1) if i not in line_numbers_set]
    return '\n'.join(result)


def rebuild_fix_from_jsonl(original_text: str, deleted_lines: List[int]) -> str:
    """根据原始文本和删除行号重建修复后的文本"""
    lines = split_dialogue_to_lines(original_text)
    if not lines or not deleted_lines:
        return original_text
    # 验证行号在有效范围内
    valid_lines = [ln for ln in deleted_lines if 1 <= ln <= len(lines)]
    if not valid_lines:
        return original_text
    return remove_lines_by_numbers(lines, valid_lines)


def clean_dialogue_with_llm(cleaned_text: str, idx: int = -1) -> Tuple[str, bool, str, List[int], List[Dict]]:
    if not cleaned_text or len(cleaned_text.strip()) < 50:
        return cleaned_text, False, "", [], []

    lines = split_dialogue_to_lines(cleaned_text)
    if len(lines) < 2:
        return cleaned_text, False, "", [], []

    numbered_dialogue = '\n'.join([f"{i}. {line}" for i, line in enumerate(lines, 1)])

    system_prompt = """你是医疗对话数据清洗专家。请检查对话中**完全相同或几乎完全相同**的重复内容。

**严格定义 - 以下情况算重复（无论间隔多少轮）**：
1. 医生问了完全相同或几乎相同的问题（仅标点或1-2个字不同），无论出现在对话的任何位置
2. 患者给出了完全相同或几乎相同的回答（仅标点或1-2个字不同），无论出现在对话的任何位置
3. 即使间隔多轮，只要内容实质相同就应标记为重复（保留首次出现的，删除后续重复的）

**不算重复，不要删除**：
- 医生追问不同的细节或角度（即使话题相关）
- 患者回答中包含任何新信息（即使部分重复）
- 语义相关但表述明显不同的内容

**输出格式（严格JSON）**：
```json
{
  "has_duplicate": true或false,
  "duplicates": [
    {"delete": 要删除的行号, "same_as": 与哪行重复}
  ],
  "reasoning": "简要说明"
}
```

示例：如果第8行和第2行完全相同（即使间隔多轮），返回 {"delete": 8, "same_as": 2}"""

    user_prompt = f"""请检查以下对话，找出**完全相同或几乎完全相同**的重复行（无论间隔多少轮）：

{numbered_dialogue}

注意：
1. 只标记真正重复的行，不要标记语义相关但表述不同的内容
2. 如果发现重复，保留首次出现的行，删除后续重复的行
3. 请仔细比对所有行，包括间隔较远的行"""

    input_tokens = estimate_tokens(system_prompt + user_prompt)
    max_tokens = min(MODEL_CONTEXT_LENGTH - input_tokens - 100, 1024)

    if max_tokens < 100:
        return cleaned_text, False, "", [], []

    try:
        result = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.2
        )
        result = strip_thinking(result)

        try:
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                has_duplicate = parsed.get("has_duplicate", False)
                duplicates = parsed.get("duplicates", [])
                reasoning = parsed.get("reasoning", "")

                if not has_duplicate or not duplicates:
                    return cleaned_text, False, "", [], []

                valid_lines = []
                verified_pairs = []

                for dup in duplicates:
                    if not isinstance(dup, dict):
                        continue
                    delete_idx = dup.get("delete", 0)
                    same_as_idx = dup.get("same_as", 0)

                    if not (1 <= delete_idx <= len(lines) and 1 <= same_as_idx <= len(lines)):
                        continue

                    sim = text_similarity(lines[delete_idx - 1], lines[same_as_idx - 1])
                    if sim >= MIN_SIMILARITY:
                        valid_lines.append(delete_idx)
                        verified_pairs.append({
                            "delete": delete_idx,
                            "same_as": same_as_idx,
                            "similarity": f"{sim:.0%}"
                        })

                if not valid_lines:
                    return cleaned_text, False, "", [], []

                delete_ratio = len(valid_lines) / len(lines)
                if delete_ratio > MAX_DELETE_RATIO:
                    logging.warning(f"[可疑] idx={idx} | 删除比例过高: {delete_ratio:.1%}")
                    write_realtime({
                        "idx": idx,
                        "status": "skipped_high_delete_ratio",
                        "delete_ratio": f"{delete_ratio:.1%}"
                    })
                    return cleaned_text, False, "", [], []

                new_text = remove_lines_by_numbers(lines, valid_lines)
                return new_text, True, reasoning, valid_lines, verified_pairs
            else:
                return cleaned_text, False, "", [], []
        except json.JSONDecodeError:
            return cleaned_text, False, "", [], []

    except Exception as e:
        logging.error(f"LLM调用失败 idx={idx}: {e}")
        raise  # 向上抛异常，让调用方处理


def process_single(args: Tuple[int, Dict]) -> Dict:
    idx, record = args
    patient_id = record.get("patient_id", f"unknown_{idx}")
    cleaned_text = record.get("cleaned_text", "")

    try:
        new_text, has_change, reasoning, deleted_lines, verified_pairs = clean_dialogue_with_llm(cleaned_text, idx)
    except Exception as e:
        # API调用失败，标记为error
        return {"idx": idx, "record": record, "fixed": False, "error": str(e)}

    updated_record = record.copy()

    if has_change:
        updated_record["cleaned_text_original"] = cleaned_text
        updated_record["cleaned_text"] = new_text
        updated_record["cleaned_text_reasoning"] = reasoning
        updated_record["deleted_lines"] = deleted_lines

        with log_lock:
            logging.info(f"[修复] idx={idx} | {patient_id} | 删除行: {deleted_lines} | 长度: {len(cleaned_text)} -> {len(new_text)}")
            if reasoning:
                logging.info(f"       原因: {reasoning[:100]}")

        write_realtime({
            "idx": idx,
            "patient_id": patient_id,
            "status": "fixed",
            "original_length": len(cleaned_text),
            "fixed_length": len(new_text),
            "deleted_lines": deleted_lines,
            "reasoning": reasoning
        })
    else:
        # 无重复，也写入jsonl，标记为已处理
        write_realtime({
            "idx": idx,
            "patient_id": patient_id,
            "status": "no_duplicate"
        })

    return {"idx": idx, "record": updated_record, "fixed": has_change}


def load_fix_records(jsonl_files: List[str]) -> Dict[int, Dict]:
    """从历史jsonl文件加载修复记录（只加载有deleted_lines的）"""
    fixes = {}
    for path in jsonl_files:
        if not os.path.exists(path):
            logging.warning(f"jsonl文件不存在: {path}")
            continue
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        idx = record.get('idx')
                        deleted_lines = record.get('deleted_lines')
                        if idx is not None and deleted_lines:
                            fixes[idx] = record
                            count += 1
                    except json.JSONDecodeError:
                        pass
        logging.info(f"  从 {os.path.basename(path)} 加载 {count} 条修复记录")
    return fixes


def load_all_processed_indices_from_previous(jsonl_files: List[str]) -> Set[int]:
    """从历史jsonl文件加载所有已处理的索引（包括no_duplicate，排除error）"""
    processed = set()
    for path in jsonl_files:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        idx = record.get('idx')
                        status = record.get('status', '')
                        if idx is not None and status != 'error':
                            processed.add(idx)
                    except json.JSONDecodeError:
                        pass
    return processed


def load_all_processed_indices(jsonl_file: str) -> Set[int]:
    """从本次运行的jsonl加载所有已处理的索引（包括无修改的）"""
    processed = set()
    if not os.path.exists(jsonl_file):
        return processed
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    idx = record.get('idx')
                    status = record.get('status', '')
                    # 只要有idx且状态不是error，就视为已处理
                    if idx is not None and status != 'error':
                        processed.add(idx)
                except json.JSONDecodeError:
                    pass
    return processed


def main():
    parser = argparse.ArgumentParser(description="v3→v4: 对话重复修复")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSON文件")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="并发数")
    parser.add_argument("--log-dir", type=str, default=None, help="日志目录（默认为output同目录下的log/）")
    parser.add_argument("--previous-jsonls", nargs="*", default=[],
                        help="历史修复JSONL文件（用于重建已有修复，从零跑时不需要）")
    args = parser.parse_args()

    global PREVIOUS_JSONLS, LOG_DIR
    PREVIOUS_JSONLS = args.previous_jsonls
    LOG_DIR = args.log_dir or os.path.join(os.path.dirname(args.output), "log")

    log_file = setup_logging()

    logging.info("=" * 60)
    logging.info("对话重复修复 - 最终版")
    logging.info(f"并发数: {args.workers}")
    logging.info("=" * 60)

    # ====== 第一步：加载原始数据 ======
    logging.info(f"加载原始数据: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"总记录数: {len(data)}")

    # ====== 第二步：从历史jsonl重建修复 ======
    logging.info("加载历史修复记录...")
    historical_fixes = load_fix_records(PREVIOUS_JSONLS)
    logging.info(f"历史修复总数: {len(historical_fixes)}")

    rebuilt_count = 0
    rebuild_failed = 0
    for idx, fix_info in historical_fixes.items():
        if idx >= len(data):
            continue
        original_text = data[idx].get("cleaned_text", "")
        deleted_lines = fix_info.get("deleted_lines", [])
        if not deleted_lines:
            continue

        new_text = rebuild_fix_from_jsonl(original_text, deleted_lines)
        if new_text != original_text:
            data[idx]["cleaned_text_original"] = original_text
            data[idx]["cleaned_text"] = new_text
            data[idx]["cleaned_text_reasoning"] = fix_info.get("reasoning", "")
            data[idx]["deleted_lines"] = deleted_lines
            rebuilt_count += 1
        else:
            rebuild_failed += 1

    logging.info(f"成功重建修复: {rebuilt_count} 条")
    if rebuild_failed:
        logging.warning(f"重建失败（可能行号不匹配）: {rebuild_failed} 条")

    # 已通过历史修复处理的idx
    historical_fixed_indices = set(historical_fixes.keys())

    # ====== 第三步：检查本次运行的jsonl（支持断点续传） ======
    current_processed = set()
    if REALTIME_FILE and os.path.exists(REALTIME_FILE):
        current_processed = load_all_processed_indices(REALTIME_FILE)
        logging.info(f"本次已处理（从断点恢复）: {len(current_processed)} 条")

    # ====== 第四步：确定待处理记录 ======
    # 从历史jsonl加载所有已处理索引（包括no_duplicate）
    previous_processed = load_all_processed_indices_from_previous(PREVIOUS_JSONLS)
    logging.info(f"历史已处理索引数: {len(previous_processed)}")
    # 跳过：历史已修复的 + 历史已处理的 + 本次已处理的
    skip_indices = historical_fixed_indices | previous_processed | current_processed
    indices_to_process = [i for i in range(len(data)) if i not in skip_indices]
    logging.info(f"待LLM处理: {len(indices_to_process)} 条")

    if not indices_to_process:
        logging.info("所有记录已处理完成，直接保存！")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"已保存到: {args.output}")
        return

    # ====== 第五步：测试API ======
    logging.info("测试 Qwen32B API 连接...")
    try:
        test_result = call_llm([{"role": "user", "content": "test"}], max_tokens=10)
        logging.info("API连接成功")
    except Exception as e:
        logging.error(f"API连接失败: {e}")
        logging.error("请先启动API服务再运行本脚本")
        # 即使API不可用，也保存已重建的修复
        logging.info("先保存已重建的历史修复结果...")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"已保存到: {args.output}（含 {rebuilt_count} 条历史修复）")
        return

    # ====== 第六步：并行处理 ======
    tasks = [(idx, data[idx]) for idx in indices_to_process]
    stats = {"total": len(indices_to_process), "fixed": 0, "no_change": 0, "error": 0}

    logging.info("-" * 60)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single, task): task[0] for task in tasks}

        with tqdm(total=len(indices_to_process), desc="处理中") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    idx = result["idx"]

                    if result.get("error"):
                        with stats_lock:
                            stats["error"] += 1
                        # 不更新data，保持原始数据
                    else:
                        data[idx] = result["record"]
                        with stats_lock:
                            if result.get("fixed"):
                                stats["fixed"] += 1
                            else:
                                stats["no_change"] += 1

                except Exception as e:
                    with stats_lock:
                        stats["error"] += 1
                    logging.error(f"处理异常: {e}")

                pbar.update(1)

    # ====== 第七步：保存 ======
    logging.info(f"保存到: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 统计总修复数
    total_fixed = sum(1 for d in data if 'cleaned_text_original' in d)

    logging.info("=" * 60)
    logging.info("完成!")
    logging.info(f"  历史重建修复: {rebuilt_count}")
    logging.info(f"  本次LLM处理: {stats['total']}")
    logging.info(f"    - 有修改: {stats['fixed']}")
    logging.info(f"    - 无修改: {stats['no_change']}")
    logging.info(f"    - 失败:   {stats['error']}")
    logging.info(f"  输出文件总修复数: {total_fixed}")
    logging.info(f"日志: {log_file}")
    logging.info(f"JSONL: {REALTIME_FILE}")

    if stats["error"] > 0:
        logging.warning(f"有 {stats['error']} 条记录处理失败，可重新运行本脚本继续处理")


if __name__ == "__main__":
    main()
