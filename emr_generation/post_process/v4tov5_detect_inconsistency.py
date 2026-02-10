#!/usr/bin/env python3
"""
跨字段一致性矛盾检测 - 使用本地 Qwen-32B
对每条记录，将所有字段喂给LLM，让其找出字段间的事实性矛盾。
先跑500条样本。
"""

import json
import re
import os
import logging
import random
import threading
import time
import requests
from datetime import datetime
from typing import Dict, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import Counter, defaultdict

# ============== 配置 ==============
API_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3-32b"
MAX_WORKERS = 32
MAX_RETRIES = 3
RETRY_DELAY = 5

INPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/Single-Field_LLM_Restoration/LingxiDiag-16K_dialogue_fixed_v2.json"
LOG_DIR = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/Cross-Field Restoration/log"

SAMPLE_SIZE = 500

stats_lock = threading.Lock()
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
    log_file = f"{LOG_DIR}/cross_field_eval_{timestamp}.log"
    REALTIME_FILE = f"{LOG_DIR}/cross_field_eval_{timestamp}.jsonl"

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


def call_llm(messages: List[Dict], max_tokens: int = 2048, temperature: float = 0.1) -> str:
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
            response = requests.post(url, json=payload, timeout=180)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
    raise last_error


SYSTEM_PROMPT = """你是一个医疗数据质量审核员。你的任务是检查一条精神科病历中各字段之间是否存在事实性矛盾。

注意：
- 只报告明确的事实性矛盾（如年龄不符、性别不符、婚姻状态不符、病程时间严重不符、家族史矛盾等）
- 不要报告信息缺失（一个字段提到但另一个没提到，不算矛盾）
- 不要报告表述方式差异（口语vs书面语不算矛盾）
- 不要报告细节详略差异（对话更详细而现病史更简略不算矛盾）
- 病程时间差异需要>=3倍才算矛盾（如主诉说3年，现病史说2个月）
- 对话中患者的口头表述是最可信的信息源"""


def build_user_prompt(record: Dict) -> str:
    """构建用户prompt，包含所有字段"""
    fields = []
    fields.append(f"【年龄】{record.get('Age', '无')}")
    fields.append(f"【性别】{record.get('Gender', '无')}")
    fields.append(f"【陪诊人】{record.get('AccompanyingPerson', '无')}")
    fields.append(f"【主诉】{record.get('ChiefComplaint', '无')}")
    fields.append(f"【现病史】{record.get('PresentIllnessHistory', '无')}")
    fields.append(f"【个人史】{record.get('PersonalHistory', '无')}")
    fields.append(f"【家族史】{record.get('FamilyHistory', '无')}")
    fields.append(f"【躯体疾病史】{record.get('ImportantRelevantPhysicalIllnessHistory', '无')}")
    fields.append(f"【药物过敏史】{record.get('DrugAllergyHistory', '无')}")
    fields.append(f"【诊断】{record.get('Diagnosis', '无')}")

    # 对话截断，避免太长
    dialogue = record.get('cleaned_text', '')
    if len(dialogue) > 3000:
        dialogue = dialogue[:3000] + "\n...(截断)"
    fields.append(f"【医患对话】\n{dialogue}")

    prompt = "请检查以下病历各字段之间是否存在事实性矛盾。\n\n"
    prompt += "\n".join(fields)
    prompt += """

请按以下JSON格式输出（不要输出其他内容）:
{
  "has_contradiction": true/false,
  "contradictions": [
    {
      "type": "矛盾类型（如：年龄矛盾/性别矛盾/病程矛盾/婚姻状态矛盾/家族史矛盾/陪诊人矛盾/症状矛盾/其他）",
      "field_a": "字段A名称",
      "field_b": "字段B名称",
      "value_a": "字段A中的相关内容",
      "value_b": "字段B中的相关内容",
      "description": "简要描述矛盾"
    }
  ]
}

如果没有矛盾，输出: {"has_contradiction": false, "contradictions": []}"""

    return prompt


def parse_llm_response(raw: str) -> Dict:
    """解析LLM的JSON输出"""
    text = strip_thinking(raw)

    # 尝试提取JSON块
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # 如果解析失败，返回空结果
    return {"has_contradiction": False, "contradictions": [], "parse_error": True, "raw": text[:500]}


def process_single(idx: int, record: Dict) -> Dict:
    """处理单条记录"""
    patient_id = record.get("patient_id", f"unknown_{idx}")

    try:
        user_prompt = build_user_prompt(record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        raw_response = call_llm(messages)
        result = parse_llm_response(raw_response)

        output = {
            "idx": idx,
            "patient_id": patient_id,
            "has_contradiction": result.get("has_contradiction", False),
            "contradictions": result.get("contradictions", []),
            "status": "success"
        }

        if result.get("parse_error"):
            output["status"] = "parse_error"
            output["raw_response"] = result.get("raw", "")

        if output["has_contradiction"]:
            n = len(output["contradictions"])
            types = [c.get("type", "?") for c in output["contradictions"]]
            logging.info(f"[矛盾] idx={idx} | {patient_id} | {n}处: {', '.join(types)}")

        write_realtime(output)
        return output

    except Exception as e:
        logging.error(f"[错误] idx={idx} | {patient_id} | {e}")
        output = {
            "idx": idx,
            "patient_id": patient_id,
            "has_contradiction": False,
            "contradictions": [],
            "status": "error",
            "error": str(e)
        }
        write_realtime(output)
        return output


def main():
    log_file = setup_logging()

    logging.info("=" * 60)
    logging.info("跨字段一致性矛盾检测 (Qwen-32B)")
    logging.info("=" * 60)

    # 加载数据
    logging.info(f"加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"总记录数: {len(data)}")

    # 均匀抽样500条
    random.seed(42)
    indices = sorted(random.sample(range(len(data)), min(SAMPLE_SIZE, len(data))))
    logging.info(f"抽样 {len(indices)} 条进行检测")

    # 并发处理
    stats = {"total": len(indices), "success": 0, "error": 0,
             "has_contradiction": 0, "no_contradiction": 0, "parse_error": 0}
    all_results = []

    logging.info(f"开始处理 (workers={MAX_WORKERS})...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, idx, data[idx]): idx for idx in indices}

        with tqdm(total=len(indices), desc="检测矛盾") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)

                with stats_lock:
                    if result["status"] == "success":
                        stats["success"] += 1
                        if result["has_contradiction"]:
                            stats["has_contradiction"] += 1
                        else:
                            stats["no_contradiction"] += 1
                    elif result["status"] == "parse_error":
                        stats["parse_error"] += 1
                    else:
                        stats["error"] += 1

                pbar.update(1)

    # ====== 统计报告 ======
    logging.info("\n" + "=" * 60)
    logging.info("检测完成!")
    logging.info(f"  总样本:       {stats['total']}")
    logging.info(f"  成功:         {stats['success']}")
    logging.info(f"  解析失败:     {stats['parse_error']}")
    logging.info(f"  API错误:      {stats['error']}")
    logging.info(f"  有矛盾:       {stats['has_contradiction']}")
    logging.info(f"  无矛盾:       {stats['no_contradiction']}")

    if stats['has_contradiction'] > 0:
        rate = stats['has_contradiction'] / (stats['success'] + stats['parse_error'])
        logging.info(f"  矛盾率:       {rate:.2%}")

    # 按矛盾类型统计
    type_counter = Counter()
    field_pair_counter = Counter()
    all_contradictions = []

    for r in all_results:
        for c in r.get("contradictions", []):
            ctype = c.get("type", "未知")
            type_counter[ctype] += 1
            pair = f"{c.get('field_a', '?')} vs {c.get('field_b', '?')}"
            field_pair_counter[pair] += 1
            all_contradictions.append({**c, "idx": r["idx"], "patient_id": r["patient_id"]})

    if type_counter:
        logging.info("\n矛盾类型分布:")
        for t, cnt in type_counter.most_common():
            logging.info(f"  {t}: {cnt}")

        logging.info("\n字段对分布:")
        for pair, cnt in field_pair_counter.most_common():
            logging.info(f"  {pair}: {cnt}")

    # 每类矛盾展示几个案例
    by_type = defaultdict(list)
    for c in all_contradictions:
        by_type[c.get("type", "未知")].append(c)

    for ctype, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        logging.info(f"\n{'─' * 60}")
        logging.info(f"【{ctype}】共 {len(items)} 处")
        for item in items[:3]:
            logging.info(f"  idx={item['idx']}, patient_id={item['patient_id']}")
            logging.info(f"    {item.get('field_a', '?')}: {str(item.get('value_a', ''))[:80]}")
            logging.info(f"    {item.get('field_b', '?')}: {str(item.get('value_b', ''))[:80]}")
            logging.info(f"    说明: {item.get('description', '')}")

    logging.info(f"\n日志: {log_file}")
    logging.info(f"JSONL: {REALTIME_FILE}")


if __name__ == "__main__":
    main()
