#!/usr/bin/env python3
"""
精神科病历不合理之处检测脚本
使用本地部署的 Qwen3-32B (vLLM) 对每条病例进行全面质量审查
将不合理之处分为：单字段不合理 (single_field) 和 字段间不合理 (cross_field)
支持并发请求、断点续跑
"""

import json
import os
import sys
import time
import re
import logging
import asyncio
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import Counter
from openai import AsyncOpenAI

# ============ 配置 ============
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "not-needed"
MODEL_NAME = "qwen3-32b"

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "LingxiDiag-16K_cross_field_fixed_v3.json")

BASE_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# 运行时动态赋值
OUTPUT_DIR = None
OUTPUT_FILE = None
SUMMARY_FILE = None
IO_LOG_FILE = None         # 记录每次发送和返回的完整内容

MAX_CONCURRENT = 32
MAX_TOKENS = 2048
TEMPERATURE = 0.0  # 消除检测非确定性，保证结果可复现
SAMPLE_SIZE = 0            # 0 表示全量处理，不抽样
RANDOM_SEED = 42
# 不截断对话，加载全部内容

# ============ Prompt ============
SYSTEM_PROMPT = """你是一位精神科临床数据质控专家。你的任务是对一条精神科病历记录进行全面质量审查，找出所有不合理之处，并严格区分为两大类：

## A. 单字段不合理（single_field）
某一个字段内部自身存在的问题，不涉及与其他字段的对比：
- 对话中有完全重复的行（同一句话连续出现多次）
- 对话中患者前后陈述自相矛盾
- 对话有明显机器生成痕迹（不自然的重复、模板化表达、医生反复用相同句式提问）
- 现病史内部前后不一致（如起病时间前后矛盾）
- 个人史内部矛盾（如同时写"未婚"和"已育"）
- 诊断文本与ICD编码不对应

## B. 字段间不合理（cross_field）
两个或多个字段之间存在事实性矛盾：
- 年龄字段 vs 现病史/对话中描述的年龄
- 性别字段 vs 现病史/对话中的性别线索
- 主诉中的病程时长 vs 现病史中的病程时长（差异≥3倍才算）
- 主诉（ChiefComplaint）vs Patient info中的主诉（重点检查时长差异）
- 婚恋状态（个人史）vs 陪诊人关系（如未婚但陪诊人是妻子）
- 工作学习情况 vs 年龄或陪诊人关系（如学生但陪诊人是同事）
- 结构化字段 vs 对话中患者的陈述（时长、症状、家族史、用药等）
- 诊断 vs 对话中呈现的症状

审查原则：
- 对话中患者的口头表述是最可信的信息源
- 只报告明确的不合理之处，不报告信息缺失或表述详略差异
- 口语与书面语的风格差异不算不合理
- 病程时间差异需≥3倍才报告（如主诉说3年，现病史说2个月）
- 已婚患者独自就诊（无陪同人）是正常情况，不算婚恋陪诊矛盾
- 已婚患者由子女、父母或其他亲属陪同是正常情况，不要求必须由配偶陪同
- 家族史仅检查明确的事实矛盾（如家族史写"无"但对话提及家族精神病史），不检查"诊断未体现家族史"这类关联性推测
- Patient_info 中的主诉与 ChiefComplaint 字段内容一致时不算矛盾
- 诊断严重程度（轻度/中度/重度）与症状表现的程度差异属于临床判断范畴，不算诊断症状矛盾
- 现病史使用临床术语概括对话中的口语化描述不算矛盾（如对话说"脑子里停不下来"，现病史写"思维奔逸"）
- 现病史与对话中药物/治疗措施描述的措辞差异不算矛盾（如"安神的中药" vs "安神类中成药"）
- "否认XX"与对话中"没有XX"/"没想过XX"是同义表述，不算矛盾
- 现病史是对对话的选择性总结，不要求涵盖对话所有细节；仅当现病史有明确的错误事实（与对话直接矛盾的陈述）才算病例对话矛盾
- 主诉↔现病史的病程时长差异严格执行≥3倍阈值
- 现病史将对话口语量化为具体数值（如对话"瘦了好多"→现病史"体重下降约5kg"），只要变化方向一致就不算矛盾
- "无明确X行为/想法"与对话中模糊消极表述（如"活着没意思"）属临床判断差异，不算事实矛盾
- 诊断是临床综合判断，现病史中症状描述不完全匹配诊断名称属正常临床实践，仅当现病史症状与诊断完全无关时才报「诊断症状矛盾」
- 现病史遗漏对话中某些细节（如具体药名、某次就诊经历）属选择性总结，仅当现病史记录了与对话直接矛盾的错误事实才报「病例对话矛盾」"""


USER_PROMPT_TEMPLATE = """请审查以下精神科病历，找出所有不合理之处，区分为单字段不合理和字段间不合理。

## 患者基本信息
- 患者ID：{patient_id}
- 年龄：{Age}
- 性别：{Gender}
- 就诊科室：{Department}
- 陪同人：{AccompanyingPerson}

## 病史信息
- 个人史：{PersonalHistory}
- 主诉（ChiefComplaint）：{ChiefComplaint}
- 现病史：{PresentIllnessHistory}
- 躯体疾病史：{ImportantRelevantPhysicalIllnessHistory}
- 药物过敏史：{DrugAllergyHistory}
- 家族史：{FamilyHistory}

## 诊断信息
- 诊断：{Diagnosis}
- ICD编码：{DiagnosisCode}

## Patient info（汇总字段，注意对比其中的主诉与上方ChiefComplaint是否一致）
{Patient_info}

## 医患对话
{cleaned_text}

---

请严格按以下JSON格式输出，不要输出任何其他内容：
{{"has_issue": <true/false>, "single_field_issues": [{{"type": "<对话重复|对话自相矛盾|对话不自然|现病史内部矛盾|个人史矛盾|诊断编码不匹配>", "field": "<涉及的字段名>", "severity": "<严重|中等|轻微>", "evidence": "<引用原文中的具体内容>", "detail": "<问题描述>"}}], "cross_field_issues": [{{"type": "<年龄矛盾|性别矛盾|病程时长矛盾|Patient_info主诉矛盾|婚恋陪诊矛盾|身份陪诊矛盾|病例对话矛盾|诊断症状矛盾|家族史矛盾>", "field_a": "<字段A名>", "field_b": "<字段B名>", "value_a": "<字段A中的相关内容>", "value_b": "<字段B中的相关内容>", "severity": "<严重|中等|轻微>", "detail": "<问题描述>"}}], "summary": "<一句话总结该样本的主要问题>"}}

如果没有任何不合理之处，输出: {{"has_issue": false, "single_field_issues": [], "cross_field_issues": [], "summary": "未发现明显不合理之处"}}"""


# ============ 工具函数 ============

def strip_thinking(text: str) -> str:
    """去除 Qwen3 的 <think>...</think> 标签"""
    if not text:
        return text
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)
    return result.strip()


def build_user_prompt(record: dict) -> str:
    """构建单条记录的 user prompt"""
    dialogue = record.get("cleaned_text", "")

    return USER_PROMPT_TEMPLATE.format(
        patient_id=record.get("patient_id", ""),
        Age=record.get("Age", ""),
        Gender=record.get("Gender", ""),
        Department=record.get("Department", ""),
        AccompanyingPerson=record.get("AccompanyingPerson", ""),
        PersonalHistory=record.get("PersonalHistory", ""),
        ChiefComplaint=record.get("ChiefComplaint", ""),
        PresentIllnessHistory=record.get("PresentIllnessHistory", ""),
        ImportantRelevantPhysicalIllnessHistory=record.get("ImportantRelevantPhysicalIllnessHistory", ""),
        DrugAllergyHistory=record.get("DrugAllergyHistory", ""),
        FamilyHistory=record.get("FamilyHistory", ""),
        Diagnosis=record.get("Diagnosis", ""),
        DiagnosisCode=record.get("DiagnosisCode", ""),
        Patient_info=record.get("Patient info", ""),
        cleaned_text=dialogue,
    )


def parse_response(text: str) -> dict:
    """解析 LLM 返回的 JSON，带多级容错"""
    cleaned = strip_thinking(text)

    # 尝试直接解析
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown code block 中提取
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试找第一个 { 到最后一个 }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 解析失败
    return {
        "has_issue": False,
        "single_field_issues": [],
        "cross_field_issues": [],
        "summary": f"[解析失败] 原始输出: {cleaned[:500]}",
        "_parse_error": True
    }


def load_completed(output_file: str) -> set:
    """加载已完成的 patient_id 集合（断点续跑）"""
    completed = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        completed.add(result["patient_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return completed


# ============ 异步处理 ============

import threading
_io_log_lock = threading.Lock()


def write_io_log(patient_id: str, messages: list, raw_output: str):
    """将每次发送给大模型的完整内容和大模型的完整返回写入 io_log.jsonl"""
    if IO_LOG_FILE is None:
        return
    record = {
        "patient_id": patient_id,
        "request": messages,
        "response": raw_output,
    }
    with _io_log_lock:
        with open(IO_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


async def eval_single(client: AsyncOpenAI, record: dict, semaphore: asyncio.Semaphore) -> dict:
    """评估单条记录"""
    async with semaphore:
        patient_id = record.get("patient_id", "unknown")
        user_prompt = build_user_prompt(record)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw_output = response.choices[0].message.content.strip()
            eval_result = parse_response(raw_output)
        except Exception as e:
            eval_result = {
                "has_issue": False,
                "single_field_issues": [],
                "cross_field_issues": [],
                "summary": f"[API错误] {str(e)}",
                "_api_error": True
            }
            raw_output = str(e)

        # 记录完整的输入输出
        write_io_log(patient_id, messages, raw_output)

        return {
            "patient_id": patient_id,
            "eval": eval_result,
            "raw_output": raw_output,
        }


async def main_async(args):
    """异步主函数"""
    # 加载数据
    logger.info(f"加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"共 {len(data)} 条记录")

    # 抽样
    sample_size = args.sample_size
    if sample_size and sample_size < len(data):
        random.seed(RANDOM_SEED)
        indices = sorted(random.sample(range(len(data)), sample_size))
        sampled_data = [data[i] for i in indices]
        logger.info(f"随机抽样 {len(sampled_data)} 条 (seed={RANDOM_SEED})")
    else:
        sampled_data = data
        logger.info(f"处理全部 {len(sampled_data)} 条")

    # --limit 限制条数（在抽样基础上进一步限制）
    if args.limit and args.limit > 0:
        sampled_data = sampled_data[:args.limit]
        logger.info(f"--limit={args.limit}，只检测前 {len(sampled_data)} 条")

    # 断点续跑
    completed = load_completed(OUTPUT_FILE)
    if completed:
        logger.info(f"已完成 {len(completed)} 条，跳过继续")

    todo = [r for r in sampled_data if r.get("patient_id") not in completed]
    if not todo:
        logger.info("所有记录已检测完毕!")
        return

    total_target = len(sampled_data)
    logger.info(f"待检测: {len(todo)} 条 | 并发数: {args.concurrent}")

    # 创建客户端
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    semaphore = asyncio.Semaphore(args.concurrent)

    # 分批处理
    batch_size = args.concurrent * 4
    total_done = len(completed)
    start_time = time.time()

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start:batch_start + batch_size]
        tasks = [eval_single(client, record, semaphore) for record in batch]
        results = await asyncio.gather(*tasks)

        # 写入结果
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        total_done += len(results)
        elapsed = time.time() - start_time
        speed = (total_done - len(completed)) / elapsed if elapsed > 0 else 0
        eta = (total_target - total_done) / speed if speed > 0 else 0

        # 统计当前批次
        errors = sum(1 for r in results if r["eval"].get("_api_error") or r["eval"].get("_parse_error"))
        issues_found = sum(1 for r in results if r["eval"].get("has_issue"))
        sf_count = sum(len(r["eval"].get("single_field_issues", [])) for r in results)
        cf_count = sum(len(r["eval"].get("cross_field_issues", [])) for r in results)

        logger.info(
            f"[{total_done}/{total_target}] "
            f"速度: {speed:.1f}条/s | "
            f"ETA: {eta / 60:.1f}min | "
            f"有问题: {issues_found} | "
            f"单字段: {sf_count} | "
            f"跨字段: {cf_count} | "
            f"错误: {errors}"
        )

    await client.close()
    logger.info(f"检测完成! 结果: {OUTPUT_FILE}")


# ============ 统计汇总 ============

def generate_summary():
    """生成汇总统计"""
    log = logger or logging.getLogger("detect")
    log.info("生成汇总统计...")

    results = []
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    total = len(results)
    parse_errors = sum(1 for r in results if r["eval"].get("_parse_error"))
    api_errors = sum(1 for r in results if r["eval"].get("_api_error"))
    has_issue_count = sum(1 for r in results if r["eval"].get("has_issue"))

    # ---- 单字段不合理统计 ----
    sf_type_counter = Counter()
    sf_severity_counter = Counter()
    sf_field_counter = Counter()
    all_sf_issues = []

    for r in results:
        for issue in r["eval"].get("single_field_issues", []):
            sf_type_counter[issue.get("type", "未知")] += 1
            sf_severity_counter[issue.get("severity", "未知")] += 1
            sf_field_counter[issue.get("field", "未知")] += 1
            all_sf_issues.append({
                **issue,
                "patient_id": r["patient_id"]
            })

    # ---- 字段间不合理统计 ----
    cf_type_counter = Counter()
    cf_severity_counter = Counter()
    cf_field_pair_counter = Counter()
    all_cf_issues = []

    for r in results:
        for issue in r["eval"].get("cross_field_issues", []):
            cf_type_counter[issue.get("type", "未知")] += 1
            cf_severity_counter[issue.get("severity", "未知")] += 1
            fa = issue.get("field_a", "?")
            fb = issue.get("field_b", "?")
            pair = f"{fa} <-> {fb}"
            cf_field_pair_counter[pair] += 1
            all_cf_issues.append({
                **issue,
                "patient_id": r["patient_id"]
            })

    # 有问题的样本列表
    issue_patient_ids = [r["patient_id"] for r in results if r["eval"].get("has_issue")]

    summary = {
        "total": total,
        "parse_errors": parse_errors,
        "api_errors": api_errors,
        "has_issue_count": has_issue_count,
        "has_issue_ratio": round(has_issue_count / total * 100, 2) if total else 0,
        "single_field": {
            "total_issues": len(all_sf_issues),
            "type_distribution": dict(sf_type_counter.most_common()),
            "severity_distribution": dict(sf_severity_counter.most_common()),
            "field_distribution": dict(sf_field_counter.most_common()),
        },
        "cross_field": {
            "total_issues": len(all_cf_issues),
            "type_distribution": dict(cf_type_counter.most_common()),
            "severity_distribution": dict(cf_severity_counter.most_common()),
            "field_pair_distribution": dict(cf_field_pair_counter.most_common()),
        },
        "issue_patient_ids": issue_patient_ids,
    }

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印报告
    log.info("=" * 60)
    log.info("检测汇总报告")
    log.info("=" * 60)
    log.info(f"总条数: {total}")
    log.info(f"解析错误: {parse_errors} | API错误: {api_errors}")
    log.info(f"有不合理之处: {has_issue_count} 条 ({summary['has_issue_ratio']}%)")

    log.info("")
    log.info(f"{'─' * 60}")
    log.info(f"【单字段不合理】共 {len(all_sf_issues)} 处")
    log.info(f"{'─' * 60}")
    if sf_type_counter:
        log.info("  类型分布:")
        for t, c in sf_type_counter.most_common():
            log.info(f"    {t}: {c}")
    if sf_severity_counter:
        log.info("  严重度分布:")
        for s, c in sf_severity_counter.most_common():
            log.info(f"    {s}: {c}")
    if sf_field_counter:
        log.info("  涉及字段:")
        for f_name, c in sf_field_counter.most_common():
            log.info(f"    {f_name}: {c}")

    log.info("")
    log.info(f"{'─' * 60}")
    log.info(f"【字段间不合理】共 {len(all_cf_issues)} 处")
    log.info(f"{'─' * 60}")
    if cf_type_counter:
        log.info("  类型分布:")
        for t, c in cf_type_counter.most_common():
            log.info(f"    {t}: {c}")
    if cf_severity_counter:
        log.info("  严重度分布:")
        for s, c in cf_severity_counter.most_common():
            log.info(f"    {s}: {c}")
    if cf_field_pair_counter:
        log.info("  字段对分布 (Top 15):")
        for pair, c in cf_field_pair_counter.most_common(15):
            log.info(f"    {pair}: {c}")

    # 展示典型案例
    if all_sf_issues:
        log.info("")
        log.info("典型单字段不合理案例 (前5条):")
        for item in all_sf_issues[:5]:
            log.info(f"  [{item.get('type')}] patient_id={item['patient_id']}")
            log.info(f"    字段: {item.get('field', '?')}")
            log.info(f"    证据: {str(item.get('evidence', ''))[:100]}")
            log.info(f"    说明: {item.get('detail', '')}")

    if all_cf_issues:
        log.info("")
        log.info("典型字段间不合理案例 (前5条):")
        for item in all_cf_issues[:5]:
            log.info(f"  [{item.get('type')}] patient_id={item['patient_id']}")
            log.info(f"    {item.get('field_a', '?')}: {str(item.get('value_a', ''))[:80]}")
            log.info(f"    {item.get('field_b', '?')}: {str(item.get('value_b', ''))[:80]}")
            log.info(f"    说明: {item.get('detail', '')}")

    log.info(f"\n汇总保存在: {SUMMARY_FILE}")


# ============ 日志配置 ============

def setup_logging():
    """配置日志：同时输出到终端和文件"""
    log_file = os.path.join(OUTPUT_DIR, "detect.log")
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    _logger = logging.getLogger("detect")
    _logger.setLevel(logging.INFO)
    _logger.handlers.clear()
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger


# 全局 logger
logger = None


# ============ 入口 ============

def main():
    global logger, OUTPUT_DIR, OUTPUT_FILE, SUMMARY_FILE, IO_LOG_FILE

    parser = argparse.ArgumentParser(description="精神科病历不合理之处检测")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"并发请求数 (默认 {MAX_CONCURRENT})")
    parser.add_argument("--limit", type=int, default=0,
                        help="只检测前 N 条 (默认 0=按抽样数)")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE,
                        help=f"抽样数量 (默认 {SAMPLE_SIZE}，设为0表示全量)")
    parser.add_argument("--input", type=str, default=None,
                        help="输入数据文件 (覆盖默认 INPUT_FILE)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="指定输出目录 (覆盖自动时间戳)")
    parser.add_argument("--summary-only", type=str, default=None, metavar="DIR",
                        help="仅对指定目录生成汇总统计")
    parser.add_argument("--resume", type=str, default=None, metavar="DIR",
                        help="断点续跑指定目录")
    args = parser.parse_args()

    # 覆盖输入文件
    if args.input:
        global INPUT_FILE
        INPUT_FILE = args.input

    # 确定输出目录
    if args.summary_only:
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, args.summary_only)
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.jsonl")
        SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.json")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger = setup_logging()
        generate_summary()
        return

    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    elif args.resume:
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, args.resume)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, timestamp)

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.jsonl")
    SUMMARY_FILE = os.path.join(OUTPUT_DIR, "summary.json")
    IO_LOG_FILE = os.path.join(OUTPUT_DIR, "io_log.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("精神科病历不合理之处检测")
    logger.info("=" * 60)
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"参数: concurrent={args.concurrent}, sample_size={args.sample_size}, limit={args.limit}")
    logger.info(f"模型: {MODEL_NAME} @ {API_BASE_URL}")

    asyncio.run(main_async(args))
    generate_summary()


if __name__ == "__main__":
    main()
