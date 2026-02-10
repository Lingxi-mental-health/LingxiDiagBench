#!/usr/bin/env python3
"""
基于锚点的精神科病历修复脚本（支持迭代修复）
锚点（不可修改）：cleaned_text（医患对话） + Diagnosis + DiagnosisCode
路径一（级联）：对话+诊断 → 修复现病史 → 修复主诉
路径二（独立）：修复性别、年龄、家族史、躯体病史、个人史等
使用 AsyncOpenAI + Semaphore 并发，支持断点续跑
"""

import json
import os
import sys
import time
import re
import logging
import asyncio
import argparse
import threading
from pathlib import Path
from datetime import datetime
from collections import Counter
from openai import AsyncOpenAI

# ============ 配置 ============
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "not-needed"
MODEL_NAME = "qwen3-32b"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_INPUT_FILE = os.path.join(PROJECT_DIR, "Cross-Field_Restoration",
                                  "LingxiDiag-16K_cross_field_fixed_v3.json")
DEFAULT_DETECTION_FILE = os.path.join(PROJECT_DIR, "Detect_inconsistencies",
                                      "output", "2026-02-07_17-02", "results.jsonl")
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# 运行时动态赋值
OUTPUT_DIR = None
OUTPUT_FILE = None
SUMMARY_FILE = None
IO_LOG_FILE = None

MAX_CONCURRENT = 16
MAX_TOKENS = 4096
TEMPERATURE = 0.1

# 不可修复的问题类型（涉及锚点字段或不可操作）
UNFIXABLE_SF_TYPES = {
    "诊断编码不匹配", "对话不自然", "对话重复",
    "对话自相矛盾", "对话模板化", "对话格式错误",
}
UNFIXABLE_CF_TYPES = {"诊断编码不匹配"}  # 诊断症状矛盾中有可修复案例(诊断↔现病史)，不再排除

# 需要修复的字段列表
FIXABLE_FIELDS = [
    "PresentIllnessHistory", "ChiefComplaint",
    "Age", "Gender", "AccompanyingPerson",
    "PersonalHistory", "FamilyHistory",
    "ImportantRelevantPhysicalIllnessHistory", "DrugAllergyHistory",
]

# ============ Prompt ============
SYSTEM_PROMPT = """你是一位精神科临床数据质量修复专家。你的任务是根据锚点字段（不可修改的权威信息源）修复病历中存在不合理之处的字段。

## 核心原则

1. **锚点不可修改**：以下字段为锚点，绝对不能修改：
   - cleaned_text（医患对话原文）—— 最高权威
   - Diagnosis（诊断）
   - DiagnosisCode（ICD编码）

2. **信息可信度排序**：对话原文 > 现病史 > 主诉 > 其他结构化字段

3. **修复顺序（级联修复）**：
   - **第一步**：根据对话原文 + 诊断信息，修复「现病史」(PresentIllnessHistory)
   - **第二步**：根据对话原文 + 诊断信息 + 修复后的现病史，修复「主诉」(ChiefComplaint)
   - **第三步**：根据全部上下文（含修复后的现病史和主诉），独立修复其他字段

4. **修复原则（优先级链是最高准则）**：
   - **无条件对齐**：只要低优先级字段与高优先级字段有任何事实性不一致（时间、数字、症状、事件、人物关系），必须修改低优先级字段，没有例外
   - 对话原文中的每一个事实性陈述都是权威。现病史、主诉、个人史等字段如有任何事实性出入，必须修改以对齐对话
   - 无问题的字段保持原值不变
   - 修复结果必须与锚点字段（对话原文、诊断、ICD编码）保持一致
   - 年龄、性别等字段以对话原文为准，对话中没有提及的则保留原值
   - **可以删除**：如果某段描述与对话直接矛盾且难以改写，直接删除该矛盾句子，保留不矛盾的部分即可。总体合理比保留原文更重要
   - **注意单位**：仔细区分"月"与"年"、"周"与"月"、"斤"与"公斤"、"天"与"周"，绝不能混淆
   - **误报判定**：仅当两个字段"措辞不同"但事实完全相同时（如"中药"="中成药"、"否认"="没有"）才算误报。时间/数值/事实性差异不算误报，一律修复

5. **冲突解决规则（严格执行）**：
   低优先级字段必须向高优先级字段对齐，不可犹豫：
   a. **病程时长矛盾**：对话说"2个月"但主诉说"3个月" → 修改主诉为"2个月"（对话是最高权威）
   b. **病例对话矛盾**：对话中提到某症状、行为或诱因，但现病史遗漏 → 补充到现病史中
   c. **陪同人矛盾**：对话或现病史提到"同事陪同就诊"但陪同人字段写"无" → 修改陪同人字段
   d. **工作/身份矛盾**：对话说患者有工作/在上学，但个人史写"无业" → 修改现病史措辞以匹配对话内容
   e. **年龄/个人史矛盾**：年龄与婚恋状况、工作等不符 → 以对话中的信息为准调整个人史
   f. **诊断-现病史矛盾**：诊断（锚点）与现病史中的症状描述不一致 → 调整现病史措辞使其与诊断方向一致。现病史是对对话的选择性总结，不必复述对话每个细节，应侧重于支持诊断结论的症状描述
   g. **诊断症状矛盾（现病史侧）**：不要删除症状描述，而是调整叙述重心。例如：
      - 诊断"非器质性失眠症"但现病史提到"情绪低落、兴趣减退" → 以失眠为核心叙述，其他症状改为"伴有..."
      - 诊断"焦虑状态"但现病史侧重抑郁 → 补充焦虑相关表述，使整体与焦虑方向一致
   h. **身份陪诊矛盾**：以对话为准确认真实身份，再修复对应字段
      - 对话确认在上学 → 陪同人"同事"改"同学"或"家长"
      - 对话确认在工作 → 个人史"学生"改为对应工作身份
   i. **时长矛盾系统性修复**：先从对话提取权威时间线，再统一修复
      - 如果主诉和现病史描述不同层面（总病程 vs 近期加重），保留各自时间描述但确保不矛盾
      - 主诉格式参考："核心症状+总病程，伴XX+加重时长"

## 修复示例（参考）

### 示例1：病程时长矛盾
问题：主诉说"3个月"，对话说"大概两个月"
修复前 ChiefComplaint: "情绪低落伴睡眠障碍3个月"
修复后 ChiefComplaint: "情绪低落伴睡眠障碍2个月"
理由：对话是最高权威，主诉时长必须对齐对话

### 示例2：病例对话矛盾（现病史遗漏诱因）
问题：对话提到"工作压力大"作为诱因，但现病史写"无明显诱因"
修复前 PresentIllnessHistory: "...无明显诱因出现情绪低落..."
修复后 PresentIllnessHistory: "...因工作压力出现情绪低落..."
理由：对话明确提到诱因，现病史不应写"无明显诱因"

### 示例3：陪同人矛盾
问题：现病史写"由母亲陪同就诊"，但陪同人字段为"无"
修复前 AccompanyingPerson: "无"
修复后 AccompanyingPerson: "母亲"
理由：现病史明确记载陪同人信息，陪同人字段需一致

### 示例4：诊断症状矛盾
问题：诊断"非器质性失眠症"，但现病史大段描述焦虑抑郁症状
修复前 PresentIllnessHistory: "...持续性多虑、情绪低落、兴趣减退。入睡困难、早醒..."
修复后 PresentIllnessHistory: "...入睡困难、早醒，睡眠质量差，伴有情绪低落、多虑等表现..."
理由：诊断是非器质性失眠症（锚点），现病史应以失眠为核心，其他症状作伴随

### 示例5：身份陪诊矛盾
问题：个人史写"学生"，陪同人写"同事"，对话显示患者在上学
修复前 AccompanyingPerson: "同事"
修复后 AccompanyingPerson: "同学"
理由：对话确认学生身份，同事应改为同学

### 示例6：病程时长矛盾（多时间线）
问题：主诉"情绪低落伴自伤行为1周"，对话说"半个月前开始不太好"
修复前 ChiefComplaint: "情绪低落伴自伤行为1周"
修复后 ChiefComplaint: "情绪低落半月余，伴自伤行为1周"
理由：主诉应区分总病程（对话=半个月）和自伤时间线（1周）"""

USER_PROMPT_TEMPLATE = """请根据以下病历信息和检测到的问题，按照级联修复顺序修复不合理之处。

## 锚点字段（不可修改）

### 诊断信息
- 诊断：{Diagnosis}
- ICD编码：{DiagnosisCode}

### 医患对话
{cleaned_text}

## 待修复字段（当前值）

- 现病史：{PresentIllnessHistory}
- 主诉：{ChiefComplaint}
- 年龄：{Age}
- 性别：{Gender}
- 陪同人：{AccompanyingPerson}
- 个人史：{PersonalHistory}
- 家族史：{FamilyHistory}
- 躯体疾病史：{ImportantRelevantPhysicalIllnessHistory}
- 药物过敏史：{DrugAllergyHistory}

## 检测到的问题

### 单字段问题
{single_field_issues_text}

### 跨字段问题
{cross_field_issues_text}

## 检测总结
{summary}

---

请按以下JSON格式输出修复结果（不要输出其他内容）：
{{"PresentIllnessHistory": "<修复后的现病史，无需修复则填原值>", "ChiefComplaint": "<修复后的主诉，无需修复则填原值>", "Age": "<修复后的年龄，无需修复则填原值>", "Gender": "<修复后的性别，无需修复则填原值>", "AccompanyingPerson": "<修复后的陪同人，无需修复则填原值>", "PersonalHistory": "<修复后的个人史，无需修复则填原值>", "FamilyHistory": "<修复后的家族史，无需修复则填原值>", "ImportantRelevantPhysicalIllnessHistory": "<修复后的躯体疾病史，无需修复则填原值>", "DrugAllergyHistory": "<修复后的药物过敏史，无需修复则填原值>", "change_log": [{{"field": "<修改的字段名>", "original": "<原值摘要>", "fixed": "<修改后值摘要>", "reason": "<修改理由，引用对话原文作为依据>"}}]}}

如果所有字段均无需修复，change_log填空数组[]，各字段填原值。"""

REVIEW_SYSTEM_PROMPT = """你是精神科病历质量审核专家。你的任务是检查修复后的病历字段是否与对话原文一致。"""

REVIEW_USER_TEMPLATE = """请检查以下修复后的病历字段是否与对话原文一致。

## 对话原文（最高权威，不可修改）
{cleaned_text}

## 诊断信息（不可修改）
- 诊断：{Diagnosis}
- ICD编码：{DiagnosisCode}

## 修复后的字段（需要检查）
- 现病史：{PresentIllnessHistory}
- 主诉：{ChiefComplaint}
- 年龄：{Age}
- 性别：{Gender}
- 陪同人：{AccompanyingPerson}
- 个人史：{PersonalHistory}
- 家族史：{FamilyHistory}
- 躯体疾病史：{ImportantRelevantPhysicalIllnessHistory}
- 药物过敏史：{DrugAllergyHistory}

## 检查要求
1. 逐句对比每个字段与对话原文，找出任何事实性不一致（时间、数字、症状、人物关系、事件）
2. 如果发现不一致，修正该字段使其与对话一致；如果某句难以修正，删除该句
3. 如果所有字段与对话一致，原样返回即可
4. 可以适当编造使整体通顺，但不能与对话原文矛盾
5. 注意单位：区分"月"与"年"、"周"与"月"、"斤"与"公斤"

请按以下JSON格式输出（不要输出其他内容）：
{{"PresentIllnessHistory": "<检查/修正后>", "ChiefComplaint": "<检查/修正后>", "Age": "<检查/修正后>", "Gender": "<检查/修正后>", "AccompanyingPerson": "<检查/修正后>", "PersonalHistory": "<检查/修正后>", "FamilyHistory": "<检查/修正后>", "ImportantRelevantPhysicalIllnessHistory": "<检查/修正后>", "DrugAllergyHistory": "<检查/修正后>", "change_log": [{{"field": "<字段名>", "original": "<修正前>", "fixed": "<修正后>", "reason": "<理由>"}}]}}"""

REVIEW_MAX_TOKENS = 2048
REVIEW_TEMPERATURE = 0.0

ITERATION_PREAMBLE = """## 重要提示 - 第{round}轮迭代修复

这是迭代修复的第{round}轮。之前已修复过{prev_round}轮，但检测发现仍存在以下问题。

**本轮修复要求**：
1. 你看到的「待修复字段」是**原始数据**（未经修复）
2. 「检测到的问题」来自上一轮修复结果的检测，描述的是修复后数据的残留问题
3. 你的任务：从原始数据出发，产出更好的修复，同时解决这些残留问题
4. 避免上一轮的错误：参考下方「上一轮修复尝试」，理解之前哪里做得不够好

{prev_fix_context}
"""


# ============ 工具函数 ============

def strip_thinking(text: str) -> str:
    """去除 Qwen3 的 <think>...</think> 标签"""
    if not text:
        return text
    result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)
    return result.strip()


def repair_json_quotes(json_str: str) -> str:
    """修复 JSON 字符串值中未转义的双引号（如中文引述 "有人想害自己"）"""
    result = []
    in_string = False
    escape_next = False
    i = 0
    while i < len(json_str):
        c = json_str[i]
        if escape_next:
            result.append(c)
            escape_next = False
        elif c == '\\':
            result.append(c)
            escape_next = True
        elif c == '"':
            if not in_string:
                result.append(c)
                in_string = True
            else:
                # 检查这个引号是结束字符串还是内部引号
                rest = json_str[i + 1:].lstrip()
                if not rest or rest[0] in ':,}]':
                    # 后面是 JSON 结构字符 → 结束字符串
                    result.append(c)
                    in_string = False
                else:
                    # 后面不是结构字符 → 内部引号，需要转义
                    result.append('\\"')
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def try_parse_json(text: str) -> dict:
    """尝试解析 JSON，失败则尝试修复未转义引号后重试"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 尝试修复未转义的引号
    repaired = repair_json_quotes(text)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def parse_response(text: str) -> dict:
    """解析 LLM 返回的 JSON，带多级容错"""
    cleaned = strip_thinking(text)

    # 尝试直接解析（含引号修复）
    result = try_parse_json(cleaned)
    if result is not None:
        return result

    # 尝试从 markdown code block 中提取
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if match:
        result = try_parse_json(match.group(1))
        if result is not None:
            return result

    # 尝试找第一个 { 到最后一个 }
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start != -1 and end != -1 and end > start:
        result = try_parse_json(cleaned[start:end + 1])
        if result is not None:
            return result

    # 解析失败
    return {"_parse_error": True}


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


def load_detection_results(detection_file: str) -> dict:
    """加载检测结果，返回 {patient_id: eval_result}"""
    results = {}
    with open(detection_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    pid = record.get("patient_id")
                    eval_data = record.get("eval", {})
                    if pid and eval_data.get("has_issue"):
                        results[pid] = eval_data
                except (json.JSONDecodeError, KeyError):
                    pass
    return results


def load_prev_fix_results(prev_fix_file: str) -> dict:
    """加载上一轮修复结果，返回 {patient_id: fix_result}"""
    results = {}
    if not prev_fix_file or not os.path.exists(prev_fix_file):
        return results
    with open(prev_fix_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if r.get("status") == "fixed" and r.get("num_changes", 0) > 0:
                        results[r["patient_id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
    return results


def format_issues(eval_result: dict) -> tuple:
    """格式化检测到的问题，过滤不可修复的类型"""
    sf_issues = eval_result.get("single_field_issues", [])
    cf_issues = eval_result.get("cross_field_issues", [])

    # 过滤不可修复类型
    sf_issues = [i for i in sf_issues if i.get("type") not in UNFIXABLE_SF_TYPES]
    cf_issues = [i for i in cf_issues if i.get("type") not in UNFIXABLE_CF_TYPES]

    if not sf_issues:
        sf_text = "无"
    else:
        sf_lines = []
        for i, issue in enumerate(sf_issues, 1):
            sf_lines.append(
                f"{i}. [{issue.get('type')}] 字段:{issue.get('field')} | "
                f"严重度:{issue.get('severity')} | {issue.get('detail', '')}"
            )
        sf_text = "\n".join(sf_lines)

    if not cf_issues:
        cf_text = "无"
    else:
        cf_lines = []
        for i, issue in enumerate(cf_issues, 1):
            cf_lines.append(
                f"{i}. [{issue.get('type')}] {issue.get('field_a', '?')} vs "
                f"{issue.get('field_b', '?')} | 严重度:{issue.get('severity')} | "
                f"{issue.get('detail', '')}"
            )
        cf_text = "\n".join(cf_lines)

    return sf_text, cf_text


def should_fix(eval_result: dict) -> bool:
    """判断记录是否有可修复的问题"""
    sf = eval_result.get("single_field_issues", [])
    cf = eval_result.get("cross_field_issues", [])

    # 有跨字段问题，基本都值得修复
    if cf:
        fixable_cf = [i for i in cf if i.get("type") not in UNFIXABLE_CF_TYPES]
        if fixable_cf:
            return True

    # 有可修复的单字段问题
    for issue in sf:
        if issue.get("type") not in UNFIXABLE_SF_TYPES:
            return True

    return False


def build_fix_prompt(record: dict, eval_result: dict,
                     round_num: int = 1, prev_fix: dict = None,
                     use_no_think: bool = False) -> str:
    """构建修复 prompt，支持迭代轮次"""
    sf_text, cf_text = format_issues(eval_result)
    summary = eval_result.get("summary", "")

    # 迭代前缀
    preamble = ""
    if round_num >= 2:
        prev_fix_context = ""
        if prev_fix and prev_fix.get("actual_changes"):
            changes_lines = []
            for ch in prev_fix["actual_changes"]:
                changes_lines.append(
                    f"- {ch['field']}: \"{ch.get('original', '')[:80]}\" → "
                    f"\"{ch.get('fixed', '')[:80]}\""
                )
            prev_fix_context = (
                "### 上一轮修复尝试（参考）\n"
                "上一轮将以下字段做了修改，但检测仍发现残留问题：\n" +
                "\n".join(changes_lines) +
                "\n\n请分析上一轮修改为何未能解决问题，产出更好的修复。"
            )
        preamble = ITERATION_PREAMBLE.format(
            round=round_num,
            prev_round=round_num - 1,
            prev_fix_context=prev_fix_context,
        )

    base_prompt = USER_PROMPT_TEMPLATE.format(
        Diagnosis=record.get("Diagnosis", ""),
        DiagnosisCode=record.get("DiagnosisCode", ""),
        cleaned_text=record.get("cleaned_text", ""),
        PresentIllnessHistory=record.get("PresentIllnessHistory", ""),
        ChiefComplaint=record.get("ChiefComplaint", ""),
        Age=record.get("Age", ""),
        Gender=record.get("Gender", ""),
        AccompanyingPerson=record.get("AccompanyingPerson", ""),
        PersonalHistory=record.get("PersonalHistory", ""),
        FamilyHistory=record.get("FamilyHistory", ""),
        ImportantRelevantPhysicalIllnessHistory=record.get(
            "ImportantRelevantPhysicalIllnessHistory", ""),
        DrugAllergyHistory=record.get("DrugAllergyHistory", ""),
        single_field_issues_text=sf_text,
        cross_field_issues_text=cf_text,
        summary=summary,
    )

    prompt = preamble + base_prompt

    if use_no_think:
        prompt += "\n/no_think"

    return prompt


def build_review_prompt(record: dict, fixed_fields: dict,
                        use_no_think: bool = False) -> str:
    """构建自查 prompt：用修复后的字段 + 原始对话让模型检查一致性"""
    # 用修复后的字段值构建 prompt
    prompt = REVIEW_USER_TEMPLATE.format(
        cleaned_text=record.get("cleaned_text", ""),
        Diagnosis=record.get("Diagnosis", ""),
        DiagnosisCode=record.get("DiagnosisCode", ""),
        PresentIllnessHistory=fixed_fields.get("PresentIllnessHistory", ""),
        ChiefComplaint=fixed_fields.get("ChiefComplaint", ""),
        Age=fixed_fields.get("Age", ""),
        Gender=fixed_fields.get("Gender", ""),
        AccompanyingPerson=fixed_fields.get("AccompanyingPerson", ""),
        PersonalHistory=fixed_fields.get("PersonalHistory", ""),
        FamilyHistory=fixed_fields.get("FamilyHistory", ""),
        ImportantRelevantPhysicalIllnessHistory=fixed_fields.get(
            "ImportantRelevantPhysicalIllnessHistory", ""),
        DrugAllergyHistory=fixed_fields.get("DrugAllergyHistory", ""),
    )
    if use_no_think:
        prompt += "\n/no_think"
    return prompt


def compare_fields(record: dict, fixed: dict) -> list:
    """比对原始字段和修复后的字段，生成变更列表"""
    changes = []
    for field in FIXABLE_FIELDS:
        original = record.get(field, "")
        new_val = fixed.get(field, "")
        if new_val and str(new_val).strip() != str(original).strip():
            changes.append({
                "field": field,
                "original": str(original)[:200],
                "fixed": str(new_val)[:200],
            })
    return changes


# ============ 异步处理 ============

_io_log_lock = threading.Lock()


def write_io_log(patient_id: str, messages: list, raw_output: str):
    """将完整 I/O 写入调试日志"""
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


async def fix_single(client: AsyncOpenAI, record: dict, eval_result: dict,
                     semaphore: asyncio.Semaphore,
                     round_num: int = 1, prev_fix: dict = None,
                     use_no_think: bool = False,
                     enable_review: bool = True) -> dict:
    """修复单条记录（含可选的自查步骤）"""
    async with semaphore:
        patient_id = record.get("patient_id", "unknown")
        user_prompt = build_fix_prompt(record, eval_result,
                                       round_num=round_num,
                                       prev_fix=prev_fix,
                                       use_no_think=use_no_think)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_output = ""
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                raw_output = response.choices[0].message.content.strip()
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    write_io_log(patient_id, messages, str(e))
                    return {
                        "patient_id": patient_id,
                        "status": "error",
                        "fixed_fields": {},
                        "change_log": [],
                        "num_changes": 0,
                        "raw_output": f"[API错误] {str(e)}",
                    }

        # 记录 I/O
        write_io_log(patient_id, messages, raw_output)

        # 解析响应
        parsed = parse_response(raw_output)
        if parsed.get("_parse_error"):
            return {
                "patient_id": patient_id,
                "status": "parse_error",
                "fixed_fields": {},
                "change_log": [],
                "num_changes": 0,
                "raw_output": raw_output,
            }

        # 提取修复后的字段
        fixed_fields = {}
        for field in FIXABLE_FIELDS:
            val = parsed.get(field)
            if val is not None:
                fixed_fields[field] = str(val)
            else:
                # LLM 没返回该字段，保留原值
                fixed_fields[field] = record.get(field, "")

        # LLM 返回的 change_log
        llm_change_log = parsed.get("change_log", [])

        # 实际对比变更（以实际字段值差异为准）
        actual_changes = compare_fields(record, fixed_fields)
        num_changes = len(actual_changes)

        # ---- Step 2: 自查（仅当有改动且启用 review 时） ----
        review_changes = []
        if enable_review and num_changes > 0:
            review_prompt = build_review_prompt(record, fixed_fields,
                                                use_no_think=use_no_think)
            review_messages = [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": review_prompt},
            ]
            review_raw = ""
            try:
                review_response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=review_messages,
                    max_tokens=REVIEW_MAX_TOKENS,
                    temperature=REVIEW_TEMPERATURE,
                )
                review_raw = review_response.choices[0].message.content.strip()
            except Exception as e:
                review_raw = f"[Review API错误] {str(e)}"

            write_io_log(f"{patient_id}_review", review_messages, review_raw)

            review_parsed = parse_response(review_raw)
            if not review_parsed.get("_parse_error"):
                # 将 review 的修正应用到 fixed_fields
                for field in FIXABLE_FIELDS:
                    review_val = review_parsed.get(field)
                    if review_val is not None:
                        review_val_str = str(review_val).strip()
                        current_val = fixed_fields.get(field, "").strip()
                        if review_val_str and review_val_str != current_val:
                            fixed_fields[field] = review_val_str
                            review_changes.append({
                                "field": field,
                                "before_review": current_val[:200],
                                "after_review": review_val_str[:200],
                            })

                # 重新计算 actual_changes
                if review_changes:
                    actual_changes = compare_fields(record, fixed_fields)
                    num_changes = len(actual_changes)
                    llm_change_log.extend(review_parsed.get("change_log", []))

        status = "fixed" if num_changes > 0 else "unchanged"

        return {
            "patient_id": patient_id,
            "status": status,
            "fixed_fields": fixed_fields,
            "change_log": llm_change_log,
            "actual_changes": actual_changes,
            "num_changes": num_changes,
            "review_changes": review_changes,
            "raw_output": raw_output,
        }


async def main_async(args, input_file: str, detection_file: str):
    """异步主函数"""
    round_num = args.round
    use_no_think = args.no_think
    enable_review = not args.no_review

    # 加载原始数据
    logger.info(f"加载原始数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_index = {r["patient_id"]: r for r in data}
    logger.info(f"共 {len(data)} 条原始记录")

    # 加载检测结果
    logger.info(f"加载检测结果: {detection_file}")
    detection = load_detection_results(detection_file)
    logger.info(f"共 {len(detection)} 条有问题的检测结果")

    # 加载上一轮修复结果（如果指定了）
    prev_fixes = {}
    if args.prev_fix:
        logger.info(f"加载上一轮修复结果: {args.prev_fix}")
        prev_fixes = load_prev_fix_results(args.prev_fix)
        logger.info(f"其中 {len(prev_fixes)} 条有修复记录")

    # 过滤有可修复问题的记录
    fixable_pids = [pid for pid, ev in detection.items() if should_fix(ev)]
    logger.info(f"其中 {len(fixable_pids)} 条有可修复的问题")

    # --limit 限制条数
    if args.limit and args.limit > 0:
        fixable_pids = fixable_pids[:args.limit]
        logger.info(f"--limit={args.limit}，只修复前 {len(fixable_pids)} 条")

    # 断点续跑
    completed = load_completed(OUTPUT_FILE)
    if completed:
        logger.info(f"已完成 {len(completed)} 条，跳过继续")

    todo_pids = [pid for pid in fixable_pids if pid not in completed]
    if not todo_pids:
        logger.info("所有记录已修复完毕!")
        return

    total_target = len(fixable_pids)
    logger.info(f"待修复: {len(todo_pids)} 条 | 并发数: {args.concurrent}")
    logger.info(f"迭代轮次: {round_num} | /no_think: {use_no_think} | 自查: {enable_review}")

    # 创建客户端
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    semaphore = asyncio.Semaphore(args.concurrent)

    # 分批处理
    batch_size = args.concurrent * 4
    total_done = len(completed)
    start_time = time.time()

    for batch_start in range(0, len(todo_pids), batch_size):
        batch_pids = todo_pids[batch_start:batch_start + batch_size]
        tasks = []
        for pid in batch_pids:
            record = data_index.get(pid)
            eval_result = detection.get(pid)
            if record and eval_result:
                prev_fix = prev_fixes.get(pid)
                tasks.append(fix_single(
                    client, record, eval_result, semaphore,
                    round_num=round_num, prev_fix=prev_fix,
                    use_no_think=use_no_think,
                    enable_review=enable_review,
                ))

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
        n_fixed = sum(1 for r in results if r["status"] == "fixed")
        n_unchanged = sum(1 for r in results if r["status"] == "unchanged")
        n_errors = sum(1 for r in results if r["status"] in ("error", "parse_error"))
        total_changes = sum(r["num_changes"] for r in results)

        logger.info(
            f"[{total_done}/{total_target}] "
            f"速度: {speed:.1f}条/s | "
            f"ETA: {eta / 60:.1f}min | "
            f"已修复: {n_fixed} | "
            f"无变化: {n_unchanged} | "
            f"变更数: {total_changes} | "
            f"错误: {n_errors}"
        )

    await client.close()
    logger.info(f"修复完成! 结果: {OUTPUT_FILE}")


# ============ 统计汇总 ============

def generate_summary():
    """生成修复统计汇总"""
    log = logger or logging.getLogger("fix")
    log.info("生成修复汇总...")

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
    n_fixed = sum(1 for r in results if r.get("status") == "fixed")
    n_unchanged = sum(1 for r in results if r.get("status") == "unchanged")
    n_error = sum(1 for r in results if r.get("status") == "error")
    n_parse_error = sum(1 for r in results if r.get("status") == "parse_error")

    # 按字段统计变更
    field_change_counter = Counter()
    total_changes = 0
    for r in results:
        for change in r.get("actual_changes", []):
            field_change_counter[change["field"]] += 1
            total_changes += 1

    # 统计 review 自查修正
    n_review_corrected = sum(1 for r in results if r.get("review_changes"))
    total_review_changes = sum(len(r.get("review_changes", [])) for r in results)

    summary = {
        "total": total,
        "fixed": n_fixed,
        "unchanged": n_unchanged,
        "error": n_error,
        "parse_error": n_parse_error,
        "total_changes": total_changes,
        "review_corrected": n_review_corrected,
        "review_changes": total_review_changes,
        "field_change_distribution": dict(field_change_counter.most_common()),
    }

    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 打印报告
    log.info("=" * 60)
    log.info("修复汇总报告")
    log.info("=" * 60)
    log.info(f"总条数: {total}")
    log.info(f"已修复: {n_fixed} ({n_fixed/total*100:.1f}%)" if total else "已修复: 0")
    log.info(f"无变化: {n_unchanged}")
    log.info(f"错误: {n_error} | 解析错误: {n_parse_error}")
    log.info(f"总变更数: {total_changes}")
    log.info(f"自查修正: {n_review_corrected} 条记录, {total_review_changes} 处修正")
    log.info("")
    log.info("各字段变更次数:")
    for field, count in field_change_counter.most_common():
        log.info(f"  {field}: {count}")
    log.info(f"\n汇总保存在: {SUMMARY_FILE}")


# ============ 日志配置 ============

def setup_logging():
    """配置日志：同时输出到终端和文件"""
    log_file = os.path.join(OUTPUT_DIR, "fix.log")
    formatter = logging.Formatter("%(asctime)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    _logger = logging.getLogger("fix")
    _logger.setLevel(logging.INFO)
    # 清理旧 handler（迭代调用时避免重复）
    _logger.handlers.clear()
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger


# 全局 logger
logger = None


# ============ 入口 ============

def main():
    global logger, OUTPUT_DIR, OUTPUT_FILE, SUMMARY_FILE, IO_LOG_FILE

    parser = argparse.ArgumentParser(description="基于锚点的精神科病历修复（支持迭代）")
    parser.add_argument("--concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"并发请求数 (默认 {MAX_CONCURRENT})")
    parser.add_argument("--limit", type=int, default=0,
                        help="只修复前 N 条 (默认 0=全量)")
    parser.add_argument("--input", type=str, default=None,
                        help=f"原始数据文件 (默认 {DEFAULT_INPUT_FILE})")
    parser.add_argument("--detection", type=str, default=None,
                        help=f"检测结果文件 (默认 {DEFAULT_DETECTION_FILE})")
    parser.add_argument("--round", type=int, default=1,
                        help="迭代轮次 (>=2 时启用迭代 prompt，默认 1)")
    parser.add_argument("--prev-fix", type=str, default=None,
                        help="上一轮 fix_results.jsonl (供迭代 prompt 参考)")
    parser.add_argument("--no-think", action="store_true",
                        help="禁用 Qwen3 思考模式 (追加 /no_think)")
    parser.add_argument("--no-review", action="store_true",
                        help="禁用修复后自查步骤")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="指定输出目录 (覆盖自动时间戳)")
    parser.add_argument("--summary-only", type=str, default=None, metavar="DIR",
                        help="仅对指定目录生成汇总统计")
    parser.add_argument("--resume", type=str, default=None, metavar="DIR",
                        help="断点续跑指定目录")
    args = parser.parse_args()

    # 确定输入文件
    input_file = args.input or DEFAULT_INPUT_FILE
    detection_file = args.detection or DEFAULT_DETECTION_FILE

    # 确定输出目录
    if args.summary_only:
        OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, args.summary_only)
        OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fix_results.jsonl")
        SUMMARY_FILE = os.path.join(OUTPUT_DIR, "fix_summary.json")
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

    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "fix_results.jsonl")
    SUMMARY_FILE = os.path.join(OUTPUT_DIR, "fix_summary.json")
    IO_LOG_FILE = os.path.join(OUTPUT_DIR, "fix_io_log.jsonl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("基于锚点的精神科病历修复")
    logger.info("=" * 60)
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"参数: concurrent={args.concurrent}, limit={args.limit}, "
                f"round={args.round}, no_think={args.no_think}, "
                f"review={not args.no_review}")
    logger.info(f"模型: {MODEL_NAME} @ {API_BASE_URL}")
    logger.info(f"原始数据: {input_file}")
    logger.info(f"检测结果: {detection_file}")
    if args.prev_fix:
        logger.info(f"上一轮修复: {args.prev_fix}")

    asyncio.run(main_async(args, input_file, detection_file))
    generate_summary()


if __name__ == "__main__":
    main()
