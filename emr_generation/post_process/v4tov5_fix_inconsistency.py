#!/usr/bin/env python3
"""
跨字段一致性修复脚本 - 使用本地 Qwen-32B
优先级: 对话 > 现病史 > 其他

修复类型:
1. 病程矛盾: 修改主诉（以现病史/对话为准）       → LLM
2. 年龄矛盾: 修改Age字段（以现病史为准）          → 规则
3. 婚姻状态矛盾: 修改个人史（以对话>现病史为准）  → LLM
4. 陪诊人矛盾: 修改陪诊人（以对话>现病史为准）    → LLM
5. 诊断vs年龄: "童年青少年期发病"→ Age改为10-18   → 规则
6. 性别vs躯体疾病史 / 个人史内部矛盾              → 标记待人工审核
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
from typing import Dict, List, Optional, Tuple
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
OUTPUT_FILE = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/Cross-Field_Restoration/LingxiDiag-16K_cross_field_fixed.json"
LOG_DIR = "/tcci_mnt/xiaoming/Lingxi_annotation_0111/QWEN-32B_for_LingxiDiag-16k/Cross-Field_Restoration/log"

stats_lock = threading.Lock()
file_lock = threading.Lock()
REALTIME_FILE = None


# ============== 工具函数 ==============

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
    log_file = f"{LOG_DIR}/fix_cross_field_{timestamp}.log"
    REALTIME_FILE = f"{LOG_DIR}/fix_cross_field_{timestamp}.jsonl"

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


def get_dialogue_excerpt(text: str, max_turns: int = 10) -> str:
    """取对话前N轮"""
    if not text:
        return ""
    lines = text.split('\n')
    return '\n'.join(lines[:max_turns * 2])


# ============== 规则修复 ==============

def rule_fix_age_from_phi(record: Dict) -> Optional[Dict]:
    """规则: 从现病史提取年龄，覆盖Age字段"""
    phi = record.get('PresentIllnessHistory', '')
    m = re.search(r'患者(?:为)?(\d{1,3})岁', phi)
    if not m:
        return None

    phi_age = int(m.group(1))
    try:
        field_age = int(record.get('Age', ''))
    except (ValueError, TypeError):
        return None

    if abs(phi_age - field_age) > 2:
        return {
            'field': 'Age',
            'original': str(field_age),
            'fixed': str(phi_age),
            'reason': f'现病史写"患者{phi_age}岁"，Age字段为{field_age}，以现病史为准',
            'type': '年龄矛盾'
        }
    return None


def rule_fix_age_from_diagnosis(record: Dict) -> Optional[Dict]:
    """规则: 诊断含"童年青少年期发病"但年龄不符，改Age为10-18"""
    diagnosis = record.get('Diagnosis', '')
    if not re.search(r'童年|青少年', diagnosis):
        return None

    try:
        age = int(record.get('Age', ''))
    except (ValueError, TypeError):
        return None

    if age > 18:
        new_age = random.randint(10, 18)
        return {
            'field': 'Age',
            'original': str(age),
            'fixed': str(new_age),
            'reason': f'诊断为"{diagnosis[:30]}"(童年青少年期发病)，Age={age}不符，随机修改为{new_age}',
            'type': '诊断vs年龄矛盾'
        }
    return None


def rule_fix_gender_body(record: Dict) -> Optional[Dict]:
    """规则: 性别与躯体疾病史矛盾，标记待人工审核"""
    gender = record.get('Gender', '')
    body = record.get('ImportantRelevantPhysicalIllnessHistory', '')
    if not gender or not body:
        return None

    female_diseases = ['卵巢', '子宫', '多囊', '巧克力囊肿', '宫颈', '经期', '月经']
    male_diseases = ['前列腺', '睾丸']

    if gender == '男':
        for d in female_diseases:
            if d in body:
                return {
                    'field': '_MANUAL_REVIEW',
                    'original': f'Gender={gender}, 躯体疾病史={body}',
                    'fixed': '',
                    'reason': f'性别为男但躯体疾病史含"{d}"，需人工判断',
                    'type': '性别vs躯体疾病史'
                }

    if gender == '女':
        for d in male_diseases:
            if d in body:
                return {
                    'field': '_MANUAL_REVIEW',
                    'original': f'Gender={gender}, 躯体疾病史={body}',
                    'fixed': '',
                    'reason': f'性别为女但躯体疾病史含"{d}"，需人工判断',
                    'type': '性别vs躯体疾病史'
                }
    return None


def rule_fix_personality_conflict(record: Dict) -> Optional[Dict]:
    """规则: 个人史中性格同时包含内向和外向，标记待人工审核"""
    personal = record.get('PersonalHistory', '')
    if '内向' in personal and '外向' in personal:
        return {
            'field': '_MANUAL_REVIEW',
            'original': personal,
            'fixed': '',
            'reason': '个人史中性格同时包含"内向"和"外向"，需人工判断',
            'type': '个人史内部矛盾'
        }
    return None


# ============== 规则预检测（判断是否需要LLM修复） ==============

def needs_llm_fix(record: Dict) -> List[str]:
    """快速预检测，判断哪些矛盾需要LLM修复"""
    needed = []

    cc = record.get('ChiefComplaint', '')
    phi = record.get('PresentIllnessHistory', '')
    personal = record.get('PersonalHistory', '')
    acc = record.get('AccompanyingPerson', '')
    dialogue = record.get('cleaned_text', '')

    # 1. 病程矛盾预检：主诉和现病史都有时间表达
    cc_times = re.findall(r'(\d+)\s*(?:个)?\s*(年|月|周|天)', cc)
    phi_times = re.findall(r'(?:约|近)?(\d+)\s*(?:个)?\s*(年|月|周|天)(?:前|来)', phi[:200])
    if cc_times and phi_times:
        # 简单估算是否差异大
        def to_days(val, unit):
            val = float(val)
            if unit == '年': return val * 365
            if unit == '月': return val * 30
            if unit == '周': return val * 7
            return val
        cc_max = max(to_days(v, u) for v, u in cc_times)
        phi_max = max(to_days(v, u) for v, u in phi_times)
        if cc_max > 0 and phi_max > 0:
            ratio = max(cc_max, phi_max) / min(cc_max, phi_max)
            if ratio >= 3:
                needed.append('duration')

    # 2. 婚姻状态预检
    personal_married = bool(re.search(r'已婚|未婚|离婚|丧偶', personal))
    phi_marriage = bool(re.search(r'离婚|结婚|婚姻|丧偶|丈夫|妻子|配偶|爱人', phi))
    dlg_marriage = bool(re.search(r'离婚|离了婚|分手|结婚|老公|老婆|丈夫|妻子|前夫|前妻|对象|男朋友|女朋友', dialogue))
    if personal_married and (phi_marriage or dlg_marriage):
        needed.append('marriage')

    # 3. 陪诊人预检
    if acc:
        acc_alone = acc in ['自来', '本人']
        acc_has = '有' in acc and not acc_alone
        dlg_companion = bool(re.search(r'(?:陪我来|一起来|带我来|在外面等|陪着来|陪同)', dialogue))
        dlg_alone = bool(re.search(r'(?:自己来|一个人来|我自己过来)', dialogue))
        phi_companion = bool(re.search(r'(?:陪同就诊|家属陪同|由.*陪同)', phi))
        phi_alone = bool(re.search(r'(?:独自就诊|自行就诊|自行前来)', phi))

        if (acc_alone and (dlg_companion or phi_companion)) or \
           (acc_has and (dlg_alone or phi_alone)):
            needed.append('accompanying')

    return needed


# ============== LLM 修复 Prompts ==============

PROMPT_DURATION = """你是一个医疗数据质量修复员。以下病历的【主诉】中的病程时间与【现病史】存在明显矛盾（差异>=3倍）。

请根据【现病史】和【医患对话】中的实际病程信息，修改【主诉】中的时间表述，使其与现病史一致。

规则：
- 仅修改主诉中的时间/病程部分，保留原有症状描述不变
- 以现病史的起病时间为准
- 如果对话中有更明确的时间信息，以对话为准
- 输出格式固定为JSON

【主诉】{chief_complaint}
【现病史】{present_illness}
【医患对话（前10轮）】
{dialogue_excerpt}

请按JSON格式输出（不要输出其他内容）:
{{"fixed_chief_complaint": "修改后的完整主诉文本", "reason": "简要说明修改了什么"}}"""

PROMPT_MARRIAGE = """你是一个医疗数据质量修复员。以下病历的【个人史】中的婚恋情况可能与【医患对话】或【现病史】存在矛盾。

请根据对话和现病史中的实际信息，判断是否需要修改个人史中的"婚恋情况"。

规则：
- 优先以对话中患者的自述为准，其次以现病史为准
- 仅修改"婚恋情况：xxx"部分，个人史中其他信息保持完全不变
- 如果没有矛盾，fixed_personal_history填原文即可

【个人史】{personal_history}
【现病史】{present_illness}
【医患对话（前10轮）】
{dialogue_excerpt}

请按JSON格式输出（不要输出其他内容）:
{{"has_fix": true/false, "fixed_personal_history": "修改后的完整个人史文本", "reason": "简要说明"}}"""

PROMPT_ACCOMPANYING = """你是一个医疗数据质量修复员。以下病历的【陪诊人】字段可能与【医患对话】或【现病史】存在矛盾。

请根据对话和现病史中的实际信息，判断是否需要修正陪诊人字段。

规则：
- 优先以对话中患者的自述为准，其次以现病史为准
- 格式举例："有 关系：父亲"、"有 关系：丈夫"、"自来"
- 如果没有矛盾，fixed_accompanying填原文即可

【陪诊人】{accompanying}
【现病史】{present_illness}
【医患对话（前10轮）】
{dialogue_excerpt}

请按JSON格式输出（不要输出其他内容）:
{{"has_fix": true/false, "fixed_accompanying": "修改后的陪诊人", "reason": "简要说明"}}"""


def parse_json_response(raw: str) -> Dict:
    text = strip_thinking(raw)
    json_match = re.search(r'\{[\s\S]*?\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ============== 单条记录处理 ==============

def process_single(idx: int, record: Dict) -> Dict:
    """处理单条记录：先规则修复，再LLM修复"""
    patient_id = record.get("patient_id", f"unknown_{idx}")
    all_fixes = []
    errors = []

    # ====== 规则修复 ======
    for rule_fn in [rule_fix_age_from_phi, rule_fix_age_from_diagnosis,
                    rule_fix_gender_body, rule_fix_personality_conflict]:
        try:
            fix = rule_fn(record)
            if fix:
                all_fixes.append(fix)
        except Exception as e:
            errors.append(f"rule_{rule_fn.__name__}: {e}")

    # ====== LLM修复预检 ======
    llm_needed = needs_llm_fix(record)

    dialogue_excerpt = get_dialogue_excerpt(record.get('cleaned_text', ''))

    # 病程矛盾修复
    if 'duration' in llm_needed:
        try:
            prompt = PROMPT_DURATION.format(
                chief_complaint=record.get('ChiefComplaint', ''),
                present_illness=record.get('PresentIllnessHistory', ''),
                dialogue_excerpt=dialogue_excerpt
            )
            raw = call_llm([{"role": "user", "content": prompt}])
            result = parse_json_response(raw)
            fixed_cc = result.get('fixed_chief_complaint', '')
            if fixed_cc and fixed_cc != record.get('ChiefComplaint', ''):
                all_fixes.append({
                    'field': 'ChiefComplaint',
                    'original': record.get('ChiefComplaint', ''),
                    'fixed': fixed_cc,
                    'reason': result.get('reason', '病程矛盾修复'),
                    'type': '病程矛盾'
                })
        except Exception as e:
            errors.append(f"llm_duration: {e}")

    # 婚姻状态矛盾修复
    if 'marriage' in llm_needed:
        try:
            prompt = PROMPT_MARRIAGE.format(
                personal_history=record.get('PersonalHistory', ''),
                present_illness=record.get('PresentIllnessHistory', ''),
                dialogue_excerpt=dialogue_excerpt
            )
            raw = call_llm([{"role": "user", "content": prompt}])
            result = parse_json_response(raw)
            if result.get('has_fix') and result.get('fixed_personal_history'):
                fixed_ph = result['fixed_personal_history']
                if fixed_ph != record.get('PersonalHistory', ''):
                    all_fixes.append({
                        'field': 'PersonalHistory',
                        'original': record.get('PersonalHistory', ''),
                        'fixed': fixed_ph,
                        'reason': result.get('reason', '婚姻状态矛盾修复'),
                        'type': '婚姻状态矛盾'
                    })
        except Exception as e:
            errors.append(f"llm_marriage: {e}")

    # 陪诊人矛盾修复
    if 'accompanying' in llm_needed:
        try:
            prompt = PROMPT_ACCOMPANYING.format(
                accompanying=record.get('AccompanyingPerson', ''),
                present_illness=record.get('PresentIllnessHistory', ''),
                dialogue_excerpt=dialogue_excerpt
            )
            raw = call_llm([{"role": "user", "content": prompt}])
            result = parse_json_response(raw)
            if result.get('has_fix') and result.get('fixed_accompanying'):
                fixed_acc = result['fixed_accompanying']
                if fixed_acc != record.get('AccompanyingPerson', ''):
                    all_fixes.append({
                        'field': 'AccompanyingPerson',
                        'original': record.get('AccompanyingPerson', ''),
                        'fixed': fixed_acc,
                        'reason': result.get('reason', '陪诊人矛盾修复'),
                        'type': '陪诊人矛盾'
                    })
        except Exception as e:
            errors.append(f"llm_accompanying: {e}")

    # 构建输出
    output = {
        "idx": idx,
        "patient_id": patient_id,
        "fixes": all_fixes,
        "errors": errors,
        "status": "error" if errors and not all_fixes else ("fixed" if all_fixes else "ok")
    }

    if all_fixes:
        fix_types = [f['type'] for f in all_fixes]
        logging.info(f"[修复] idx={idx} | {patient_id} | {len(all_fixes)}处: {', '.join(fix_types)}")

    write_realtime(output)
    return output


# ============== 主程序 ==============

def main():
    log_file = setup_logging()
    random.seed(42)

    logging.info("=" * 60)
    logging.info("跨字段一致性修复 (规则 + Qwen-32B)")
    logging.info("=" * 60)

    # 加载数据
    logging.info(f"加载数据: {INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"总记录数: {len(data)}")

    # 检查断点
    processed_indices = set()
    if REALTIME_FILE and os.path.exists(REALTIME_FILE):
        with open(REALTIME_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get('idx') is not None:
                            processed_indices.add(r['idx'])
                    except json.JSONDecodeError:
                        pass
        logging.info(f"断点恢复: 已处理 {len(processed_indices)} 条")

    indices = [i for i in range(len(data)) if i not in processed_indices]
    logging.info(f"待处理: {len(indices)} 条")

    # 统计
    stats = Counter()
    all_results = []

    logging.info(f"开始处理 (workers={MAX_WORKERS})...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, idx, data[idx]): idx for idx in indices}

        with tqdm(total=len(indices), desc="修复矛盾") as pbar:
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                stats[result['status']] += 1
                for fix in result.get('fixes', []):
                    stats[f"type_{fix['type']}"] += 1
                pbar.update(1)

    # ====== 应用修复 ======
    logging.info("应用修复到数据...")

    # 重新加载所有结果（包括断点恢复的）
    all_fix_results = {}
    if REALTIME_FILE and os.path.exists(REALTIME_FILE):
        with open(REALTIME_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get('idx') is not None:
                            all_fix_results[r['idx']] = r
                    except json.JSONDecodeError:
                        pass

    applied = 0
    manual_review = []
    for idx, result in all_fix_results.items():
        for fix in result.get('fixes', []):
            field = fix.get('field', '')
            if field == '_MANUAL_REVIEW':
                manual_review.append({
                    'idx': idx,
                    'patient_id': result.get('patient_id', ''),
                    'type': fix.get('type', ''),
                    'reason': fix.get('reason', ''),
                    'original': fix.get('original', '')
                })
                continue

            if field and fix.get('fixed'):
                data[idx][field] = fix['fixed']
                applied += 1

    logging.info(f"已应用 {applied} 处修复")
    logging.info(f"待人工审核 {len(manual_review)} 处")

    # 保存修复后数据
    logging.info(f"保存到: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 保存人工审核列表
    if manual_review:
        manual_file = OUTPUT_FILE.replace('.json', '_manual_review.json')
        with open(manual_file, 'w', encoding='utf-8') as f:
            json.dump(manual_review, f, ensure_ascii=False, indent=2)
        logging.info(f"人工审核列表: {manual_file}")

    # ====== 统计报告 ======
    logging.info("\n" + "=" * 60)
    logging.info("修复完成!")
    logging.info(f"  总记录:        {len(data)}")
    logging.info(f"  有修复:        {stats.get('fixed', 0)}")
    logging.info(f"  无矛盾:        {stats.get('ok', 0)}")
    logging.info(f"  有错误:        {stats.get('error', 0)}")
    logging.info(f"  待人工审核:    {len(manual_review)}")

    logging.info("\n修复类型分布:")
    for key, cnt in stats.most_common():
        if key.startswith('type_'):
            logging.info(f"  {key[5:]}: {cnt}")

    logging.info(f"\n日志: {log_file}")
    logging.info(f"JSONL: {REALTIME_FILE}")
    logging.info(f"输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
