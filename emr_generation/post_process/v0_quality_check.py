#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
病例数据质量检测脚本
检测16K合成病例数据中的各类问题
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import os

# ============== 问题检测函数 ==============

def extract_duration_years(text: str) -> List[int]:
    """从文本中提取病程年数"""
    if not text:
        return []

    years = []
    # 匹配 "X年" 模式
    patterns = [
        r'(\d+)\s*年',
        r'(\d+)\s*余年',
        r'(\d+)\s*多年',
        r'近(\d+)\s*年',
        r'约(\d+)\s*年',
        r'(\d+)\s*年余',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                years.append(int(m))
            except:
                pass
    return years

def extract_age_from_text(text: str) -> List[int]:
    """从现病史中提取提到的年龄"""
    if not text:
        return []

    ages = []
    patterns = [
        r'患者为?(\d+)岁',
        r'(\d+)岁(?:女|男|患者)',
        r'患者(\d+)岁',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                age = int(m)
                if 1 <= age <= 120:
                    ages.append(age)
            except:
                pass
    return ages

def extract_duration_months(text: str) -> List[int]:
    """从文本中提取病程月数"""
    if not text:
        return []

    months = []
    patterns = [
        r'(\d+)\s*个?月',
        r'近(\d+)\s*个?月',
        r'约(\d+)\s*个?月',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                months.append(int(m))
            except:
                pass
    return months

def check_age_duration_conflict(record: dict) -> Optional[str]:
    """检测年龄与病史时长矛盾"""
    try:
        age = int(record.get('Age', 0))
    except:
        return None

    if age <= 0:
        return None

    # 从主诉中提取病程
    chief_complaint = record.get('ChiefComplaint', '')
    durations = extract_duration_years(chief_complaint)

    for duration in durations:
        if duration >= age:
            return f"年龄{age}岁，但主诉中病程{duration}年（不可能）"
        # 检查是否病程过长（比如8岁以前就发病的情况）
        if age - duration < 5 and duration > 5:
            return f"年龄{age}岁，主诉病程{duration}年，意味着{age-duration}岁发病（可疑）"

    return None

def check_age_inconsistency(record: dict) -> Optional[str]:
    """检测Age字段与现病史中年龄不一致"""
    try:
        age = int(record.get('Age', 0))
    except:
        return None

    present_illness = record.get('PresentIllnessHistory', '')
    ages_in_text = extract_age_from_text(present_illness)

    for text_age in ages_in_text:
        if abs(text_age - age) > 2:  # 允许1-2岁误差
            return f"Age字段={age}岁，但现病史中写'{text_age}岁'"

    return None

def check_present_illness_empty(record: dict) -> Optional[str]:
    """检测现病史为空或过短"""
    present_illness = record.get('PresentIllnessHistory', '')

    if not present_illness or present_illness.strip() == '':
        return "现病史为空"

    # 去除前缀后检查
    content = present_illness.replace('现病史：', '').replace('现病史:', '').strip()

    if len(content) < 10:
        return f"现病史过短（仅{len(content)}字符）：'{content}'"

    # 检测被截断的情况（以常见的未完成词结尾）
    truncation_patterns = [
        r'及$', r'与$', r'和$', r'因$', r'由于$', r'导致$',
        r'出现$', r'表现为$', r'伴有$', r'伴随$',
    ]
    for pattern in truncation_patterns:
        if re.search(pattern, content):
            return f"现病史疑似被截断，以'{content[-10:]}'结尾"

    return None

def check_chief_vs_present_duration(record: dict) -> Optional[str]:
    """检测主诉与现病史时间不一致"""
    chief = record.get('ChiefComplaint', '')
    present = record.get('PresentIllnessHistory', '')

    chief_years = extract_duration_years(chief)
    present_years = extract_duration_years(present)

    chief_months = extract_duration_months(chief)
    present_months = extract_duration_months(present)

    # 如果主诉说X年，现病史说几个月，可能有问题
    if chief_years and present_months:
        max_chief_year = max(chief_years)
        max_present_month = max(present_months) if present_months else 0

        # 主诉说1年以上，但现病史说几个月
        if max_chief_year >= 1 and max_present_month < 12 and not present_years:
            return f"主诉病程{max_chief_year}年，但现病史仅提到{max_present_month}个月"

    return None

def check_age_occupation_conflict(record: dict) -> Optional[str]:
    """检测年龄与工作学习情况矛盾"""
    try:
        age = int(record.get('Age', 0))
    except:
        return None

    personal_history = record.get('PersonalHistory', '')

    # 检测学习阶段与年龄
    if '高二' in personal_history or '高中二年级' in personal_history:
        if age < 15 or age > 18:
            return f"年龄{age}岁，但工作学习情况显示'高二'（通常16-17岁）"

    if '高一' in personal_history or '高中一年级' in personal_history:
        if age < 14 or age > 17:
            return f"年龄{age}岁，但工作学习情况显示'高一'（通常15-16岁）"

    if '高三' in personal_history or '高中三年级' in personal_history:
        if age < 16 or age > 19:
            return f"年龄{age}岁，但工作学习情况显示'高三'（通常17-18岁）"

    if '初中' in personal_history:
        if age < 11 or age > 16:
            return f"年龄{age}岁，但工作学习情况显示'初中'（通常12-15岁）"

    if '小学' in personal_history:
        if age < 5 or age > 13:
            return f"年龄{age}岁，但工作学习情况显示'小学'（通常6-12岁）"

    return None

def check_family_history_format(record: dict) -> Optional[str]:
    """检测家族史格式问题"""
    family_history = record.get('FamilyHistory', '')

    if '家族史：家族史：' in family_history or '家族史:家族史:' in family_history:
        return "家族史字段重复表述'家族史：家族史：'"

    return None

def check_physical_illness_vague(record: dict) -> Optional[str]:
    """检测躯体疾病史描述模糊"""
    illness = record.get('ImportantRelevantPhysicalIllnessHistory', '')

    vague_patterns = ['已痊愈', '不详', '有（）', '有()']
    for pattern in vague_patterns:
        if pattern in illness and len(illness) < 30:
            return f"躯体疾病史描述模糊：'{illness}'"

    return None

def check_dialogue_repetition(record: dict) -> Optional[str]:
    """检测对话中医生重复提问"""
    dialogue = record.get('cleaned_text', '')
    if not dialogue:
        return None

    # 分割对话
    lines = dialogue.split('\n')
    doctor_questions = []

    for line in lines:
        if line.startswith('医生：'):
            question = line[3:].strip()
            if len(question) > 20:  # 只检查较长的问句
                doctor_questions.append(question)

    # 检测连续重复
    consecutive_repeats = 0
    for i in range(1, len(doctor_questions)):
        if doctor_questions[i] == doctor_questions[i-1]:
            consecutive_repeats += 1

    if consecutive_repeats >= 2:
        return f"对话中医生连续重复提问{consecutive_repeats+1}次"

    return None

def check_patient_label_in_dialogue(record: dict) -> Optional[str]:
    """检测对话中患者回复包含角色标注"""
    dialogue = record.get('cleaned_text', '')
    if not dialogue:
        return None

    problematic_patterns = [
        '患者本人：',
        '患者：患者',
    ]

    for pattern in problematic_patterns:
        if pattern in dialogue:
            return f"对话中包含不当标注'{pattern}'"

    return None

def check_diagnosis_age_mismatch(record: dict) -> Optional[str]:
    """检测诊断与年龄不匹配"""
    try:
        age = int(record.get('Age', 0))
    except:
        return None

    diagnosis = record.get('Diagnosis', '')

    # 童年/青少年期障碍但患者是成年人
    if '童年' in diagnosis or '青少年期发病' in diagnosis:
        if age > 30:
            return f"诊断'{diagnosis}'通常指童年/青少年期，但患者{age}岁"

    return None

def check_gender_pregnancy_conflict(record: dict) -> Optional[str]:
    """检测性别与孕产情况矛盾"""
    gender = record.get('Gender', '')
    personal_history = record.get('PersonalHistory', '')

    if gender == '男':
        pregnancy_keywords = ['孕产情况', '月经情况', '足月顺产', '剖腹产']
        for kw in pregnancy_keywords:
            if kw in personal_history and '无' not in personal_history.split(kw)[0][-5:]:
                # 检查是否有孕产相关描述
                if '足月' in personal_history or '剖腹' in personal_history or '月经' in personal_history:
                    return f"性别为男，但个人史中包含'{kw}'相关描述"

    return None

def check_marital_accompany_conflict(record: dict) -> Optional[str]:
    """检测婚姻状况与陪同者矛盾"""
    personal_history = record.get('PersonalHistory', '')
    accompanying = record.get('AccompanyingPerson', '')

    # 未婚但陪同者是配偶
    if '未婚' in personal_history:
        if '妻子' in accompanying or '丈夫' in accompanying or '老婆' in accompanying or '老公' in accompanying:
            return f"个人史显示未婚，但陪同者是'{accompanying}'"

    return None

def check_icd_label_consistency(record: dict) -> Optional[str]:
    """检测ICD标签与诊断码是否一致"""
    diagnosis_code = record.get('DiagnosisCode', '')
    icd_label = record.get('icd_clf_label', [])

    if not diagnosis_code or not icd_label:
        return None

    # 提取诊断码前缀
    code_prefix = diagnosis_code.split('.')[0] if '.' in diagnosis_code else diagnosis_code[:3]

    # 检查是否匹配
    if icd_label and isinstance(icd_label, list):
        label_prefixes = [l[:3] if len(l) >= 3 else l for l in icd_label]
        if code_prefix not in label_prefixes and code_prefix[:2] not in [l[:2] for l in label_prefixes]:
            # 特殊情况：多诊断
            if ',' in diagnosis_code:
                return None  # 多诊断情况暂不检测
            if 'Others' in icd_label:
                return None  # Others是通用标签
            return f"诊断码'{diagnosis_code}'与icd_clf_label'{icd_label}'可能不一致"

    return None

# ============== 主检测函数 ==============

def check_single_record(record: dict, idx: int) -> Dict[str, List[str]]:
    """检测单条记录的所有问题"""
    issues = defaultdict(list)
    patient_id = record.get('patient_id', f'unknown_{idx}')

    checkers = [
        ('年龄与病史时长矛盾', check_age_duration_conflict),
        ('年龄字段不一致', check_age_inconsistency),
        ('现病史为空或截断', check_present_illness_empty),
        ('主诉与现病史时间矛盾', check_chief_vs_present_duration),
        ('年龄与学历/职业矛盾', check_age_occupation_conflict),
        ('家族史格式问题', check_family_history_format),
        ('躯体疾病史描述模糊', check_physical_illness_vague),
        ('对话医生重复提问', check_dialogue_repetition),
        ('对话包含不当标注', check_patient_label_in_dialogue),
        ('诊断与年龄不匹配', check_diagnosis_age_mismatch),
        ('性别与孕产情况矛盾', check_gender_pregnancy_conflict),
        ('婚姻状况与陪同者矛盾', check_marital_accompany_conflict),
        ('ICD标签与诊断码不一致', check_icd_label_consistency),
    ]

    for issue_type, checker in checkers:
        try:
            result = checker(record)
            if result:
                issues[issue_type].append({
                    'patient_id': patient_id,
                    'detail': result,
                    'index': idx
                })
        except Exception as e:
            issues['检测异常'].append({
                'patient_id': patient_id,
                'detail': f"{issue_type}检测出错: {str(e)}",
                'index': idx
            })

    return issues

def merge_issues(all_issues: Dict, new_issues: Dict):
    """合并问题字典"""
    for issue_type, items in new_issues.items():
        all_issues[issue_type].extend(items)

def main():
    """主函数"""
    input_file = '/tcci_mnt/xiaoming/Lingxi_annotation_0111/emr_quality_filter_for_Lingxi/version/v0/LingxiDiag-16K_all_data.json'
    output_dir = '/tcci_mnt/xiaoming/Lingxi_annotation_0111/emr_quality_filter_for_Lingxi/version/v0'

    print("=" * 60)
    print("病例数据质量检测")
    print("=" * 60)

    # 加载数据
    print(f"\n正在加载数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_records = len(data)
    print(f"共加载 {total_records} 条记录")

    # 检测所有记录
    all_issues = defaultdict(list)
    problem_records = set()  # 有问题的记录索引

    print("\n正在检测...")
    for idx, record in enumerate(data):
        if idx % 1000 == 0:
            print(f"  已检测 {idx}/{total_records} 条...")

        issues = check_single_record(record, idx)
        if issues:
            problem_records.add(idx)
            merge_issues(all_issues, issues)

    print(f"\n检测完成！")

    # ============== 生成报告 ==============

    # 1. 问题汇总
    print("\n" + "=" * 60)
    print("问题汇总")
    print("=" * 60)

    summary = []
    for issue_type, items in sorted(all_issues.items(), key=lambda x: -len(x[1])):
        count = len(items)
        summary.append({
            'issue_type': issue_type,
            'count': count,
            'percentage': f"{count/total_records*100:.2f}%"
        })
        print(f"  {issue_type}: {count} 条 ({count/total_records*100:.2f}%)")

    print(f"\n有问题的记录总数: {len(problem_records)} 条 ({len(problem_records)/total_records*100:.2f}%)")
    print(f"无问题的记录数: {total_records - len(problem_records)} 条")

    # 2. 保存详细报告
    report = {
        'summary': {
            'total_records': total_records,
            'problem_records_count': len(problem_records),
            'clean_records_count': total_records - len(problem_records),
            'issue_types': summary
        },
        'details': {k: v for k, v in all_issues.items()},
        'problem_record_indices': sorted(list(problem_records))
    }

    report_file = os.path.join(output_dir, 'quality_check_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n详细报告已保存至: {report_file}")

    # 3. 生成按问题类型分组的详细清单
    detail_file = os.path.join(output_dir, 'quality_check_details.txt')
    with open(detail_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("病例数据质量检测详细报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总记录数: {total_records}\n")
        f.write(f"有问题记录数: {len(problem_records)} ({len(problem_records)/total_records*100:.2f}%)\n")
        f.write(f"无问题记录数: {total_records - len(problem_records)}\n\n")

        for issue_type, items in sorted(all_issues.items(), key=lambda x: -len(x[1])):
            f.write("-" * 80 + "\n")
            f.write(f"问题类型: {issue_type}\n")
            f.write(f"问题数量: {len(items)}\n")
            f.write("-" * 80 + "\n")

            # 只显示前50个示例
            for i, item in enumerate(items[:50]):
                f.write(f"  [{i+1}] patient_id: {item['patient_id']}, index: {item['index']}\n")
                f.write(f"      详情: {item['detail']}\n")

            if len(items) > 50:
                f.write(f"  ... 还有 {len(items) - 50} 条记录\n")
            f.write("\n")

    print(f"详细清单已保存至: {detail_file}")

    # 4. 生成有问题记录的patient_id列表
    problem_ids_file = os.path.join(output_dir, 'problem_patient_ids.json')
    problem_ids = {}
    for issue_type, items in all_issues.items():
        problem_ids[issue_type] = [item['patient_id'] for item in items]

    with open(problem_ids_file, 'w', encoding='utf-8') as f:
        json.dump(problem_ids, f, ensure_ascii=False, indent=2)
    print(f"问题记录ID列表已保存至: {problem_ids_file}")

    print("\n" + "=" * 60)
    print("检测完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()
