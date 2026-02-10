#!/usr/bin/env python3
"""
分析真实病例数据，提取分布和关键词映射

使用方法：
    python scripts/analyze_data.py --input real_emrs/input_real_emrs.json
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.analyzers.distribution_analyzer import DistributionAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer


def load_data(filepath: Path) -> list:
    """加载数据文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_distributions(records: list, output_file: Path = None):
    """分析分布"""
    print("=" * 60)
    print("开始分析数据分布...")
    print("=" * 60)
    
    analyzer = DistributionAnalyzer()
    summary = analyzer.analyze_records(records)
    
    # 打印摘要
    print(f"\n总记录数: {summary['total_records']}")
    
    print("\n--- 性别分布 ---")
    gender_dist = summary["distributions"].get("gender", {})
    for gender, prob in sorted(gender_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {gender}: {prob:.1%}")
    
    print("\n--- 诊断编码大类分布 (Top 15) ---")
    diag_dist = summary["distributions"].get("diagnosis_code_category", {})
    sorted_diags = sorted(diag_dist.items(), key=lambda x: x[1], reverse=True)[:15]
    for diag, prob in sorted_diags:
        print(f"  {diag}: {prob:.1%}")
    
    print("\n--- 年龄统计 ---")
    age_stats = summary.get("age_stats", {})
    if age_stats:
        print(f"  最小年龄: {age_stats.get('min')}")
        print(f"  最大年龄: {age_stats.get('max')}")
        print(f"  平均年龄: {age_stats.get('mean')}")
    
    print("\n--- 症状分布 (Top 10) ---")
    symptom_dist = summary["distributions"].get("symptoms", {})
    sorted_symptoms = sorted(symptom_dist.items(), key=lambda x: x[1], reverse=True)[:10]
    for symptom, prob in sorted_symptoms:
        print(f"  {symptom}: {prob:.1%}")
    
    print("\n--- 诊断编码大类-症状关联 (Top 10 诊断) ---")
    diag_symptoms = summary.get("diagnosis_code_symptoms", {})
    # 按症状数量排序
    sorted_diag_symptoms = sorted(diag_symptoms.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for diag, symptoms in sorted_diag_symptoms:
        print(f"\n  [{diag}] 常见症状:")
        sorted_symptoms = sorted(symptoms.items(), key=lambda x: x[1], reverse=True)[:5]
        for symptom, prob in sorted_symptoms:
            print(f"    - {symptom}: {prob:.1%}")
    
    # 保存分布映射
    output_file = output_file or Config.DISTRIBUTION_MAPPING_FILE
    analyzer.save_mapping(output_file)
    
    # 保存诊断编码映射（单独文件）
    diag_mapping = analyzer.save_diagnosis_mapping()
    
    print(f"\n--- 诊断编码映射 (共 {len(diag_mapping.get('diagnosis_code_to_name', {}))} 个) ---")
    code_to_name = diag_mapping.get("diagnosis_code_to_name", {})
    # 按编码排序显示前15个
    sorted_codes = sorted(code_to_name.items())[:15]
    for code, name in sorted_codes:
        print(f"  {code} -> {name}")
    if len(code_to_name) > 15:
        print(f"  ... (还有 {len(code_to_name) - 15} 个)")
    
    # 显示文本长度分布（Top 5 诊断编码）
    print(f"\n--- 文本长度分布 (Top 5 诊断编码) ---")
    length_dists = summary.get("diagnosis_length_distributions", {})
    # 按诊断编码分布排序获取前5个
    diag_dist = summary.get("distributions", {}).get("diagnosis_code_category", {})
    top_diags = sorted(diag_dist.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for diag_code, _ in top_diags:
        if diag_code in length_dists:
            print(f"\n  [{diag_code}]")
            diag_length = length_dists[diag_code]
            for field in ["chief_complaint", "present_illness", "dialogue"]:
                if field in diag_length:
                    field_name = {"chief_complaint": "主诉", "present_illness": "现病史", "dialogue": "对话"}[field]
                    dist_str = ", ".join([f"{k}: {v:.1%}" for k, v in diag_length[field].items()])
                    print(f"    {field_name}: {dist_str}")
    
    # 显示 ICD 编码列表长度分布（Top 5 编码组合）
    print(f"\n--- ICD 编码列表长度分布 (Top 5 编码组合) ---")
    icd_codes_dists = summary.get("icd_codes_length_distribution", {})
    # 计算每个编码组合的总样本数（以 chief_complaint 字段为准）
    icd_codes_sample_counts = []
    for codes_key, field_data in icd_codes_dists.items():
        # 计算该组合的样本权重（用第一个字段的概率和为1来估算）
        sample_weight = 1.0
        if "chief_complaint" in field_data:
            sample_weight = sum(field_data["chief_complaint"].values())
        icd_codes_sample_counts.append((codes_key, sample_weight))
    
    # 按样本权重排序取前5个
    top_icd_codes = sorted(icd_codes_sample_counts, key=lambda x: x[1], reverse=True)[:5]
    
    for codes_key, _ in top_icd_codes:
        if codes_key in icd_codes_dists:
            print(f"\n  {codes_key}")
            codes_length = icd_codes_dists[codes_key]
            for field in ["chief_complaint", "present_illness", "dialogue_turns", "doctor_avg_chars", "patient_avg_chars"]:
                if field in codes_length:
                    field_name = {
                        "chief_complaint": "主诉", 
                        "present_illness": "现病史", 
                        "dialogue_turns": "对话轮数",
                        "doctor_avg_chars": "医生平均字数",
                        "patient_avg_chars": "患者平均字数"
                    }[field]
                    dist_str = ", ".join([f"{k}: {v:.1%}" for k, v in codes_length[field].items()])
                    print(f"    {field_name}: {dist_str}")
    
    # 显示年龄-个人史字段分布
    print(f"\n--- 年龄组-个人史分布 ---")
    age_ph = summary.get("age_personal_history", {})
    field_names = {
        "marriage_status": "婚恋",
        "occupation": "职业",
        "pregnancy_status": "孕产",
        "menstrual_status": "月经",
        "special_habits": "嗜好",
        "personality": "性格",
        "development_status": "发育",
    }
    for age_group in ["0-18", "18-30", "30-45", "45-60", "60+"]:
        if age_group in age_ph:
            print(f"\n  [{age_group}岁]")
            for field, display_name in field_names.items():
                if field in age_ph[age_group]:
                    top3 = sorted(age_ph[age_group][field].items(), key=lambda x: x[1], reverse=True)[:3]
                    dist_str = ", ".join([f"{k}: {v:.1%}" for k, v in top3])
                    print(f"    {display_name}: {dist_str}")
    
    # 显示年龄+性别-陪同人分布
    print(f"\n--- 年龄组+性别-陪同人分布 ---")
    age_gender_acc = summary.get("age_gender_accompanying", {})
    for age_group in ["0-18", "18-30", "30-45", "45-60", "60+"]:
        if age_group in age_gender_acc:
            print(f"\n  [{age_group}岁]")
            for gender in ["男", "女"]:
                if gender in age_gender_acc[age_group]:
                    gender_data = age_gender_acc[age_group][gender]
                    # 是否有陪同
                    has_acc = gender_data.get("has_accompanying", {})
                    has_str = ", ".join([f"{k}: {v:.1%}" for k, v in has_acc.items()])
                    # 陪同人关系 (Top 5)
                    rel = gender_data.get("relation", {})
                    top5_rel = sorted(rel.items(), key=lambda x: x[1], reverse=True)[:5]
                    rel_str = ", ".join([f"{k}: {v:.1%}" for k, v in top5_rel])
                    print(f"    {gender}: 陪同({has_str})")
                    if rel_str:
                        print(f"        关系: {rel_str}")
    
    # 显示共病分布
    comorbidity = summary.get("comorbidity", {})
    if comorbidity:
        print(f"\n--- 共病分布 ---")
        count_dist = comorbidity.get("count_distribution", {})
        if count_dist:
            print("  诊断数量分布:")
            for n, prob in sorted(count_dist.items(), key=lambda x: int(x[0])):
                print(f"    {n}个诊断: {prob:.1%}")
        
        pairs = comorbidity.get("pairs_by_primary", {})
        if pairs:
            print("\n  常见共病对 (按主诊断, Top 5):")
            top5_pairs = list(pairs.items())[:5]
            for primary, secondaries in top5_pairs:
                sec_str = ", ".join([f"{k}: {v:.1%}" for k, v in list(secondaries.items())[:3]])
                print(f"    {primary} -> {sec_str}")
        
        combos = comorbidity.get("combinations", {})
        if combos:
            print("\n  常见共病组合 (Top 10):")
            for combo_str, prob in list(combos.items())[:10]:
                print(f"    {combo_str}: {prob:.1%}")
    
    return summary


def analyze_keywords(records: list, output_file: Path = None):
    """分析关键词"""
    print("\n" + "=" * 60)
    print("开始分析关键词...")
    print("=" * 60)
    
    analyzer = KeywordAnalyzer()
    summary = analyzer.analyze_records(records)
    
    # 打印摘要
    print("\n--- 诱因关键词 (按诊断编码, Top 5) ---")
    trigger_info = summary.get("trigger_keywords", {})
    # 按样本量排序取前5个诊断
    sorted_diags = sorted(trigger_info.items(), key=lambda x: sum(x[1].values()) if isinstance(x[1], dict) else 0, reverse=True)[:5]
    for diag, triggers in sorted_diags:
        if isinstance(triggers, dict):
            print(f"\n  [{diag}]")
            sorted_triggers = sorted(triggers.items(), key=lambda x: x[1], reverse=True)[:5]
            for trigger, prob in sorted_triggers:
                print(f"    - {trigger}: {prob:.1%}")
    
    print("\n--- 时间表达式 (按诊断编码, Top 5) ---")
    time_info = summary.get("time_templates", {})
    sorted_diags = sorted(time_info.items(), key=lambda x: sum(x[1].values()) if isinstance(x[1], dict) else 0, reverse=True)[:5]
    for diag, times in sorted_diags:
        if isinstance(times, dict):
            print(f"\n  [{diag}]")
            sorted_times = sorted(times.items(), key=lambda x: x[1], reverse=True)[:5]
            for time_expr, prob in sorted_times:
                print(f"    - {time_expr}: {prob:.1%}")
    
    print("\n--- 现病史关键词 (按诊断编码大类, Top 10) ---")
    pi_keywords = summary.get("present_illness_keywords", {})
    # 按关键词数量排序，取前10个诊断编码
    sorted_diags = sorted(pi_keywords.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for diag, keywords in sorted_diags:
        print(f"\n  [{diag}] Top 5 关键词:")
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:5]
        for kw, prob in sorted_keywords:
            print(f"    - {kw}: {prob:.1%}")
    
    # 保存映射
    output_file = output_file or Config.KEYWORD_MAPPING_FILE
    analyzer.save_mapping(output_file)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="分析真实病例数据")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Config.DEFAULT_DATA_FILE,
        help="输入数据文件路径"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Config.MAPPING_DIR,
        help="输出目录"
    )
    parser.add_argument(
        "--distribution-only",
        action="store_true",
        help="只分析分布"
    )
    parser.add_argument(
        "--keyword-only",
        action="store_true",
        help="只分析关键词"
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"加载数据: {args.input}")
    records = load_data(args.input)
    print(f"共加载 {len(records)} 条记录")
    
    # 分析
    if not args.keyword_only:
        dist_file = args.output_dir / "distribution_mapping.json"
        analyze_distributions(records, dist_file)
    
    if not args.distribution_only:
        keyword_file = args.output_dir / "keyword_mapping.json"
        analyze_keywords(records, keyword_file)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print(f"映射文件已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
