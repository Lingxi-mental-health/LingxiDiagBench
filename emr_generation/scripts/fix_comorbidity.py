#!/usr/bin/env python3
"""
修复已生成病例的共病信息

根据真实数据的共病分布，为已生成的合成病例添加共病诊断。

使用方法：
    python scripts/fix_comorbidity.py \
        --input outputs/generated_emrs_20251218_010011.json \
        --output outputs/generated_emrs_with_comorbidity.json
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config


def parse_diagnosis_code_to_category(diagnosis_code: str) -> str:
    """
    将完整诊断编码解析为大类
    
    例如:
    - "F32.900" -> "F32.9"
    - "F41.100" -> "F41.1"
    """
    if not diagnosis_code:
        return ""
    
    match = re.match(r'^([A-Z]\d+)(?:\.(\d))?', diagnosis_code)
    if match:
        main_code = match.group(1)
        sub_code = match.group(2)
        if sub_code:
            return f"{main_code}.{sub_code}"
        return main_code
    return diagnosis_code


def generate_full_diagnosis_code(category: str) -> str:
    """
    从诊断编码大类生成完整编码
    
    例如:
    - "F32.9" -> "F32.900"
    - "F41.1" -> "F41.100"
    """
    if "." in category:
        parts = category.split(".")
        main = parts[0]
        sub = parts[1]
        return f"{main}.{sub}00"
    return f"{category}.900"


def get_overall_diagnosis(diag_code: str) -> str:
    """根据诊断编码推断总体诊断分类"""
    if diag_code.startswith("F32") or diag_code.startswith("F33"):
        return "Depression"
    elif diag_code.startswith("F41.2"):
        return "Mix"
    elif diag_code.startswith("F41"):
        return "Anxiety"
    elif diag_code.startswith("F42"):
        return "OCD"
    elif diag_code.startswith("F51") or diag_code.startswith("G47"):
        return "Sleep"
    elif diag_code.startswith("F20"):
        return "Schizophrenia"
    elif diag_code.startswith("F39"):
        return "Mood"
    else:
        return "Other"


def combine_overall_diagnosis(diagnoses: List[str]) -> str:
    """
    合并多个诊断的总体分类
    
    规则:
    - 如果包含 Depression 和 Anxiety，返回 Mix
    - 否则返回第一个诊断的分类
    """
    overall_set = set()
    for diag in diagnoses:
        overall = get_overall_diagnosis(diag)
        overall_set.add(overall)
    
    # 检查是否为混合型
    if "Depression" in overall_set and "Anxiety" in overall_set:
        return "Mix"
    if "Depression" in overall_set and "OCD" in overall_set:
        return "Mix"
    
    # 返回第一个诊断的分类
    return get_overall_diagnosis(diagnoses[0]) if diagnoses else "Other"


class ComorbidityFixer:
    """共病修复器"""
    
    def __init__(self, distribution_file: Path = None):
        """
        初始化修复器
        
        Args:
            distribution_file: 分布映射文件路径
        """
        dist_file = distribution_file or Config.DISTRIBUTION_MAPPING_FILE
        
        with open(dist_file, 'r', encoding='utf-8') as f:
            self.distribution = json.load(f)
        
        # 加载诊断编码映射
        with open(Config.DIAGNOSIS_CODE_MAPPING_FILE, 'r', encoding='utf-8') as f:
            self.diagnosis_mapping = json.load(f)
        
        self.code_to_name = self.diagnosis_mapping.get("diagnosis_code_to_name", {})
        
        # 提取共病分布
        self.comorbidity = self.distribution.get("comorbidity", {})
        self.count_distribution = self.comorbidity.get("count_distribution", {})
        self.pairs_by_primary = self.comorbidity.get("pairs_by_primary", {})
        
        # 如果没有共病分布，使用默认值
        if not self.count_distribution:
            print("警告：分布映射中没有共病信息，使用默认分布")
            self.count_distribution = {"1": 0.842, "2": 0.135, "3": 0.022, "4": 0.002}
        
        print(f"共病数量分布: {self.count_distribution}")
        print(f"共病对分布: {len(self.pairs_by_primary)} 个主诊断")
    
    def sample_diagnosis_count(self) -> int:
        """采样诊断数量"""
        items = list(self.count_distribution.keys())
        probs = list(self.count_distribution.values())
        count_str = random.choices(items, weights=probs, k=1)[0]
        return int(count_str)
    
    def sample_secondary_diagnoses(
        self, 
        primary_code: str, 
        n: int
    ) -> List[str]:
        """
        采样共病诊断
        
        Args:
            primary_code: 主诊断编码大类
            n: 需要采样的共病数量
            
        Returns:
            共病诊断编码大类列表
        """
        if n <= 0:
            return []
        
        # 尝试从主诊断的共病对分布采样
        if primary_code in self.pairs_by_primary:
            pairs_dist = self.pairs_by_primary[primary_code]
            items = list(pairs_dist.keys())
            probs = list(pairs_dist.values())
            
            # 加权随机采样（不放回）
            result = []
            remaining_items = items.copy()
            remaining_probs = probs.copy()
            
            for _ in range(min(n, len(remaining_items))):
                if not remaining_items:
                    break
                
                # 归一化概率
                total = sum(remaining_probs)
                normalized = [p / total for p in remaining_probs]
                
                # 采样
                idx = random.choices(range(len(remaining_items)), weights=normalized, k=1)[0]
                result.append(remaining_items[idx])
                
                # 移除已选项
                remaining_items.pop(idx)
                remaining_probs.pop(idx)
            
            return result
        
        # 如果没有对应的共病对分布，从全局诊断分布采样
        diag_dist = self.distribution.get("distributions", {}).get("diagnosis_code_category", {})
        if diag_dist:
            # 过滤掉主诊断
            filtered = {k: v for k, v in diag_dist.items() if k != primary_code}
            if filtered:
                items = list(filtered.keys())
                probs = list(filtered.values())
                
                result = []
                remaining_items = items.copy()
                remaining_probs = probs.copy()
                
                for _ in range(min(n, len(remaining_items))):
                    if not remaining_items:
                        break
                    
                    total = sum(remaining_probs)
                    normalized = [p / total for p in remaining_probs]
                    
                    idx = random.choices(range(len(remaining_items)), weights=normalized, k=1)[0]
                    result.append(remaining_items[idx])
                    
                    remaining_items.pop(idx)
                    remaining_probs.pop(idx)
                
                return result
        
        return []
    
    def fix_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        修复单条记录的共病信息
        
        Args:
            record: 原始记录
            
        Returns:
            修复后的记录
        """
        # 复制记录
        fixed = record.copy()
        
        # 获取当前诊断编码
        current_code = record.get("DiagnosisCode", "")
        primary_category = parse_diagnosis_code_to_category(current_code)
        
        if not primary_category:
            return fixed
        
        # 采样诊断数量
        target_count = self.sample_diagnosis_count()
        
        # 如果目标数量为1，不需要添加共病
        if target_count <= 1:
            return fixed
        
        # 采样共病诊断
        secondary_categories = self.sample_secondary_diagnoses(
            primary_category, 
            target_count - 1
        )
        
        if not secondary_categories:
            return fixed
        
        # 生成完整诊断编码
        all_categories = [primary_category] + secondary_categories
        all_codes = [generate_full_diagnosis_code(cat) for cat in all_categories]
        
        # 更新 DiagnosisCode
        fixed["DiagnosisCode"] = ",".join(all_codes)
        
        # 更新 Diagnosis（诊断名称）
        all_names = []
        for cat in all_categories:
            name = self.code_to_name.get(cat, f"诊断{cat}")
            all_names.append(name)
        fixed["Diagnosis"] = ",".join(all_names)
        
        # 更新 OverallDiagnosis
        fixed["OverallDiagnosis"] = combine_overall_diagnosis(all_categories)
        
        return fixed
    
    def fix_batch(
        self, 
        records: List[Dict[str, Any]],
        progress_callback=None
    ) -> List[Dict[str, Any]]:
        """
        批量修复记录
        
        Args:
            records: 记录列表
            progress_callback: 进度回调
            
        Returns:
            修复后的记录列表
        """
        fixed_records = []
        
        for i, record in enumerate(records):
            fixed = self.fix_record(record)
            fixed_records.append(fixed)
            
            if progress_callback and (i + 1) % 1000 == 0:
                progress_callback(i + 1, len(records))
        
        return fixed_records


def print_statistics(records: List[Dict[str, Any]], title: str):
    """打印诊断数量统计"""
    count_dist = Counter()
    for record in records:
        codes = record.get("DiagnosisCode", "").split(",")
        count_dist[len(codes)] += 1
    
    print(f"\n{title}:")
    for n, c in sorted(count_dist.items()):
        print(f"  {n}个诊断: {c}条 ({c/len(records)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="修复已生成病例的共病信息")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="输入文件路径（已生成的病例）"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="输出文件路径（默认覆盖输入文件）"
    )
    parser.add_argument(
        "--distribution", "-d",
        type=Path,
        default=None,
        help="分布映射文件路径"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 输出文件默认覆盖输入文件
    output_file = args.output or args.input
    
    # 加载输入数据
    print(f"加载数据: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        records = json.load(f)
    print(f"共加载 {len(records)} 条记录")
    
    # 打印修复前统计
    print_statistics(records, "修复前诊断数量分布")
    
    # 初始化修复器
    print("\n初始化共病修复器...")
    fixer = ComorbidityFixer(distribution_file=args.distribution)
    
    # 修复记录
    print("\n开始修复共病信息...")
    
    def progress_callback(current, total):
        print(f"  进度: {current}/{total} ({current/total*100:.1f}%)")
    
    fixed_records = fixer.fix_batch(records, progress_callback)
    
    # 打印修复后统计
    print_statistics(fixed_records, "修复后诊断数量分布")
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_records, f, ensure_ascii=False, indent=2)
    
    print("\n修复完成！")


if __name__ == "__main__":
    main()


