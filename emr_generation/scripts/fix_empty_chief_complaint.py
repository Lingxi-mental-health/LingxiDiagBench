#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复空主诉的脚本

读取生成的病例文件，为空主诉的病例重新生成主诉内容。
"""

import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzers.distribution_analyzer import DistributionSampler
from src.analyzers.keyword_analyzer import KeywordSampler
from src.generators.chief_complaint_generator import ChiefComplaintGenerator, GenerationContext
from src.utils.llm_client import LLMClient
from src.config import Config


def is_empty_chief_complaint(cc: str) -> bool:
    """判断主诉是否为空"""
    if not cc:
        return True
    # 去除前缀后检查
    cleaned = cc.replace("主诉：", "").replace("主诉:", "").strip()
    return len(cleaned) == 0


def generate_chief_complaint_for_record(
    record: Dict[str, Any],
    generator: ChiefComplaintGenerator,
    use_llm: bool = False,
    max_retries: int = 3
) -> str:
    """为单条记录生成主诉"""
    import re
    
    # 构建上下文
    context = GenerationContext(
        age=int(record.get("Age", 30)),
        gender=record.get("Gender", "男"),
        department=record.get("Department", "普通精神科"),
        diagnosis=record.get("DiagnosisCode", "F32.9").split(",")[0].strip(),
    )
    
    # 尝试解析诊断编码
    diag_code = record.get("DiagnosisCode", "")
    if diag_code:
        # 解析为大类，如 F32.100 -> F32.1
        match = re.match(r'^([A-Z]\d+)(?:\.(\d))?', diag_code.split(",")[0].strip())
        if match:
            main_code = match.group(1)
            sub_code = match.group(2)
            context.diagnosis = f"{main_code}.{sub_code}" if sub_code else main_code
    
    # 尝试生成主诉（带重试）
    for attempt in range(max_retries):
        result = generator.generate(context, use_llm=use_llm)
        
        # 验证结果是否有效
        if result:
            cleaned = result.replace("主诉：", "").replace("主诉:", "").strip()
            if cleaned:
                return result
        
        # 如果使用LLM失败，尝试不使用LLM
        if use_llm and attempt == max_retries - 1:
            result = generator.generate(context, use_llm=False)
            if result:
                cleaned = result.replace("主诉：", "").replace("主诉:", "").strip()
                if cleaned:
                    return result
    
    # 最终回退：生成一个基础主诉
    default_symptoms = ["情绪问题", "睡眠不佳"]
    return f"主诉：{default_symptoms[0]}、{default_symptoms[1]} 数月"


def fix_empty_chief_complaints(
    input_file: str,
    output_file: str = None,
    use_llm: bool = False,
    llm_host: str = None,
    llm_port: int = None,
    llm_model: str = None,
    num_workers: int = 4,
    limit: int = None,
):
    """
    修复空主诉
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（默认覆盖原文件）
        use_llm: 是否使用LLM润色
        llm_host: LLM服务地址
        llm_port: LLM服务端口
        llm_model: LLM模型名称
        num_workers: 并行线程数
        limit: 限制处理数量（用于测试）
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"错误: 文件不存在 {input_file}")
        return
    
    output_path = Path(output_file) if output_file else input_path
    
    # 加载数据
    print(f"加载文件: {input_file}")
    with open(input_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print(f"总记录数: {len(records)}")
    
    # 统计空主诉
    empty_indices = []
    for i, record in enumerate(records):
        cc = record.get("ChiefComplaint", "")
        if is_empty_chief_complaint(cc):
            empty_indices.append(i)
    
    print(f"空主诉数量: {len(empty_indices)}")
    
    if not empty_indices:
        print("没有空主诉需要修复")
        return
    
    if limit:
        empty_indices = empty_indices[:limit]
        print(f"限制处理数量: {len(empty_indices)}")
    
    # 初始化生成器
    print("\n初始化生成器...")
    dist_sampler = DistributionSampler()
    keyword_sampler = KeywordSampler()
    
    llm_client = None
    if use_llm:
        llm_client = LLMClient(
            host=llm_host or Config.LLM_DEFAULT_HOST,
            port=llm_port or Config.LLM_DEFAULT_PORT,
            model=llm_model or Config.LLM_DEFAULT_MODEL,
        )
    
    generator = ChiefComplaintGenerator(
        llm_client=llm_client,
        distribution_sampler=dist_sampler,
        keyword_sampler=keyword_sampler,
    )
    
    # 修复空主诉
    print(f"\n开始修复 {len(empty_indices)} 条空主诉...")
    print(f"使用LLM: {'是' if use_llm else '否'}")
    print(f"并行线程数: {num_workers}")
    
    fixed_count = 0
    lock = threading.Lock()
    
    def fix_one(idx: int) -> tuple:
        """修复单条记录"""
        nonlocal fixed_count
        try:
            record = records[idx]
            new_cc = generate_chief_complaint_for_record(record, generator, use_llm)
            
            with lock:
                fixed_count += 1
                if fixed_count % 100 == 0 or fixed_count == len(empty_indices):
                    print(f"进度: {fixed_count}/{len(empty_indices)} ({fixed_count/len(empty_indices)*100:.1f}%)")
            
            return idx, new_cc
        except Exception as e:
            print(f"修复记录 #{idx} 失败: {e}")
            # 异常时返回默认主诉，确保不会返回 None
            default_cc = "主诉：情绪问题、睡眠不佳 数月"
            with lock:
                fixed_count += 1
            return idx, default_cc
    
    # 并行处理
    results = []
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(fix_one, idx): idx for idx in empty_indices}
            for future in as_completed(futures):
                result = future.result()
                if result[1]:
                    results.append(result)
    else:
        for idx in empty_indices:
            result = fix_one(idx)
            if result[1]:
                results.append(result)
    
    # 更新记录
    print(f"\n成功修复: {len(results)} 条")
    for idx, new_cc in results:
        records[idx]["ChiefComplaint"] = new_cc
    
    # 保存结果
    print(f"\n保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    
    # 验证结果
    remaining_empty = 0
    for idx in empty_indices:
        if is_empty_chief_complaint(records[idx].get("ChiefComplaint", "")):
            remaining_empty += 1
    
    print(f"\n修复完成!")
    print(f"  原空主诉: {len(empty_indices)}")
    print(f"  成功修复: {len(results)}")
    print(f"  仍为空: {remaining_empty}")
    
    # 显示几个修复后的示例
    print("\n修复示例:")
    for idx, new_cc in results[:3]:
        record = records[idx]
        print(f"  #{idx}: {new_cc[:60]}...")
        print(f"        诊断: {record.get('Diagnosis', '')}")


def main():
    parser = argparse.ArgumentParser(description="修复空主诉")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="输入文件路径"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认覆盖原文件）"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="使用LLM润色"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="LLM服务地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="LLM服务端口"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM模型名称"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并行线程数"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理数量（用于测试）"
    )
    
    args = parser.parse_args()
    
    fix_empty_chief_complaints(
        input_file=args.input,
        output_file=args.output,
        use_llm=args.use_llm,
        llm_host=args.host,
        llm_port=args.port,
        llm_model=args.model,
        num_workers=args.workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
