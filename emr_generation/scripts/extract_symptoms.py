#!/usr/bin/env python
"""
症状提取脚本 - 使用LLM从病例中提取251个标准症状
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.analyzers.symptom_extractor import extract_symptoms_from_data


def main():
    parser = argparse.ArgumentParser(description="从病例中提取251个标准症状")
    
    parser.add_argument(
        "--data-file", "-d",
        type=str,
        default=None,
        help="数据文件路径（默认使用配置中的训练数据）"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出文件路径（默认 mapping/symptom_occurrence.json）"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default=Config.LLM_DEFAULT_HOST,
        help=f"LLM服务地址（默认 {Config.LLM_DEFAULT_HOST}）"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=Config.LLM_DEFAULT_PORT,
        help=f"LLM服务端口（默认 {Config.LLM_DEFAULT_PORT}）"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=Config.LLM_DEFAULT_MODEL,
        help=f"LLM模型名称（默认 {Config.LLM_DEFAULT_MODEL}）"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并行工作线程数（默认 4）"
    )
    
    parser.add_argument(
        "--max-records", "-n",
        type=int,
        default=None,
        help="最大处理记录数（用于测试，默认处理全部）"
    )
    
    args = parser.parse_args()
    
    # 处理路径
    data_file = Path(args.data_file) if args.data_file else None
    output_file = Path(args.output) if args.output else None
    
    print("=" * 60)
    print("症状提取")
    print("=" * 60)
    print(f"数据文件: {data_file or Config.DEFAULT_DATA_FILE}")
    print(f"输出文件: {output_file or Config.MAPPING_DIR / 'symptom_occurrence.json'}")
    print(f"LLM服务: {args.host}:{args.port}")
    print(f"模型: {args.model}")
    print(f"并行线程: {args.workers}")
    if args.max_records:
        print(f"最大记录数: {args.max_records}")
    print("=" * 60)
    
    # 执行提取
    try:
        summary = extract_symptoms_from_data(
            data_file=data_file,
            output_file=output_file,
            llm_host=args.host,
            llm_port=args.port,
            llm_model=args.model,
            num_workers=args.workers,
            max_records=args.max_records,
        )
        
        print("\n" + "=" * 60)
        print("提取完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
