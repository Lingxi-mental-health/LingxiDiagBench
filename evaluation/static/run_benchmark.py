#!/usr/bin/env python3
"""
运行静态Benchmark的主脚本

Usage:
    # 运行完整benchmark
    python run_benchmark.py
    
    # 只运行TF-IDF
    python run_benchmark.py --tfidf-only
    
    # 跳过LLM（节省时间）
    python run_benchmark.py --skip-llm
    
    # 使用不同的测试集
    python run_benchmark.py --test-file /path/to/test.json
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env 文件中的环境变量（包括 OPENROUTER_API_KEY）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from static.config import (
    TRAIN_DATA_FILE, TEST_DATA_100_FILE, TEST_DATA_500_FILE,
    OUTPUT_DIR
)
from static.benchmark_runner import BenchmarkRunner

# 增加hugginfgace的源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def main():
    parser = argparse.ArgumentParser(
        description='精神疾病诊断静态Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 运行完整benchmark
    python run_benchmark.py
    
    # 只运行TF-IDF (最快)
    python run_benchmark.py --tfidf-only
    
    # 只运行BERT
    python run_benchmark.py --bert-only
    
    # 只运行RoBERTa
    python run_benchmark.py --roberta-only
    
    # 只运行LLM
    python run_benchmark.py --llm-only
    
    # 跳过BERT和LLM
    python run_benchmark.py --skip-bert --skip-llm
    
    # 使用500样本测试集
    python run_benchmark.py --test-file %(test_500)s
        """ % {'test_500': TEST_DATA_500_FILE}
    )
    
    parser.add_argument(
        '--train-file', type=str, default=TRAIN_DATA_FILE,
        help='训练数据文件路径'
    )
    parser.add_argument(
        '--test-file', type=str, default=TEST_DATA_100_FILE,
        help='测试数据文件路径'
    )
    parser.add_argument(
        '--output-dir', type=str, default=OUTPUT_DIR,
        help='输出目录'
    )
    parser.add_argument(
        '--skip-bert', action='store_true',
        help='跳过BERT benchmark'
    )
    parser.add_argument(
        '--skip-roberta', action='store_true',
        help='跳过RoBERTa benchmark'
    )
    parser.add_argument(
        '--skip-llm', action='store_true',
        help='跳过LLM benchmark'
    )
    parser.add_argument(
        '--skip-next-utterance', action='store_true',
        help='跳过下一句预测benchmark'
    )
    parser.add_argument(
        '--llm-model', type=str, default='qwen3-32b:9041',
        help='LLM模型名称，支持多种格式: '
             '1) vLLM简化格式: "model:port" (如 qwen3-32b:9041); '
             '2) vLLM完整格式: "model@host:port" (如 qwen3-30b@10.119.28.185:9041); '
             '3) OpenRouter格式: "provider/model-name" (如 qwen/qwen3-30b-a3b-instruct-2507)'
    )
    parser.add_argument(
        '--tfidf-only', action='store_true',
        help='只运行TF-IDF benchmark'
    )
    parser.add_argument(
        '--bert-only', action='store_true',
        help='只运行BERT benchmark'
    )
    parser.add_argument(
        '--roberta-only', action='store_true',
        help='只运行RoBERTa benchmark'
    )
    parser.add_argument(
        '--llm-only', action='store_true',
        help='只运行LLM benchmark'
    )
    parser.add_argument(
        '--next-utterance-prediction-only', action='store_true',
        help='只运行医生提问下一句预测benchmark'
    )
    parser.add_argument(
        '--tfidf-classifiers', type=str, nargs='+',
        default=['logistic', 'svm', 'rf'],
        choices=['logistic', 'svm', 'rf'],
        help='TF-IDF使用的分类器 (默认: logistic svm rf)'
    )
    parser.add_argument(
        '--classification-types', type=str, nargs='+',
        default=['2class', '4class', '12class'],
        choices=['2class', '4class', '12class'],
        help='分类类型 (默认: 2class 4class 12class)'
    )
    parser.add_argument(
        '--summary-file', type=str, default='benchmark_summary.xlsx',
        help='汇总结果Excel文件名，支持追加模式 (默认: benchmark_summary.xlsx)'
    )
    parser.add_argument(
        '--next-utterance-summary-file', type=str, default='next_utterance_summary.xlsx',
        help='医生提问预测汇总Excel文件名，支持追加模式 (默认: next_utterance_summary.xlsx)'
    )
    parser.add_argument(
        '--next-utterance-eval-interval', type=int, default=5,
        help='医生提问预测的采样间隔，每隔多少轮采样一次 (默认: 5)'
    )
    parser.add_argument(
        '--next-utterance-limit', type=int, default=5000,
        help='医生提问预测的最大样本数限制，固定seed随机采样 (默认: 5000)'
    )
    
    args = parser.parse_args()
    
    # 打印配置
    print("\n" + "="*80)
    print("精神疾病诊断静态Benchmark")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练数据: {args.train_file}")
    print(f"测试数据: {args.test_file}")
    print(f"输出目录: {args.output_dir}")
    print("-"*80)
    print(f"TF-IDF分类器: {args.tfidf_classifiers}")
    print(f"分类类型: {args.classification_types}")
    print(f"跳过BERT: {args.skip_bert}")
    print(f"跳过RoBERTa: {args.skip_roberta}")
    print(f"跳过LLM: {args.skip_llm}")
    print(f"跳过下一句预测: {args.skip_next_utterance}")
    if not args.skip_next_utterance:
        print(f"下一句预测采样间隔: 每 {args.next_utterance_eval_interval} 轮")
        print(f"下一句预测样本限制: {args.next_utterance_limit}")
    if not args.skip_llm:
        print(f"LLM模型: {args.llm_model}")
    print("="*80 + "\n")
    
    # 创建运行器
    runner = BenchmarkRunner(
        train_file=args.train_file,
        test_file=args.test_file,
        output_dir=args.output_dir
    )
    
    if args.tfidf_only:
        # 只运行TF-IDF
        runner.run_tfidf_benchmark(
            classifier_types=args.tfidf_classifiers,
            classification_types=args.classification_types
        )
    elif args.bert_only:
        # 只运行BERT
        runner.run_bert_benchmark(
            classification_types=args.classification_types
        )
    elif args.roberta_only:
        # 只运行RoBERTa
        runner.run_roberta_benchmark(
            classification_types=args.classification_types
        )
    elif args.llm_only:
        # 只运行LLM
        runner.run_llm_benchmark(
            model_name=args.llm_model,
            classification_types=args.classification_types
        )
    elif args.next_utterance_prediction_only:
        # 只运行医生提问下一句预测（使用 LLM 模型生成预测）
        runner.run_next_utterance_benchmark(
            eval_interval=args.next_utterance_eval_interval,
            model_name=args.llm_model,
            limit=args.next_utterance_limit
        )
    else:
        # 运行所有benchmark
        runner.run_all(
            skip_bert=args.skip_bert,
            skip_roberta=args.skip_roberta,
            skip_llm=args.skip_llm,
            skip_next_utterance=args.skip_next_utterance,
            llm_model=args.llm_model,
            next_utterance_eval_interval=args.next_utterance_eval_interval,
            next_utterance_limit=args.next_utterance_limit
        )
    
    # 保存结果
    runner.save_results()
    
    # 导出Excel（详细版本）
    try:
        runner.export_to_excel()
    except Exception as e:
        print(f"导出详细Excel失败: {e}")
        print("请确保已安装pandas和openpyxl: pip install pandas openpyxl")
    
    # 导出汇总Excel（按图示格式，支持追加）
    try:
        runner.export_summary_excel(
            filename=args.summary_file,
            append=True
        )
    except Exception as e:
        print(f"导出汇总Excel失败: {e}")
    
    # 导出Next Utterance结果到专门的Excel（支持追加）
    try:
        # 提取模型名称用于标识行
        model_name_for_next_utterance = args.llm_model.split('@')[0].split('/')[-1] if args.llm_model else "Default"
        runner.export_next_utterance_excel(
            filename=args.next_utterance_summary_file,
            model_name=model_name_for_next_utterance,
            append=True
        )
    except Exception as e:
        print(f"导出Next Utterance Excel失败: {e}")
    
    print("\n" + "="*80)
    print("Benchmark完成！")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()

