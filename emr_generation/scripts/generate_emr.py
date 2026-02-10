#!/usr/bin/env python3
"""
生成虚拟病例

使用方法：
    # 先运行分析脚本生成映射
    python scripts/analyze_data.py
    
    # 然后生成病例
    python scripts/generate_emr.py --num 10 --output outputs/generated_emrs.json
    
    # 使用LLM生成更自然的病例
    python scripts/generate_emr.py --num 5 --use-llm --host localhost --port 8000
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.generators.emr_generator import EMRGenerator


def main():
    parser = argparse.ArgumentParser(description="生成虚拟病例")
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=10,
        help="生成病例数量"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="输出文件路径"
    )
    parser.add_argument(
        "--distribution-mapping",
        type=Path,
        default=Config.DISTRIBUTION_MAPPING_FILE,
        help="分布映射文件"
    )
    parser.add_argument(
        "--keyword-mapping",
        type=Path,
        default=Config.KEYWORD_MAPPING_FILE,
        help="关键词映射文件"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="使用LLM生成更自然的文本"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=Config.LLM_DEFAULT_HOST,
        help="LLM 服务地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=Config.LLM_DEFAULT_PORT,
        help="LLM 服务端口"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=Config.LLM_DEFAULT_MODEL,
        help="LLM 模型名称"
    )
    parser.add_argument(
        "--diagnosis",
        type=str,
        choices=["Depression", "Anxiety", "Mix", "Other"],
        default=None,
        help="指定诊断类型（可选）"
    )
    parser.add_argument(
        "--gender",
        type=str,
        choices=["男", "女"],
        default=None,
        help="指定性别（可选）"
    )
    parser.add_argument(
        "--age-range",
        type=str,
        default=None,
        help="指定年龄范围，如 '20-40'（可选）"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并行工作线程数（默认1为串行，建议4-8）"
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Config.ensure_dirs()
    
    # 设置输出文件
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = Config.OUTPUTS_DIR / f"generated_emrs_{timestamp}.json"
    
    # 加载映射
    dist_mapping = None
    keyword_mapping = None
    
    if args.distribution_mapping.exists():
        print(f"加载分布映射: {args.distribution_mapping}")
        with open(args.distribution_mapping, 'r', encoding='utf-8') as f:
            dist_mapping = json.load(f)
    else:
        print("警告: 未找到分布映射文件，将使用默认分布")
    
    if args.keyword_mapping.exists():
        print(f"加载关键词映射: {args.keyword_mapping}")
        with open(args.keyword_mapping, 'r', encoding='utf-8') as f:
            keyword_mapping = json.load(f)
    else:
        print("警告: 未找到关键词映射文件，将使用默认关键词")
    
    # 初始化生成器
    print("\n初始化生成器...")
    generator = EMRGenerator(
        distribution_mapping=dist_mapping,
        keyword_mapping=keyword_mapping,
        llm_host=args.host,
        llm_port=args.port,
        llm_model=args.model,
        use_llm=args.use_llm,
    )
    
    # 构建约束条件
    constraints = {}
    if args.diagnosis:
        constraints["diagnosis"] = args.diagnosis
    if args.gender:
        constraints["gender"] = args.gender
    if args.age_range:
        age_min, age_max = map(int, args.age_range.split("-"))
        # 在生成时会在此范围内随机选择
        import random
        constraints["age"] = random.randint(age_min, age_max)
    
    # 生成病例
    print(f"\n开始生成 {args.num} 条虚拟病例...")
    print(f"使用LLM: {'是' if args.use_llm else '否'}")
    print(f"并行线程数: {args.workers}")
    if constraints:
        print(f"约束条件: {constraints}")
    
    def progress_callback(current, total):
        print(f"\r生成进度: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)
    
    import time
    start_time = time.time()
    
    emrs = generator.generate_batch(
        n=args.num,
        constraints=constraints if constraints else None,
        use_llm=args.use_llm,
        progress_callback=progress_callback,
        num_workers=args.workers,
    )
    
    elapsed_time = time.time() - start_time
    
    print()  # 换行
    
    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(emrs, f, ensure_ascii=False, indent=2)
    
    print(f"\n生成完成！共 {len(emrs)} 条病例")
    print(f"耗时: {elapsed_time:.2f} 秒")
    print(f"平均: {elapsed_time/len(emrs):.2f} 秒/条" if emrs else "")
    print(f"输出文件: {args.output}")
    
    # 打印示例
    if emrs:
        print("\n" + "=" * 60)
        print("示例病例预览:")
        print("=" * 60)
        
        example = emrs[0]
        print(f"\n患者ID: {example.get('patient_id')}")
        print(f"年龄: {example.get('Age')} 岁")
        print(f"性别: {example.get('Gender')}")
        print(f"科室: {example.get('Department')}")
        print(f"诊断: {example.get('Diagnosis')}")
        print(f"\n个人史: {example.get('PersonalHistory')[:100]}...")
        print(f"\n主诉: {example.get('ChiefComplaint')}")
        print(f"\n现病史: {example.get('PresentIllnessHistory')[:200]}...")


if __name__ == "__main__":
    main()
