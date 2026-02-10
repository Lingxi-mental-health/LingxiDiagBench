#!/usr/bin/env python3
"""
完整流水线：分析 -> 提取映射 -> 生成病例

使用方法：
    # 完整流水线（不使用LLM）
    python scripts/run_pipeline.py --input real_emrs/input_real_emrs.json --num 20
    
    # 使用LLM
    python scripts/run_pipeline.py --input real_emrs/input_real_emrs.json --num 10 \
        --use-llm --host localhost --port 8000
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
from src.analyzers.distribution_analyzer import DistributionAnalyzer
from src.analyzers.keyword_analyzer import KeywordAnalyzer
from src.generators.emr_generator import EMRGenerator


def run_pipeline(
    input_file: Path,
    output_dir: Path,
    num_generate: int,
    use_llm: bool = False,
    llm_host: str = None,
    llm_port: int = None,
    llm_model: str = None,
):
    """运行完整流水线"""
    
    # Step 1: 加载数据
    print("=" * 70)
    print("Step 1: 加载数据")
    print("=" * 70)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print(f"  加载了 {len(records)} 条真实病例")
    
    # Step 2: 分析分布
    print("\n" + "=" * 70)
    print("Step 2: 分析数据分布")
    print("=" * 70)
    
    dist_analyzer = DistributionAnalyzer()
    dist_summary = dist_analyzer.analyze_records(records)
    
    dist_file = output_dir / "distribution_mapping.json"
    dist_analyzer.save_mapping(dist_file)
    
    print(f"  总记录数: {dist_summary['total_records']}")
    print(f"  性别分布: {dict(dist_summary['distributions'].get('gender', {}).get('counts', {}))}")
    print(f"  诊断分布: {dict(dist_summary['distributions'].get('overall_diagnosis', {}).get('counts', {}))}")
    
    # Step 3: 分析关键词
    print("\n" + "=" * 70)
    print("Step 3: 分析关键词")
    print("=" * 70)
    
    keyword_analyzer = KeywordAnalyzer()
    keyword_summary = keyword_analyzer.analyze_records(records)
    
    keyword_file = output_dir / "keyword_mapping.json"
    keyword_analyzer.save_mapping(keyword_file)
    
    trigger_keywords = keyword_summary.get("trigger_keywords", {}).get("keywords", {})
    print(f"  提取诱因关键词: {len(trigger_keywords)} 个")
    print(f"  Top 5 诱因: {list(trigger_keywords.keys())[:5]}")
    
    # Step 4: 生成虚拟病例
    print("\n" + "=" * 70)
    print("Step 4: 生成虚拟病例")
    print("=" * 70)
    
    generator = EMRGenerator(
        distribution_mapping=dist_summary,
        keyword_mapping=keyword_summary,
        llm_host=llm_host,
        llm_port=llm_port,
        llm_model=llm_model,
        use_llm=use_llm,
    )
    
    print(f"  目标生成数量: {num_generate}")
    print(f"  使用LLM: {'是' if use_llm else '否'}")
    
    def progress_callback(current, total):
        print(f"\r  生成进度: {current}/{total}", end="", flush=True)
    
    generated_emrs = generator.generate_batch(
        n=num_generate,
        use_llm=use_llm,
        progress_callback=progress_callback,
    )
    
    print()  # 换行
    
    # 保存生成结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"generated_emrs_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(generated_emrs, f, ensure_ascii=False, indent=2)
    
    print(f"  成功生成 {len(generated_emrs)} 条虚拟病例")
    print(f"  保存到: {output_file}")
    
    # Step 5: 质量评估
    print("\n" + "=" * 70)
    print("Step 5: 质量评估")
    print("=" * 70)
    
    # 统计生成病例的分布
    gen_gender = {}
    gen_diagnosis = {}
    
    for emr in generated_emrs:
        gender = emr.get("Gender", "未知")
        diag = emr.get("OverallDiagnosis", "未知")
        
        gen_gender[gender] = gen_gender.get(gender, 0) + 1
        gen_diagnosis[diag] = gen_diagnosis.get(diag, 0) + 1
    
    print(f"  生成病例性别分布: {gen_gender}")
    print(f"  生成病例诊断分布: {gen_diagnosis}")
    
    # 对比原始分布
    orig_gender = dist_summary['distributions'].get('gender', {}).get('counts', {})
    orig_diag = dist_summary['distributions'].get('overall_diagnosis', {}).get('counts', {})
    
    print(f"  原始病例性别分布: {dict(orig_gender)}")
    print(f"  原始病例诊断分布: {dict(orig_diag)}")
    
    print("\n" + "=" * 70)
    print("流水线完成!")
    print("=" * 70)
    
    # 打印示例
    if generated_emrs:
        print("\n" + "-" * 70)
        print("生成病例示例:")
        print("-" * 70)
        
        example = generated_emrs[0]
        print(f"\n患者ID: {example.get('patient_id')}")
        print(f"年龄/性别: {example.get('Age')}岁/{example.get('Gender')}")
        print(f"诊断: {example.get('Diagnosis')}")
        print(f"\n主诉: {example.get('ChiefComplaint')}")
        print(f"\n现病史: {example.get('PresentIllnessHistory')}")
    
    return {
        "distribution_mapping": dist_file,
        "keyword_mapping": keyword_file,
        "generated_emrs": output_file,
        "num_generated": len(generated_emrs),
    }


def main():
    parser = argparse.ArgumentParser(description="运行完整的虚拟病例生成流水线")
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
        "--num", "-n",
        type=int,
        default=10,
        help="生成病例数量"
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
    
    args = parser.parse_args()
    
    # 确保目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 运行流水线
    run_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        num_generate=args.num,
        use_llm=args.use_llm,
        llm_host=args.host,
        llm_port=args.port,
        llm_model=args.model,
    )


if __name__ == "__main__":
    main()
