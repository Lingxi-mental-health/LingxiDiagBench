#!/usr/bin/env python3
"""
使用 LLM 提取结构化映射信息

使用方法：
    python scripts/extract_mapping.py --input real_emrs/input_real_emrs.json \
        --host localhost --port 8000
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.extractors.llm_extractor import LLMExtractor, BatchLLMExtractor
from src.extractors.rule_extractor import RuleExtractor

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def load_data(filepath: Path) -> list:
    """加载数据文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_with_rules(records: list, output_file: Path):
    """使用规则提取"""
    print("=" * 60)
    print("使用规则提取结构化信息...")
    print("=" * 60)
    
    extractor = RuleExtractor()
    results = []
    
    for record in tqdm(records, desc="规则提取"):
        extracted = extractor.extract_all(record)
        
        # 转换 Pydantic 模型为字典
        if extracted.get("personal_history"):
            extracted["personal_history"] = extracted["personal_history"].model_dump()
        if extracted.get("chief_complaint"):
            extracted["chief_complaint"] = extracted["chief_complaint"].model_dump()
        if extracted.get("physical_illness"):
            extracted["physical_illness"] = extracted["physical_illness"].model_dump()
        
        results.append(extracted)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n规则提取结果已保存到: {output_file}")
    return results


def extract_with_llm(
    records: list,
    output_file: Path,
    host: str,
    port: int,
    model: str,
    fields: list = None,
    limit: int = None,
):
    """使用 LLM 提取"""
    print("\n" + "=" * 60)
    print("使用 LLM 提取结构化信息...")
    print(f"LLM 服务: http://{host}:{port}")
    print(f"模型: {model}")
    print("=" * 60)
    
    # 限制处理数量
    if limit:
        records = records[:limit]
        print(f"注意: 只处理前 {limit} 条记录")
    
    extractor = BatchLLMExtractor(host=host, port=port, model=model)
    
    def progress_callback(current, total):
        print(f"\r进度: {current}/{total} ({current/total*100:.1f}%)", end="", flush=True)
    
    results = extractor.extract_batch(
        records=records,
        fields=fields,
        progress_callback=progress_callback,
    )
    
    print()  # 换行
    
    # 转换结果
    for result in results:
        for key, value in result.items():
            if hasattr(value, 'model_dump'):
                result[key] = value.model_dump()
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nLLM 提取结果已保存到: {output_file}")
    return results


def merge_mappings(rule_file: Path, llm_file: Path, output_file: Path):
    """合并规则和LLM提取结果"""
    print("\n" + "=" * 60)
    print("合并提取结果...")
    print("=" * 60)
    
    with open(rule_file, 'r', encoding='utf-8') as f:
        rule_results = json.load(f)
    
    with open(llm_file, 'r', encoding='utf-8') as f:
        llm_results = json.load(f)
    
    llm_map = {r.get("patient_id"): r for r in llm_results}
    
    merged = []
    for rule_result in rule_results:
        patient_id = rule_result.get("patient_id")
        llm_result = llm_map.get(patient_id, {})
        
        combined = {**rule_result}
        
        if llm_result.get("personal_history"):
            combined["personal_history_llm"] = llm_result["personal_history"]
        if llm_result.get("chief_complaint"):
            combined["chief_complaint_llm"] = llm_result["chief_complaint"]
        if llm_result.get("present_illness"):
            combined["present_illness_llm"] = llm_result["present_illness"]
        
        merged.append(combined)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"合并结果已保存到: {output_file}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="提取结构化映射信息")
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
        "--rule-only",
        action="store_true",
        help="只使用规则提取"
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="只使用LLM提取"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="LLM提取的最大记录数"
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["personal_history", "chief_complaint", "present_illness"],
        help="LLM提取的字段列表"
    )
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"加载数据: {args.input}")
    records = load_data(args.input)
    print(f"共加载 {len(records)} 条记录")
    
    rule_file = args.output_dir / "rule_extracted.json"
    llm_file = args.output_dir / "llm_extracted.json"
    merged_file = args.output_dir / "merged_mapping.json"
    
    if not args.llm_only:
        extract_with_rules(records, rule_file)
    
    if not args.rule_only:
        try:
            extract_with_llm(
                records=records,
                output_file=llm_file,
                host=args.host,
                port=args.port,
                model=args.model,
                fields=args.fields,
                limit=args.limit,
            )
            
            if not args.llm_only and rule_file.exists():
                merge_mappings(rule_file, llm_file, merged_file)
                
        except Exception as e:
            print(f"\nLLM 提取失败: {e}")
            print("请确保 LLM 服务已启动，或使用 --rule-only 选项只进行规则提取")
    
    print("\n" + "=" * 60)
    print("提取完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
