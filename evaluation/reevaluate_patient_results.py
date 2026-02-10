#!/usr/bin/env python3
"""
Patient 评估结果重新评估脚本

用于从已保存的 JSON 文件中读取 patient 回复，使用新的评估模型重新评估，
并生成新的评估结果和聚合报告。

使用场景：
- 之前评估时评估模型 IP 地址填错导致评估失败
- 需要用不同的评估模型重新评估已有的 patient 回复

使用方法:
    # 重新评估单个文件
    python reevaluate_patient_results.py \
        --input-files "./evaluation_results/static_patient_eval/patient_eval_v3_*.json" \
        --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
        --output-dir ./evaluation_results/static_patient_eval \
        --max-workers 16

    # 从目录重新评估所有文件
    python reevaluate_patient_results.py \
        --input-dir ./evaluation_results/static_patient_eval \
        --eval-models "gemma-3-27b-it@10.119.29.220:9051,qwen3-30b@10.119.29.220:9041,gpt-oss-20b@10.119.29.220:9042" \
        --max-workers 16
"""

import argparse
import json
import os
import sys
import concurrent.futures
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import glob
import re

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from evaluation.llm_client import (
    create_llm_client,
    PromptLoader,
    parse_evaluation_result,
    parse_realness_multi_result,
    RealnessMultiEvaluation,
    extract_message_text,
)

# 导入 unified_patient_eval 中的类和函数
from evaluation.unified_patient_eval import (
    UnifiedPatientEvaluator,
    TurnEvaluationResult,
    SampleEvaluationResult,
    EvalTask,
    compute_statistics,
    print_summary,
    result_to_dict,
    sanitize_model_name,
    calc_average,
    calc_std,
    aggregate_multi_model_results,
    print_patient_aggregated_summary,
    save_patient_aggregated_results,
    save_patient_aggregated_summary_txt,
)


@dataclass
class ReevalTask:
    """重新评估任务数据结构"""
    patient_id: str
    turn_index: int
    doctor_question: str
    patient_reply: str
    patient_info: str  # 从原样本信息恢复
    dialogue_history: str  # 可能为空
    information_exists: bool  # 从原始结果保留
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_existing_result(file_path: str) -> Dict[str, Any]:
    """加载已有的评估结果文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_reeval_tasks(result_data: Dict[str, Any]) -> List[ReevalTask]:
    """
    从已有的评估结果中提取重新评估任务
    
    Args:
        result_data: 已有的评估结果数据
    
    Returns:
        重新评估任务列表
    """
    tasks = []
    
    for sample_result in result_data.get("results", []):
        patient_id = sample_result.get("patient_id", "unknown")
        metadata = sample_result.get("metadata", {})
        
        # 构建基本的 patient_info（从 metadata 恢复）
        patient_info_parts = []
        if metadata.get("age"):
            patient_info_parts.append(f"年龄: {metadata['age']}")
        if metadata.get("gender"):
            patient_info_parts.append(f"性别: {metadata['gender']}")
        if metadata.get("diagnosis"):
            patient_info_parts.append(f"诊断: {metadata['diagnosis']}")
        patient_info = "\n".join(patient_info_parts) if patient_info_parts else "无详细信息"
        
        # 处理每个轮次
        for turn in sample_result.get("turns", []):
            doctor_question = turn.get("doctor_question", "")
            patient_reply = turn.get("patient_reply", "")
            
            if not doctor_question or not patient_reply:
                continue
            
            # 获取 information_exists（如果原始结果有）
            information_exists = turn.get("information_exists", True)
            
            tasks.append(ReevalTask(
                patient_id=patient_id,
                turn_index=turn.get("turn_index", 0),
                doctor_question=doctor_question,
                patient_reply=patient_reply,
                patient_info=patient_info,
                dialogue_history="",  # 对话历史在原始文件中可能不完整，设为空
                information_exists=information_exists,
                metadata=metadata,
            ))
    
    return tasks


def reevaluate_tasks_parallel(
    tasks: List[ReevalTask],
    eval_models: List[str],
    max_workers: int = 8,
    api_key: Optional[str] = None,
) -> Dict[str, Dict[str, List[Tuple[int, TurnEvaluationResult]]]]:
    """
    并行重新评估所有任务
    
    Args:
        tasks: 重新评估任务列表
        eval_models: 评估模型列表
        max_workers: 每个模型的并行线程数
        api_key: API Key
    
    Returns:
        模型名称 -> (patient_id -> [(turn_index, TurnEvaluationResult), ...]) 的映射
    """
    print(f"\n[评估] 开始并行评估...")
    print(f"[评估] 评估任务数: {len(tasks)}")
    print(f"[评估] 评估模型: {', '.join(eval_models)}")
    print(f"[评估] 每模型并行线程数: {max_workers}")
    
    # 初始化所有评估器
    evaluators = {}
    for model_name in eval_models:
        evaluators[model_name] = UnifiedPatientEvaluator(
            eval_model=model_name,
            api_key=api_key,
        )
    
    # 结果存储
    all_model_results: Dict[str, Dict[str, List[Tuple[int, TurnEvaluationResult]]]] = {
        model: {} for model in eval_models
    }
    
    def evaluate_single_model(model_name: str) -> Dict[str, List[Tuple[int, TurnEvaluationResult]]]:
        """评估单个模型"""
        evaluator = evaluators[model_name]
        model_results: Dict[str, List[Tuple[int, TurnEvaluationResult]]] = {}
        
        lock = threading.Lock()
        progress = {"count": 0}
        
        def evaluate_single_task(task: ReevalTask) -> Tuple[str, int, TurnEvaluationResult]:
            """评估单个任务"""
            turn_result = evaluator.evaluate_turn(
                turn_index=task.turn_index,
                doctor_question=task.doctor_question,
                patient_reply=task.patient_reply,
                patient_info=task.patient_info,
                dialogue_history=task.dialogue_history,
            )
            
            with lock:
                progress["count"] += 1
                if progress["count"] % 20 == 0 or progress["count"] == len(tasks):
                    print(f"  [{model_name}] 评估进度: {progress['count']}/{len(tasks)}")
            
            return task.patient_id, task.turn_index, turn_result
        
        print(f"\n  [{model_name}] 开始评估 {len(tasks)} 个任务...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_single_task, task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    patient_id, turn_idx, turn_result = future.result()
                    
                    with lock:
                        if patient_id not in model_results:
                            model_results[patient_id] = []
                        model_results[patient_id].append((turn_idx, turn_result))
                except Exception as e:
                    print(f"  [{model_name}] 评估失败: {e}")
        
        print(f"  [{model_name}] 完成! 评估 {len(model_results)} 个样本")
        return model_results
    
    # 并行评估所有模型
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(eval_models)) as executor:
        model_futures = {
            executor.submit(evaluate_single_model, model): model 
            for model in eval_models
        }
        
        for future in concurrent.futures.as_completed(model_futures):
            model_name = model_futures[future]
            try:
                all_model_results[model_name] = future.result()
            except Exception as e:
                print(f"[评估] 模型 {model_name} 评估失败: {e}")
    
    print(f"\n[评估] 所有模型评估完成!")
    return all_model_results


def build_sample_results(
    tasks: List[ReevalTask],
    eval_results: Dict[str, List[Tuple[int, TurnEvaluationResult]]],
) -> Dict[str, SampleEvaluationResult]:
    """
    将评估结果构建为 SampleEvaluationResult 格式
    
    Args:
        tasks: 原始任务列表
        eval_results: 评估结果
    
    Returns:
        patient_id -> SampleEvaluationResult 的映射
    """
    # 收集每个 patient 的 metadata
    patient_metadata = {}
    patient_turns_count = {}
    for task in tasks:
        if task.patient_id not in patient_metadata:
            patient_metadata[task.patient_id] = task.metadata
            patient_turns_count[task.patient_id] = 0
        patient_turns_count[task.patient_id] += 1
    
    # 构建结果
    results: Dict[str, SampleEvaluationResult] = {}
    
    for patient_id, turn_list in eval_results.items():
        # 按 turn_idx 排序
        turn_list.sort(key=lambda x: x[0])
        turn_results = [tr for _, tr in turn_list]
        
        results[patient_id] = SampleEvaluationResult(
            patient_id=patient_id,
            total_turns=patient_turns_count.get(patient_id, len(turn_results)),
            evaluated_turns=len(turn_results),
            turn_results=turn_results,
            metadata=patient_metadata.get(patient_id, {}),
        )
    
    return results


def save_reevaluated_results(
    output_dir: Path,
    model_name: str,
    all_results: Dict[str, SampleEvaluationResult],
    stats: Dict[str, Any],
    original_metadata: Dict[str, Any],
    timestamp: str,
):
    """保存重新评估的结果"""
    model_safe_name = sanitize_model_name(model_name)
    
    # 从原始 metadata 获取信息
    patient_model = original_metadata.get("patient_model", "unknown")
    patient_version = original_metadata.get("patient_version", "unknown")
    patient_safe_name = sanitize_model_name(patient_model)
    
    # 生成文件名
    prefix = f"patient_eval_{patient_version}_{patient_safe_name}"
    output_file = output_dir / f"{prefix}_reevaluated_{model_safe_name}_{timestamp}.json"
    
    output_data = {
        "metadata": {
            "data_file": original_metadata.get("data_file", ""),
            "eval_model": model_name,
            "timestamp": timestamp,
            "use_original_reply": original_metadata.get("use_original_reply", False),
            "patient_model": patient_model,
            "patient_version": patient_version,
            "reevaluated": True,
            "original_timestamp": original_metadata.get("timestamp", ""),
        },
        "statistics": stats,
        "results": [result_to_dict(r) for r in all_results.values()],
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"[{model_name}] 重新评估结果已保存到: {output_file}")
    
    return output_file


def find_input_files(
    input_files: Optional[str],
    input_dir: Optional[str],
    pattern: str = "patient_eval_*.json",
) -> List[str]:
    """查找输入文件"""
    files = []
    
    if input_files:
        # 支持 glob 模式
        for pattern_str in input_files.split(","):
            pattern_str = pattern_str.strip()
            if "*" in pattern_str or "?" in pattern_str:
                files.extend(glob.glob(pattern_str))
            elif os.path.exists(pattern_str):
                files.append(pattern_str)
    
    if input_dir:
        input_path = Path(input_dir)
        if input_path.exists():
            files.extend([str(f) for f in input_path.glob(pattern)])
    
    # 去重并排序
    files = sorted(set(files))
    
    return files


def group_files_by_patient_model(files: List[str]) -> Dict[str, List[str]]:
    """
    按 patient_model 分组文件
    
    返回: patient_model -> [file_paths] 的映射
    """
    groups: Dict[str, List[str]] = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            patient_model = metadata.get("patient_model", "unknown")
            patient_version = metadata.get("patient_version", "unknown")
            key = f"{patient_version}_{patient_model}"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(file_path)
        except Exception as e:
            print(f"[警告] 无法读取文件 {file_path}: {e}")
    
    return groups


def main():
    parser = argparse.ArgumentParser(
        description="Patient 评估结果重新评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 输入相关参数
    parser.add_argument(
        "--input-files",
        type=str,
        help="输入文件路径（支持 glob 模式，如 'path/*.json'，多个用逗号分隔）"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="输入目录，自动查找目录下的 patient_eval_*.json 文件"
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="patient_eval_*.json",
        help="文件匹配模式（与 --input-dir 配合使用，默认: patient_eval_*.json）"
    )
    
    # 评估模型相关参数
    parser.add_argument(
        "--eval-models",
        type=str,
        required=True,
        help="评估模型列表，逗号分隔（例如: gemma-3-27b-it@10.119.29.220:9051）"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API Key（仅用于 OpenRouter 模型）"
    )
    
    # 输出相关参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results/static_patient_eval",
        help="输出目录（默认: ./evaluation_results/static_patient_eval）"
    )
    
    # 并行相关参数
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="最大并行工作线程数（默认: 16）"
    )
    
    # 其他选项
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要处理的文件，不实际运行"
    )
    # 注意：现在默认按 patient_model 分组，同一个 patient_model 的多个文件会合并处理
    
    args = parser.parse_args()
    
    # 查找输入文件
    input_files = find_input_files(args.input_files, args.input_dir, args.file_pattern)
    
    if not input_files:
        print("[错误] 未找到输入文件")
        print("请使用 --input-files 或 --input-dir 指定输入")
        sys.exit(1)
    
    print(f"\n{'#'*60}")
    print(f"# Patient 评估结果重新评估")
    print(f"{'#'*60}")
    print(f"\n找到 {len(input_files)} 个输入文件:")
    for i, f in enumerate(input_files, 1):
        print(f"  {i}. {os.path.basename(f)}")
    
    # 解析评估模型列表
    eval_models = [m.strip() for m in args.eval_models.split(",") if m.strip()]
    print(f"\n评估模型: {', '.join(eval_models)}")
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        print("\n[Dry Run] 以下是将要执行的操作：")
        for file_path in input_files:
            print(f"\n  - 输入: {os.path.basename(file_path)}")
            print(f"    评估模型: {', '.join(eval_models)}")
            print(f"    输出目录: {output_dir}")
        print("\n[Dry Run] 结束")
        sys.exit(0)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 按 patient_model 分组处理（默认行为）
    # 这样同一个 patient_model 的所有 patient_reply 会被合并，
    # 然后用所有评估模型评估后生成一个综合的聚合报告
    file_groups = group_files_by_patient_model(input_files)
    print(f"\n按 patient_model 分组: {len(file_groups)} 组")
    for key, files in file_groups.items():
        print(f"  - {key}: {len(files)} 个文件")
    
    # 处理每组文件
    all_aggregated_results = {}
    
    for group_key, group_files in file_groups.items():
        print(f"\n\n{'='*60}")
        print(f"处理组: {group_key}")
        print(f"{'='*60}")
        
        # 合并所有文件的任务（去重）
        all_tasks: List[ReevalTask] = []
        seen_tasks: set = set()  # 用于去重
        original_metadata: Dict[str, Any] = {}
        
        for file_path in group_files:
            print(f"\n加载文件: {os.path.basename(file_path)}")
            result_data = load_existing_result(file_path)
            
            # 保存原始 metadata
            if not original_metadata:
                original_metadata = result_data.get("metadata", {})
            
            # 提取任务
            tasks = extract_reeval_tasks(result_data)
            
            # 去重：同一个 patient_id + turn_index 的任务只保留一个
            new_tasks = 0
            for task in tasks:
                task_key = (task.patient_id, task.turn_index)
                if task_key not in seen_tasks:
                    seen_tasks.add(task_key)
                    all_tasks.append(task)
                    new_tasks += 1
            
            print(f"  提取 {len(tasks)} 个任务，新增 {new_tasks} 个（去重后）")
        
        if not all_tasks:
            print(f"[警告] 组 {group_key} 没有可评估的任务，跳过")
            continue
        
        print(f"\n组 {group_key} 共 {len(all_tasks)} 个评估任务")
        
        # 并行评估
        eval_results = reevaluate_tasks_parallel(
            tasks=all_tasks,
            eval_models=eval_models,
            max_workers=args.max_workers,
            api_key=args.api_key,
        )
        
        # 保存每个模型的结果
        group_model_results = {}
        
        for model_name in eval_models:
            model_eval_results = eval_results.get(model_name, {})
            
            if not model_eval_results:
                print(f"[警告] 模型 {model_name} 没有评估结果")
                continue
            
            # 构建 SampleEvaluationResult
            sample_results = build_sample_results(all_tasks, model_eval_results)
            
            # 计算统计
            stats = compute_statistics(sample_results)
            
            # 打印摘要
            print_summary(stats, model_name, original_metadata.get("data_file", ""))
            
            # 保存结果
            output_file = save_reevaluated_results(
                output_dir=output_dir,
                model_name=model_name,
                all_results=sample_results,
                stats=stats,
                original_metadata=original_metadata,
                timestamp=timestamp,
            )
            
            # 加载保存的结果用于聚合
            with open(output_file, 'r', encoding='utf-8') as f:
                group_model_results[model_name] = json.load(f)
        
        # 聚合多模型结果
        if len(group_model_results) > 0:
            print(f"\n正在聚合组 {group_key} 的多模型评估结果...")
            
            aggregated = aggregate_multi_model_results(group_model_results, "average")
            
            # 打印摘要
            print_patient_aggregated_summary(aggregated)
            
            # 生成安全的组名
            safe_group_key = sanitize_model_name(group_key)
            
            # 保存 JSON 格式结果
            json_output = output_dir / f"patient_aggregated_results_{safe_group_key}_{timestamp}.json"
            save_patient_aggregated_results(aggregated, json_output)
            
            # 保存文本格式摘要报告
            summary_output = output_dir / f"patient_aggregated_summary_{safe_group_key}_{timestamp}.txt"
            save_patient_aggregated_summary_txt(aggregated, summary_output)
            
            all_aggregated_results[group_key] = aggregated
    
    print(f"\n\n{'#'*60}")
    print(f"# 重新评估完成")
    print(f"{'#'*60}")
    print(f"\n处理了 {len(file_groups)} 组文件")
    print(f"结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

