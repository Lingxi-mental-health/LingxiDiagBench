"""
TF-IDF不同样本数对比实验

对比不同训练样本数（2000, 5000, 10000, 全量）对TF-IDF分类器性能的影响
"""

import json
import os
import time
from typing import Dict, List, Any
import pandas as pd
import numpy as np

try:
    from .config import (
        TRAIN_DATA_FILE, TEST_DATA_100_FILE, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS
    )
    from .data_utils import load_and_process_data
    from .tfidf_classifier import train_and_evaluate_tfidf
except ImportError:
    from config import (
        TRAIN_DATA_FILE, TEST_DATA_100_FILE, OUTPUT_DIR,
        TWO_CLASS_LABELS, FOUR_CLASS_LABELS, TWELVE_CLASS_LABELS
    )
    from data_utils import load_and_process_data
    from tfidf_classifier import train_and_evaluate_tfidf


def compare_sample_sizes(
    train_file: str,
    test_file: str,
    sample_sizes: List[int] = None,
    classifier_type: str = "logistic",
    classification_types: List[str] = None,
    output_dir: str = None,
    additional_train_file: str = None,
    external_val_file: str = None
) -> Dict[str, Any]:
    """
    对比不同样本数对TF-IDF分类器的影响
    
    Args:
        train_file: 主训练数据文件路径（LingxiDiag-16K）
        test_file: 测试数据文件路径（LingxiDiag-16K）
        sample_sizes: 要测试的样本数列表（None会被替换为全量），仅针对主训练数据
        classifier_type: 分类器类型
        classification_types: 分类类型列表
        output_dir: 输出目录
        additional_train_file: 额外的训练数据文件路径（SMHC_collection train），将与主训练数据混合
        external_val_file: 外部验证集文件路径（SMHC_collection validation）
        
    Returns:
        对比结果字典
    """
    if sample_sizes is None:
        sample_sizes = [2000, 5000, 10000, None]  # None表示全量
    
    if classification_types is None:
        classification_types = ["2class", "4class", "12class"]
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # 加载数据
    print("正在加载数据...")
    main_train_data = load_and_process_data(train_file)
    test_data = load_and_process_data(test_file)
    
    # 加载额外的训练数据（如果提供）
    additional_train_data = None
    if additional_train_file:
        print(f"加载额外训练数据: {additional_train_file}")
        additional_train_data = load_and_process_data(additional_train_file)
        print(f"  额外训练样本数: {len(additional_train_data)}")
    
    # 加载外部验证集（如果提供）
    external_val_data = None
    if external_val_file:
        print(f"加载外部验证集: {external_val_file}")
        external_val_data = load_and_process_data(external_val_file)
        print(f"  外部验证集样本数: {len(external_val_data)}")
    
    total_samples = len(main_train_data)
    print(f"\n主训练集样本数 (LingxiDiag-16K): {total_samples}")
    print(f"测试集样本数 (LingxiDiag-16K): {len(test_data)}")
    print(f"\n将对比以下主训练集样本数: {[s if s else total_samples for s in sample_sizes]}")
    if additional_train_data:
        print(f"每次实验都会混合 {len(additional_train_data)} 个额外训练样本 (SMHC_collection)\n")
    else:
        print()
    
    # 存储所有结果
    all_results = []
    comparison_data = {
        'sample_sizes': [],
        'results_by_task': {}
    }
    
    # 对每个样本数进行实验
    for sample_size in sample_sizes:
        sample_str = str(sample_size) if sample_size else "full"
        actual_main_size = sample_size if sample_size else total_samples
        
        # 准备训练数据：主训练集（可能采样） + 额外训练集（全量）
        if sample_size is not None and sample_size < len(main_train_data):
            print(f"\n从主训练集随机采样 {sample_size} 个样本...")
            from config import RANDOM_SEED
            np.random.seed(RANDOM_SEED)
            sample_indices = np.random.choice(len(main_train_data), sample_size, replace=False)
            sampled_main_train = [main_train_data[i] for i in sample_indices]
        else:
            sampled_main_train = main_train_data
        
        # 混合训练数据
        if additional_train_data:
            mixed_train_data = sampled_main_train + additional_train_data
            total_train_size = len(mixed_train_data)
            print(f"\n{'='*80}")
            print(f"{'实验': ^20} 主训练集: {actual_main_size}, 额外: {len(additional_train_data)}, 总计: {total_train_size}")
            print("="*80)
        else:
            mixed_train_data = sampled_main_train
            total_train_size = len(mixed_train_data)
            print(f"\n{'='*80}")
            print(f"{'实验': ^20} 训练集样本数: {actual_main_size} ({sample_str})")
            print("="*80)
        
        comparison_data['sample_sizes'].append(actual_main_size)
        
        # 对每种分类任务进行评估
        for class_type in classification_types:
            start_time = time.time()
            
            try:
                # 在LingxiDiag-16K测试集上评估
                print(f"\n训练 {class_type}...")
                result_test = train_and_evaluate_tfidf(
                    train_data=mixed_train_data,
                    test_data=test_data,
                    classification_type=class_type,
                    classifier_type=classifier_type,
                    save_model=False,  # 不保存中间模型
                    sample_size=None  # 已经在外部采样了
                )
                
                training_time = time.time() - start_time
                
                # 构建结果
                result = {
                    'classification_type': class_type,
                    'classifier_type': classifier_type,
                    'method': 'TF-IDF',
                    'sample_size_label': sample_str,
                    'actual_main_sample_size': actual_main_size,  # 主训练集样本数
                    'additional_sample_size': len(additional_train_data) if additional_train_data else 0,
                    'total_train_size': total_train_size,  # 总训练样本数
                    'training_time': training_time,
                    'LingxiDiag_metrics': result_test['metrics']  # LingxiDiag-16K测试集指标
                }
                
                print(f"  LingxiDiag-16K测试集 Macro-F1: {result_test['metrics']['macro_f1']:.4f}")
                
                # 在外部验证集上评估（如果提供）
                if external_val_data:
                    result_ext = train_and_evaluate_tfidf(
                        train_data=mixed_train_data,
                        test_data=external_val_data,
                        classification_type=class_type,
                        classifier_type=classifier_type,
                        save_model=False,
                        sample_size=None
                    )
                    result['smhc_metrics'] = result_ext['metrics']  # SMHC验证集指标
                    print(f"  SMHC验证集 Macro-F1: {result_ext['metrics']['macro_f1']:.4f}")
                
                all_results.append(result)
                
                # 按任务类型组织数据
                if class_type not in comparison_data['results_by_task']:
                    comparison_data['results_by_task'][class_type] = []
                comparison_data['results_by_task'][class_type].append(result)
                
                print(f"✓ {class_type} 完成，耗时: {training_time:.2f}秒")
                
            except Exception as e:
                print(f"✗ {class_type} 失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 生成对比报告
    print(f"\n{'='*80}")
    print(f"{'对比结果汇总': ^40}")
    print("="*80)
    
    report = generate_comparison_report(
        comparison_data, 
        has_external_val=external_val_data is not None,
        has_additional_train=additional_train_data is not None
    )
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_subdir = os.path.join(output_dir, f"sample_size_comparison_{timestamp}")
    os.makedirs(output_subdir, exist_ok=True)
    
    # 保存详细结果（JSON）
    results_file = os.path.join(output_subdir, "detailed_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment_info': {
                'train_file': train_file,
                'test_file': test_file,
                'classifier_type': classifier_type,
                'total_samples': total_samples,
                'sample_sizes_tested': [s if s else total_samples for s in sample_sizes]
            },
            'all_results': convert_to_serializable(all_results),
            'comparison_data': convert_to_serializable(comparison_data)
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {results_file}")
    
    # 保存对比报告（Markdown）
    report_file = os.path.join(output_subdir, "comparison_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"对比报告已保存到: {report_file}")
    
    # 保存对比表格（Excel）
    excel_file = os.path.join(output_subdir, "comparison_metrics.xlsx")
    save_comparison_excel(comparison_data, excel_file)
    print(f"对比表格已保存到: {excel_file}")
    
    return {
        'all_results': all_results,
        'comparison_data': comparison_data,
        'report': report,
        'output_dir': output_subdir
    }


def generate_comparison_report(
    comparison_data: Dict[str, Any], 
    has_external_val: bool = False,
    has_additional_train: bool = False
) -> str:
    """生成对比报告"""
    report_lines = []
    report_lines.append("# TF-IDF样本数对比实验报告\n")
    report_lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    sample_sizes = comparison_data['sample_sizes']
    report_lines.append(f"## 实验配置\n")
    report_lines.append(f"- **主训练集样本数 (LingxiDiag-16K)**: {', '.join(map(str, sample_sizes))}\n")
    
    # 获取额外训练集信息
    if has_additional_train and comparison_data['results_by_task']:
        first_task = list(comparison_data['results_by_task'].keys())[0]
        first_result = comparison_data['results_by_task'][first_task][0]
        additional_size = first_result.get('additional_sample_size', 0)
        if additional_size > 0:
            report_lines.append(f"- **额外训练集样本数 (SMHC_collection)**: {additional_size} (每次实验都会混合)\n")
    
    report_lines.append(f"- **评估数据集**: LingxiDiag-16K 测试集")
    if has_external_val:
        report_lines.append(" + SMHC_collection 验证集")
    report_lines.append("\n")
    
    # 对每个任务生成对比表格
    for class_type, results in comparison_data['results_by_task'].items():
        report_lines.append(f"\n## {class_type} 分类结果\n")
        
        # LingxiDiag-16K测试集结果
        report_lines.append(f"\n### LingxiDiag-16K 测试集\n")
        
        # 创建表格
        if class_type == "12class":
            # 多标签任务
            report_lines.append("| 主训练集样本数 | 总训练样本数 | Macro-F1 | Micro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | 训练时间(s) |")
            report_lines.append("|---------------|-------------|----------|----------|-------------|-----------------|--------------|-------------|")
            
            for result in results:
                metrics = result['LingxiDiag_metrics']
                main_size = result['actual_main_sample_size']
                total_size = result['total_train_size']
                report_lines.append(
                    f"| {main_size} | {total_size} | "
                    f"{metrics['macro_f1']:.4f} | "
                    f"{metrics['micro_f1']:.4f} | "
                    f"{metrics['weighted_f1']:.4f} | "
                    f"{metrics['macro_precision']:.4f} | "
                    f"{metrics['macro_recall']:.4f} | "
                    f"{result['training_time']:.2f} |"
                )
        else:
            # 单标签任务
            report_lines.append("| 主训练集样本数 | 总训练样本数 | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall | 训练时间(s) |")
            report_lines.append("|---------------|-------------|----------|----------|-------------|-----------------|--------------|-------------|")
            
            for result in results:
                metrics = result['LingxiDiag_metrics']
                main_size = result['actual_main_sample_size']
                total_size = result['total_train_size']
                report_lines.append(
                    f"| {main_size} | {total_size} | "
                    f"{metrics['accuracy']:.4f} | "
                    f"{metrics['macro_f1']:.4f} | "
                    f"{metrics['weighted_f1']:.4f} | "
                    f"{metrics['macro_precision']:.4f} | "
                    f"{metrics['macro_recall']:.4f} | "
                    f"{result['training_time']:.2f} |"
                )
        
        # SMHC验证集结果（如果有）
        if has_external_val and 'smhc_metrics' in results[0]:
            report_lines.append(f"\n### SMHC_collection 验证集\n")
            
            if class_type == "12class":
                report_lines.append("| 主训练集样本数 | 总训练样本数 | Macro-F1 | Micro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |")
                report_lines.append("|---------------|-------------|----------|----------|-------------|-----------------|--------------|")
                
                for result in results:
                    metrics = result['smhc_metrics']
                    main_size = result['actual_main_sample_size']
                    total_size = result['total_train_size']
                    report_lines.append(
                        f"| {main_size} | {total_size} | "
                        f"{metrics['macro_f1']:.4f} | "
                        f"{metrics['micro_f1']:.4f} | "
                        f"{metrics['weighted_f1']:.4f} | "
                        f"{metrics['macro_precision']:.4f} | "
                        f"{metrics['macro_recall']:.4f} |"
                    )
            else:
                report_lines.append("| 主训练集样本数 | 总训练样本数 | Accuracy | Macro-F1 | Weighted-F1 | Macro-Precision | Macro-Recall |")
                report_lines.append("|---------------|-------------|----------|----------|-------------|-----------------|--------------|")
                
                for result in results:
                    metrics = result['smhc_metrics']
                    main_size = result['actual_main_sample_size']
                    total_size = result['total_train_size']
                    report_lines.append(
                        f"| {main_size} | {total_size} | "
                        f"{metrics['accuracy']:.4f} | "
                        f"{metrics['macro_f1']:.4f} | "
                        f"{metrics['weighted_f1']:.4f} | "
                        f"{metrics['macro_precision']:.4f} | "
                        f"{metrics['macro_recall']:.4f} |"
                    )
        
        # 分析趋势
        report_lines.append(f"\n### 趋势分析\n")
        if len(results) >= 2:
            first_result = results[0]
            last_result = results[-1]
            
            # LingxiDiag-16K测试集趋势
            first_f1_ed = first_result['LingxiDiag_metrics']['macro_f1']
            last_f1_ed = last_result['LingxiDiag_metrics']['macro_f1']
            f1_improvement_ed = ((last_f1_ed - first_f1_ed) / first_f1_ed) * 100
            
            report_lines.append(f"**LingxiDiag-16K 测试集:**")
            report_lines.append(f"- Macro-F1 从 {first_f1_ed:.4f} 提升到 {last_f1_ed:.4f} (提升 {f1_improvement_ed:+.2f}%)")
            
            # SMHC验证集趋势（如果有）
            if has_external_val and 'smhc_metrics' in results[0]:
                first_f1_smhc = first_result['smhc_metrics']['macro_f1']
                last_f1_smhc = last_result['smhc_metrics']['macro_f1']
                f1_improvement_smhc = ((last_f1_smhc - first_f1_smhc) / first_f1_smhc) * 100
                
                report_lines.append(f"\n**SMHC_collection 验证集:**")
                report_lines.append(f"- Macro-F1 从 {first_f1_smhc:.4f} 提升到 {last_f1_smhc:.4f} (提升 {f1_improvement_smhc:+.2f}%)")
                
                # 泛化能力分析
                report_lines.append(f"\n**泛化能力对比 (LingxiDiag vs SMHC):**")
                for result in results:
                    ed_f1 = result['LingxiDiag_metrics']['macro_f1']
                    smhc_f1 = result['smhc_metrics']['macro_f1']
                    gap = ed_f1 - smhc_f1
                    main_size = result['actual_main_sample_size']
                    report_lines.append(
                        f"- 主训练集{main_size}样本: LingxiDiag={ed_f1:.4f}, SMHC={smhc_f1:.4f}, "
                        f"差距={gap:+.4f}"
                    )
            
            first_time = first_result['training_time']
            last_time = last_result['training_time']
            time_increase = ((last_time - first_time) / first_time) * 100
            
            report_lines.append(f"\n**训练时间:**")
            report_lines.append(f"- 从 {first_time:.2f}s 增加到 {last_time:.2f}s (增加 {time_increase:+.2f}%)")
    
    # 总结
    report_lines.append("\n## 总结\n")
    report_lines.append("### 性能 vs 样本数权衡\n")
    report_lines.append("- **小样本 (2000)**: 训练速度快，适合快速原型验证\n")
    report_lines.append("- **中样本 (5000)**: 性能和速度的良好平衡点\n")
    report_lines.append("- **大样本 (10000)**: 接近全量性能，训练时间可接受\n")
    report_lines.append("- **全量**: 最佳性能，但训练时间最长\n")
    
    if has_additional_train:
        report_lines.append("\n### 混合训练数据的影响\n")
        report_lines.append("- 每次实验都混合了SMHC_collection训练数据\n")
        report_lines.append("- 观察不同LingxiDiag-16K训练集大小对混合训练效果的影响\n")
    
    if has_external_val:
        report_lines.append("\n### 跨数据集泛化能力\n")
        report_lines.append("- 对比在LingxiDiag-16K和SMHC_collection上的性能\n")
        report_lines.append("- 性能差距可反映模型的泛化能力和数据集差异\n")
        report_lines.append("- 观察主训练集大小如何影响跨数据集性能\n")
    
    return "\n".join(report_lines)


def save_comparison_excel(comparison_data: Dict[str, Any], output_file: str) -> None:
    """保存对比结果到Excel"""
    writer = pd.ExcelWriter(output_file, engine='openpyxl')
    
    # 检查是否有外部验证集
    has_external_val = False
    if comparison_data['results_by_task']:
        first_task = list(comparison_data['results_by_task'].keys())[0]
        first_result = comparison_data['results_by_task'][first_task][0]
        has_external_val = 'smhc_metrics' in first_result
    
    # 为每个任务创建一个sheet
    for class_type, results in comparison_data['results_by_task'].items():
        rows = []
        
        for result in results:
            # 基本信息
            row = {
                '主训练集样本数': result['actual_main_sample_size'],
                '额外训练样本数': result.get('additional_sample_size', 0),
                '总训练样本数': result['total_train_size'],
                '训练时间(秒)': round(result['training_time'], 2),
            }
            
            # LingxiDiag-16K测试集指标
            ed_metrics = result['LingxiDiag_metrics']
            row['ED-Macro-F1'] = round(ed_metrics['macro_f1'], 4)
            row['ED-Macro-Precision'] = round(ed_metrics['macro_precision'], 4)
            row['ED-Macro-Recall'] = round(ed_metrics['macro_recall'], 4)
            row['ED-Weighted-F1'] = round(ed_metrics['weighted_f1'], 4)
            row['ED-Weighted-Precision'] = round(ed_metrics['weighted_precision'], 4)
            row['ED-Weighted-Recall'] = round(ed_metrics['weighted_recall'], 4)
            
            if class_type == "12class":
                row['ED-Micro-F1'] = round(ed_metrics['micro_f1'], 4)
                row['ED-Micro-Precision'] = round(ed_metrics['micro_precision'], 4)
                row['ED-Micro-Recall'] = round(ed_metrics['micro_recall'], 4)
            else:
                row['ED-Accuracy'] = round(ed_metrics['accuracy'], 4)
            
            # SMHC验证集指标（如果有）
            if has_external_val and 'smhc_metrics' in result:
                smhc_metrics = result['smhc_metrics']
                row['SMHC-Macro-F1'] = round(smhc_metrics['macro_f1'], 4)
                row['SMHC-Macro-Precision'] = round(smhc_metrics['macro_precision'], 4)
                row['SMHC-Macro-Recall'] = round(smhc_metrics['macro_recall'], 4)
                row['SMHC-Weighted-F1'] = round(smhc_metrics['weighted_f1'], 4)
                row['SMHC-Weighted-Precision'] = round(smhc_metrics['weighted_precision'], 4)
                row['SMHC-Weighted-Recall'] = round(smhc_metrics['weighted_recall'], 4)
                
                if class_type == "12class":
                    row['SMHC-Micro-F1'] = round(smhc_metrics['micro_f1'], 4)
                    row['SMHC-Micro-Precision'] = round(smhc_metrics['micro_precision'], 4)
                    row['SMHC-Micro-Recall'] = round(smhc_metrics['micro_recall'], 4)
                else:
                    row['SMHC-Accuracy'] = round(smhc_metrics['accuracy'], 4)
                
                # 计算性能差距
                row['F1-Gap(ED-SMHC)'] = round(ed_metrics['macro_f1'] - smhc_metrics['macro_f1'], 4)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name=class_type, index=False)
    
    writer.close()
    print(f"Excel文件已保存: {output_file}")


def convert_to_serializable(obj):
    """转换numpy类型为Python原生类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='TF-IDF样本数对比实验')
    parser.add_argument('--train', type=str, help='主训练数据文件路径 (LingxiDiag-16K)')
    parser.add_argument('--test', type=str, help='测试数据文件路径 (LingxiDiag-16K)')
    parser.add_argument('--additional-train', type=str, 
                        help='额外的训练数据文件路径 (SMHC_collection train)，将与主训练数据混合')
    parser.add_argument('--external-val', type=str, 
                        help='外部验证集文件路径 (SMHC_collection validation)')
    parser.add_argument('--sample-sizes', type=int, nargs='+',
                        help='要测试的主训练集样本数列表，例如: --sample-sizes 2000 5000 10000')
    parser.add_argument('--classifier', type=str, default='logistic',
                        choices=['logistic', 'svm', 'rf'],
                        help='分类器类型')
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['2class', '4class', '12class'],
                        help='分类任务类型')
    parser.add_argument('--output', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 使用默认配置或命令行参数
    train_file = args.train if args.train else TRAIN_DATA_FILE
    test_file = args.test if args.test else TEST_DATA_100_FILE
    
    # 设置样本数（添加None表示全量）
    sample_sizes = args.sample_sizes if args.sample_sizes else [2000, 5000, 10000]
    sample_sizes.append(None)  # 添加全量
    
    print("="*80)
    print(" TF-IDF样本数对比实验 ".center(80, "="))
    print("="*80)
    print(f"\n配置信息:")
    print(f"  主训练数据 (LingxiDiag-16K): {train_file}")
    print(f"  测试数据 (LingxiDiag-16K): {test_file}")
    if args.additional_train:
        print(f"  额外训练数据 (SMHC_collection): {args.additional_train}")
    if args.external_val:
        print(f"  外部验证集 (SMHC_collection): {args.external_val}")
    print(f"  主训练集样本数: {sample_sizes}")
    print(f"  分类器: {args.classifier}")
    print(f"  任务类型: {args.tasks}")
    print()
    
    # 运行对比实验
    result = compare_sample_sizes(
        train_file=train_file,
        test_file=test_file,
        sample_sizes=sample_sizes,
        classifier_type=args.classifier,
        classification_types=args.tasks,
        output_dir=args.output,
        additional_train_file=args.additional_train,
        external_val_file=args.external_val
    )
    
    print("\n" + "="*80)
    print(" 实验完成! ".center(80, "="))
    print("="*80)
    print(f"\n所有结果已保存到: {result['output_dir']}")
    print("\n生成的文件:")
    print(f"  - detailed_results.json: 详细的JSON格式结果")
    print(f"  - comparison_report.md: Markdown格式的对比报告")
    print(f"  - comparison_metrics.xlsx: Excel格式的指标对比表")
    
    if args.additional_train and args.external_val:
        print(f"\n实验类型: 混合训练 + 跨数据集评估")
        print(f"  - 混合了LingxiDiag-16K和SMHC_collection训练数据")
        print(f"  - 在两个数据集上评估了模型性能")
        print(f"  - 可以分析不同主训练集大小对泛化能力的影响")


if __name__ == "__main__":
    main()


