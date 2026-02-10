#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断数据预加载脚本

功能：
1. 读取患者对话数据
2. 使用指定的模型（离线VLLM或OpenRouter）批量生成诊断
3. 将诊断结果缓存到文件
4. 支持断点续传，跳过已生成的诊断
"""

import os
import sys
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from patient_test_ui.backend.agents import AutoDiagnosisVerifier


def load_patient_data(data_path: str, max_patients: Optional[int] = None) -> List[Dict]:
    """加载患者数据"""
    print(f"正在加载患者数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if max_patients:
        data = data[:max_patients]
    
    print(f"成功加载 {len(data)} 个患者数据")
    return data


def parse_conversation_text(text: str) -> List[Dict[str, str]]:
    """解析对话文本为结构化格式，支持角色后跟数字的情况"""
    if not text:
        return []
    
    lines = text.split('\n')
    conversation = []
    
    # 定义角色识别模式（支持角色后跟数字，如"医生1："、"家属2："等）
    role_patterns = [
        (r'^医生\d*[：:]', 'doctor'),
        (r'^患者\d*[：:]', 'patient'),
        (r'^患者本人\d*[：:]', 'patient'),
        (r'^病人\d*[：:]', 'patient'),
        (r'^家属\d*[：:]', 'family'),
        (r'^患者家属\d*[：:]', 'family'),
        (r'^未知发言人\d*[：:]', 'unknown'),
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        role = "others"
        content = line
        
        # 尝试匹配各种角色模式
        matched = False
        for pattern, role_type in role_patterns:
            match = re.match(pattern, line)
            if match:
                role = role_type
                # 提取冒号后的内容
                content = line[match.end():].strip()
                matched = True
                break
        
        # 如果没有匹配到任何角色，保持原始内容
        if not matched:
            content = line
        
        if content:
            conversation.append({
                "role": role,
                "content": content
            })
    
    return conversation


def validate_conversation_roles(
    conversation: List[Dict[str, str]], 
    patient_id: str = "unknown"
) -> None:
    """
    验证对话数据中是否包含有效的角色标识
    
    Args:
        conversation: 对话列表，每个元素包含role和content
        patient_id: 患者ID，用于错误提示
        
    Raises:
        ValueError: 如果对话为空或所有角色都是unknown
    """
    if not conversation:
        raise ValueError(f"患者 {patient_id}: 对话数据为空")
    
    # 检查是否所有角色都是unknown
    valid_roles = [turn for turn in conversation if turn.get("role") != "others"]
    
    if not valid_roles:
        # 提供详细的错误信息
        total_turns = len(conversation)
        sample_content = conversation[0].get("content", "")[:100] if conversation else ""
        
        raise ValueError(
            f"患者 {patient_id}: 对话数据中未检测到有效角色标识。\n"
            f"  - 总对话轮数: {total_turns}\n"
            f"  - 识别到的有效角色数: 0\n"
            f"  - 示例对话内容: {sample_content}...\n"
            f"请确保对话文本包含正确的角色前缀，支持的格式包括：\n"
            f"  - 医生：/医生1：\n"
            f"  - 患者：/患者本人：/病人：\n"
            f"  - 家属：/患者家属：\n"
        )
    
    # 统计角色分布（用于信息提示）
    role_counts = {}
    for turn in conversation:
        role = turn.get("role", "others")
        role_counts[role] = role_counts.get(role, 0) + 1
    
    # 如果unknown角色占比过高，给出警告
    unknown_count = role_counts.get("others", 0)
    if unknown_count > 0:
        unknown_ratio = unknown_count / len(conversation)
        if unknown_ratio > 0.3:  # 超过30%的对话无法识别角色
            print(
                f"警告 - 患者 {patient_id}: {unknown_count}/{len(conversation)} "
                f"({unknown_ratio:.1%}) 条对话未能识别角色"
            )


def extract_conversation(patient: Dict) -> List[Dict[str, str]]:
    """从患者数据中提取对话记录"""
    # 尝试从不同字段提取对话
    conversation_text = ""
    
    if "cleaned_text" in patient and patient["cleaned_text"]:
        conversation_text = patient["cleaned_text"]
    elif "PresentIllnessHistory" in patient and patient["PresentIllnessHistory"]:
        conversation_text = patient["PresentIllnessHistory"]
    
    if not conversation_text:
        return []
    
    return parse_conversation_text(conversation_text)


def load_existing_cache(cache_file: str) -> Dict[str, Any]:
    """加载已有的缓存数据"""
    if not os.path.exists(cache_file):
        return {}
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 加载缓存文件失败: {e}")
        return {}


def save_cache(cache_file: str, cache_data: Dict[str, Any]):
    """保存缓存数据"""
    # 确保目录存在
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)


def generate_diagnosis_for_patient(
    patient: Dict,
    verifier: AutoDiagnosisVerifier,
    cache_data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """为单个患者生成诊断"""
    patient_id = str(patient.get("patient_id", patient.get("患者", "unknown")))
    
    # 检查是否已缓存
    if patient_id in cache_data:
        return None
    
    # 提取对话
    conversation = extract_conversation(patient)
    
    if not conversation:
        print(f"患者 {patient_id}: 没有有效的对话记录，跳过")
        return None
    
    try:
        # 验证对话中是否包含有效角色
        validate_conversation_roles(conversation, patient_id)
        
        # 生成诊断
        result = verifier.generate_diagnosis(conversation)
        
        # 构建缓存条目
        cache_entry = {
            "patient_id": patient_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            "conversation": conversation,
            "diagnosis_result": {
                "thought": result.get("thought", ""),
                "icd_codes": result.get("icd_codes", []),
                "icd_box": result.get("icd_box", ""),
                "reasoning": result.get("reasoning", ""),
                "model": result.get("model", "")
            },
            # 保存真实诊断信息用于对比
            "ground_truth": {
                "diagnosis": patient.get("Diagnosis", ""),
                "diagnosis_code": patient.get("DiagnosisCode", "")
            }
        }
        
        return cache_entry
        
    except Exception as e:
        print(f"患者 {patient_id}: 生成诊断失败: {e}")
        traceback.print_exc()
        return None


def process_patient_batch(
    patients: List[Dict],
    model_config: Dict[str, Any],
    openrouter_config: Dict[str, Any],
    cache_file: str,
    max_workers: int = 4
) -> Dict[str, Any]:
    """批量处理患者数据"""
    # 加载现有缓存
    cache_data = load_existing_cache(cache_file)
    initial_count = len(cache_data)
    
    print(f"已缓存 {initial_count} 个患者的诊断数据")
    
    # 创建诊断生成器
    try:
        verifier = AutoDiagnosisVerifier(model_config, openrouter_config)
    except Exception as e:
        print(f"错误: 无法初始化诊断生成器: {e}")
        traceback.print_exc()
        return cache_data
    
    # 过滤出需要处理的患者
    patients_to_process = []
    for patient in patients:
        patient_id = str(patient.get("patient_id", patient.get("患者", "unknown")))
        if patient_id not in cache_data:
            patients_to_process.append(patient)
    
    if not patients_to_process:
        print("所有患者诊断已缓存，无需处理")
        return cache_data
    
    print(f"需要处理 {len(patients_to_process)} 个患者")
    
    # 使用线程池并行处理（如果max_workers > 1）
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    generate_diagnosis_for_patient,
                    patient,
                    verifier,
                    cache_data
                ): patient
                for patient in patients_to_process
            }
            
            # 使用tqdm显示进度
            with tqdm(total=len(patients_to_process), desc="生成诊断") as pbar:
                for future in as_completed(futures):
                    patient = futures[future]
                    patient_id = str(patient.get("patient_id", patient.get("患者", "unknown")))
                    
                    try:
                        result = future.result()
                        if result:
                            cache_data[patient_id] = result
                            # 定期保存缓存
                            if len(cache_data) % 10 == 0:
                                save_cache(cache_file, cache_data)
                    except Exception as e:
                        print(f"患者 {patient_id} 处理失败: {e}")
                    
                    pbar.update(1)
    else:
        # 串行处理
        for patient in tqdm(patients_to_process, desc="生成诊断"):
            patient_id = str(patient.get("patient_id", patient.get("患者", "unknown")))
            result = generate_diagnosis_for_patient(patient, verifier, cache_data)
            
            if result:
                cache_data[patient_id] = result
                # 定期保存缓存
                if len(cache_data) % 10 == 0:
                    save_cache(cache_file, cache_data)
    
    # 最终保存
    save_cache(cache_file, cache_data)
    
    final_count = len(cache_data)
    print(f"处理完成! 新增 {final_count - initial_count} 个诊断，总计 {final_count} 个")
    
    return cache_data


def main():
    parser = argparse.ArgumentParser(description="预加载诊断数据")
    
    # 数据配置
    parser.add_argument(
        "--data-path",
        type=str,
        default="raw_data/SMHC_LingxiDiag-16K_train_data.json",
        help="患者数据文件路径"
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="patient_test_ui/data/diagnosis_cache_LingxiDiag-16K_train_data.json",
        help="诊断缓存文件路径"
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="最大处理患者数量（用于测试）"
    )
    
    # 模型配置
    parser.add_argument(
        "--use-openrouter",
        action="store_true",
        help="使用OpenRouter API"
    )
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY", ""),
        help="OpenRouter API密钥"
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="qwen/qwen3-32b",
        help="OpenRouter模型名称"
    )
    parser.add_argument(
        "--offline-model",
        type=str,
        default="../../outputs/dataset_v2/LingxiDiagnosis-8B_icd-code-prediction_kimi-k2-0905-cot_grpo_with-real-data",
        help="离线模型路径"
    )
    parser.add_argument(
        "--offline-ports",
        type=str,
        default="9088",
        help="离线模型VLLM端口（多个端口用逗号分隔）"
    )
    parser.add_argument(
        "--vllm-ip",
        type=str,
        default="",
        help="VLLM服务IP地址（留空使用localhost）"
    )
    
    # 性能配置
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="并行处理的最大线程数"
    )
    
    args = parser.parse_args()
    
    # 构建模型配置
    if args.use_openrouter:
        model_config = {
            "use_openrouter": True,
            "openrouter_model": args.openrouter_model,
        }
        openrouter_config = {
            "api_key": args.openrouter_api_key,
            "site_url": "",
            "site_name": "Diagnosis Preload Script",
            "max_parallel": 16,
        }
        
        if not args.openrouter_api_key:
            print("错误: 使用OpenRouter需要提供API密钥 (--openrouter-api-key 或设置环境变量 OPENROUTER_API_KEY)")
            sys.exit(1)
        
        print(f"使用OpenRouter API: {args.openrouter_model}")
    else:
        # 解析端口（取第一个端口）
        ports = args.offline_ports.split(",")
        port = int(ports[0])
        
        # 构建model_name，格式为 /path/to/model@host:port 或 /path/to/model:port
        if args.vllm_ip:
            model_name = f"{args.offline_model}@{args.vllm_ip}:{port}"
        else:
            model_name = f"{args.offline_model}:{port}"
        
        model_config = {
            "use_openrouter": False,
            "local_model_name": model_name,  # 包含完整地址信息的模型名称
            "local_model_port": port,  # 保持与create_client_for_diagnosis一致
            "max_parallel": 1,
        }
        openrouter_config = {}
        
        print(f"使用离线VLLM模型: {args.offline_model}")
        print(f"端口: {port}")
        if args.vllm_ip:
            print(f"IP: {args.vllm_ip}")
        print(f"模型名称: {model_name}")
    
    # 加载患者数据
    try:
        patients = load_patient_data(args.data_path, args.max_patients)
    except Exception as e:
        print(f"错误: 无法加载患者数据: {e}")
        sys.exit(1)
    
    # 处理数据
    try:
        cache_data = process_patient_batch(
            patients,
            model_config,
            openrouter_config,
            args.cache_file,
            args.max_workers
        )
        
        print(f"\n诊断数据已保存到: {args.cache_file}")
        print(f"总计缓存 {len(cache_data)} 个患者的诊断数据")
        
    except KeyboardInterrupt:
        print("\n\n用户中断，正在保存已处理的数据...")
        sys.exit(0)
    except Exception as e:
        print(f"错误: 处理失败: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

