from gpu_manager import GPUManager
import json
import os
import time
import argparse
from dotenv import load_dotenv
# 导入评估函数
from evaluation.doctor_eval_multilabel import api_llm_diagnosis_evaluation_multilabel, print_evaluation_results

# 加载.env文件配置
load_dotenv()

# # 配置模型参数供不依赖环境变量的用户使用
# os.environ['MAX_PATIENTS'] = '100'
# os.environ['DOCTOR_VERSION'] = 'base'
# os.environ['PATIENT_VERSION'] = 'v1'

# # 每个 Agent 单独控制是否使用 OpenRouter
# os.environ['PATIENT_USE_OPENROUTER'] = 'True'
# os.environ['DOCTOR_USE_OPENROUTER'] = 'True'
# os.environ['VERIFIER_USE_OPENROUTER'] = 'True'
# os.environ['OPENROUTER_DOCTOR_MODEL'] = 'qwen/qwen3-8b'
# os.environ['OPENROUTER_PATIENT_MODEL'] = 'qwen/qwen3-8b'
# os.environ['OPENROUTER_VERIFIER_MODEL'] = 'qwen/qwen3-8b'
# os.environ['OPENROUTER_MAX_PARALLEL'] = '1'
os.environ['EVALUATE_DIAGNOSIS'] = 'False'

# ================== 版本控制配置 ==================
# Doctor版本选择：'v1' = 传统诊断树，'v2' = 阶段式诊断树，'base' = 基础问诊（无诊断树）
DOCTOR_VERSION = os.getenv('DOCTOR_VERSION', 'v2')  # 可选值：'v1', 'v2', 'base'
# Patient版本选择：'cot' = Chain of Thought版本，'v1' = 基础版本
PATIENT_VERSION = os.getenv('PATIENT_VERSION', 'v3')  # 可选值：'cot', 'v1', 'v3'

# 配置参数
DOCTOR_PROMPT_PATH = './prompts/doctor/doctor_persona.json'
PATIENT_INFO_PATH = './emr_generation/outputs/generated_emrs_20251218_010011.json'
DIAGTREE_PATH = './prompts/diagtree'
NUM_ROUNDS = 1
OUTPUT_DATASYN_PATH = './LingxiDiag-16K_dataSyn'
OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION = None  # 设置为None时不生成分类版本
# OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION = './DataSyn/DataSyn_true_data_8b_923_doctor_base_with_classification'  # 需要时取消注释

# 根据Doctor版本和Patient版本导入对应的类
try:
    # 导入Doctor类
    if DOCTOR_VERSION == 'v2':
        print("=== 使用Doctor V2版本：阶段式诊断树 ===")
        from src.doctor.doctor_v2 import Doctor
        from src.doctor.diagtree_v2 import DiagTree
    elif DOCTOR_VERSION == 'v1':
        print("=== 使用Doctor V1版本：传统诊断树 ===")
        from src.doctor.doctor_v1 import Doctor
        from src.doctor.diagtree_v1 import DiagTree
    elif DOCTOR_VERSION == 'base':
        print("=== 使用Doctor基础版本：无诊断树问诊 ===")
        from src.doctor.doctor_base import DoctorBase as Doctor
        # 基础版本不需要DiagTree
        DiagTree = None
    else:
        raise ValueError(f"不支持的Doctor版本: {DOCTOR_VERSION}。请选择 'v1', 'v2' 或 'base'")
    
    # 导入Patient类
    if PATIENT_VERSION == 'cot':
        print("=== 使用Patient CoT版本：Chain of Thought ===")
        from src.patient.patient_cot import Patient
    elif PATIENT_VERSION == 'v1':
        print("=== 使用Patient V1版本：基础版本 ===")
        from src.patient.patient_v1 import Patient
    elif PATIENT_VERSION == 'v3':
        print("=== 使用Patient V3版本：优化版本 ===")
        from src.patient.patient_v3 import Patient
    else:
        raise ValueError(f"不支持的Patient版本: {PATIENT_VERSION}。请选择 'cot' 或 'v1' 或 'v3'")
        
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确的目录下运行程序")
    if DOCTOR_VERSION == 'v2':
        print("如果Doctor V2版本导入失败，请检查doctor_v2.py和diagtree_v2.py是否存在")
    elif DOCTOR_VERSION == 'v1':
        print("如果Doctor V1版本导入失败，请检查doctor_v1.py和diagtree_v1.py是否存在")
    elif DOCTOR_VERSION == 'base':
        print("如果Doctor基础版本导入失败，请检查doctor_base.py是否存在")
    if PATIENT_VERSION == 'cot':
        print("如果Patient CoT版本导入失败，请检查patient_cot.py是否存在")
    elif PATIENT_VERSION == 'v1':
        print("如果Patient V1版本导入失败，请检查patient_v1.py是否存在")
    elif PATIENT_VERSION == 'v3':
        print("如果Patient V3版本导入失败，请检查patient_v3.py是否存在")
    raise

def main():
    # ================== 输出路径配置 ==================
    # 从环境变量读取输出路径，如果未设置则使用默认值
    global OUTPUT_DATASYN_PATH, OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION
    OUTPUT_DATASYN_PATH = os.getenv('OUTPUT_DATASYN_PATH', OUTPUT_DATASYN_PATH)
    
    # ================== 模型配置 ==================
    # 每个 Agent 单独控制是否使用 OpenRouter
    
    print("="*60)
    print("=== Agent 模型配置 ===")
    print("="*60)
    
    # 辅助函数：将字符串转为布尔值
    def str_to_bool(value):
        """将字符串转为布尔值"""
        if isinstance(value, bool):
            return value
        return value.lower() in ('true', '1', 'yes')
    
    # ========== Doctor 模型配置 ==========
    doctor_use_openrouter = str_to_bool(os.getenv('DOCTOR_USE_OPENROUTER', 'false'))
    
    if doctor_use_openrouter:
        # 使用 OpenRouter
        DOCTOR_MODEL = os.getenv('OPENROUTER_DOCTOR_MODEL', 'qwen/qwen3-8b')
        DOCTOR_PORT = [9041]  # OpenRouter不使用端口，但保留占位符
        print(f"Doctor: OpenRouter ({DOCTOR_MODEL})")
    else:
        # 使用离线模型
        DOCTOR_MODEL = os.getenv('OFFLINE_DOCTOR_MODEL', '../../models/Qwen3-8B')
        doctor_ports_str = os.getenv('OFFLINE_DOCTOR_PORTS', '9041')
        DOCTOR_PORT = [int(p.strip()) for p in doctor_ports_str.split(',')]
        
        # 检查是否配置了远程vLLM服务的IP地址
        VLLM_DOCTOR_IP = os.getenv('VLLM_DOCTOR_IP', '').strip()
        if VLLM_DOCTOR_IP:
            DOCTOR_MODEL = f"{DOCTOR_MODEL}@{VLLM_DOCTOR_IP}"
            print(f"Doctor: 远程部署 ({DOCTOR_MODEL}, 端口: {DOCTOR_PORT})")
        else:
            print(f"Doctor: 本地部署 ({DOCTOR_MODEL}, 端口: {DOCTOR_PORT})")
    
    # ========== Patient 模型配置 ==========
    patient_use_openrouter = str_to_bool(os.getenv('PATIENT_USE_OPENROUTER', 'false'))
    
    if patient_use_openrouter:
        # 使用 OpenRouter
        PATIENT_MODEL = os.getenv('OPENROUTER_PATIENT_MODEL', 'qwen/qwen3-8b')
        PATIENT_PORT = [9041]  # OpenRouter不使用端口，但保留占位符
        print(f"Patient: OpenRouter ({PATIENT_MODEL})")
    else:
        # 使用离线模型
        PATIENT_MODEL = os.getenv('OFFLINE_PATIENT_MODEL', '../../models/Qwen3-8B')
        patient_ports_str = os.getenv('OFFLINE_PATIENT_PORTS', '9041')
        PATIENT_PORT = [int(p.strip()) for p in patient_ports_str.split(',')]
        
        # 检查是否配置了远程vLLM服务的IP地址
        VLLM_PATIENT_IP = os.getenv('VLLM_PATIENT_IP', '').strip()
        if VLLM_PATIENT_IP:
            PATIENT_MODEL = f"{PATIENT_MODEL}@{VLLM_PATIENT_IP}"
            print(f"Patient: 远程部署 ({PATIENT_MODEL}, 端口: {PATIENT_PORT})")
        else:
            print(f"Patient: 本地部署 ({PATIENT_MODEL}, 端口: {PATIENT_PORT})")
    
    # ========== 计算最大并行进程数 ==========
    # 根据使用 OpenRouter 的数量来决定并行度
    openrouter_count = sum([doctor_use_openrouter, patient_use_openrouter])
    
    if openrouter_count == 2:
        # 全部使用 OpenRouter，可以支持更高并发
        MAX_PARALLEL_PROCESSES = int(os.getenv('OPENROUTER_MAX_PARALLEL', '16'))
        print(f"并行度: 全部使用 OpenRouter，并行数 = {MAX_PARALLEL_PROCESSES}")
    elif openrouter_count == 0:
        # 全部使用离线模型
        offline_max_parallel_str = os.getenv('OFFLINE_MAX_PARALLEL', '').strip()
        if offline_max_parallel_str:
            MAX_PARALLEL_PROCESSES = int(offline_max_parallel_str)
            print(f"并行度: 使用配置的并行数 = {MAX_PARALLEL_PROCESSES}")
        else:
            # 自动计算：取两个端口数的最小值
            MAX_PARALLEL_PROCESSES = min(len(DOCTOR_PORT), len(PATIENT_PORT))
            print(f"并行度: 自动计算 = min(Doctor端口数={len(DOCTOR_PORT)}, Patient端口数={len(PATIENT_PORT)}) = {MAX_PARALLEL_PROCESSES}")
    else:
        # 混合模式（一部分 OpenRouter，一部分离线）
        offline_ports = []
        if not doctor_use_openrouter:
            offline_ports.extend(DOCTOR_PORT)
        if not patient_use_openrouter:
            offline_ports.extend(PATIENT_PORT)
        
        # 混合模式的并行度由离线模型的端口数决定
        if offline_ports:
            MAX_PARALLEL_PROCESSES = min(len(DOCTOR_PORT) if not doctor_use_openrouter else float('inf'),
                                        len(PATIENT_PORT) if not patient_use_openrouter else float('inf'))
            if MAX_PARALLEL_PROCESSES == float('inf'):
                MAX_PARALLEL_PROCESSES = max(len(DOCTOR_PORT) if not doctor_use_openrouter else 0,
                                            len(PATIENT_PORT) if not patient_use_openrouter else 0)
        else:
            MAX_PARALLEL_PROCESSES = int(os.getenv('OPENROUTER_MAX_PARALLEL', '16'))
        
        print(f"并行度: 混合模式，并行数 = {MAX_PARALLEL_PROCESSES}")
    
    print(f"最大并行进程数: {MAX_PARALLEL_PROCESSES}")
    print("="*60)
    
    # 患者先提出主诉功能（只在base版本启用，因为诊断树都是直接开始提问，不需要患者先提出主诉）
    enable_chief_complaint = (DOCTOR_VERSION == 'base')
    
    # ================== 自动生成输出文件名 ==================
    # 提取模型basename函数
    def extract_model_basename(model_name):
        """从模型路径或名称中提取简短的basename"""
        # 处理路径: /path/to/Qwen3-8B -> Qwen3-8B
        # 处理远程路径: /path/to/Qwen3-8B@host:port -> Qwen3-8B
        # 处理API模型名: qwen/qwen3-8b:9041 -> qwen3-8b
        # 处理API模型名: qwen/qwen3-8b -> qwen3-8b
        basename = model_name.split('/')[-1]  # 取最后一部分
        basename = basename.split('@')[0]     # 去掉主机信息（如果有）
        basename = basename.split(':')[0]     # 去掉端口号
        basename = basename.replace('_', '-') # 统一使用连字符
        return basename.lower()
    
    # 提取模型basename
    doctor_model_basename = extract_model_basename(DOCTOR_MODEL)
    patient_model_basename = extract_model_basename(PATIENT_MODEL)
    
    # 生成输出文件名
    # 格式: conversations_doctor_{版本}_{模型}_patient_{版本}_{模型}.json
    output_filename = f'conversations_doctor_{DOCTOR_VERSION}_{doctor_model_basename}_patient_{PATIENT_VERSION}_{patient_model_basename}.json'
    output_filename_with_classification = f'conversations_doctor_{DOCTOR_VERSION}_{doctor_model_basename}_patient_{PATIENT_VERSION}_{patient_model_basename}_with_classification.json'
    
    print("\n" + "="*60)
    print("=== 系统配置总览 ===")
    print("="*60)
    print(f"Doctor版本: {DOCTOR_VERSION.upper()}")
    print(f"Doctor类: {Doctor.__name__}")
    if doctor_use_openrouter:
        print(f"Doctor模型: {DOCTOR_MODEL} (OpenRouter)")
    else:
        print(f"Doctor模型: {DOCTOR_MODEL} (端口: {DOCTOR_PORT})")
    
    print(f"Patient版本: {PATIENT_VERSION.upper()}")
    print(f"Patient类: {Patient.__name__}")
    if patient_use_openrouter:
        print(f"Patient模型: {PATIENT_MODEL} (OpenRouter)")
    else:
        print(f"Patient模型: {PATIENT_MODEL} (端口: {PATIENT_PORT})")
    
    print(f"患者主诉生成功能: {'启用' if enable_chief_complaint else '禁用 (有诊断树引导)'}")
    
    print(f"最大并行进程数: {MAX_PARALLEL_PROCESSES}")
    print(f"每个患者对话轮数: {NUM_ROUNDS}")
    print(f"\n输出配置:")
    print(f"  - 输出目录: {OUTPUT_DATASYN_PATH}")
    print(f"  - 对话文件名: {output_filename}")
    if OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION:
        print(f"  - 带分类目录: {OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION}")
        print(f"  - 带分类文件名: {output_filename_with_classification}")
    else:
        print(f"  - 带分类版本: 禁用")
    print("="*60 + "\n")
    
    # 记录开始时间
    start_time = time.perf_counter()
    
    # 加载患者信息
    try:
        with open(PATIENT_INFO_PATH, 'r', encoding='utf-8') as f:
            patient_info = json.load(f) # 加载所有患者
        print(f"成功加载 {len(patient_info)} 个患者信息")
    except FileNotFoundError:
        print(f"错误: 找不到患者信息文件 {PATIENT_INFO_PATH}")
        return
    except Exception as e:
        print(f"错误: 加载患者信息时发生异常 {str(e)}")
        return
    
    # 创建GPU管理器（传递类和版本信息）
    gpu_manager = GPUManager(
        max_parallel_processes=MAX_PARALLEL_PROCESSES,
        doctor_prompt_path=DOCTOR_PROMPT_PATH,
        diagtree_path=DIAGTREE_PATH,
        num_rounds=NUM_ROUNDS,
        doctor_model_name=DOCTOR_MODEL,
        patient_model_name=PATIENT_MODEL,
        doctor_port=DOCTOR_PORT,
        patient_port=PATIENT_PORT,
        doctor_class=Doctor,  # 传递Doctor类
        patient_class=Patient,  # 传递Patient类
        doctor_version=DOCTOR_VERSION,  # 传递Doctor版本信息
        enable_chief_complaint=enable_chief_complaint,  # 传递患者主诉开关
        output_filename=output_filename,  # 动态生成的文件名
        output_filename_with_classification=output_filename_with_classification  # 动态生成的带分类文件名
    )
    
    # 执行并行处理
    try:
        total_cost, successful_patients, error_patients = gpu_manager.run_parallel_processing(
            patient_info_list=patient_info,
            output_dir=OUTPUT_DATASYN_PATH,
            output_dir_with_classification=OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION
        )
        
        # 计算总耗时
        elapsed_seconds = time.perf_counter() - start_time
        
        print("\n" + "="*50)
        print("=== 双GPU并行处理完成 ===")
        print(f"总患者数: {len(patient_info)}")
        print(f"成功处理: {successful_patients}")
        print(f"处理失败: {error_patients}")
        print(f"总费用: {total_cost}")
        print(f"总耗时: {elapsed_seconds:.2f} 秒")
        print(f"平均每个患者耗时: {elapsed_seconds/len(patient_info):.2f} 秒")
        print(f"\n输出文件:")
        print(f"  - 对话记录: {os.path.join(OUTPUT_DATASYN_PATH, output_filename)}")
        if OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION:
            print(f"  - 带分类记录: {os.path.join(OUTPUT_DATASYN_PATH_WITH_CLASSIFICATION, output_filename_with_classification)}")
        else:
            print(f"  - 带分类版本: 已跳过")
        print("="*50)
        
        # ================== 评估诊断正确率 ==================
        print("\n" + "="*50)
        ## 选择是否要评估诊断正确率
        evaluate_diagnosis = str_to_bool(os.getenv('EVALUATE_DIAGNOSIS', 'true'))
        if evaluate_diagnosis:
            print("=== 开始评估诊断正确率 ===")
            print("="*50)
        else:
            print("=== 跳过评估诊断正确率 ===")
            print("="*50)
            return
        
        print("=== 开始评估诊断正确率 ===")
        print("="*50)
        
        # 设置评估输出目录
        evaluation_output_dir = os.path.join(OUTPUT_DATASYN_PATH, 'evaluation_results')
        os.makedirs(evaluation_output_dir, exist_ok=True)
        
        # 生成评估结果文件名
        evaluation_filename = f'evaluation_doctor_{DOCTOR_VERSION}_{doctor_model_basename}_patient_{PATIENT_VERSION}_{patient_model_basename}.json'
        evaluation_output_path = os.path.join(evaluation_output_dir, evaluation_filename)
        
        try:
            # 执行评估
            dialogue_file_path = os.path.join(OUTPUT_DATASYN_PATH, output_filename)
            print(f"对话文件路径: {dialogue_file_path}")
            print(f"标签文件路径: {PATIENT_INFO_PATH}")
            
            eval_results = api_llm_diagnosis_evaluation_multilabel(
                dialogue_path=dialogue_file_path,
                labels_file_path=PATIENT_INFO_PATH
            )
            
            # 打印评估结果
            print_evaluation_results(eval_results)
            
            # 保存评估结果到JSON文件
            # 使用格式化的JSON输出（压缩matrix部分）
            json_str = json.dumps(eval_results, ensure_ascii=False, indent=2)
            
            # 使用正则表达式压缩matrix部分
            import re
            def replace_matrix(match):
                matrix_data = eval_results['confusion_matrix']['matrix']
                replacement = '    "matrix": [\n'
                for i, row in enumerate(matrix_data):
                    row_str = '      [' + ', '.join(map(str, row)) + ']'
                    if i < len(matrix_data) - 1:
                        row_str += ','
                    replacement += row_str + '\n'
                replacement += '    ]'
                return replacement
            
            pattern = r'"matrix": \[\s*(?:\[[\d\s,]*\],?\s*)*\]'
            formatted_json = re.sub(pattern, lambda m: replace_matrix(m), json_str, flags=re.MULTILINE | re.DOTALL)
            
            with open(evaluation_output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_json)
            
            print(f"\n评估结果已保存到: {evaluation_output_path}")
            print("="*50)
            
        except Exception as e:
            print(f"评估过程中发生错误: {str(e)}")
            print("跳过评估步骤，继续执行...")
        
    except Exception as e:
        print(f"并行处理过程中发生错误: {str(e)}")
        raise

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='双GPU并行处理患者对话生成')
    
    parser.add_argument('--patient_file', type=str, 
                       default='./raw_data/selected_50_samples.json',
                       help='患者信息文件路径')
    
    parser.add_argument('--output_dir', type=str,
                       default='./DataSyn_true_data_8b_828_parallel', 
                       help='正常版本输出目录')
    
    parser.add_argument('--output_dir_with_classification', type=str,
                       default='./DataSyn_true_data_8b_828_parallel_with_classification',
                       help='带分类版本输出目录')
    
    parser.add_argument('--model_path', type=str,
                       default='../../models/Qwen3-8B',
                       help='模型路径')
    
    parser.add_argument('--doctor_prompt_path', type=str,
                       default='./prompts/doctor/doctor_persona.json',
                       help='医生人格模板文件路径')
    
    parser.add_argument('--diagtree_path', type=str,
                       default='./prompts/diagtree',
                       help='诊断树目录路径')
    
    parser.add_argument('--ports', nargs='+', type=int,
                       default=[9041],
                       help='GPU端口列表（兼容性保留）')
    
    parser.add_argument('--doctor_model', type=str,
                       default='../../models/Qwen3-8B',
                       help='Doctor模型路径')
    
    parser.add_argument('--patient_model', type=str,
                       default='../../models/Qwen3-1.7B',
                       help='Patient模型路径')
    
    parser.add_argument('--doctor_port', nargs='+', type=int,
                       default=[9041],
                       help='Doctor端口列表，支持多个端口进行负载均衡')
    
    parser.add_argument('--patient_port', nargs='+', type=int,
                       default=[9040],
                       help='Patient端口列表，支持多个端口进行负载均衡')
    
    parser.add_argument('--num_rounds', type=int, default=1,
                       help='每个患者的对话轮数')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 可选：使用命令行参数
    # args = parse_arguments()
    # 然后用args中的参数替换硬编码的配置
    
    main()
