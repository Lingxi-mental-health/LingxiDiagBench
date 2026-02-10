#!/usr/bin/env python3
"""
批量 Patient Model 评估脚本

自动化流程：
1. 从模型列表中逐个部署 patient model 到指定 GPU（本地模型）
   或直接使用 OpenRouter API（远程模型）
2. 等待 vllm 服务就绪（仅本地模型）
3. 运行评估命令
4. 停止 vllm 服务（仅本地模型）
5. 继续下一个模型
6. 汇总所有结果

支持的模型格式：
- 本地模型: "Qwen3-1.7B", "Qwen3-4B" 等（自动部署 vLLM）
- OpenRouter模型: "google/gemini-3-flash-preview", "openai/gpt-5-mini" 等
  需要设置环境变量 OPENROUTER_API_KEY

支持的 Patient 版本:
    - v1: 基础版本 (src/patient/patient_v1.py)
    - cot: Chain-of-Thought 版本 (src/patient/patient_cot.py)
    - mdd5k: MDD-5K 版本 (src/patient/patient_mdd5k.py)
    - v3: 优化版本 (src/patient/patient_v3.py)

使用方法:
    # 评估本地模型（自动部署vLLM）
    python batch_patient_eval.py \\
        --patient-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B" \\
        --patient-version v3 \\
        --eval-models "gemma-3-27b-it@10.119.16.100:9051,qwen3-30b@10.119.16.100:9041" \\
        --gpu-devices "5,6" \\
        --port 9060 \\
        --limit 100

    # 评估OpenRouter模型（无需部署，直接调用API）
    python batch_patient_eval.py \\
        --patient-models "google/gemini-3-flash-preview,openai/gpt-5-mini" \\
        --patient-version mdd5k \\
        --eval-models "gemma-3-27b-it@10.119.16.100:9051" \\
        --limit 100

    # 混合评估（本地 + OpenRouter）
    python batch_patient_eval.py \\
        --patient-models "Qwen3-8B,google/gemini-3-flash-preview,openai/gpt-5-mini" \\
        --patient-version v3 \\
        --eval-models "gemma-3-27b-it@10.119.16.100:9051" \\
        --gpu-devices "5,6" \\
        --port 9060 \\
        --limit 100

    # 也可以从文件读取模型列表
    python batch_patient_eval.py \\
        --patient-models-file models_to_test.txt \\
        --patient-version v3 \\
        --eval-models "gemma-3-27b-it@10.119.16.100:9051"
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
import json

# 默认配置
DEFAULT_MODEL_BASE_PATH = "../../models"
DEFAULT_LOG_DIR = "../../logs"
DEFAULT_GPU_DEVICES = "0,1,2,3,4,5,6,7"
DEFAULT_PORT = 9060
DEFAULT_HOST = "0.0.0.0"
DEFAULT_MAX_MODEL_LEN = 20480
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


class VLLMServer:
    """vLLM 服务管理类"""
    
    def __init__(
        self,
        model_path: str,
        model_name: str,
        port: int,
        gpu_devices: str,
        host: str = DEFAULT_HOST,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        log_dir: str = DEFAULT_LOG_DIR,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.port = port
        self.gpu_devices = gpu_devices
        self.host = host
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.log_dir = log_dir
        self.process: Optional[subprocess.Popen] = None
        
        # 计算 tensor parallel size（基于 GPU 数量）
        self.tensor_parallel_size = len(gpu_devices.split(","))
    
    def start(self, timeout: int = 1200) -> bool:
        """启动 vLLM 服务器并等待就绪"""
        if self.process is not None:
            print(f"[警告] 服务器已经在运行中")
            return True
        
        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_file = os.path.join(
            self.log_dir, 
            f"vllm_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # 读取config.json文件中的num_attention_heads
        num_attention_heads = None
        config_file = os.path.join(self.model_path, "config.json")
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            num_attention_heads = config.get("num_attention_heads")
        
        if num_attention_heads is not None and num_attention_heads % self.tensor_parallel_size != 0:
            print(f"[警告] num_attention_heads {num_attention_heads} 不能被 tensor_parallel_size {self.tensor_parallel_size} 整除")
            self.tensor_parallel_size = 2  # qwen2.5-0.5b-instruct can only be run with tensor_parallel_size 2
            print(f"重置 tensor_parallel_size 为 {self.tensor_parallel_size}")
        
        # 构建启动命令
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--served-model-name", self.model_name,
            "--port", str(self.port),
            "--host", self.host,
            "--trust-remote-code",
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--dtype", "bfloat16",
        ]
        
        if "qwen2.5" not in self.model_name.lower() and "gpt-oss" not in self.model_name.lower():
            cmd.append("--reasoning-parser")
            cmd.append("deepseek_r1")
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
        
        print(f"\n{'='*60}")
        print(f"[启动] 正在启动 vLLM 服务器...")
        print(f"  模型: {self.model_name}")
        print(f"  路径: {self.model_path}")
        print(f"  端口: {self.port}")
        print(f"  GPU: {self.gpu_devices}")
        print(f"  日志: {log_file}")
        print(f"  命令: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        # 启动进程
        with open(log_file, 'w') as f:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,  # 创建新的进程组
            )
        
        # 等待服务就绪
        return self._wait_for_ready(timeout)
    
    def _wait_for_ready(self, timeout: int) -> bool:
        """等待服务就绪"""
        start_time = time.time()
        check_interval = 20  # 每20秒检查一次
        
        print(f"[等待] 等待服务就绪（最长 {timeout} 秒）...")
        
        while time.time() - start_time < timeout:
            # 检查进程是否还在运行
            if self.process.poll() is not None:
                print(f"[错误] vLLM 进程意外退出，返回码: {self.process.returncode}")
                return False
            
            # 检查服务是否可用
            if self._check_health():
                elapsed = int(time.time() - start_time)
                print(f"[成功] 服务已就绪！（耗时 {elapsed} 秒）")
                return True
            
            time.sleep(check_interval)
            elapsed = int(time.time() - start_time)
            print(f"  ... 已等待 {elapsed} 秒")
        
        print(f"[超时] 服务启动超时（{timeout} 秒）")
        self.stop()
        return False
    
    def _check_health(self) -> bool:
        """检查服务健康状态"""
        # 尝试多种方式检查服务是否就绪
        endpoints = [
            f"http://localhost:{self.port}/v1/models",  # OpenAI 兼容接口
            f"http://localhost:{self.port}/health",      # 健康检查端点
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    return True
            except (requests.exceptions.RequestException, ConnectionError):
                continue
        
        return False
    
    def stop(self):
        """停止 vLLM 服务器"""
        if self.process is None:
            return
        
        print(f"\n[停止] 正在停止 vLLM 服务器 ({self.model_name})...")
        
        try:
            # 发送 SIGTERM 到整个进程组
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # 等待进程退出
            try:
                self.process.wait(timeout=30)
                print(f"[成功] 服务器已停止")
            except subprocess.TimeoutExpired:
                print(f"[警告] 进程未响应 SIGTERM，发送 SIGKILL...")
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
                print(f"[成功] 服务器已强制停止")
        except ProcessLookupError:
            print(f"[信息] 进程已不存在")
        except Exception as e:
            print(f"[错误] 停止服务时出错: {e}")
        
        self.process = None
        
        # 额外等待确保端口释放
        time.sleep(3)


def run_evaluation(
    patient_model: str,
    patient_version: str,
    eval_models: str,
    data_file: str,
    max_workers: int,
    eval_interval: int,
    limit: Optional[int],
    output_dir: Optional[str],
    working_dir: str,
) -> Tuple[bool, str]:
    """运行评估命令"""
    
    cmd = [
        "python", "unified_patient_eval.py",
        "--data-file", data_file,
        "--patient-model", patient_model,
        "--patient-version", patient_version,
        "--eval-models", eval_models,
        "--max-workers", str(max_workers),
        "--eval-interval", str(eval_interval),
    ]
    
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    
    print(f"\n{'='*60}")
    print(f"[评估] 开始运行评估...")
    print(f"  Patient Model: {patient_model}")
    print(f"  Patient Version: {patient_version}")
    print(f"  Eval Models: {eval_models}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=working_dir,
            capture_output=False,  # 直接输出到控制台
            text=True,
        )
        
        if result.returncode == 0:
            print(f"\n[成功] 评估完成!")
            return True, ""
        else:
            print(f"\n[错误] 评估失败，返回码: {result.returncode}")
            return False, f"Exit code: {result.returncode}"
            
    except Exception as e:
        print(f"\n[错误] 运行评估时出错: {e}")
        return False, str(e)


def is_openrouter_model(model_name: str) -> bool:
    """
    判断是否是 OpenRouter 模型
    
    OpenRouter 模型格式: "provider/model-name"
    例如: "google/gemini-3-flash-preview", "openai/gpt-5-mini", "anthropic/claude-haiku-4.5"
    
    本地模型格式: "Qwen3-8B", "model_name@host:port" 等
    """
    # 包含 '/' 但不包含 '@' 且不以 '/' 开头（排除绝对路径）
    if '/' in model_name and '@' not in model_name and not model_name.startswith('/'):
        # 检查是否像 "provider/model" 格式
        parts = model_name.split('/')
        if len(parts) >= 2 and parts[0] and parts[1]:
            # 确保第一部分是合法的 provider 名称（非路径）
            provider = parts[0]
            # 如果 provider 看起来像路径组件（如 tcci_mnt），则不是 OpenRouter
            if not any(c in provider for c in ['_', '.']):
                return True
            # 常见的 OpenRouter provider
            known_providers = [
                'google', 'openai', 'anthropic', 'meta', 'mistral', 
                'qwen', 'deepseek', 'x-ai', 'moonshotai', 'cohere',
                'perplexity', 'together', 'fireworks-ai', 'alibaba'
            ]
            if provider.lower() in known_providers:
                return True
    return False


def find_model_path(model_name: str, model_base_path: str) -> Optional[str]:
    """查找模型路径（仅用于本地模型）"""
    # 如果是 OpenRouter 模型，不需要查找本地路径
    if is_openrouter_model(model_name):
        return None
    
    # 如果是完整路径，直接返回
    if os.path.isabs(model_name) and os.path.exists(model_name):
        return model_name
    
    # 在模型基础目录下查找
    model_path = os.path.join(model_base_path, model_name)
    if os.path.exists(model_path):
        return model_path
    
    # 尝试一些常见的变体
    variants = [
        model_name,
        model_name.lower(),
        model_name.replace("-", "_"),
        model_name.replace("_", "-"),
    ]
    
    for variant in variants:
        path = os.path.join(model_base_path, variant)
        if os.path.exists(path):
            return path
    
    return None


def parse_models_list(models_str: Optional[str], models_file: Optional[str]) -> List[str]:
    """解析模型列表"""
    models = []
    
    if models_str:
        models.extend([m.strip() for m in models_str.split(",") if m.strip()])
    
    if models_file and os.path.exists(models_file):
        with open(models_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    models.append(line)
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description="批量 Patient Model 评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 模型相关参数
    parser.add_argument(
        "--patient-models",
        type=str,
        help="要测试的 patient model 列表，逗号分隔（例如: Qwen3-1.7B,Qwen3-4B,Qwen3-8B）"
    )
    parser.add_argument(
        "--patient-models-file",
        type=str,
        help="包含 patient model 列表的文件，每行一个模型名"
    )
    parser.add_argument(
        "--model-base-path",
        type=str,
        default=DEFAULT_MODEL_BASE_PATH,
        help=f"模型基础路径（默认: {DEFAULT_MODEL_BASE_PATH}）"
    )
    parser.add_argument(
        "--patient-version",
        type=str,
        default="v3",
        help="Patient agent 版本（默认: v3，可选: v1, cot, mdd5k, v3）"
    )
    parser.add_argument(
        "--eval-models",
        type=str,
        required=True,
        help="评估模型列表，逗号分隔（例如: gemma-3-27b-it@10.119.16.100:9051）"
    )
    
    # vLLM 服务相关参数
    parser.add_argument(
        "--gpu-devices",
        type=str,
        default=DEFAULT_GPU_DEVICES,
        help=f"用于部署 patient model 的 GPU 设备（默认: {DEFAULT_GPU_DEVICES}）"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"vLLM 服务端口（默认: {DEFAULT_PORT}）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="vLLM 服务访问地址（默认: localhost）"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help=f"模型最大长度（默认: {DEFAULT_MAX_MODEL_LEN}）"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help=f"GPU 显存利用率（默认: {DEFAULT_GPU_MEMORY_UTILIZATION}）"
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=1200,
        help="vLLM 服务启动超时时间（秒）（默认: 1200）"
    )
    
    # 评估相关参数
    parser.add_argument(
        "--data-file",
        type=str,
        default="./raw_data/LingxiDiag-16K_validation_data.json",
        help="评估数据文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results/static_patient_eval",
        help="评估结果输出目录（默认: ./evaluation_results/static_patient_eval）"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="最大并行工作线程数（默认: 16）"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="采样间隔（默认: 5）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="评估样本数量限制"
    )
    
    # 日志相关
    parser.add_argument(
        "--log-dir",
        type=str,
        default=DEFAULT_LOG_DIR,
        help=f"日志目录（默认: {DEFAULT_LOG_DIR}）"
    )
    
    # 其他选项
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不实际运行"
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="如果某个模型评估失败，继续评估下一个模型"
    )
    
    args = parser.parse_args()
    
    # 解析模型列表
    patient_models = parse_models_list(args.patient_models, args.patient_models_file)
    
    if not patient_models:
        print("[错误] 请提供要测试的 patient model 列表（--patient-models 或 --patient-models-file）")
        sys.exit(1)
    
    # 获取工作目录
    script_dir = Path(__file__).resolve().parent
    working_dir = str(script_dir)
    
    # 区分本地模型和 OpenRouter 模型
    local_models = []
    openrouter_models = []
    for model_name in patient_models:
        if is_openrouter_model(model_name):
            openrouter_models.append(model_name)
        else:
            local_models.append(model_name)
    
    print(f"\n{'#'*60}")
    print(f"# 批量 Patient Model 评估")
    print(f"{'#'*60}")
    print(f"\n待测试模型列表 ({len(patient_models)} 个):")
    for i, model in enumerate(patient_models, 1):
        model_type = "[OpenRouter]" if is_openrouter_model(model) else "[本地]"
        print(f"  {i}. {model} {model_type}")
    
    if openrouter_models:
        print(f"\nOpenRouter 模型 ({len(openrouter_models)} 个): 将直接调用 API")
        # 检查 OpenRouter API Key
        if not os.getenv('OPENROUTER_API_KEY'):
            print(f"[警告] 未设置 OPENROUTER_API_KEY 环境变量，OpenRouter 模型可能无法正常使用")
    
    if local_models:
        print(f"\n本地模型 ({len(local_models)} 个): 将自动部署 vLLM")
    
    print(f"\nPatient Version: {args.patient_version}")
    print(f"Eval Models: {args.eval_models}")
    if local_models:
        print(f"GPU Devices: {args.gpu_devices}")
        print(f"Port: {args.port}")
    print()
    
    # 验证本地模型路径（OpenRouter 模型跳过）
    model_paths = {}
    for model_name in patient_models:
        if is_openrouter_model(model_name):
            # OpenRouter 模型不需要本地路径
            model_paths[model_name] = None
            print(f"[确认] {model_name} -> OpenRouter API")
        else:
            model_path = find_model_path(model_name, args.model_base_path)
            if model_path is None:
                print(f"[错误] 找不到本地模型: {model_name}")
                print(f"  尝试的路径: {os.path.join(args.model_base_path, model_name)}")
                sys.exit(1)
            model_paths[model_name] = model_path
            print(f"[确认] {model_name} -> {model_path}")
    
    if args.dry_run:
        print("\n[Dry Run] 以下是将要执行的操作：")
        for model_name in patient_models:
            if is_openrouter_model(model_name):
                print(f"\n  [OpenRouter] {model_name}")
                print(f"  1. 直接调用 OpenRouter API")
                print(f"  2. 运行评估")
                print(f"     - Patient: {model_name}")
                print(f"     - Version: {args.patient_version}")
            else:
                print(f"\n  [本地] {model_name}")
                print(f"  1. 部署 {model_name} 到 GPU {args.gpu_devices}")
                print(f"     - 路径: {model_paths[model_name]}")
                print(f"     - 端口: {args.port}")
                print(f"  2. 运行评估")
                print(f"     - Patient: {model_name}@{args.host}:{args.port}")
                print(f"     - Version: {args.patient_version}")
                print(f"  3. 停止 vLLM 服务")
        print("\n[Dry Run] 结束")
        sys.exit(0)
    
    # 记录结果
    results = []
    
    # 逐个处理模型
    for idx, model_name in enumerate(patient_models, 1):
        print(f"\n\n{'#'*60}")
        print(f"# [{idx}/{len(patient_models)}] 正在处理: {model_name}")
        print(f"{'#'*60}")
        
        # 检查是否是 OpenRouter 模型
        if is_openrouter_model(model_name):
            # ==========================================
            # OpenRouter 模型：直接调用 API，无需部署
            # ==========================================
            print(f"\n[OpenRouter] 使用 OpenRouter API 调用模型: {model_name}")
            
            # 对于 OpenRouter 模型，直接使用模型名称
            patient_model_addr = model_name
            
            # 运行评估
            success, error_msg = run_evaluation(
                patient_model=patient_model_addr,
                patient_version=args.patient_version,
                eval_models=args.eval_models,
                data_file=args.data_file,
                max_workers=args.max_workers,
                eval_interval=args.eval_interval,
                limit=args.limit,
                output_dir=args.output_dir,
                working_dir=working_dir,
            )
            
            results.append((model_name, success, error_msg))
            
            if not success and not args.skip_failed:
                print(f"[退出] 评估失败，终止批处理")
                break
        else:
            # ==========================================
            # 本地模型：需要部署 vLLM 服务
            # ==========================================
            model_path = model_paths[model_name]
            
            # 创建 vLLM 服务器
            server = VLLMServer(
                model_path=model_path,
                model_name=model_name,
                port=args.port,
                gpu_devices=args.gpu_devices,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                log_dir=args.log_dir,
            )
            
            try:
                # 启动服务
                if not server.start(timeout=args.startup_timeout):
                    results.append((model_name, False, "服务启动失败"))
                    if args.skip_failed:
                        print(f"[跳过] 继续下一个模型...")
                        continue
                    else:
                        print(f"[退出] 服务启动失败，终止批处理")
                        break
                
                # 构建 patient model 地址
                patient_model_addr = f"{model_name}@{args.host}:{args.port}"
                
                # 运行评估
                success, error_msg = run_evaluation(
                    patient_model=patient_model_addr,
                    patient_version=args.patient_version,
                    eval_models=args.eval_models,
                    data_file=args.data_file,
                    max_workers=args.max_workers,
                    eval_interval=args.eval_interval,
                    limit=args.limit,
                    output_dir=args.output_dir,
                    working_dir=working_dir,
                )
                
                results.append((model_name, success, error_msg))
                
                if not success and not args.skip_failed:
                    print(f"[退出] 评估失败，终止批处理")
                    break
                    
            finally:
                # 确保服务停止
                server.stop()
    
    # 打印汇总
    print(f"\n\n{'#'*60}")
    print(f"# 批量评估完成")
    print(f"{'#'*60}")
    print(f"\n结果汇总:")
    
    success_count = 0
    for model_name, success, error_msg in results:
        status = "✓ 成功" if success else f"✗ 失败 ({error_msg})"
        print(f"  - {model_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n总计: {success_count}/{len(results)} 成功")
    
    # 返回退出码
    if success_count == len(patient_models):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

