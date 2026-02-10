#!/bin/bash
# =============================================================================
# EverDiag-16K 数据集 Benchmark 运行脚本
# 
# 功能：
#   1. TF-IDF 方法辅助诊断 benchmark
#   2. LLM-only 方法辅助诊断 benchmark  
#   3. LLM-only 方法 next_utterance_prediction benchmark
#
# 支持自动部署 vLLM 服务（参考 batch_doctor_eval.py）
# 支持多个模型批量评测
# 支持 OpenRouter API 模型
#
# 使用方法：
#   chmod +x run_benchmark_static.sh
#   ./run_benchmark_static.sh
#
# 自动部署模式（单个模型）：
#   ./run_benchmark_static.sh --deploy-model Qwen3-8B --gpu-devices 0,1
#
# 批量评测模式（多个模型）：
#   ./run_benchmark_static.sh --deploy-models "Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Qwen2.5-0.5B,Qwen2.5-7B-Instruct,Qwen2.5-32B-Instruct," --gpu-devices 0,1,2,3,4,5,6,7
#
# OpenRouter 模型：
#   ./run_benchmark_static.sh --deploy-models "google/gemini-3-flash-preview,moonshotai/kimi-k2-thinking,openai/gpt-5-mini,x-ai/grok-4.1-fast,anthropic/claude-haiku-4.5" --next-utterance-limit 1000 --next-utterance-eval-interval 5
#
# 混合模式（本地 + OpenRouter）：
#   ./run_benchmark_static.sh --deploy-models "Qwen3-8B,google/gemini-2.5-flash" --gpu-devices 0,1
#
# 可选参数：
#   --llm-model MODEL_NAME        指定 LLM 模型 (默认: qwen3-32b:9041)
#   --deploy-model MODEL_NAME     自动部署的单个模型名称（如 Qwen3-8B）
#   --deploy-models MODELS        多个模型，逗号分隔（如 Qwen3-1.7B,Qwen3-4B,Qwen3-8B）
#   --deploy-models-file FILE     从文件读取模型列表，每行一个模型
#   --model-base-path PATH        模型基础路径 (默认: /tcci_mnt/shihao/models)
#   --gpu-devices DEVICES         GPU 设备列表 (默认: 0,1,2,3,4,5,6,7)
#   --port PORT                   vLLM 服务端口 (默认: 9060)
#   --max-model-len LEN           模型最大长度 (默认: 20480)
#   --startup-timeout SECONDS     启动超时时间 (默认: 600)
#   --skip-tfidf                  跳过 TF-IDF benchmark
#   --skip-llm-diagnosis          跳过 LLM 辅助诊断 benchmark
#   --skip-next-utterance         跳过 next_utterance_prediction benchmark
#   --skip-failed                 如果某个模型失败，继续评估下一个
#   --train-file FILE             训练数据文件路径
#   --test-file FILE              测试数据文件路径
#   --output-dir DIR              输出目录路径
#   --help                        显示帮助信息
#   --next-utterance-limit N      医生提问预测的最大样本数限制，固定seed随机采样 (默认: 5000)
#   --next-utterance-eval-interval I      医生提问预测的采样间隔，每隔多少轮采样一次 (默认: 5)
# =============================================================================

set -e

# 默认配置
LLM_MODEL="${LLM_MODEL:-qwen3-32b:9041}"
RUN_TFIDF=true
RUN_LLM_DIAGNOSIS=true
RUN_NEXT_UTTERANCE=true

# 数据文件配置
TRAIN_FILE=""
TEST_FILE=""
OUTPUT_DIR=""

# vLLM 部署配置
DEPLOY_MODEL=""
DEPLOY_MODELS=""
DEPLOY_MODELS_FILE=""
MODEL_BASE_PATH="${MODEL_BASE_PATH:-/tcci_mnt/shihao/models}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3,4,5,6,7}"
VLLM_PORT="${VLLM_PORT:-9060}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-20480}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
LOG_DIR="${LOG_DIR:-/tcci_mnt/shihao/logs}"
SKIP_FAILED=false
NEXT_UTTERANCE_LIMIT="${NEXT_UTTERANCE_LIMIT:-5000}"
NEXT_UTTERANCE_EVAL_INTERVAL="${NEXT_UTTERANCE_EVAL_INTERVAL:-5}"

# vLLM 进程 PID
VLLM_PID=""

# 评测结果记录
declare -a EVAL_RESULTS=()

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_deploy() {
    echo -e "${CYAN}[DEPLOY]${NC} $1"
}

print_batch() {
    echo -e "${MAGENTA}[BATCH]${NC} $1"
}

# 清理函数：停止 vLLM 服务
cleanup_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        print_deploy "正在停止 vLLM 服务 (PID: $VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null || true
        
        # 等待进程退出
        local wait_count=0
        while kill -0 "$VLLM_PID" 2>/dev/null && [ $wait_count -lt 30 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        
        # 如果还没退出，强制杀死
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            print_warning "进程未响应 SIGTERM，发送 SIGKILL..."
            kill -9 "$VLLM_PID" 2>/dev/null || true
        fi
        
        print_success "vLLM 服务已停止"
        VLLM_PID=""
        
        # 额外等待确保端口释放
        sleep 3
    fi
}

# 注册清理函数
trap cleanup_vllm EXIT INT TERM

# 显示帮助信息
show_help() {
    echo "EverDiag-16K 数据集 Benchmark 运行脚本"
    echo ""
    echo "使用方法："
    echo "  ./run_benchmark_static.sh [选项]"
    echo ""
    echo "LLM 模型选项："
    echo "  --llm-model MODEL_NAME        指定 LLM 模型 (默认: qwen3-32b:9041)"
    echo "                                支持格式:"
    echo "                                  - vLLM简化格式: model:port (如 qwen3-32b:9041)"
    echo "                                  - vLLM完整格式: model@host:port (如 qwen3-30b@10.119.28.185:9041)"
    echo ""
    echo "自动部署选项（可选）："
    echo "  --deploy-model MODEL_NAME     要部署的单个模型名称（如 Qwen3-8B, Qwen3-32B）"
    echo "  --deploy-models MODELS        多个模型，逗号分隔（如 Qwen3-1.7B,Qwen3-4B,Qwen3-8B）"
    echo "                                支持 OpenRouter 模型: google/gemini-2.5-flash,openai/gpt-4o-mini"
    echo "  --deploy-models-file FILE     从文件读取模型列表，每行一个模型"
    echo "  --model-base-path PATH        模型基础路径 (默认: /tcci_mnt/shihao/models)"
    echo "  --gpu-devices DEVICES         GPU 设备列表，逗号分隔 (默认: 0,1,2,3,4,5,6,7)"
    echo "  --port PORT                   vLLM 服务端口 (默认: 9060)"
    echo "  --max-model-len LEN           模型最大长度 (默认: 20480)"
    echo "  --gpu-memory-utilization VAL  GPU 显存利用率 (默认: 0.9)"
    echo "  --startup-timeout SECONDS     启动超时时间 (默认: 600)"
    echo "  --log-dir DIR                 日志目录 (默认: /tcci_mnt/shihao/logs)"
    echo ""
    echo "Benchmark 选项："
    echo "  --skip-tfidf                  跳过 TF-IDF 辅助诊断 benchmark"
    echo "  --skip-llm-diagnosis          跳过 LLM 辅助诊断 benchmark"
    echo "  --skip-next-utterance         跳过 next_utterance_prediction benchmark"
    echo ""
    echo "Next Utterance Prediction 选项："
    echo "  --next-utterance-limit N            最大样本数限制 (默认: 5000)"
    echo "  --next-utterance-eval-interval I    采样间隔，每隔多少轮采样一次 (默认: 5)"
    echo ""
    echo "数据文件选项："
    echo "  --train-file FILE             训练数据文件路径（默认使用配置文件中的路径）"
    echo "  --test-file FILE              测试数据文件路径（默认使用配置文件中的路径）"
    echo "  --output-dir DIR              输出目录路径（默认: evaluation_results/static_doctor_eval_linxi）"
    echo ""
    echo "其他选项："
    echo "  --skip-failed                 如果某个模型评估失败，继续评估下一个"
    echo "  --help                        显示此帮助信息"
    echo ""
    echo "示例："
    echo "  # 使用已部署的模型运行所有 benchmark"
    echo "  ./run_benchmark_static.sh --llm-model qwen3-32b:9041"
    echo ""
    echo "  # 自动部署单个模型并运行 LLM benchmark"
    echo "  ./run_benchmark_static.sh --deploy-model Qwen3-8B --gpu-devices 0,1 --port 9060 --skip-tfidf"
    echo ""
    echo "  # 批量评测多个本地模型"
    echo "  ./run_benchmark_static.sh --deploy-models 'Qwen3-1.7B,Qwen3-4B,Qwen3-8B' --gpu-devices 0,1 --skip-tfidf"
    echo ""
    echo "  # 批量评测 OpenRouter 模型"
    echo "  ./run_benchmark_static.sh --deploy-models 'google/gemini-2.5-flash,openai/gpt-4o-mini' --skip-tfidf"
    echo ""
    echo "  # 混合评测（本地 + OpenRouter）"
    echo "  ./run_benchmark_static.sh --deploy-models 'Qwen3-8B,google/gemini-2.5-flash' --gpu-devices 0,1 --skip-tfidf"
    echo ""
    echo "  # 从文件读取模型列表"
    echo "  ./run_benchmark_static.sh --deploy-models-file models.txt --gpu-devices 0,1 --skip-tfidf"
    echo ""
    echo "  # 只运行 TF-IDF benchmark（不需要 LLM）"
    echo "  ./run_benchmark_static.sh --skip-llm-diagnosis --skip-next-utterance"
    echo ""
    echo "OpenRouter 模型说明："
    echo "  - 格式: provider/model-name (如 google/gemini-2.5-flash, openai/gpt-4o-mini)"
    echo "  - 需要设置环境变量: export OPENROUTER_API_KEY=your_key"
    echo "  - 常见 provider: google, openai, anthropic, meta, mistral, qwen, deepseek"
    exit 0
}

# 判断是否是 OpenRouter 模型
is_openrouter_model() {
    local model_name=$1
    
    # 包含 '/' 但不包含 '@' 且不以 '/' 开头（排除绝对路径）
    if [[ "$model_name" == *"/"* ]] && [[ "$model_name" != *"@"* ]] && [[ "$model_name" != /* ]]; then
        # 获取 provider 部分
        local provider="${model_name%%/*}"
        
        # 常见的 OpenRouter provider
        local known_providers="google openai anthropic meta mistral qwen deepseek x-ai moonshotai cohere perplexity together fireworks-ai alibaba"
        
        for p in $known_providers; do
            if [[ "$provider" == "$p" ]]; then
                return 0
            fi
        done
        
        # 如果 provider 不包含 _ 或 . 且不像路径，也认为是 OpenRouter
        if [[ "$provider" != *"_"* ]] && [[ "$provider" != *"."* ]]; then
            return 0
        fi
    fi
    
    return 1
}

# 检查服务健康状态
check_health() {
    local port=$1
    local host=${2:-localhost}
    
    # 尝试 /v1/models 端点
    if curl -s --connect-timeout 5 "http://${host}:${port}/v1/models" > /dev/null 2>&1; then
        return 0
    fi
    
    # 尝试 /health 端点
    if curl -s --connect-timeout 5 "http://${host}:${port}/health" > /dev/null 2>&1; then
        return 0
    fi
    
    return 1
}

# 等待服务就绪
wait_for_ready() {
    local port=$1
    local timeout=$2
    local check_interval=20
    local elapsed=0
    
    print_deploy "等待服务就绪（最长 ${timeout} 秒）..."
    
    while [ $elapsed -lt $timeout ]; do
        # 检查进程是否还在运行
        if [ -n "$VLLM_PID" ] && ! kill -0 "$VLLM_PID" 2>/dev/null; then
            print_error "vLLM 进程意外退出"
            return 1
        fi
        
        # 检查服务是否可用
        if check_health "$port"; then
            print_success "服务已就绪！（耗时 ${elapsed} 秒）"
            return 0
        fi
        
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        echo "  ... 已等待 ${elapsed} 秒"
    done
    
    print_error "服务启动超时（${timeout} 秒）"
    return 1
}

# 部署 vLLM 服务
deploy_vllm() {
    local model_name=$1
    local model_path=$2
    local port=$3
    local gpu_devices=$4
    
    # 计算 tensor parallel size
    local tp_size=$(echo "$gpu_devices" | tr ',' '\n' | wc -l)
    
    # 确保日志目录存在
    mkdir -p "$LOG_DIR"
    
    local log_file="${LOG_DIR}/vllm_${model_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # 检查模型配置，获取 num_attention_heads
    local config_file="${model_path}/config.json"
    if [ -f "$config_file" ]; then
        local num_attention_heads=$(python3 -c "import json; print(json.load(open('$config_file'))['num_attention_heads'])" 2>/dev/null || echo "")
        if [ -n "$num_attention_heads" ]; then
            local remainder=$((num_attention_heads % tp_size))
            if [ $remainder -ne 0 ]; then
                print_warning "num_attention_heads ($num_attention_heads) 不能被 tensor_parallel_size ($tp_size) 整除"
                tp_size=2
                print_warning "重置 tensor_parallel_size 为 $tp_size"
            fi
        fi
    fi
    
    # 构建启动命令
    local cmd="python -m vllm.entrypoints.openai.api_server"
    cmd="$cmd --model $model_path"
    cmd="$cmd --served-model-name $model_name"
    cmd="$cmd --port $port"
    cmd="$cmd --host $VLLM_HOST"
    cmd="$cmd --trust-remote-code"
    cmd="$cmd --tensor-parallel-size $tp_size"
    cmd="$cmd --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
    cmd="$cmd --max-model-len $MAX_MODEL_LEN"
    cmd="$cmd --dtype bfloat16"
    
    # 对于非 qwen2.5 和 gpt-oss 模型，添加 reasoning parser
    local model_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    if [[ "$model_lower" != *"qwen2.5"* ]] && [[ "$model_lower" != *"gpt-oss"* ]]; then
        cmd="$cmd --reasoning-parser deepseek_r1"
    fi
    
    echo ""
    echo "=============================================="
    print_deploy "正在启动 vLLM 服务器..."
    echo "  模型: $model_name"
    echo "  路径: $model_path"
    echo "  端口: $port"
    echo "  GPU: $gpu_devices"
    echo "  Tensor Parallel Size: $tp_size"
    echo "  日志: $log_file"
    echo "  命令: $cmd"
    echo "=============================================="
    echo ""
    
    # 设置环境变量并启动
    CUDA_VISIBLE_DEVICES="$gpu_devices" nohup $cmd > "$log_file" 2>&1 &
    VLLM_PID=$!
    
    print_deploy "vLLM 进程已启动 (PID: $VLLM_PID)"
    
    # 等待服务就绪
    if ! wait_for_ready "$port" "$STARTUP_TIMEOUT"; then
        print_error "vLLM 服务启动失败"
        cleanup_vllm
        return 1
    fi
    
    return 0
}

# 查找模型路径
find_model_path() {
    local model_name=$1
    local base_path=$2
    
    # 如果是绝对路径且存在
    if [[ "$model_name" == /* ]] && [ -d "$model_name" ]; then
        echo "$model_name"
        return 0
    fi
    
    # 在基础目录下查找
    local model_path="${base_path}/${model_name}"
    if [ -d "$model_path" ]; then
        echo "$model_path"
        return 0
    fi
    
    # 尝试小写
    local model_lower=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
    model_path="${base_path}/${model_lower}"
    if [ -d "$model_path" ]; then
        echo "$model_path"
        return 0
    fi
    
    # 尝试替换 - 和 _
    local model_variant=$(echo "$model_name" | tr '-' '_')
    model_path="${base_path}/${model_variant}"
    if [ -d "$model_path" ]; then
        echo "$model_path"
        return 0
    fi
    
    model_variant=$(echo "$model_name" | tr '_' '-')
    model_path="${base_path}/${model_variant}"
    if [ -d "$model_path" ]; then
        echo "$model_path"
        return 0
    fi
    
    return 1
}

# 运行单个模型的 benchmark
run_benchmark_for_model() {
    local llm_model=$1
    local model_display_name=$2
    
    print_info "=========================================="
    print_info "开始评测模型: $model_display_name"
    print_info "LLM 地址: $llm_model"
    print_info "=========================================="
    
    local benchmark_success=true
    local benchmark_start=$(date +%s)
    
    # 构建数据文件参数
    local data_file_args=""
    if [ -n "$TRAIN_FILE" ]; then
        data_file_args="$data_file_args --train-file $TRAIN_FILE"
    fi
    if [ -n "$TEST_FILE" ]; then
        data_file_args="$data_file_args --test-file $TEST_FILE"
    fi
    if [ -n "$OUTPUT_DIR" ]; then
        data_file_args="$data_file_args --output-dir $OUTPUT_DIR"
    fi
    
    # 运行 LLM 辅助诊断 Benchmark
    if $RUN_LLM_DIAGNOSIS; then
        echo ""
        print_info "运行 LLM 辅助诊断 Benchmark..."
        
        if python run_benchmark.py \
            --llm-only \
            --llm-model "$llm_model" \
            --classification-types 2class 4class 12class \
            $data_file_args; then
            print_success "LLM 辅助诊断 Benchmark 完成"
        else
            print_error "LLM 辅助诊断 Benchmark 失败"
            benchmark_success=false
        fi
    fi
    
    # 运行 Next Utterance Prediction Benchmark
    if $RUN_NEXT_UTTERANCE && $benchmark_success; then
        echo ""
        print_info "运行 Next Utterance Prediction Benchmark..."
        print_info "  采样间隔: $NEXT_UTTERANCE_EVAL_INTERVAL, 样本限制: $NEXT_UTTERANCE_LIMIT"
        
        if python run_benchmark.py \
            --next-utterance-prediction-only \
            --llm-model "$llm_model" \
            --next-utterance-eval-interval "$NEXT_UTTERANCE_EVAL_INTERVAL" \
            --next-utterance-limit "$NEXT_UTTERANCE_LIMIT" \
            $data_file_args; then
            print_success "Next Utterance Prediction Benchmark 完成"
        else
            print_error "Next Utterance Prediction Benchmark 失败"
            benchmark_success=false
        fi
    fi
    
    local benchmark_end=$(date +%s)
    local benchmark_duration=$((benchmark_end - benchmark_start))
    
    if $benchmark_success; then
        EVAL_RESULTS+=("✓ $model_display_name (${benchmark_duration}秒)")
        return 0
    else
        EVAL_RESULTS+=("✗ $model_display_name (失败)")
        return 1
    fi
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --llm-model)
            LLM_MODEL="$2"
            shift 2
            ;;
        --deploy-model)
            DEPLOY_MODEL="$2"
            shift 2
            ;;
        --deploy-models)
            DEPLOY_MODELS="$2"
            shift 2
            ;;
        --deploy-models-file)
            DEPLOY_MODELS_FILE="$2"
            shift 2
            ;;
        --model-base-path)
            MODEL_BASE_PATH="$2"
            shift 2
            ;;
        --gpu-devices)
            GPU_DEVICES="$2"
            shift 2
            ;;
        --port)
            VLLM_PORT="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --startup-timeout)
            STARTUP_TIMEOUT="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --skip-tfidf)
            RUN_TFIDF=false
            shift
            ;;
        --skip-llm-diagnosis)
            RUN_LLM_DIAGNOSIS=false
            shift
            ;;
        --skip-next-utterance)
            RUN_NEXT_UTTERANCE=false
            shift
            ;;
        --skip-failed)
            SKIP_FAILED=true
            shift
            ;;
        --next-utterance-limit)
            NEXT_UTTERANCE_LIMIT="$2"
            shift 2
            ;;
        --next-utterance-eval-interval)
            NEXT_UTTERANCE_EVAL_INTERVAL="$2"
            shift 2
            ;;
        --train-file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 设置 benchmark 工作目录
BENCHMARK_DIR="${SCRIPT_DIR}/static"
# 设置输出目录（如果未通过参数指定，使用默认值）
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="${BENCHMARK_DIR}/../evaluation_results/static_doctor_eval_lingxi"
fi

# 检查 benchmark 目录是否存在
if [ ! -d "$BENCHMARK_DIR" ]; then
    print_error "Benchmark 目录不存在: $BENCHMARK_DIR"
    exit 1
fi

# 保存当前工作目录，用于将相对路径转换为绝对路径
ORIGINAL_PWD="$(pwd)"

# 将相对路径转换为绝对路径（在切换目录前处理）
if [ -n "$TRAIN_FILE" ] && [[ "$TRAIN_FILE" != /* ]]; then
    TRAIN_FILE="$ORIGINAL_PWD/$TRAIN_FILE"
fi
if [ -n "$TEST_FILE" ] && [[ "$TEST_FILE" != /* ]]; then
    TEST_FILE="$ORIGINAL_PWD/$TEST_FILE"
fi
if [ -n "$OUTPUT_DIR" ] && [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$ORIGINAL_PWD/$OUTPUT_DIR"
fi

# 切换到 benchmark 工作目录
cd "$BENCHMARK_DIR"

# ============================================
# 解析模型列表
# ============================================
declare -a MODEL_LIST=()

# 从 --deploy-model 添加单个模型
if [ -n "$DEPLOY_MODEL" ]; then
    MODEL_LIST+=("$DEPLOY_MODEL")
fi

# 从 --deploy-models 添加多个模型
if [ -n "$DEPLOY_MODELS" ]; then
    IFS=',' read -ra MODELS <<< "$DEPLOY_MODELS"
    for model in "${MODELS[@]}"; do
        model=$(echo "$model" | xargs)  # 去除空格
        if [ -n "$model" ]; then
            MODEL_LIST+=("$model")
        fi
    done
fi

# 从 --deploy-models-file 读取模型列表
if [ -n "$DEPLOY_MODELS_FILE" ] && [ -f "$DEPLOY_MODELS_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line=$(echo "$line" | xargs)  # 去除空格
        # 跳过空行和注释
        if [ -n "$line" ] && [[ "$line" != \#* ]]; then
            MODEL_LIST+=("$line")
        fi
    done < "$DEPLOY_MODELS_FILE"
elif [ -n "$DEPLOY_MODELS_FILE" ]; then
    print_error "模型列表文件不存在: $DEPLOY_MODELS_FILE"
    exit 1
fi

# ============================================
# 判断运行模式
# ============================================
if [ ${#MODEL_LIST[@]} -gt 0 ]; then
    # 批量评测模式
    BATCH_MODE=true
    
    # 区分本地模型和 OpenRouter 模型
    declare -a LOCAL_MODELS=()
    declare -a OPENROUTER_MODELS=()
    
    for model in "${MODEL_LIST[@]}"; do
        if is_openrouter_model "$model"; then
            OPENROUTER_MODELS+=("$model")
        else
            LOCAL_MODELS+=("$model")
        fi
    done
    
    echo ""
    echo "############################################################"
    echo "#  EverDiag-16K 批量 Benchmark 评测"
    echo "############################################################"
    echo ""
    print_batch "待评测模型列表 (${#MODEL_LIST[@]} 个):"
    for i in "${!MODEL_LIST[@]}"; do
        model="${MODEL_LIST[$i]}"
        if is_openrouter_model "$model"; then
            echo "  $((i+1)). $model [OpenRouter]"
        else
            echo "  $((i+1)). $model [本地]"
        fi
    done
    
    if [ ${#OPENROUTER_MODELS[@]} -gt 0 ]; then
        echo ""
        print_batch "OpenRouter 模型 (${#OPENROUTER_MODELS[@]} 个): 将直接调用 API"
        if [ -z "$OPENROUTER_API_KEY" ]; then
            print_warning "未设置 OPENROUTER_API_KEY 环境变量，OpenRouter 模型可能无法正常使用"
        fi
    fi
    
    if [ ${#LOCAL_MODELS[@]} -gt 0 ]; then
        echo ""
        print_batch "本地模型 (${#LOCAL_MODELS[@]} 个): 将自动部署 vLLM"
        print_batch "GPU Devices: $GPU_DEVICES"
        print_batch "Port: $VLLM_PORT"
    fi
    
else
    # 单模型模式（使用 --llm-model）
    BATCH_MODE=false
    
    echo ""
    echo "=============================================="
    echo "  EverDiag-16K Benchmark 运行配置"
    echo "=============================================="
    echo ""
    print_info "项目根目录: $SCRIPT_DIR"
    print_info "Benchmark 目录: $BENCHMARK_DIR"
    print_info "LLM 模型: $LLM_MODEL"
fi

# 显示数据文件配置
if [ -n "$TRAIN_FILE" ] || [ -n "$TEST_FILE" ] || [ -n "$OUTPUT_DIR" ]; then
    echo ""
    echo "数据文件配置:"
    if [ -n "$TRAIN_FILE" ]; then
        echo "  训练数据: $TRAIN_FILE"
    else
        echo "  训练数据: (使用默认)"
    fi
    if [ -n "$TEST_FILE" ]; then
        echo "  测试数据: $TEST_FILE"
    else
        echo "  测试数据: (使用默认)"
    fi
    echo "  输出目录: $OUTPUT_DIR"
fi

echo ""
echo "运行任务:"
if $RUN_TFIDF; then
    echo "  ✓ TF-IDF 辅助诊断 benchmark"
else
    echo "  ✗ TF-IDF 辅助诊断 benchmark (已跳过)"
fi
if $RUN_LLM_DIAGNOSIS; then
    echo "  ✓ LLM 辅助诊断 benchmark"
else
    echo "  ✗ LLM 辅助诊断 benchmark (已跳过)"
fi
if $RUN_NEXT_UTTERANCE; then
    echo "  ✓ Next Utterance Prediction benchmark (间隔: $NEXT_UTTERANCE_EVAL_INTERVAL, 限制: $NEXT_UTTERANCE_LIMIT)"
else
    echo "  ✗ Next Utterance Prediction benchmark (已跳过)"
fi
echo ""
echo "=============================================="
echo ""

print_info "工作目录: $(pwd)"

# 记录开始时间
START_TIME=$(date +%s)

# ============================================
# 1. TF-IDF 辅助诊断 Benchmark（只运行一次）
# ============================================
if $RUN_TFIDF; then
    echo ""
    print_info "=========================================="
    print_info "开始运行 TF-IDF 辅助诊断 Benchmark..."
    print_info "=========================================="
    echo ""
    
    TFIDF_START=$(date +%s)
    
    # 构建数据文件参数
    TFIDF_DATA_ARGS=""
    if [ -n "$TRAIN_FILE" ]; then
        TFIDF_DATA_ARGS="$TFIDF_DATA_ARGS --train-file $TRAIN_FILE"
    fi
    if [ -n "$TEST_FILE" ]; then
        TFIDF_DATA_ARGS="$TFIDF_DATA_ARGS --test-file $TEST_FILE"
    fi
    if [ -n "$OUTPUT_DIR" ]; then
        TFIDF_DATA_ARGS="$TFIDF_DATA_ARGS --output-dir $OUTPUT_DIR"
    fi
    
    python run_benchmark.py \
        --tfidf-only \
        --classification-types 2class 4class 12class \
        --tfidf-classifiers logistic svm rf \
        $TFIDF_DATA_ARGS
    
    TFIDF_END=$(date +%s)
    TFIDF_DURATION=$((TFIDF_END - TFIDF_START))
    
    print_success "TF-IDF 辅助诊断 Benchmark 完成！耗时: ${TFIDF_DURATION}秒"
    echo ""
fi

# ============================================
# 2. LLM Benchmark（批量或单个模型）
# ============================================
if $BATCH_MODE; then
    # 批量评测模式
    total_models=${#MODEL_LIST[@]}
    current_idx=0
    success_count=0
    
    for model_name in "${MODEL_LIST[@]}"; do
        current_idx=$((current_idx + 1))
        
        echo ""
        echo "############################################################"
        print_batch "[$current_idx/$total_models] 正在处理: $model_name"
        echo "############################################################"
        
        if is_openrouter_model "$model_name"; then
            # OpenRouter 模型：直接调用 API
            print_batch "[OpenRouter] 使用 OpenRouter API 调用模型: $model_name"
            
            llm_model_addr="$model_name"
            
            if run_benchmark_for_model "$llm_model_addr" "$model_name"; then
                success_count=$((success_count + 1))
            else
                if ! $SKIP_FAILED; then
                    print_error "评估失败，终止批处理"
                    break
                fi
                print_warning "评估失败，继续下一个模型..."
            fi
        else
            # 本地模型：需要部署 vLLM
            model_path=$(find_model_path "$model_name" "$MODEL_BASE_PATH")
            
            if [ -z "$model_path" ]; then
                print_error "找不到模型: $model_name"
                print_error "尝试的路径: ${MODEL_BASE_PATH}/${model_name}"
                EVAL_RESULTS+=("✗ $model_name (找不到模型)")
                if ! $SKIP_FAILED; then
                    break
                fi
                continue
            fi
            
            print_batch "[本地] 模型路径: $model_path"
            
            # 部署 vLLM
            if deploy_vllm "$model_name" "$model_path" "$VLLM_PORT" "$GPU_DEVICES"; then
                llm_model_addr="${model_name}@localhost:${VLLM_PORT}"
                
                if run_benchmark_for_model "$llm_model_addr" "$model_name"; then
                    success_count=$((success_count + 1))
                else
                    if ! $SKIP_FAILED; then
                        cleanup_vllm
                        print_error "评估失败，终止批处理"
                        break
                    fi
                    print_warning "评估失败，继续下一个模型..."
                fi
                
                # 停止 vLLM 服务
                cleanup_vllm
            else
                EVAL_RESULTS+=("✗ $model_name (部署失败)")
                if ! $SKIP_FAILED; then
                    print_error "部署失败，终止批处理"
                    break
                fi
                print_warning "部署失败，继续下一个模型..."
            fi
        fi
    done
    
    # 打印批量评测结果汇总
    echo ""
    echo "############################################################"
    echo "#  批量评测完成"
    echo "############################################################"
    echo ""
    print_batch "结果汇总:"
    for result in "${EVAL_RESULTS[@]}"; do
        echo "  - $result"
    done
    echo ""
    print_batch "总计: $success_count/$total_models 成功"
    
else
    # 单模型模式
    if $RUN_LLM_DIAGNOSIS || $RUN_NEXT_UTTERANCE; then
        run_benchmark_for_model "$LLM_MODEL" "$LLM_MODEL"
    fi
fi

# 计算总耗时
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_MINUTES=$((TOTAL_DURATION / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

# 打印总结
echo ""
echo "=============================================="
echo "  Benchmark 运行完成"
echo "=============================================="
echo ""
print_success "总耗时: ${TOTAL_MINUTES}分${TOTAL_SECONDS}秒"
echo ""
echo "结果文件保存在: $OUTPUT_DIR/"
echo ""
echo "主要结果文件:"
echo "  - 辅助诊断汇总: $OUTPUT_DIR/benchmark_summary.xlsx"
echo "  - 问诊预测汇总: $OUTPUT_DIR/next_utterance_summary.xlsx"
echo ""
print_info "使用以下命令查看详细帮助: ./run_benchmark_static.sh --help"
echo ""
