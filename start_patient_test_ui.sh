#!/bin/bash

# Lingxi Platform 测试界面启动脚本
# 
# 功能：
# 1. 检查 VLLM 服务状态
# 2. 启动 Lingxi Platform 测试界面后端服务
# 3. 提供便捷的启动方式

set -e  # 遇到错误时退出

echo "=== Lingxi Platform 测试界面启动脚本 ==="
echo ""

# 项目根目录
PROJECT_ROOT="."

# 默认配置 - 可通过命令行参数覆盖
BACKEND_PORT=5001
PATIENT_DATA_PATH="$PROJECT_ROOT/raw_data/SMHC_EverDiag-16K_validation_data_100samples.json"
MAX_PATIENTS=100
DIAGNOSIS_CACHE_FILE="$PROJECT_ROOT/patient_test_ui/data/diagnosis_cache_EverDiag-16K_train_data.json"
DEFAULT_MODEL_NAME="EverDiagnosis-8B_icd-code-prediction_kimi-k2-0905-cot_grpo_with-real-data"

# 从 .env 加载敏感配置（如果存在）
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +o allexport
fi

# === 全局 OpenRouter API 配置 ===
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-""}
OPENROUTER_SITE_URL=${OPENROUTER_SITE_URL:-""}
OPENROUTER_SITE_NAME=${OPENROUTER_SITE_NAME:-"Patient Agent Test"}
OPENROUTER_MAX_PARALLEL=${OPENROUTER_MAX_PARALLEL:-16}

# === 模型模式配置 ===
# 支持两种方式：
# 1. 全局 MODEL_MODE（所有 Agent 使用同一种方式）
# 2. 单独的 {PREFIX}_USE_OPENROUTER（为每个 Agent 单独指定）
MODEL_MODE=${MODEL_MODE:-""}

# 单独控制每个 Agent（优先级高于 MODEL_MODE）
PATIENT_USE_OPENROUTER=${PATIENT_USE_OPENROUTER:-""}
DOCTOR_USE_OPENROUTER=${DOCTOR_USE_OPENROUTER:-""}
VERIFIER_USE_OPENROUTER=${VERIFIER_USE_OPENROUTER:-""}

# === OpenRouter 模型配置 ===
OPENROUTER_DOCTOR_MODEL=${OPENROUTER_DOCTOR_MODEL:-"qwen/qwen3-32b"}
OPENROUTER_PATIENT_MODEL=${OPENROUTER_PATIENT_MODEL:-"qwen/qwen3-32b"}
OPENROUTER_VERIFIER_MODEL=${OPENROUTER_VERIFIER_MODEL:-"qwen/qwen3-32b"}

# === 离线模型配置 ===
OFFLINE_DOCTOR_MODEL=${OFFLINE_DOCTOR_MODEL:-"$DEFAULT_MODEL_NAME"}
OFFLINE_PATIENT_MODEL=${OFFLINE_PATIENT_MODEL:-"$DEFAULT_MODEL_NAME"}
OFFLINE_VERIFIER_MODEL=${OFFLINE_VERIFIER_MODEL:-"$DEFAULT_MODEL_NAME"}

OFFLINE_DOCTOR_PORTS=${OFFLINE_DOCTOR_PORTS:-"9041"}
OFFLINE_PATIENT_PORTS=${OFFLINE_PATIENT_PORTS:-"9040"}
OFFLINE_VERIFIER_PORTS=${OFFLINE_VERIFIER_PORTS:-"9002"}

OFFLINE_MAX_PARALLEL=${OFFLINE_MAX_PARALLEL:-1}

# === VLLM 外部部署配置 ===
VLLM_DOCTOR_IP=${VLLM_DOCTOR_IP:-""}
VLLM_PATIENT_IP=${VLLM_PATIENT_IP:-""}
VLLM_VERIFIER_IP=${VLLM_VERIFIER_IP:-""}

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 解析配置参数
parse_config_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --patient-data-path)
                PATIENT_DATA_PATH="$2"
                shift 2
                ;;
            --backend-port)
                BACKEND_PORT="$2"
                shift 2
                ;;
            --max-patients)
                MAX_PATIENTS="$2"
                shift 2
                ;;
            --diagnosis-cache-file)
                DIAGNOSIS_CACHE_FILE="$2"
                shift 2
                ;;
            # === 模型模式 ===
            --model-mode)
                MODEL_MODE="$2"
                shift 2
                ;;
            # === 单独控制每个 Agent ===
            --patient-use-openrouter)
                PATIENT_USE_OPENROUTER="true"
                shift 1
                ;;
            --patient-use-offline)
                PATIENT_USE_OPENROUTER="false"
                shift 1
                ;;
            --doctor-use-openrouter)
                DOCTOR_USE_OPENROUTER="true"
                shift 1
                ;;
            --doctor-use-offline)
                DOCTOR_USE_OPENROUTER="false"
                shift 1
                ;;
            --verifier-use-openrouter)
                VERIFIER_USE_OPENROUTER="true"
                shift 1
                ;;
            --verifier-use-offline)
                VERIFIER_USE_OPENROUTER="false"
                shift 1
                ;;
            # === OpenRouter API 配置 ===
            --openrouter-api-key)
                OPENROUTER_API_KEY="$2"
                shift 2
                ;;
            --openrouter-site-url)
                OPENROUTER_SITE_URL="$2"
                shift 2
                ;;
            --openrouter-site-name)
                OPENROUTER_SITE_NAME="$2"
                shift 2
                ;;
            --openrouter-max-parallel)
                OPENROUTER_MAX_PARALLEL="$2"
                shift 2
                ;;
            # === OpenRouter 模型配置 ===
            --openrouter-doctor-model)
                OPENROUTER_DOCTOR_MODEL="$2"
                shift 2
                ;;
            --openrouter-patient-model)
                OPENROUTER_PATIENT_MODEL="$2"
                shift 2
                ;;
            --openrouter-verifier-model)
                OPENROUTER_VERIFIER_MODEL="$2"
                shift 2
                ;;
            # === 离线模型配置 ===
            --offline-doctor-model)
                OFFLINE_DOCTOR_MODEL="$2"
                shift 2
                ;;
            --offline-patient-model)
                OFFLINE_PATIENT_MODEL="$2"
                shift 2
                ;;
            --offline-verifier-model)
                OFFLINE_VERIFIER_MODEL="$2"
                shift 2
                ;;
            --offline-doctor-ports)
                OFFLINE_DOCTOR_PORTS="$2"
                shift 2
                ;;
            --offline-patient-ports)
                OFFLINE_PATIENT_PORTS="$2"
                shift 2
                ;;
            --offline-verifier-ports)
                OFFLINE_VERIFIER_PORTS="$2"
                shift 2
                ;;
            --offline-max-parallel)
                OFFLINE_MAX_PARALLEL="$2"
                shift 2
                ;;
            # === VLLM 外部部署配置 ===
            --vllm-doctor-ip)
                VLLM_DOCTOR_IP="$2"
                shift 2
                ;;
            --vllm-patient-ip)
                VLLM_PATIENT_IP="$2"
                shift 2
                ;;
            --vllm-verifier-ip)
                VLLM_VERIFIER_IP="$2"
                shift 2
                ;;
            *)
                # 其他参数传递给主参数处理函数
                return 0
                ;;
        esac
    done
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":${port} "; then
        return 0  # 端口被占用
    else
        return 1  # 端口未被占用
    fi
}

# 检查患者数据
check_patient_data() {
    print_message $BLUE "检查患者数据..."
    
    if [ ! -f "$PATIENT_DATA_PATH" ]; then
        print_message $RED "✗ 找不到患者数据文件: $PATIENT_DATA_PATH"
        return 1
    fi
    
    # 检查文件大小
    local file_size=$(stat -f%z "$PATIENT_DATA_PATH" 2>/dev/null || stat -c%s "$PATIENT_DATA_PATH" 2>/dev/null)
    if [ "$file_size" -lt 1000 ]; then
        print_message $RED "✗ 患者数据文件太小，可能损坏"
        return 1
    fi
    
    print_message $GREEN "✓ 患者数据文件检查通过: $PATIENT_DATA_PATH"
    return 0
}

# 检查OpenRouter API配置
check_openrouter_config() {
    if [ "$MODEL_MODE" = "openrouter" ]; then
        print_message $BLUE "检查OpenRouter API配置..."
        
        if [ -z "$OPENROUTER_DOCTOR_MODEL" ] || [ -z "$OPENROUTER_PATIENT_MODEL" ]; then
            print_message $RED "✗ 使用OpenRouter API需要指定所有模型"
            return 1
        fi
        
        print_message $GREEN "✓ OpenRouter API配置检查通过"
        print_message $YELLOW "  - Doctor模型: $OPENROUTER_DOCTOR_MODEL"
        print_message $YELLOW "  - Patient模型: $OPENROUTER_PATIENT_MODEL"
        print_message $YELLOW "  - Verifier模型: $OPENROUTER_VERIFIER_MODEL"
        return 0
    fi
    return 0
}

# 启动后端服务
start_backend_service() {
    print_message $BLUE "启动 Patient Agent 测试界面后端服务..."
    
    # 检查端口是否被占用
    if check_port $BACKEND_PORT; then
        print_message $YELLOW "⚠ 端口 $BACKEND_PORT 已被占用"
        read -p "是否要终止现有服务并重新启动？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # 终止占用端口的进程
            local pids=$(lsof -ti:$BACKEND_PORT)
            if [ ! -z "$pids" ]; then
                print_message $YELLOW "终止进程: $pids"
                kill -9 $pids
                sleep 2
            fi
        else
            print_message $YELLOW "保持现有服务运行"
            return 1
        fi
    fi
    
    # 进入项目目录
    cd "$PROJECT_ROOT"
    
    # 设置 Python 路径
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # 设置基础配置环境变量
    export BACKEND_PORT="$BACKEND_PORT"
    export PATIENT_DATA_PATH="$PATIENT_DATA_PATH"
    export MAX_PATIENTS="$MAX_PATIENTS"
    export DIAGNOSIS_CACHE_FILE="$DIAGNOSIS_CACHE_FILE"
    
    # 全局 OpenRouter API 配置
    export OPENROUTER_API_KEY="$OPENROUTER_API_KEY"
    export OPENROUTER_SITE_URL="$OPENROUTER_SITE_URL"
    export OPENROUTER_SITE_NAME="$OPENROUTER_SITE_NAME"
    export OPENROUTER_MAX_PARALLEL="$OPENROUTER_MAX_PARALLEL"
    
    # 模型模式配置
    export MODEL_MODE="$MODEL_MODE"
    
    # 单独控制每个 Agent
    export PATIENT_USE_OPENROUTER="$PATIENT_USE_OPENROUTER"
    export DOCTOR_USE_OPENROUTER="$DOCTOR_USE_OPENROUTER"
    export VERIFIER_USE_OPENROUTER="$VERIFIER_USE_OPENROUTER"
    
    # OpenRouter 模型配置
    export OPENROUTER_DOCTOR_MODEL="$OPENROUTER_DOCTOR_MODEL"
    export OPENROUTER_PATIENT_MODEL="$OPENROUTER_PATIENT_MODEL"
    export OPENROUTER_VERIFIER_MODEL="$OPENROUTER_VERIFIER_MODEL"
    
    # 离线模型配置
    export OFFLINE_DOCTOR_MODEL="$OFFLINE_DOCTOR_MODEL"
    export OFFLINE_PATIENT_MODEL="$OFFLINE_PATIENT_MODEL"
    export OFFLINE_VERIFIER_MODEL="$OFFLINE_VERIFIER_MODEL"
    export OFFLINE_DOCTOR_PORTS="$OFFLINE_DOCTOR_PORTS"
    export OFFLINE_PATIENT_PORTS="$OFFLINE_PATIENT_PORTS"
    export OFFLINE_VERIFIER_PORTS="$OFFLINE_VERIFIER_PORTS"
    export OFFLINE_MAX_PARALLEL="$OFFLINE_MAX_PARALLEL"
    
    # VLLM 外部部署配置
    export VLLM_DOCTOR_IP="$VLLM_DOCTOR_IP"
    export VLLM_PATIENT_IP="$VLLM_PATIENT_IP"
    export VLLM_VERIFIER_IP="$VLLM_VERIFIER_IP"
    
    # 清除代理设置，避免影响本地服务访问
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    
    print_message $GREEN "启动后端服务..."
    print_message $YELLOW "项目目录: $PROJECT_ROOT"
    print_message $BLUE "使用配置:"
    print_message $YELLOW "  - 患者数据: $PATIENT_DATA_PATH"
    print_message $YELLOW "  - 最大患者数: $MAX_PATIENTS"
    print_message $YELLOW "  - 诊断缓存文件: $DIAGNOSIS_CACHE_FILE"
    if [ -n "$MODEL_MODE" ]; then
        print_message $YELLOW "  - 模型模式: $MODEL_MODE"
    fi
    echo ""
    
    # 检查是否有任何 Agent 使用 OpenRouter
    if [ "$PATIENT_USE_OPENROUTER" = "true" ] || [ "$DOCTOR_USE_OPENROUTER" = "true" ] || [ "$VERIFIER_USE_OPENROUTER" = "true" ] || [ "$MODEL_MODE" = "openrouter" ]; then
        print_message $BLUE "OpenRouter API 配置:"
        if [ -n "$OPENROUTER_API_KEY" ]; then
            print_message $YELLOW "  - API密钥: ${OPENROUTER_API_KEY:0:12}..."
        else
            print_message $YELLOW "  - API密钥: (未设置)"
        fi
        print_message $YELLOW "  - 最大并行数: $OPENROUTER_MAX_PARALLEL"
        echo ""
    fi
    
    print_message $BLUE "模型配置:"
    
    # EverPsychiatrist (Doctor)
    echo ""
    print_message $YELLOW "  EverPsychiatrist (医生问诊):"
    if [ "$DOCTOR_USE_OPENROUTER" = "true" ] || ([ -z "$DOCTOR_USE_OPENROUTER" ] && [ "$MODEL_MODE" = "openrouter" ]); then
        print_message $YELLOW "    类型: OpenRouter API"
        print_message $YELLOW "    模型: $OPENROUTER_DOCTOR_MODEL"
    else
        print_message $YELLOW "    类型: 本地 VLLM"
        print_message $YELLOW "    模型: $OFFLINE_DOCTOR_MODEL"
        print_message $YELLOW "    端口: $OFFLINE_DOCTOR_PORTS"
        if [ -n "$VLLM_DOCTOR_IP" ]; then
            print_message $YELLOW "    IP: $VLLM_DOCTOR_IP"
        fi
    fi
    
    # EverPsychosis (Patient)
    echo ""
    print_message $YELLOW "  EverPsychosis (患者模拟):"
    if [ "$PATIENT_USE_OPENROUTER" = "true" ] || ([ -z "$PATIENT_USE_OPENROUTER" ] && [ "$MODEL_MODE" = "openrouter" ]); then
        print_message $YELLOW "    类型: OpenRouter API"
        print_message $YELLOW "    模型: $OPENROUTER_PATIENT_MODEL"
    else
        print_message $YELLOW "    类型: 本地 VLLM"
        print_message $YELLOW "    模型: $OFFLINE_PATIENT_MODEL"
        print_message $YELLOW "    端口: $OFFLINE_PATIENT_PORTS"
        if [ -n "$VLLM_PATIENT_IP" ]; then
            print_message $YELLOW "    IP: $VLLM_PATIENT_IP"
        fi
    fi
    
    # EverDiagnosis (Verifier)
    echo ""
    print_message $YELLOW "  EverDiagnosis (诊断生成):"
    if [ "$VERIFIER_USE_OPENROUTER" = "true" ] || ([ -z "$VERIFIER_USE_OPENROUTER" ] && [ "$MODEL_MODE" = "openrouter" ]); then
        print_message $YELLOW "    类型: OpenRouter API"
        print_message $YELLOW "    模型: $OPENROUTER_VERIFIER_MODEL"
    else
        print_message $YELLOW "    类型: 本地 VLLM"
        print_message $YELLOW "    模型: $OFFLINE_VERIFIER_MODEL"
        print_message $YELLOW "    端口: $OFFLINE_VERIFIER_PORTS"
        if [ -n "$VLLM_VERIFIER_IP" ]; then
            print_message $YELLOW "    IP: $VLLM_VERIFIER_IP"
        fi
    fi
    echo ""
    
    # 获取本机IP地址
    LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || ip route get 1 | sed -n 's/^.*src \([0-9.]*\) .*$/\1/p' 2>/dev/null || echo "127.0.0.1")
    
    print_message $YELLOW "本机访问: http://localhost:$BACKEND_PORT"
    print_message $YELLOW "局域网访问: http://$LOCAL_IP:$BACKEND_PORT"
    print_message $BLUE "局域网内其他设备可通过 $LOCAL_IP:$BACKEND_PORT 访问"
    print_message $YELLOW "按 Ctrl+C 停止服务"
    echo ""
    
    # 启动 Flask 应用
    python patient_test_ui/backend/app.py
}

# 主函数
main() {
    echo ""
    print_message $BLUE "开始系统检查..."
    echo ""
    
    # 1. 检查OpenRouter API配置 (如果使用的话)
    if ! check_openrouter_config; then
        print_message $RED "OpenRouter API配置检查失败"
        exit 1
    fi
    
    # 2. 检查患者数据
    if ! check_patient_data; then
        print_message $RED "患者数据检查失败，请确认数据文件存在且完整"
        exit 1
    fi
    
    echo ""
    print_message $GREEN "✓ 所有检查通过，准备启动测试界面"
    echo ""
    
    # 3. 启动后端服务
    start_backend_service
}

# 信号处理
trap 'print_message $YELLOW "\n正在停止服务..."; exit 0' INT TERM

# 帮助信息
show_help() {
    echo "Patient Agent 测试界面启动脚本"
    echo ""
    echo "用法: $0 [选项] [配置选项]"
    echo ""
    echo "控制选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  --check-only            只进行系统检查，不启动服务"
    echo ""
    echo "基础配置选项:"
    echo "  --patient-data-path PATH        患者数据文件路径 (默认: $PATIENT_DATA_PATH)"
    echo "  --backend-port PORT             后端服务端口 (默认: $BACKEND_PORT)"
    echo "  --max-patients NUM              最大患者数量 (默认: $MAX_PATIENTS)"
    echo "  --diagnosis-cache-file PATH     诊断预加载缓存文件路径"
    echo "                                  (默认: patient_test_ui/data/diagnosis_cache.json)"
    echo ""
    echo "模型模式选项:"
    echo "  --model-mode MODE           全局模型模式: openrouter 或 offline"
    echo "                              (所有 Agent 使用同一种方式)"
    echo ""
    echo "单独控制每个 Agent (优先级高于 --model-mode):"
    echo "  --patient-use-openrouter    患者模拟使用 OpenRouter API"
    echo "  --patient-use-offline       患者模拟使用本地 VLLM"
    echo "  --doctor-use-openrouter     医生问诊使用 OpenRouter API"
    echo "  --doctor-use-offline        医生问诊使用本地 VLLM"
    echo "  --verifier-use-openrouter   诊断生成使用 OpenRouter API"
    echo "  --verifier-use-offline      诊断生成使用本地 VLLM"
    echo ""
    echo "OpenRouter API 配置选项:"
    echo "  --openrouter-api-key KEY        OpenRouter API 密钥（也可在 .env 中设置）"
    echo "  --openrouter-site-url URL       网站URL (可选)"
    echo "  --openrouter-site-name NAME     网站名称 (默认: Patient Agent Test)"
    echo "  --openrouter-max-parallel NUM   最大并行数 (默认: 16)"
    echo ""
    echo "  --openrouter-doctor-model MODEL    医生模型 (默认: qwen/qwen3-32b)"
    echo "  --openrouter-patient-model MODEL   患者模型 (默认: qwen/qwen3-32b)"
    echo "  --openrouter-verifier-model MODEL  诊断模型 (默认: qwen/qwen3-32b)"
    echo ""
    echo "离线模型配置选项:"
    echo "  --offline-doctor-model PATH      医生模型路径"
    echo "  --offline-patient-model PATH     患者模型路径"
    echo "  --offline-verifier-model PATH    诊断模型路径"
    echo ""
    echo "  --offline-doctor-ports PORTS     医生模型端口 (默认: 9041)"
    echo "  --offline-patient-ports PORTS    患者模型端口 (默认: 9040)"
    echo "  --offline-verifier-ports PORTS   诊断模型端口 (默认: 9002)"
    echo ""
    echo "  --offline-max-parallel NUM       最大并行数 (默认: 1)"
    echo ""
    echo "VLLM 外部部署配置:"
    echo "  --vllm-doctor-ip IP       医生模型VLLM服务IP (留空使用localhost)"
    echo "  --vllm-patient-ip IP      患者模型VLLM服务IP (留空使用localhost)"
    echo "  --vllm-verifier-ip IP     诊断模型VLLM服务IP (留空使用localhost)"
    echo ""
    echo "使用示例:"
    echo ""
    echo "  # 示例 1: 全部使用 OpenRouter API (推荐):"
    echo "  $0 --model-mode openrouter --openrouter-api-key sk-or-v1-xxx"
    echo ""
    echo "  # 示例 2: 全部使用离线模型 (需要预先启动VLLM服务):"
    echo "  $0 --model-mode offline \\"
    echo "     --offline-doctor-ports 9041 \\"
    echo "     --offline-patient-ports 9040 \\"
    echo "     --offline-verifier-ports 9002"
    echo ""
    echo "  # 示例 3: 混合部署 - Patient 和 Doctor 用 OpenRouter，Verifier 用本地:"
    echo "  $0 --openrouter-api-key sk-or-v1-xxx \\"
    echo "     --patient-use-openrouter \\"
    echo "     --doctor-use-openrouter \\"
    echo "     --verifier-use-offline \\"
    echo "     --offline-verifier-ports 9002"
    echo ""
    echo "  # 示例 4: 使用不同的 OpenRouter 模型:"
    echo "  $0 --model-mode openrouter --openrouter-api-key sk-or-v1-xxx \\"
    echo "     --openrouter-doctor-model anthropic/claude-3.5-sonnet \\"
    echo "     --openrouter-patient-model qwen/qwen3-32b \\"
    echo "     --openrouter-verifier-model qwen/qwen3-32b"
    echo ""
    echo "  # 示例 5: 使用远程 VLLM 服务:"
    echo "  $0 --patient-use-offline \\"
    echo "     --vllm-patient-ip 10.119.28.185 \\"
    echo "     --offline-patient-ports 9028,9029 \\"
    echo "     --doctor-use-offline --offline-doctor-ports 9041 \\"
    echo "     --verifier-use-offline --offline-verifier-ports 9002"
    echo ""
    echo "环境要求:"
    echo "  - Python 3.10+"
    echo "  - 项目依赖已安装"
    echo "  - 离线模式: VLLM 服务已启动"
    echo "  - OpenRouter模式: 有效的 OpenRouter API 密钥"
    echo ""
    echo "配置文件:"
    echo "  可以在 .env 文件中设置所有配置项，参考 .env_example"
    echo ""
}

# 显示当前配置
show_config() {
    print_message $BLUE "当前配置:"
    echo "  患者数据路径: $PATIENT_DATA_PATH"
    echo "  后端端口: $BACKEND_PORT"
    echo "  最大患者数: $MAX_PATIENTS"
    echo "  诊断缓存文件: $DIAGNOSIS_CACHE_FILE"
    if [ -n "$MODEL_MODE" ]; then
        echo "  模型模式: $MODEL_MODE"
    fi
    echo ""
    
    # 检查是否有任何 Agent 使用 OpenRouter
    if [ "$PATIENT_USE_OPENROUTER" = "true" ] || [ "$DOCTOR_USE_OPENROUTER" = "true" ] || [ "$VERIFIER_USE_OPENROUTER" = "true" ] || [ "$MODEL_MODE" = "openrouter" ]; then
        print_message $BLUE "OpenRouter API 配置:"
        if [ -n "$OPENROUTER_API_KEY" ]; then
            echo "  API密钥: ${OPENROUTER_API_KEY:0:12}..."
        else
            echo "  API密钥: (未设置)"
        fi
        echo "  网站名称: $OPENROUTER_SITE_NAME"
        echo "  最大并行数: $OPENROUTER_MAX_PARALLEL"
        echo ""
    fi
    
    print_message $BLUE "模型配置:"
    
    # EverPsychiatrist (Doctor)
    echo ""
    echo "  EverPsychiatrist (医生问诊):"
    if [ "$DOCTOR_USE_OPENROUTER" = "true" ] || ([ -z "$DOCTOR_USE_OPENROUTER" ] && [ "$MODEL_MODE" = "openrouter" ]); then
        echo "    类型: OpenRouter API"
        echo "    模型: $OPENROUTER_DOCTOR_MODEL"
    else
        echo "    类型: 本地 VLLM"
        echo "    模型: $OFFLINE_DOCTOR_MODEL"
        echo "    端口: $OFFLINE_DOCTOR_PORTS"
        if [ -n "$VLLM_DOCTOR_IP" ]; then
            echo "    IP: $VLLM_DOCTOR_IP"
        fi
    fi
    
    # EverPsychosis (Patient)
    echo ""
    echo "  EverPsychosis (患者模拟):"
    if [ "$PATIENT_USE_OPENROUTER" = "true" ] || ([ -z "$PATIENT_USE_OPENROUTER" ] && [ "$MODEL_MODE" = "openrouter" ]); then
        echo "    类型: OpenRouter API"
        echo "    模型: $OPENROUTER_PATIENT_MODEL"
    else
        echo "    类型: 本地 VLLM"
        echo "    模型: $OFFLINE_PATIENT_MODEL"
        echo "    端口: $OFFLINE_PATIENT_PORTS"
        if [ -n "$VLLM_PATIENT_IP" ]; then
            echo "    IP: $VLLM_PATIENT_IP"
        fi
    fi
    
    # EverDiagnosis (Verifier)
    echo ""
    echo "  EverDiagnosis (诊断生成):"
    if [ "$VERIFIER_USE_OPENROUTER" = "true" ] || ([ -z "$VERIFIER_USE_OPENROUTER" ] && [ "$MODEL_MODE" = "openrouter" ]); then
        echo "    类型: OpenRouter API"
        echo "    模型: $OPENROUTER_VERIFIER_MODEL"
    else
        echo "    类型: 本地 VLLM"
        echo "    模型: $OFFLINE_VERIFIER_MODEL"
        echo "    端口: $OFFLINE_VERIFIER_PORTS"
        if [ -n "$VLLM_VERIFIER_IP" ]; then
            echo "    IP: $VLLM_VERIFIER_IP"
        fi
    fi
    echo ""
}

# 参数处理
# 首先解析配置参数
parse_config_args "$@"

# 然后处理控制参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    --check-only)
        print_message $BLUE "执行系统检查模式"
        echo ""
        show_config
        check_patient_data 
        exit $?
        ;;
    ""|--*)
        # 默认行为：完整启动（包括只有配置参数的情况）
        show_config
        main
        ;;
    *)
        print_message $RED "未知选项: $1"
        show_help
        exit 1
        ;;
esac
