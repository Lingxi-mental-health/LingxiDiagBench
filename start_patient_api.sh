#!/bin/bash

# Patient Agent API 启动脚本
# 
# 功能：
# 1. 根据环境变量选择使用 OpenRouter 或本地 VLLM 模型
# 2. 启动 Patient Agent FastAPI 服务
# 3. 支持命令行参数覆盖配置

## 请求示例：
## patient_version 可选值: v1, mdd5k, v3, cot (默认: cot)

# curl -X POST "http://10.119.29.220:8001/api/v1/patient/chat" -H "Content-Type: application/json" -d '{"patient_id": "300005853", "messages": [{"role": "user", "content": "你好，最近感觉怎么样？"}], "patient_version": "v1", "model_name": "Qwen3-1.7B"}'

set -e  # 遇到错误时退出

echo "=== Patient Agent API 启动脚本 ==="
echo ""

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 默认配置
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT=8001
DEFAULT_DATA_FILE="./raw_data/LingxiDiag-16K_train_data.json"
DEFAULT_PATIENT_VERSION="cot"
RELOAD_MODE=false

# 从 .env 加载配置（如果存在）
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -o allexport
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +o allexport
    echo "✓ 已加载 .env 配置文件"
else
    echo "⚠ 未找到 .env 文件，使用默认配置"
    echo "  提示：可以复制 .env_example 为 .env 并修改配置"
fi

# === 从环境变量读取配置 ===
HOST=${HOST:-$DEFAULT_HOST}
PORT=${PORT:-$DEFAULT_PORT}
PATIENT_DATA_FILE=${PATIENT_DATA_FILE:-$DEFAULT_DATA_FILE}
PATIENT_VERSION=${PATIENT_VERSION:-$DEFAULT_PATIENT_VERSION}

# Patient 模型配置
PATIENT_USE_OPENROUTER=${PATIENT_USE_OPENROUTER:-true}
OPENROUTER_PATIENT_MODEL=${OPENROUTER_PATIENT_MODEL:-qwen/qwen3-32b}
OFFLINE_PATIENT_MODEL=${OFFLINE_PATIENT_MODEL:-../models/qwen3-32b}
OFFLINE_PATIENT_PORTS=${OFFLINE_PATIENT_PORTS:-9040}
VLLM_PATIENT_IP=${VLLM_PATIENT_IP:-}

# OpenRouter API Key
OPENROUTER_API_KEY=${OPENROUTER_API_KEY:-}

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

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --host)
                HOST="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --data-file)
                PATIENT_DATA_FILE="$2"
                shift 2
                ;;
            --reload)
                RELOAD_MODE=true
                shift
                ;;
            --openrouter)
                PATIENT_USE_OPENROUTER=true
                shift
                ;;
            --offline)
                PATIENT_USE_OPENROUTER=false
                shift
                ;;
            --openrouter-model)
                OPENROUTER_PATIENT_MODEL="$2"
                shift 2
                ;;
            --offline-model)
                OFFLINE_PATIENT_MODEL="$2"
                shift 2
                ;;
            --offline-ports)
                OFFLINE_PATIENT_PORTS="$2"
                shift 2
                ;;
            --vllm-ip)
                VLLM_PATIENT_IP="$2"
                shift 2
                ;;
            --patient-version)
                PATIENT_VERSION="$2"
                # 验证 patient_version 参数
                if [[ ! "$PATIENT_VERSION" =~ ^(v1|mdd5k|v3|cot)$ ]]; then
                    print_message "$RED" "错误: 无效的 patient_version: $PATIENT_VERSION"
                    print_message "$YELLOW" "  有效值: v1, mdd5k, v3, cot"
                    exit 1
                fi
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                print_message "$RED" "错误: 未知参数 $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 显示帮助信息
show_help() {
    cat << EOF
使用方法: $0 [选项]

选项:
  --host HOST                服务器地址 (默认: $DEFAULT_HOST)
  --port PORT                服务器端口 (默认: $DEFAULT_PORT)
  --data-file PATH           患者数据文件路径 (默认: $DEFAULT_DATA_FILE)
  --reload                   启用开发模式（自动重载）
  
  模型配置:
  --openrouter               使用 OpenRouter 模式
  --offline                  使用本地 VLLM 模式
  --openrouter-model MODEL   OpenRouter 模型名称 (默认: qwen/qwen3-32b)
  --offline-model PATH       本地模型路径 (默认: ../models/qwen3-32b)
  --offline-ports PORTS      本地 VLLM 端口 (默认: 9040)
  --vllm-ip IP               远程 VLLM 服务器 IP (可选)
  --patient-version VERSION  Patient 版本 (v1, mdd5k, v3, cot) (默认: cot)
  
  -h, --help                 显示此帮助信息

环境变量:
  可以通过 .env 文件配置所有选项，参考 .env_example

示例:
  # 使用 OpenRouter 模式启动
  $0 --openrouter
  
  # 使用本地 VLLM 模式启动
  $0 --offline --offline-ports 9040
  
  # 使用远程 VLLM 服务
  $0 --offline --vllm-ip 10.119.28.185 --offline-ports 9028
  
  # 使用指定的 Patient 版本启动
  $0 --patient-version v1
  $0 --patient-version mdd5k
  $0 --patient-version v3
  $0 --patient-version cot
  
  # 开发模式（自动重载）
  $0 --reload

EOF
}

# 检查 Python 环境
check_python_env() {
    print_message "$BLUE" "检查 Python 环境..."
    
    if ! command -v python3 &> /dev/null; then
        print_message "$RED" "✗ 错误: 未找到 python3"
        exit 1
    fi
    
    print_message "$GREEN" "✓ Python 环境正常"
}

# 检查数据文件
check_data_file() {
    print_message "$BLUE" "检查数据文件..."
    
    if [ ! -f "$PATIENT_DATA_FILE" ]; then
        print_message "$RED" "✗ 错误: 数据文件不存在: $PATIENT_DATA_FILE"
        exit 1
    fi
    
    print_message "$GREEN" "✓ 数据文件: $PATIENT_DATA_FILE"
}

# 检查 OpenRouter 配置
check_openrouter_config() {
    if [ "${PATIENT_USE_OPENROUTER,,}" = "true" ]; then
        print_message "$BLUE" "检查 OpenRouter 配置..."
        
        if [ -z "$OPENROUTER_API_KEY" ]; then
            print_message "$RED" "✗ 错误: 使用 OpenRouter 模式但未配置 OPENROUTER_API_KEY"
            print_message "$YELLOW" "  请在 .env 文件中设置 OPENROUTER_API_KEY"
            exit 1
        fi
        
        print_message "$GREEN" "✓ OpenRouter 配置正常"
        print_message "$GREEN" "  - 模型: $OPENROUTER_PATIENT_MODEL"
        print_message "$GREEN" "  - API Key: ${OPENROUTER_API_KEY:0:10}..."
    fi
}

# 检查本地 VLLM 配置
check_vllm_config() {
    if [ "${PATIENT_USE_OPENROUTER,,}" != "true" ]; then
        print_message "$BLUE" "检查本地 VLLM 配置..."
        
        print_message "$GREEN" "✓ 本地 VLLM 配置:"
        print_message "$GREEN" "  - 模型: $OFFLINE_PATIENT_MODEL"
        print_message "$GREEN" "  - 端口: $OFFLINE_PATIENT_PORTS"
        if [ -n "$VLLM_PATIENT_IP" ]; then
            print_message "$GREEN" "  - IP: $VLLM_PATIENT_IP"
        else
            print_message "$GREEN" "  - IP: 127.0.0.1 (本地)"
        fi
        
        print_message "$YELLOW" "⚠ 注意: 请确保 VLLM 服务已经启动"
    fi
}

# 显示配置摘要
show_config() {
    echo ""
    print_message "$BLUE" "==================== 配置摘要 ===================="
    echo "服务地址: $HOST:$PORT"
    echo "数据文件: $PATIENT_DATA_FILE"
    echo "Patient 版本: $PATIENT_VERSION"
    echo "开发模式: $([ "$RELOAD_MODE" = true ] && echo "是" || echo "否")"
    echo ""
    echo "模型配置:"
    if [ "${PATIENT_USE_OPENROUTER,,}" = "true" ]; then
        echo "  - 模式: OpenRouter"
        echo "  - 模型: $OPENROUTER_PATIENT_MODEL"
    else
        echo "  - 模式: 本地 VLLM"
        echo "  - 模型: $OFFLINE_PATIENT_MODEL"
        echo "  - 端口: $OFFLINE_PATIENT_PORTS"
        [ -n "$VLLM_PATIENT_IP" ] && echo "  - IP: $VLLM_PATIENT_IP"
    fi
    print_message "$BLUE" "=================================================="
    echo ""
}

# 启动服务
start_service() {
    print_message "$BLUE" "启动 Patient Agent API 服务..."
    
    # 构建启动命令
    local cmd="python3 src/patient/patient_api.py"
    cmd="$cmd --host $HOST"
    cmd="$cmd --port $PORT"
    cmd="$cmd --data-file $PATIENT_DATA_FILE"
    
    if [ "$RELOAD_MODE" = true ]; then
        cmd="$cmd --reload"
    fi
    
    # 导出环境变量
    export PATIENT_USE_OPENROUTER
    export OPENROUTER_PATIENT_MODEL
    export OFFLINE_PATIENT_MODEL
    export OFFLINE_PATIENT_PORTS
    export VLLM_PATIENT_IP
    export OPENROUTER_API_KEY
    export PATIENT_DATA_FILE
    export PATIENT_VERSION
    
    print_message "$GREEN" "执行命令: $cmd"
    echo ""
    
    # 执行命令
    eval "$cmd"
}

# ==================== 主程序 ====================

# 解析命令行参数
parse_args "$@"

# 显示配置
show_config

# 检查环境
check_python_env
check_data_file

# 根据模式检查配置
if [ "${PATIENT_USE_OPENROUTER,,}" = "true" ]; then
    check_openrouter_config
else
    check_vllm_config
fi

# 启动服务
echo ""
print_message "$GREEN" "准备就绪，正在启动服务..."
echo ""

start_service

