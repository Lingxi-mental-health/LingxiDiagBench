#!/bin/bash
# ============================================================
# Lingxi Annotation 项目环境安装脚本
# 使用 uv 包管理器进行依赖安装
# ============================================================

set -e  # 遇到错误时退出

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

print_step() {
    print_message "$BLUE" "\n=== $1 ===\n"
}

print_success() {
    print_message "$GREEN" "✓ $1"
}

print_warning() {
    print_message "$YELLOW" "⚠ $1"
}

print_error() {
    print_message "$RED" "✗ $1"
}

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "========================================"
echo "  Lingxi Annotation 环境安装"
echo "========================================"
echo ""

# 检查 uv 是否已安装
print_step "检查 uv 安装状态"
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    print_success "uv 已安装: $UV_VERSION"
else
    print_warning "uv 未安装，正在安装..."
    # 安装 uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 添加 uv 到 PATH（当前 session）
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &> /dev/null; then
        print_success "uv 安装成功"
    else
        print_error "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

# 解析命令行参数
INSTALL_TYPE="default"
PYTHON_VERSION="3.10"
USE_MIRROR=true
FORCE_REINSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            INSTALL_TYPE="cuda"
            shift
            ;;
        --vllm)
            INSTALL_TYPE="vllm"
            shift
            ;;
        --all)
            INSTALL_TYPE="all"
            shift
            ;;
        --dev)
            INSTALL_TYPE="dev"
            shift
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --no-mirror)
            USE_MIRROR=false
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --cuda        安装 GPU 版本的 PyTorch"
            echo "  --vllm        安装 vLLM 依赖"
            echo "  --all         安装所有可选依赖"
            echo "  --dev         安装开发依赖"
            echo "  --python VER  指定 Python 版本 (默认: 3.10)"
            echo "  --no-mirror   不使用国内镜像"
            echo "  --force       强制重新安装所有依赖"
            echo "  -h, --help    显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                  # 安装基础依赖"
            echo "  $0 --cuda           # 安装 GPU 版本"
            echo "  $0 --all --dev      # 安装所有依赖（包括开发工具）"
            exit 0
            ;;
        *)
            print_error "未知选项: $1"
            exit 1
            ;;
    esac
done

# 检查并创建虚拟环境
print_step "创建/检查虚拟环境"

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ] && [ "$FORCE_REINSTALL" = false ]; then
    print_success "虚拟环境已存在: $VENV_DIR"
else
    if [ "$FORCE_REINSTALL" = true ] && [ -d "$VENV_DIR" ]; then
        print_warning "强制重新创建虚拟环境..."
        rm -rf "$VENV_DIR"
    fi
    print_message "$BLUE" "创建虚拟环境 (Python $PYTHON_VERSION)..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    print_success "虚拟环境创建成功: $VENV_DIR"
fi

# 激活虚拟环境
print_step "激活虚拟环境"
source "$VENV_DIR/bin/activate"
print_success "虚拟环境已激活"

# 配置 pip 源（如果使用镜像）
if [ "$USE_MIRROR" = true ]; then
    print_step "配置国内镜像源"
    export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    print_success "使用清华镜像源"
fi

# 安装依赖
print_step "安装项目依赖"

# 基础安装命令
INSTALL_CMD="uv pip install -e ."

case $INSTALL_TYPE in
    cuda)
        print_message "$BLUE" "安装 GPU 版本依赖..."
        # 先安装 GPU 版本的 PyTorch
        uv pip install torch --index-url https://download.pytorch.org/whl/cu118
        $INSTALL_CMD
        ;;
    vllm)
        print_message "$BLUE" "安装 vLLM 依赖..."
        $INSTALL_CMD"[vllm]"
        ;;
    all)
        print_message "$BLUE" "安装所有依赖..."
        $INSTALL_CMD"[all]"
        ;;
    dev)
        print_message "$BLUE" "安装开发依赖..."
        $INSTALL_CMD"[dev]"
        ;;
    *)
        print_message "$BLUE" "安装基础依赖..."
        $INSTALL_CMD
        ;;
esac

print_success "依赖安装完成"

# 下载 NLTK 数据（用于 BLEU 评估）
print_step "下载 NLTK 数据"
python -c "
import nltk
import os
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print('NLTK 数据下载完成')
" 2>/dev/null || print_warning "NLTK 数据下载失败（可选，不影响主要功能）"

# 配置 HuggingFace 镜像
print_step "配置 HuggingFace 镜像"
export HF_ENDPOINT="https://hf-mirror.com"
echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> "$VENV_DIR/bin/activate"
print_success "HuggingFace 镜像已配置"

# 验证安装
print_step "验证安装"

python -c "
import sys
print(f'Python 版本: {sys.version}')

# 检查核心依赖
deps = {
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'torch': 'PyTorch',
    'transformers': 'transformers',
    'fastapi': 'FastAPI',
    'flask': 'Flask',
    'openai': 'OpenAI',
    'pydantic': 'Pydantic',
    'openpyxl': 'openpyxl',
    'jieba': 'jieba',
    'faiss': 'faiss-cpu',
    'tqdm': 'tqdm',
}

print('\\n已安装的依赖:')
for module, name in deps.items():
    try:
        __import__(module)
        print(f'  ✓ {name}')
    except ImportError:
        print(f'  ✗ {name} (未安装)')
"

echo ""
print_success "环境安装完成！"
echo ""
echo "========================================"
echo "  使用说明"
echo "========================================"
echo ""
echo "激活环境:"
echo "  source .venv/bin/activate"
echo ""
echo "运行 Benchmark:"
echo "  python evaluation/static/run_benchmark.py --help"
echo ""
echo "运行 Doctor 评估:"
echo "  bash run_doctor_eval.sh"
echo ""
echo "运行 Patient 评估:"
echo "  bash run_patient_eval.sh"
echo ""
echo "启动 Patient API:"
echo "  bash start_patient_api.sh"
echo ""
echo "启动测试 UI:"
echo "  bash start_patient_test_ui.sh"
echo ""



