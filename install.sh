#!/bin/bash
# ============================================================
# 快速安装脚本 - 使用 uv 进行一键环境配置
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  Lingxi Annotation 一键安装"
echo "================================================"
echo ""

# 检查并安装 uv
if ! command -v uv &> /dev/null; then
    echo "正在安装 uv 包管理器..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 创建虚拟环境并安装依赖
echo "创建虚拟环境并安装依赖..."
uv venv --python 3.10 .venv
source .venv/bin/activate

# 使用清华镜像
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export HF_ENDPOINT="https://hf-mirror.com"

# 安装项目
uv pip install -e .

# 下载 NLTK 数据
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

echo ""
echo "================================================"
echo "  安装完成！"
echo "================================================"
echo ""
echo "使用方法:"
echo "  source .venv/bin/activate"
echo ""
echo "运行 Benchmark:"
echo "  python evaluation/static/run_benchmark.py"
echo ""

