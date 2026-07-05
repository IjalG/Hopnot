#!/bin/bash
# Kaggle 环境一键安装脚本
# 在 Kaggle Notebook 的第一个 cell 运行:
# !bash /kaggle/working/kaggle/install.sh

set -e

echo "=== 安装系统依赖 ==="
apt-get update -qq && apt-get install -y -qq graphviz 2>/dev/null
pip install -q modelscope sentence-transformers faiss-cpu

echo "=== 验证 Hopnot 导入 ==="
cd /kaggle/working
python3 -c "from hopnot import HippocampusMemorySystem; print('Hopnot OK')"

echo "=== 验证 GPU ==="
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=== 就绪 ==="
