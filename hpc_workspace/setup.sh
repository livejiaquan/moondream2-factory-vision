#!/bin/bash
# =============================================================
# Moondream HPC 環境建立腳本
# 在 iService / SLURM 節點上執行一次即可
# 用法：bash setup.sh
# =============================================================

set -e  # 任何錯誤就停止

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR="$PROJECT_DIR/.venv"

echo "=================================================="
echo "  Moondream HPC 環境設定"
echo "  專案路徑：$PROJECT_DIR"
echo "=================================================="

# ── 1. 確認 Python 版本 ──────────────────────────────────────
echo ""
echo "[1/5] 檢查 Python..."
python3 --version || { echo "❌ 找不到 python3，請先 module load python"; exit 1; }

# ── 2. 建立虛擬環境 ──────────────────────────────────────────
echo ""
echo "[2/5] 建立虛擬環境 (.venv)..."
if [ -d "$VENV_DIR" ]; then
    echo "  ⚠️  .venv 已存在，跳過建立"
else
    python3 -m venv "$VENV_DIR"
    echo "  ✓ 虛擬環境建立完成"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q

# ── 3. 安裝 CUDA 版本 PyTorch ────────────────────────────────
echo ""
echo "[3/5] 安裝 PyTorch（CUDA 12.1）..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
echo "  ✓ PyTorch 安裝完成"

# ── 4. 安裝其他套件 ──────────────────────────────────────────
echo ""
echo "[4/5] 安裝其他依賴..."
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "  ✓ 所有套件安裝完成"

# ── 5. 驗證安裝 ──────────────────────────────────────────────
echo ""
echo "[5/5] 驗證安裝..."
python3 - <<'EOF'
import torch
print(f"  PyTorch:  {torch.__version__}")
print(f"  CUDA:     {torch.version.cuda}")
print(f"  GPU 數量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}:   {p.name}  VRAM={p.total_memory/1e9:.0f}GB")
import transformers
print(f"  transformers: {transformers.__version__}")
print("  ✅ 所有套件正常！")
EOF

echo ""
echo "=================================================="
echo "  ✅ 環境設定完成！"
echo ""
echo "  之後使用方式："
echo "  source .venv/bin/activate"
echo "  sbatch jobs/run_moondream2.sh    # 跑 2B 模型"
echo "  sbatch jobs/run_moondream3.sh    # 跑 3.0 旗艦版"
echo "=================================================="
