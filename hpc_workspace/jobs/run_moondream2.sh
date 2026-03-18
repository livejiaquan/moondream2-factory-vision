#!/bin/bash
#SBATCH --job-name=moondream2
#SBATCH --output=results/moondream2_%j.log
#SBATCH --error=results/moondream2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# -------------------------------------------------------
# 如果學校有分 partition，在這裡修改：
# #SBATCH --partition=gpu
# -------------------------------------------------------

echo "========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start     : $(date)"
echo "========================================="

# 切換到專案目錄
PROJECT_DIR="$HOME/moondream_workspace"
cd "$PROJECT_DIR"

# 啟動虛擬環境
source .venv/bin/activate

# 確認 GPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 執行主程式
python3 scripts/run_moondream2.py \
    --image configs/test_images.txt \
    --output results/output_moondream2_${SLURM_JOB_ID}.json

echo "========================================="
echo "End: $(date)"
echo "========================================="
