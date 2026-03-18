#!/bin/bash
#SBATCH --job-name=moondream3
#SBATCH --output=results/moondream3_%j.log
#SBATCH --error=results/moondream3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# -------------------------------------------------------
# Moondream 3 MoE (9B) 需要 ~20GB VRAM
# H100 80GB 完全足夠
# -------------------------------------------------------

echo "========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURM_NODELIST"
echo "Start     : $(date)"
echo "========================================="

PROJECT_DIR="$HOME/moondream_workspace"
cd "$PROJECT_DIR"

source .venv/bin/activate

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 scripts/run_moondream3.py \
    --image configs/test_images.txt \
    --output results/output_moondream3_${SLURM_JOB_ID}.json

echo "========================================="
echo "End: $(date)"
echo "========================================="
