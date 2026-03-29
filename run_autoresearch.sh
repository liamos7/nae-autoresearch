#!/bin/bash
#SBATCH --job-name=nae-autoresearch
#SBATCH --output=/scratch/network/lo8603/thesis/logs/autoresearch_%j.out
#SBATCH --error=/scratch/network/lo8603/thesis/logs/autoresearch_%j.err
#SBATCH --nodelist=adroit-h11g1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Load your environment (same as your existing script)
module purge
module load anaconda3/2024.10
module load cudatoolkit/12.6
conda activate /scratch/network/lo8603/thesis/conda/envs/myenv

echo "=============================="
echo "NAE Autoresearch"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================="
nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Navigate to autoresearch directory
cd /scratch/network/lo8603/thesis/fast-ad/autoresearch-nae

# Your paths
DATA_ROOT="/scratch/network/lo8603/thesis/fast-ad/data/h5_files/"
PRETRAINED="/scratch/network/lo8603/thesis/fast-ad/outputs/ae_zb_npv_geq10_dim80/model_best.pkl"

# Init git for experiment tracking
if [ ! -d .git ]; then
    git init
    git add train.py evaluate.py program.md
    git commit -m "Initial autoresearch setup"
fi

# Run baseline
echo ">>> Running baseline..."
python evaluate.py \
    --dataset CICADA \
    --holdout-class "1,2,3,4,5,6,7,8,9,10" \
    --pretrained-path $PRETRAINED \
    --data-root $DATA_ROOT \
    --time-budget 600 \
    --verbose

echo "Baseline done. GPU allocated on $(hostname)."
echo "SSH in and run your coding agent (claude, cursor, etc)."

# Keep GPU alive for interactive agent use
sleep infinity