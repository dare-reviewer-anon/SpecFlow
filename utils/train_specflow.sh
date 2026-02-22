#!/bin/bash
#SBATCH --job-name=specflow-qwen-a100-4g
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --exclusive
#SBATCH --mail-type=BEGIN,END
# Cluster: <CLUSTER_PROVIDER>

set -euo pipefail

mkdir -p logs

#############################
# 1. Load modules
#############################
module load 2023
module load CUDA/12.4.0
module load Miniconda3/23.5.2-0

#############################
# 2. Environment setup
#############################
export PATH=/home/<USER>/.conda/envs/DARE/bin:$PATH
export PYTHONNOUSERSITE=1

export HF_HOME=<FS_ROOT>/hfcache
export HF_DATASETS_TRUST_REMOTE_CODE=1
export HF_HUB_DISABLE_TELEMETRY=1

export WANDB_MODE=disabled
# export WANDB_API_KEY=YOUR_KEY

export TRITON_CACHE_DIR=<FS_ROOT>/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR"

export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

#############################
# 3. Move to project root
#############################
cd <FS_ROOT>/DARE
mkdir -p logs

echo "=== Environment check ==="
which python
python --version
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'gpus:', torch.cuda.device_count())"
which torchrun || true
echo "========================="

#############################
# 4. Training (4 GPUs)
#############################
echo "[$(date)] Starting torchrun..."

torchrun --nproc_per_node=4 train_specflow.py \
  --model qwen \
  --data interleaved_maze \
  --data_dir <FS_ROOT>/data-samples \
  --decoder_type qwen \
  --input_format qwen \
  --do_train \
  --do_eval \
  --cfg_path cfg \
  --output outputs/specflow-qwen-maze \
  --note "specflow-maze-image_seq_len-1024-" \
  --image_seq_length 1024 \
  --report_to "none" \
  --train_bz 2 \
  --val_bz 2 \
  --grad_acc 32 \
  --enable_specflow \
  --model_ckpt <FS_ROOT>/DARE/outputs/qwen_zero3_4gpusoutput \
  --load_last_checkpoint