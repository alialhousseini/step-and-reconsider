#!/bin/bash
# GPU training run for VNE, sharing the GPU via MPS (uses a portion, not the
# whole card, so other cluster users can co-run).
#
# Submit (partition + MPS share are passed at submit time), e.g.:
#   sbatch --partition=gpuISIN --gres=mps:30 \
#     --export=ALL,VNE_RUN_TAG=scaled50k,VNE_CUSTOM_NUM_INSTANCES=,VNE_NUM_EPOCHS=40 \
#     scripts/vne_train.sbatch
#
#SBATCH --job-name=vne_train
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --output=logs/vne_train_%j.out
#SBATCH --error=logs/vne_train_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Avoid fragmentation under memory-fraction caps — grow segments incrementally.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Pin to a specific GPU on multi-GPU nodes (set via VNE_GPU_DEVICE env var).
# Must be set BEFORE PyTorch/CUDA init, so we export it as the real env var.
export CUDA_VISIBLE_DEVICES="${VNE_GPU_DEVICE:-0}"

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"
cd "$REPO"; mkdir -p logs

VENV="${VNE_VENV:-$REPO/.venv}"; PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then echo "venv python not found at $PY" >&2; exit 1; fi

# Polite sharing: cap our slice of GPU memory (model is tiny; leaves room for others).
export VNE_GPU_MEM_FRACTION="${VNE_GPU_MEM_FRACTION:-0.15}"

# Isolate Ray per-job so multiple jobs on the same node don't conflict on ports/tmp.
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

RUN_TAG="${VNE_RUN_TAG:-run}"
export VNE_RESULTS_PATH="${VNE_RESULTS_PATH:-$REPO/model_checkpoints/vne/results/${RUN_TAG}_${SLURM_JOB_ID:-local}}"

# Pass through network/training overrides (read by vne_main.py::_apply_env_overrides).
export VNE_ARCHITECTURE="${VNE_ARCHITECTURE:-}"
export VNE_EMBEDDING_DIM="${VNE_EMBEDDING_DIM:-}"
export VNE_HIDDEN_DIM="${VNE_HIDDEN_DIM:-}"
export VNE_NUM_HEADS="${VNE_NUM_HEADS:-}"
export VNE_NUM_DECODER_LAYERS="${VNE_NUM_DECODER_LAYERS:-}"
export VNE_NUM_ENCODER_LAYERS="${VNE_NUM_ENCODER_LAYERS:-}"
export VNE_FF_DIM="${VNE_FF_DIM:-}"
export VNE_LR="${VNE_LR:-}"
export VNE_LEARNING_TYPE="${VNE_LEARNING_TYPE:-}"
export VNE_BEAM_WIDTH="${VNE_BEAM_WIDTH:-}"
export VNE_REPLAN_STEPS="${VNE_REPLAN_STEPS:-}"
export VNE_NUM_GENERATE="${VNE_NUM_GENERATE:-}"
export VNE_NUM_CPU_WORKERS="${VNE_NUM_CPU_WORKERS:-}"
export VNE_CPU_BATCH_SIZE="${VNE_CPU_BATCH_SIZE:-}"
export VNE_EVAL_WORKERS="${VNE_EVAL_WORKERS:-}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-NA} tag=$RUN_TAG"
"$PY" -c "import torch;print('cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
echo "results -> $VNE_RESULTS_PATH  | train_instances=${VNE_CUSTOM_NUM_INSTANCES:-ALL} epochs=${VNE_NUM_EPOCHS:-config}"

LOGFILE="logs/vne_train_${RUN_TAG}_${SLURM_JOB_ID:-local}.log"
stdbuf -oL -eL "$PY" -u vne_main.py 2>&1 | tee -a "$LOGFILE"
echo "Done. results: $VNE_RESULTS_PATH | log: $LOGFILE"
