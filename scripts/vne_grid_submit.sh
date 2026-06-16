#!/bin/bash
# Submit the VNE Gumbeldore/TaSaR grid search.
# Grid: beam_width ∈ {8, 16, 32, 64} × replan_steps ∈ {2, 4, 8}
#
# Usage: bash scripts/vne_grid_submit.sh [--dry-run]
#   --dry-run  Print commands without submitting.

set -euo pipefail
DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=true; fi

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"

# Spread across available partitions. With MPS, multiple jobs share each GPU.
# gpu: A100×2, V100×2 → 4 jobs
# gpuISIN: A30, L40S×2 → 4 jobs
# gpu_AMD: H200×2, RTX PRO 6000×4 → 4 jobs
PARTITIONS=("gpu" "gpu" "gpu" "gpu" "gpuISIN" "gpuISIN" "gpuISIN" "gpuISIN" "gpu_AMD" "gpu_AMD" "gpu_AMD" "gpu_AMD")
GRES=("mps:30" "mps:30" "mps:30" "mps:30" "mps:30" "mps:30" "mps:30" "mps:30" "mps:25" "mps:25" "mps:25" "mps:25")

BEAM_WIDTHS=(8 16 32 64)
REPLAN_STEPS=(2 4 8)
NUM_EPOCHS=10
NUM_GENERATE=64
NUM_CPU_WORKERS=4
CPU_BATCH_SIZE=8
TRAIN_BATCH_SIZE=32
EVAL_WORKERS=4
GPU_MEM_FRACTION=0.30

echo "=== VNE TaSaR Grid Submission ==="
echo "Beam widths: ${BEAM_WIDTHS[*]}"
echo "Replan steps: ${REPLAN_STEPS[*]}"
echo "Total runs: $(( ${#BEAM_WIDTHS[@]} * ${#REPLAN_STEPS[@]} ))"
echo "Epochs/run: $NUM_EPOCHS"
echo "Instances/epoch: $NUM_GENERATE"
echo ""

job_idx=0
total=$(( ${#BEAM_WIDTHS[@]} * ${#REPLAN_STEPS[@]} ))

for bw in "${BEAM_WIDTHS[@]}"; do
  for rs in "${REPLAN_STEPS[@]}"; do
    tag="gumbeldore_b${bw}_s${rs}"
    partition="${PARTITIONS[$job_idx]}"
    gres="${GRES[$job_idx]}"

    cmd="sbatch --partition=$partition --gres=$gres \
      --cpus-per-task=$((NUM_CPU_WORKERS * 2)) \
      --export=ALL,\
VNE_RUN_TAG=$tag,\
VNE_LEARNING_TYPE=gumbeldore,\
VNE_BEAM_WIDTH=$bw,\
VNE_REPLAN_STEPS=$rs,\
VNE_NUM_EPOCHS=$NUM_EPOCHS,\
VNE_NUM_GENERATE=$NUM_GENERATE,\
VNE_NUM_CPU_WORKERS=$NUM_CPU_WORKERS,\
VNE_CPU_BATCH_SIZE=$CPU_BATCH_SIZE,\
VNE_BATCH_SIZE=$TRAIN_BATCH_SIZE,\
VNE_EVAL_WORKERS=$EVAL_WORKERS,\
VNE_GPU_MEM_FRACTION=$GPU_MEM_FRACTION \
      $REPO/scripts/vne_train.sbatch"

    echo "[$((job_idx + 1))/$total] $tag → $partition ($gres)"
    if $DRY_RUN; then
      echo "  DRY: $cmd"
    else
      eval "$cmd"
      sleep 1  # Don't flood the scheduler
    fi
    job_idx=$((job_idx + 1))
  done
done

echo ""
echo "Submitted $total jobs. Monitor with: squeue -u \$USER"
echo "Logs: $REPO/logs/vne_train_gumbeldore_b*_s*_*.log"
