#!/bin/bash
# Phase 1: Launch all 5 supervised architecture baselines in parallel.
#
# Submits BQ-128, BQ-192, LEHD-128, LEHD-192, LEHD-256 as independent SLURM jobs.
# Each runs on 2.5k ILP-labeled instances (60-80n substrates, 10-20 requests).
#
# PREREQUISITE: Split checkpoint dataset first.
#   python scripts/vne_split_checkpoint.py --train 2500 --val 500 --test 1000
#
# Usage:
#   bash scripts/vne_phase1_launch.sh
#
# GPU partition strategy:
#   gnode06 (2×L40S 48GB):
#     GPU 0: BQ-128 (~8 GiB) + LEHD-128 (~15 GiB) = ~23 GiB / 48 GiB
#     GPU 1: BQ-192 (~14 GiB) + LEHD-192 (~20 GiB) = ~34 GiB / 48 GiB
#   gpu_HIGH (A100 80GB):
#     GPU 0: LEHD-256 (~24 GiB) = ~24 GiB / 80 GiB
#   Distinct seeds per architecture (100, 200, 300, 400, 500).
set -euo pipefail

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"
cd "$REPO"
mkdir -p logs

# Common settings
EPOCHS=30
GPU_MEM="0.50"       # 50% of GPU via MPS (only user on gnode06)
TRAIN_PATH="data/vne/vne_supervised_training_dataset_4k.pickle"
VAL_PATH="data/vne/vne_validation_dataset_1k.pickle"
TEST_PATH="data/vne/vne_test_dataset_2k.pickle"

echo "================================================================"
echo " Phase 1: Supervised Architecture Baselines"
echo " Train: $TRAIN_PATH (2.5k ILP-labeled, 60-80n)"
echo " Epochs: $EPOCHS | GPU mem: ${GPU_MEM} | all batch_size=16 (LEHD-256=32)"
echo " GPUs:  gnode06 GPU0 (BQ-128+LEHD-128) GPU1 (BQ-192+LEHD-192)"
echo "        gpu_HIGH GPU0 (LEHD-256 on A100)"
echo "================================================================"
echo ""

declare -A JOB_IDS

# Helper: build a comma-separated --export string.
# Usage: make_export <arch> <dim> <heads> <ff> <tag> <seed> <gpu> [extra] [gpu_mem]
make_export() {
    local arch="$1" dim="$2" heads="$3" ff="$4" tag="$5" seed="$6" gpu="$7"
    local extra="${8:-}"
    local gpu_mem="${9:-$GPU_MEM}"
    echo "ALL,VNE_ARCHITECTURE=${arch},VNE_EMBEDDING_DIM=${dim},VNE_NUM_HEADS=${heads},VNE_FF_DIM=${ff},VNE_LEARNING_TYPE=supervised,VNE_NUM_EPOCHS=${EPOCHS},VNE_GPU_MEM_FRACTION=${gpu_mem},VNE_TRAINING_SET_PATH=${TRAIN_PATH},VNE_VALIDATION_SET_PATH=${VAL_PATH},VNE_TEST_SET_PATH=${TEST_PATH},VNE_RUN_TAG=${tag},VNE_SEED=${seed},VNE_GPU_DEVICE=${gpu}${extra}"
}

# ---- BQ-128 (9 blocks, dim=128, heads=8, ff=512, ~1.78M params) ----
# GPU 0 on gnode06 L40S. Shares with LEHD-128.
echo ">>> BQ-128 (seed=100, GPU 0)"
EXP=$(make_export "bq" "128" "8" "512" "phase1_bq128" "100" "0" ",VNE_BATCH_SIZE=16")
JOB_BQ128=$(sbatch --parsable --partition=gpuISIN --exclude=gnode05 --gres=mps:30 --job-name=bq128 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["BQ-128"]=$JOB_BQ128
echo "  Job: $JOB_BQ128"

# ---- BQ-192 (9 blocks, dim=192, heads=12, ff=768, ~4.00M params) ----
# GPU 1 on gnode06 L40S. Shares with LEHD-192.
echo ">>> BQ-192 (seed=200, GPU 1)"
EXP=$(make_export "bq" "192" "12" "768" "phase1_bq192" "200" "1" ",VNE_BATCH_SIZE=16")
JOB_BQ192=$(sbatch --parsable --partition=gpuISIN --exclude=gnode05 --gres=mps:30 --job-name=bq192 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["BQ-192"]=$JOB_BQ192
echo "  Job: $JOB_BQ192"

# ---- LEHD-128 (6e+6d, dim=128, heads=8, ff=512, ~2.43M params) ----
# GPU 0 on gnode06 L40S. Shares with BQ-128.
echo ">>> LEHD-128 (seed=300, GPU 0)"
EXP=$(make_export "lehd" "128" "8" "512" "phase1_lehd128" "300" "0" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6,VNE_BATCH_SIZE=16")
JOB_L128=$(sbatch --parsable --partition=gpuISIN --exclude=gnode05 --gres=mps:30 --job-name=lehd128 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["LEHD-128"]=$JOB_L128
echo "  Job: $JOB_L128"

# ---- LEHD-192 (6e+6d, dim=192, heads=12, ff=768, ~5.45M params) ----
# GPU 1 on gnode06 L40S. Shares with BQ-192.
echo ">>> LEHD-192 (seed=400, GPU 1)"
EXP=$(make_export "lehd" "192" "12" "768" "phase1_lehd192" "400" "1" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6,VNE_BATCH_SIZE=16")
JOB_L192=$(sbatch --parsable --partition=gpuISIN --exclude=gnode05 --gres=mps:30 --job-name=lehd192 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["LEHD-192"]=$JOB_L192
echo "  Job: $JOB_L192"

# ---- LEHD-256 (6e+6d, dim=256, heads=8, ff=1024, ~9.67M params) ----
# A100 (80GB) on gpu_HIGH. Single job, higher mem fraction.
echo ">>> LEHD-256 (seed=500, A100, gpu_mem=0.40)"
EXP=$(make_export "lehd" "256" "8" "1024" "phase1_lehd256" "500" "0" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6,VNE_BATCH_SIZE=32" "0.40")
JOB_L256=$(sbatch --parsable --partition=gpu_HIGH --gres=mps:a100:50 --job-name=lehd256 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["LEHD-256"]=$JOB_L256
echo "  Job: $JOB_L256"

echo ""
echo "================================================================"
echo " Phase 1 — All jobs submitted:"
echo "================================================================"
for name in BQ-128 BQ-192 LEHD-128 LEHD-192 LEHD-256; do
    echo "  $name: ${JOB_IDS[$name]}"
done
echo ""
echo "Monitor:  squeue -j ${JOB_IDS[BQ-128]},${JOB_IDS[BQ-192]},${JOB_IDS[LEHD-128]},${JOB_IDS[LEHD-192]},${JOB_IDS[LEHD-256]}"
echo "Logs:     ls logs/vne_train_phase1_*"
echo "Results:  ls model_checkpoints/vne/results/phase1_*/"
