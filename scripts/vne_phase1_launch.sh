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
#   - gpuISIN / gpuISIN_HIGH (A30 24GB / L40S 48GB) with mps:30 → 4 concurrent jobs
#   - gpu_HIGH (A100 80GB) with mps:a100:50 → LEHD-256 (~9.7M params)
#   Distinct seeds per architecture for statistical validity (100, 200, 300, 400, 500).
set -euo pipefail

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"
cd "$REPO"
mkdir -p logs

# Common settings
EPOCHS=30
GPU_MEM="0.50"       # 50% of GPU via MPS (single user on gnode06, 48GB → 24GB)
TRAIN_PATH="data/vne/vne_supervised_training_dataset_4k.pickle"
VAL_PATH="data/vne/vne_validation_dataset_1k.pickle"
TEST_PATH="data/vne/vne_test_dataset_2k.pickle"

echo "================================================================"
echo " Phase 1: Supervised Architecture Baselines"
echo " Train: $TRAIN_PATH (2.5k ILP-labeled, 60-80n)"
echo " Epochs: $EPOCHS | GPU mem: ${GPU_MEM} | BQ batch=32, LEHD batch=128/64"
echo "================================================================"
echo ""

declare -A JOB_IDS

# Helper: build a comma-separated --export string.
# Usage: make_export <arch> <dim> <heads> <ff> <tag> <seed> [extra] [gpu_mem]
make_export() {
    local arch="$1" dim="$2" heads="$3" ff="$4" tag="$5" seed="$6"
    local extra="${7:-}"
    local gpu_mem="${8:-$GPU_MEM}"
    echo "ALL,VNE_ARCHITECTURE=${arch},VNE_EMBEDDING_DIM=${dim},VNE_NUM_HEADS=${heads},VNE_FF_DIM=${ff},VNE_LEARNING_TYPE=supervised,VNE_NUM_EPOCHS=${EPOCHS},VNE_GPU_MEM_FRACTION=${gpu_mem},VNE_TRAINING_SET_PATH=${TRAIN_PATH},VNE_VALIDATION_SET_PATH=${VAL_PATH},VNE_TEST_SET_PATH=${TEST_PATH},VNE_RUN_TAG=${tag},VNE_SEED=${seed}${extra}"
}

# ---- BQ-128 (9 blocks, dim=128, heads=8, ff=512, ~1.78M params) ----
# Reduced batch_size=32: BQ's unified attention over all tokens is O(seq_len²)
echo ">>> BQ-128 (seed=100)"
EXP=$(make_export "bq" "128" "8" "512" "phase1_bq128" "100" ",VNE_BATCH_SIZE=32")
JOB_BQ128=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=bq128 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["BQ-128"]=$JOB_BQ128
echo "  Job: $JOB_BQ128"

# ---- BQ-192 (9 blocks, dim=192, heads=12, ff=768, ~4.00M params) ----
echo ">>> BQ-192 (seed=200)"
EXP=$(make_export "bq" "192" "12" "768" "phase1_bq192" "200" ",VNE_BATCH_SIZE=32")
JOB_BQ192=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=bq192 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["BQ-192"]=$JOB_BQ192
echo "  Job: $JOB_BQ192"

# ---- LEHD-128 (6e+6d, dim=128, heads=8, ff=512, ~2.43M params) ----
echo ">>> LEHD-128 (seed=300)"
EXP=$(make_export "lehd" "128" "8" "512" "phase1_lehd128" "300" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6")
JOB_L128=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=lehd128 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["LEHD-128"]=$JOB_L128
echo "  Job: $JOB_L128"

# ---- LEHD-192 (6e+6d, dim=192, heads=12, ff=768, ~5.45M params) ----
echo ">>> LEHD-192 (seed=400)"
EXP=$(make_export "lehd" "192" "12" "768" "phase1_lehd192" "400" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6")
JOB_L192=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=lehd192 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["LEHD-192"]=$JOB_L192
echo "  Job: $JOB_L192"

# ---- LEHD-256 (6e+6d, dim=256, heads=8, ff=1024, ~9.67M params) ----
# Pinned to A100 (80GB) via qualified MPS type; higher mem fraction for the larger model.
echo ">>> LEHD-256 (seed=500, gpu_mem=0.40, pinned to A100)"
EXP=$(make_export "lehd" "256" "8" "1024" "phase1_lehd256" "500" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6,VNE_BATCH_SIZE=64" "0.40")
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
