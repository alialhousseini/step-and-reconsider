#!/bin/bash
# Phase 1: Launch all 5 supervised architecture baselines in parallel.
#
# Submits BQ-128, BQ-192, LEHD-128, LEHD-192, LEHD-256 as independent SLURM jobs.
# Each runs on 10k ILP-labeled instances (60-80n substrates, 10-20 requests).
#
# PREREQUISITE: Datasets must be generated and merged first.
#   bash scripts/vne_gen_datasets.sh   # then merge when complete
#
# Usage:
#   bash scripts/vne_phase1_launch.sh
#
# GPU partition strategy:
#   - gpuISIN (A30/L40S, 24GB) with mps:30 → 4 concurrent jobs (BQ-128/192, LEHD-128/192)
#   - gpu_HIGH (A100, 80GB) with mps:50 → 1 job (LEHD-256, ~9.7M params needs more memory)
#   Adjust --partition and --gres below as needed for available resources.
set -euo pipefail

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"
cd "$REPO"
mkdir -p logs

# Common settings
EPOCHS=30
GPU_MEM="0.30"       # 30% of GPU via MPS (share with other jobs)
TRAIN_PATH="data/vne/vne_supervised_training_dataset_10k.pickle"
VAL_PATH="data/vne/vne_validation_dataset_1k.pickle"
TEST_PATH="data/vne/vne_test_dataset_2k.pickle"

echo "================================================================"
echo " Phase 1: Supervised Architecture Baselines"
echo " Train: $TRAIN_PATH (10k ILP-labeled, 60-80n)"
echo " Epochs: $EPOCHS | GPU mem: ${GPU_MEM}"
echo "================================================================"
echo ""

declare -A JOB_IDS

# Helper: build a comma-separated --export string
make_export() {
    local arch="$1" dim="$2" heads="$3" ff="$4" tag="$5"
    local extra="${6:-}"
    echo "ALL,VNE_ARCHITECTURE=${arch},VNE_EMBEDDING_DIM=${dim},VNE_NUM_HEADS=${heads},VNE_FF_DIM=${ff},VNE_LEARNING_TYPE=supervised,VNE_NUM_EPOCHS=${EPOCHS},VNE_GPU_MEM_FRACTION=${GPU_MEM},VNE_TRAINING_SET_PATH=${TRAIN_PATH},VNE_VALIDATION_SET_PATH=${VAL_PATH},VNE_TEST_SET_PATH=${TEST_PATH},VNE_RUN_TAG=${tag}${extra}"
}

# ---- BQ-128 (9 blocks, dim=128, heads=8, ff=512, ~1.78M params) ----
echo ">>> BQ-128"
EXP=$(make_export "bq" "128" "8" "512" "phase1_bq128")
JOB_BQ128=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=bq128 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["BQ-128"]=$JOB_BQ128
echo "  Job: $JOB_BQ128"

# ---- BQ-192 (9 blocks, dim=192, heads=12, ff=768, ~4.00M params) ----
echo ">>> BQ-192"
EXP=$(make_export "bq" "192" "12" "768" "phase1_bq192")
JOB_BQ192=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=bq192 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["BQ-192"]=$JOB_BQ192
echo "  Job: $JOB_BQ192"

# ---- LEHD-128 (6e+6d, dim=128, heads=8, ff=512, ~2.43M params) ----
echo ">>> LEHD-128"
EXP=$(make_export "lehd" "128" "8" "512" "phase1_lehd128" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6")
JOB_L128=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=lehd128 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["LEHD-128"]=$JOB_L128
echo "  Job: $JOB_L128"

# ---- LEHD-192 (6e+6d, dim=192, heads=12, ff=768, ~5.45M params) ----
echo ">>> LEHD-192"
EXP=$(make_export "lehd" "192" "12" "768" "phase1_lehd192" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6")
JOB_L192=$(sbatch --parsable --partition=gpuISIN --gres=mps:30 --job-name=lehd192 --export="${EXP}" scripts/vne_train.sbatch)
JOB_IDS["LEHD-192"]=$JOB_L192
echo "  Job: $JOB_L192"

# ---- LEHD-256 (6e+6d, dim=256, heads=8, ff=1024, ~9.67M params) ----
echo ">>> LEHD-256"
EXP=$(make_export "lehd" "256" "8" "1024" "phase1_lehd256" ",VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6,VNE_GPU_MEM_FRACTION=0.40")
JOB_L256=$(sbatch --parsable --partition=gpu_HIGH --gres=mps:50 --job-name=lehd256 --export="${EXP}" scripts/vne_train.sbatch)
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
