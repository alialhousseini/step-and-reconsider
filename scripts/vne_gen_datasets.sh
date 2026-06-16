#!/bin/bash
# Submit all VNE dataset generation jobs for Phase 1 (new 60-80n, 10-20rq scale).
#
# Training:   10 shards x 1,000 = 10,000 instances (seed band 1,000,000)
# Validation:  1 shard  x 1,000 =  1,000 instances (seed band 2,000,000)
# Test (ID):   2 shards x 1,000 =  2,000 instances (seed band 3,000,000)
#
# Usage:
#   bash scripts/vne_gen_datasets.sh
#
# After all arrays complete, merge with:
#   .venv/bin/python scripts/vne_merge_shards.py \
#       --shard-glob "data/vne/shards/train_shard*.pickle" \
#       --out data/vne/vne_supervised_training_dataset_10k.pickle
#   .venv/bin/python scripts/vne_merge_shards.py \
#       --shard-glob "data/vne/shards/val_shard*.pickle" \
#       --out data/vne/vne_validation_dataset_1k.pickle
#   .venv/bin/python scripts/vne_merge_shards.py \
#       --shard-glob "data/vne/shards/test_shard*.pickle" \
#       --out data/vne/vne_test_dataset_2k.pickle
set -euo pipefail

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"
cd "$REPO"
mkdir -p logs data/vne/shards

# Clean shard directory from previous runs
echo "Cleaning shard directory..."
rm -f data/vne/shards/train_shard*.pickle data/vne/shards/val_shard*.pickle data/vne/shards/test_shard*.pickle

# Common settings for 60-80n, 10-20rq scale (from vne/config.py)
# --time-limit 120 is generous for the ID scale; HiGHS solves most in 5-60s
SOLVER="highs"
TIME_LIMIT=120
MIP_GAP=0.0

# Build export string once to avoid line-continuation issues
EXPORT_TRAIN="ALL,VNE_SPLIT=train,VNE_NUM_PER_SHARD=1000,VNE_SEED_BASE=1000000,VNE_SOLVER=${SOLVER},VNE_TIME_LIMIT=${TIME_LIMIT},VNE_MIP_GAP=${MIP_GAP}"
EXPORT_VAL="ALL,VNE_SPLIT=val,VNE_NUM_PER_SHARD=1000,VNE_SEED_BASE=2000000,VNE_SOLVER=${SOLVER},VNE_TIME_LIMIT=${TIME_LIMIT},VNE_MIP_GAP=${MIP_GAP}"
EXPORT_TEST="ALL,VNE_SPLIT=test,VNE_NUM_PER_SHARD=1000,VNE_SEED_BASE=3000000,VNE_SOLVER=${SOLVER},VNE_TIME_LIMIT=${TIME_LIMIT},VNE_MIP_GAP=${MIP_GAP}"

echo "=== Submitting TRAINING set: 10 shards x 1,000 = 10,000 ==="
JOB_TRAIN=$(sbatch --parsable --array=0-9 --export="${EXPORT_TRAIN}" scripts/vne_gen_array.sbatch)
echo "  Training job: $JOB_TRAIN"

echo "=== Submitting VALIDATION set: 1 shard x 1,000 = 1,000 ==="
JOB_VAL=$(sbatch --parsable --array=0-0 --export="${EXPORT_VAL}" scripts/vne_gen_array.sbatch)
echo "  Validation job: $JOB_VAL"

echo "=== Submitting TEST set: 2 shards x 1,000 = 2,000 ==="
JOB_TEST=$(sbatch --parsable --array=0-1 --export="${EXPORT_TEST}" scripts/vne_gen_array.sbatch)
echo "  Test job: $JOB_TEST"

echo ""
echo "=== Submitted ==="
echo "Train:     $JOB_TRAIN (array 0-9)"
echo "Val:       $JOB_VAL  (array 0)"
echo "Test:      $JOB_TEST (array 0-1)"
echo ""
echo "Check status:  squeue -j $JOB_TRAIN,$JOB_VAL,$JOB_TEST"
echo "Merge when all complete:"
echo "  .venv/bin/python scripts/vne_merge_shards.py --shard-glob 'data/vne/shards/train_shard*.pickle' --out data/vne/vne_supervised_training_dataset_10k.pickle"
echo "  .venv/bin/python scripts/vne_merge_shards.py --shard-glob 'data/vne/shards/val_shard*.pickle' --out data/vne/vne_validation_dataset_1k.pickle"
echo "  .venv/bin/python scripts/vne_merge_shards.py --shard-glob 'data/vne/shards/test_shard*.pickle' --out data/vne/vne_test_dataset_2k.pickle"
