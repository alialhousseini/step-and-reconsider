#!/bin/bash
# Parallel VNE dataset generation as a SLURM job array (CPU / ILP-only).
#
# Each array task solves one shard with the shared venv and writes a pickle into
# $VNE_OUT_DIR. After the array finishes, merge with:
#   .venv/bin/python scripts/vne_merge_shards.py \
#       --shard-glob "$VNE_OUT_DIR/${VNE_SPLIT}_shard*.pickle" --out <final.pickle>
#
# Submit (50 tasks x 1000 = 50k train, default):
#   sbatch scripts/vne_gen_array.sbatch
# Override knobs at submit time, e.g.:
#   sbatch --array=0-49 --export=ALL,VNE_SPLIT=train,VNE_NUM_PER_SHARD=1000,\
#VNE_SEED_BASE=1000000,VNE_SOLVER=auto,VNE_TIME_LIMIT=60 scripts/vne_gen_array.sbatch
#
#SBATCH --job-name=vne_gen
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --array=0-49
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/vne_gen_%A_%a.out
#SBATCH --error=logs/vne_gen_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"
cd "$REPO"
mkdir -p logs

VENV="${VNE_VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "venv python not found at $PY" >&2
  exit 1
fi

# Defaults (overridable via --export); the shard script also reads these envs.
export VNE_SPLIT="${VNE_SPLIT:-train}"
export VNE_NUM_PER_SHARD="${VNE_NUM_PER_SHARD:-1000}"
export VNE_SEED_BASE="${VNE_SEED_BASE:-1000000}"
export VNE_SOLVER="${VNE_SOLVER:-auto}"
export VNE_TIME_LIMIT="${VNE_TIME_LIMIT:-60}"
export VNE_OUT_DIR="${VNE_OUT_DIR:-$REPO/data/vne/shards}"
export VNE_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "host=$(hostname) array_task=${SLURM_ARRAY_TASK_ID:-NA} split=$VNE_SPLIT n=$VNE_NUM_PER_SHARD solver=$VNE_SOLVER"

srun "$PY" scripts/vne_gen_shard.py
