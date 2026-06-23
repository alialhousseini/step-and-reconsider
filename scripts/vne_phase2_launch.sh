#!/bin/bash
# Phase 2: TaSaR SIL k×s grid on best Phase 1 architectures (BQ-128 + LEHD-128).
#
# Pilot first (2 runs × 3 epochs), then full grid (12-24 runs × 15 epochs).
#
# Usage:
#   bash scripts/vne_phase2_launch.sh          # pilot
#   bash scripts/vne_phase2_launch.sh --full   # full grid
set -euo pipefail

REPO="${VNE_REPO:-/mnt/beegfs/scratch/ali.alhousseini/step-and-reconsider}"
cd "$REPO"
mkdir -p logs

MODE="${1:-pilot}"
EPOCHS=3
if [[ "$MODE" == "--full" ]]; then
    EPOCHS=15
fi

# Phase 1 checkpoints
BQ128_CKPT="$REPO/model_checkpoints/vne/results/phase1_bq128_42476/last_model.pt"
L128_CKPT="$REPO/model_checkpoints/vne/results/phase1_lehd128_42487/last_model.pt"

echo "=================================================================="
echo " Phase 2: TaSaR SIL — ${MODE} mode (${EPOCHS} Gumbeldore epochs)"
echo "=================================================================="
echo " BQ-128 ckpt: $BQ128_CKPT"
echo " LEHD-128 ckpt: $L128_CKPT"
echo ""

make_export() {
    local arch="$1" dim="$2" heads="$3" ff="$4" tag="$5" seed="$6"
    local k="$7" s="$8" gpu="$9" ckpt="${10}" extra="${11:-}"
    echo "ALL,VNE_ARCHITECTURE=${arch},VNE_EMBEDDING_DIM=${dim},VNE_NUM_HEADS=${heads},VNE_FF_DIM=${ff},VNE_LEARNING_TYPE=gumbeldore,VNE_SEARCH_TYPE=tasar,VNE_NUM_EPOCHS=${EPOCHS},VNE_GPU_MEM_FRACTION=0.50,VNE_RUN_TAG=${tag},VNE_SEED=${seed},VNE_GPU_DEVICE=${gpu},VNE_BEAM_WIDTH=${k},VNE_REPLAN_STEPS=${s},VNE_LOAD_CHECKPOINT_FROM_PATH=${ckpt},VNE_NUM_GENERATE=256,VNE_NUM_CPU_WORKERS=8,VNE_LR=2e-4${extra}"
}

declare -A JOB_IDS

if [[ "$MODE" == "pilot" ]]; then
    # ---- Pilot: BQ-128, k=32, s=4 ----
    echo ">>> BQ-128 SIL pilot: k=32, s=4 (A100)"
    EXP=$(make_export "bq" "128" "8" "512" "phase2_bq128_k32_s4" "600" "32" "4" "0" "$BQ128_CKPT" ",VNE_BATCH_SIZE=16")
    JOB=$(sbatch --parsable --partition=gpu --gres=mps:a100:50 --job-name=sil_bq128 --export="${EXP}" scripts/vne_train.sbatch)
    JOB_IDS["BQ-128_k32_s4"]=$JOB
    echo "  Job: $JOB"

    # ---- Pilot: LEHD-128, k=32, s=4 ----
    echo ">>> LEHD-128 SIL pilot: k=32, s=4 (GPU 0, A100)"
    EXP=$(make_export "lehd" "128" "8" "512" "phase2_lehd128_k32_s4" "700" "32" "4" "0" "$L128_CKPT" ",VNE_BATCH_SIZE=16,VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6")
    JOB=$(sbatch --parsable --partition=gpu --gres=mps:a100:50 --job-name=sil_l128 --export="${EXP}" scripts/vne_train.sbatch)
    JOB_IDS["LEHD-128_k32_s4"]=$JOB
    echo "  Job: $JOB"

elif [[ "$MODE" == "--full" ]]; then
    # Full k × s grid: k ∈ {16,32,64} × s ∈ {4,8,16}
    # BQ-128: all 9 cells, LEHD-128: key 6 cells
    KS_LIST=("16 4" "16 8" "16 16" "32 4" "32 8" "32 16" "64 4" "64 8" "64 16")
    SEED_BASE=600
    for KS in "${KS_LIST[@]}"; do
        k=$(echo "$KS" | awk '{print $1}')
        s=$(echo "$KS" | awk '{print $2}')
        seed=$((SEED_BASE + k * 10 + s))
        tag="phase2_bq128_k${k}_s${s}"

        echo ">>> BQ-128: k=$k, s=$s"
        EXP=$(make_export "bq" "128" "8" "512" "$tag" "$seed" "$k" "$s" "0" "$BQ128_CKPT" ",VNE_BATCH_SIZE=16")
        JOB=$(sbatch --parsable --partition=gpu --gres=mps:a100:50 --job-name="bq_${k}_${s}" --export="${EXP}" scripts/vne_train.sbatch)
        JOB_IDS["BQ_k${k}_s${s}"]=$JOB
        echo "  Job: $JOB"
    done

    # LEHD-128: sparse grid — fewer runs since it's slower
    LEHD_KS=("16 4" "16 8" "32 4" "32 8" "64 4" "64 8")
    for KS in "${LEHD_KS[@]}"; do
        k=$(echo "$KS" | awk '{print $1}')
        s=$(echo "$KS" | awk '{print $2}')
        seed=$((700 + k * 10 + s))
        tag="phase2_lehd128_k${k}_s${s}"

        echo ">>> LEHD-128: k=$k, s=$s"
        EXP=$(make_export "lehd" "128" "8" "512" "$tag" "$seed" "$k" "$s" "0" "$L128_CKPT" ",VNE_BATCH_SIZE=16,VNE_NUM_ENCODER_LAYERS=6,VNE_NUM_DECODER_LAYERS=6")
        JOB=$(sbatch --parsable --partition=gpu --gres=mps:a100:50 --job-name="lhd_${k}_${s}" --export="${EXP}" scripts/vne_train.sbatch)
        JOB_IDS["LEHD_k${k}_s${s}"]=$JOB
        echo "  Job: $JOB"
    done
fi

echo ""
echo "=================================================================="
echo " Phase 2 (${MODE}) — Submitted:"
echo "=================================================================="
for name in "${!JOB_IDS[@]}"; do
    echo "  $name: ${JOB_IDS[$name]}"
done
echo ""
echo "Monitor: squeue -j ${JOB_IDS[*]}"
echo "Logs:    ls logs/vne_train_phase2_*"
