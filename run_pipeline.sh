#!/usr/bin/env bash
# =============================================================================
# Full Pipeline: Raw WAV → Mel-GADF → PCNN → Train/Val/Test Split
# =============================================================================
# Automates the complete data preparation and training pipeline for the
# PINN-based transformer fault diagnosis framework.
#
# Usage:
#   bash run_pipeline.sh                          # Full pipeline
#   bash run_pipeline.sh --skip-build             # Skip dataset building
#   bash run_pipeline.sh --skip-train             # Only prepare data
#   bash run_pipeline.sh --quick                  # Quick test (5 epochs)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_DIR="${RAW_DIR:-raw-data}"
FUSED_DIR="${FUSED_DIR:-datasets/mel_gadf_fused}"
PCNN_DIR="${PCNN_DIR:-datasets/mel_gadf_pcnn}"
GAMMA="${GAMMA:-1.7}"
IMG_SIZE="${IMG_SIZE:-224}"
PCNN_STEPS="${PCNN_STEPS:-10}"
PCNN_VT="${PCNN_VT:-0.8}"
SEED="${SEED:-42}"

SKIP_BUILD=false
SKIP_TRAIN=false
QUICK_MODE=false

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
for arg in "$@"; do
    case $arg in
        --skip-build) SKIP_BUILD=true ;;
        --skip-train) SKIP_TRAIN=true ;;
        --quick)      QUICK_MODE=true ;;
        *)            echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

echo "============================================"
echo "  PINN Transformer Fault Diagnosis Pipeline"
echo "============================================"
echo "  Raw data dir:    $RAW_DIR"
echo "  Fused output:    $FUSED_DIR"
echo "  PCNN output:     $PCNN_DIR"
echo "  Gamma:           $GAMMA"
echo "  PCNN steps:      $PCNN_STEPS (VT=$PCNN_VT)"
echo "  Seed:            $SEED"
echo "  Skip build:      $SKIP_BUILD"
echo "  Quick mode:      $QUICK_MODE"
echo "============================================"
echo ""

# ===========================================================================
# Step 1: Build Mel-GADF fused images
# ===========================================================================
if [ "$SKIP_BUILD" = false ]; then
    echo ">>> Step 1/4: Building Mel-GADF fused images..."
    python scripts/build_dataset.py \
        --input_dir "$RAW_DIR" \
        --output_dir "$FUSED_DIR" \
        --gamma "$GAMMA" \
        --img_size "$IMG_SIZE" \
        --seed "$SEED"
    echo ""
else
    echo ">>> Step 1/4: SKIPPED (--skip-build)"
    echo ""
fi

# ===========================================================================
# Step 2: Apply PCNN enhancement
# ===========================================================================
echo ">>> Step 2/4: Applying PCNN enhancement..."
python scripts/apply_pcnn.py \
    --input_dir "$FUSED_DIR" \
    --output_dir "${PCNN_DIR}_flat" \
    --mode rgb \
    --num_steps "$PCNN_STEPS" \
    --VT "$PCNN_VT" \
    --img_size "$IMG_SIZE"
echo ""

# ===========================================================================
# Step 3: Split into train/val/test
# ===========================================================================
echo ">>> Step 3/4: Splitting into train/val/test..."
python scripts/build_dataset.py \
    --split_only \
    --input_dir "${PCNN_DIR}_flat" \
    --output_dir "$PCNN_DIR" \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed "$SEED"
echo ""

# ===========================================================================
# Step 4: Train
# ===========================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo ">>> Step 4/4: Training PINN model..."

    TRAIN_ARGS=(
        --config configs/default.yaml
        "dataset.root_dir=./${PCNN_DIR}"
        "dataset.normalize_mean=[0.485,0.456,0.406]"
        "dataset.normalize_std=[0.229,0.224,0.225]"
        "model.in_channels=3"
        "model.use_pcnn=false"
        "output.root_dir=./experiments/pipeline_run"
    )

    if [ "$QUICK_MODE" = true ]; then
        TRAIN_ARGS+=("train.epochs=5")
    fi

    python scripts/train.py "${TRAIN_ARGS[@]}"
    echo ""
else
    echo ">>> Step 4/4: SKIPPED (--skip-train)"
    echo ""
fi

echo "============================================"
echo "  Pipeline complete!"
echo "  PCNN dataset:  $PCNN_DIR"
echo "  Experiment:    experiments/pipeline_run/"
echo "============================================"
