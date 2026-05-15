#!/bin/bash
#
# Run all ACE evaluations (PG19, Needle-in-Haystack, LongBench)
#
# Usage:
#   ./scripts/run_all_evals.sh
#   ./scripts/run_all_evals.sh --model meta-llama/Llama-2-7b-hf --cache-budget 1024
#   ./scripts/run_all_evals.sh --ablation no_mlp
#

set -e  # Exit on error

# Default values
MODEL="meta-llama/Llama-2-7b-hf"
CACHE_BUDGET=1024
UTILITY_MLP="checkpoints/utility_mlp.pt"
NUM_SAMPLES=100
ABLATION=""
OUTPUT_DIR="results"
DATE=$(date +%Y%m%d_%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --cache-budget)
            CACHE_BUDGET="$2"
            shift 2
            ;;
        --utility-mlp)
            UTILITY_MLP="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --ablation)
            ABLATION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model NAME        Model name/path (default: meta-llama/Llama-2-7b-hf)"
            echo "  --cache-budget N    Cache budget in tokens (default: 1024)"
            echo "  --utility-mlp PATH  Path to utility MLP (default: checkpoints/utility_mlp.pt)"
            echo "  --num-samples N     Number of samples per task (default: 100)"
            echo "  --ablation TYPE     Ablation variant (no_mlp, linear_w, square_age, sqrt_age)"
            echo "  --output-dir DIR    Output directory (default: results)"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ACE: Full Evaluation Suite"
echo "============================================================"
echo "Model:          $MODEL"
echo "Cache budget:   $CACHE_BUDGET"
echo "Utility MLP:    $UTILITY_MLP"
echo "Num samples:    $NUM_SAMPLES"
echo "Ablation:       ${ABLATION:-none}"
echo "Output dir:     $OUTPUT_DIR"
echo "============================================================"
echo ""

# Build common arguments
COMMON_ARGS="--model $MODEL --cache-budget $CACHE_BUDGET --num-samples $NUM_SAMPLES"
if [ -n "$ABLATION" ]; then
    COMMON_ARGS="$COMMON_ARGS --ablation $ABLATION"
else
    COMMON_ARGS="$COMMON_ARGS --utility-mlp $UTILITY_MLP"
fi

# ============================================================
# 1. PG19 Perplexity
# ============================================================
echo ""
echo "============================================================"
echo "[1/3] PG19 Perplexity Evaluation"
echo "============================================================"

PG19_OUTPUT="$OUTPUT_DIR/pg19_${DATE}.json"
python experiments/run_pg19.py $COMMON_ARGS --output "$PG19_OUTPUT"

# ============================================================
# 2. Needle-in-a-Haystack
# ============================================================
echo ""
echo "============================================================"
echo "[2/3] Needle-in-a-Haystack Evaluation"
echo "============================================================"

NEEDLE_OUTPUT="$OUTPUT_DIR/needle_${DATE}.json"
python experiments/run_needle_test.py $COMMON_ARGS --num-tests 10 --output "$NEEDLE_OUTPUT"

# ============================================================
# 3. LongBench
# ============================================================
echo ""
echo "============================================================"
echo "[3/3] LongBench Evaluation"
echo "============================================================"

LONGBENCH_OUTPUT="$OUTPUT_DIR/longbench_${DATE}.json"
python experiments/run_longbench.py $COMMON_ARGS --output "$LONGBENCH_OUTPUT"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Summary files:"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "  (no JSON files found)"
echo ""
echo "To view results:"
echo "  cat $PG19_OUTPUT"
echo "  cat $NEEDLE_OUTPUT"
echo "  cat $LONGBENCH_OUTPUT"