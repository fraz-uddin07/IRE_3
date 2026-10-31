#!/bin/bash
# ==============================================================================
# run_all.sh - One-command reproduction for IRE Assignment 3
#
# Usage:
#   bash run_all.sh                          # Full benchmark on complete datasets (~45-60 min)
#   bash run_all.sh --quick                  # Fast benchmark on 100k subsets (< 15 min on laptop/Colab)
#   bash run_all.sh --from-step 3            # Resume execution starting from Step 3
#   bash run_all.sh --quick --from-step 3    # Resume execution starting from Step 3 (quick mode)
# ==============================================================================

set -e

QUICK_FLAG=""
START_STEP=0
MODE_NAME="FULL BENCHMARK"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_FLAG="--quick"
            MODE_NAME="QUICK SUBSET BENCHMARK"
            shift
            ;;
        --from-step)
            START_STEP="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "======================================================================"
echo "STARTING IRE ASSIGNMENT 3: BENCHMARKING APPROXIMATE NEAREST NEIGHBOUR"
echo "Mode: $MODE_NAME (Starting from Step $START_STEP)"
echo "Timestamp: $(date)"
echo "======================================================================"

# Ensure directories exist
mkdir -p results/figures data
mkdir -p ../results/figures ../data 2>/dev/null || true

# Step 0: Run Unit Tests for Recall Metric
if [ "$START_STEP" -le 0 ]; then
    echo ""
    echo "[Step 0/5] Running Unit Tests for recall@10 tie-handling..."
    python test_metrics.py
    echo ">> All unit tests PASSED successfully."
fi

# Step 1: Part 1 Exact Baseline
if [ "$START_STEP" -le 1 ]; then
    echo ""
    echo "[Step 1/5] Running Part 1: Exact Baseline & Verification..."
    python part1_exact.py --dataset sift-128-euclidean $QUICK_FLAG
    python part1_exact.py --dataset glove-100-angular $QUICK_FLAG
fi

# Step 2: Part 2 Benchmarking HNSW, IVF-Flat, IVF-PQ on SIFT & GloVe
if [ "$START_STEP" -le 2 ]; then
    echo ""
    echo "[Step 2/5] Running Part 2: Benchmarking Three Indexes..."
    python part2_benchmark.py --dataset both $QUICK_FLAG
fi

# Step 3: Part 3 Scaling Experiment on SIFT
if [ "$START_STEP" -le 3 ]; then
    echo ""
    echo "[Step 3/5] Running Part 3: Standard Data Experiment (Scaling on SIFT)..."
    python part3_scaling.py $QUICK_FLAG
fi

# Step 4: Part 4 Hard Geometry Experiment on Synthetic Data
if [ "$START_STEP" -le 4 ]; then
    echo ""
    echo "[Step 4/5] Running Part 4: Custom Data Experiment (Hard Geometry & Distance Concentration)..."
    python part4_geometry.py $QUICK_FLAG
fi


echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS AND DELIVERABLES COMPLETED SUCCESSFULLY!"
echo "Generated outputs:"
echo "  - results/results.csv"
echo "  - results/figures/part2_sift_tradeoff.png"
echo "  - results/figures/part2_glove_tradeoff.png"
echo "  - results/figures/part3_scaling.png"
echo "  - results/figures/part4_geometry.png"
echo "======================================================================"
