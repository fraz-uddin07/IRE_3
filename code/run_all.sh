#!/bin/bash
# ==============================================================================
# run_all.sh - One-command reproduction for IRE Assignment 3
#
# Usage:
#   bash run_all.sh          # Full benchmark on complete datasets (~45-60 min)
#   bash run_all.sh --quick  # Fast benchmark on 100k subsets (< 15 min on laptop/Colab)
# ==============================================================================

set -e

QUICK_FLAG=""
MODE_NAME="FULL BENCHMARK"

if [ "$1" == "--quick" ]; then
    QUICK_FLAG="--quick"
    MODE_NAME="QUICK SUBSET BENCHMARK (<15 min)"
fi

echo "======================================================================"
echo "STARTING IRE ASSIGNMENT 3: BENCHMARKING APPROXIMATE NEAREST NEIGHBOUR"
echo "Mode: $MODE_NAME"
echo "Timestamp: $(date)"
echo "======================================================================"

# Ensure directories exist
mkdir -p results/figures data
mkdir -p ../results/figures ../data 2>/dev/null || true

# Step 0: Run Unit Tests for Recall Metric
echo ""
echo "[Step 0/5] Running Unit Tests for recall@10 tie-handling..."
python test_metrics.py
echo ">> All unit tests PASSED successfully."

# Step 1: Part 1 Exact Baseline
echo ""
echo "[Step 1/5] Running Part 1: Exact Baseline & Verification..."
python part1_exact.py --dataset sift-128-euclidean $QUICK_FLAG
python part1_exact.py --dataset glove-100-angular $QUICK_FLAG

# Step 2: Part 2 Benchmarking HNSW, IVF-Flat, IVF-PQ on SIFT & GloVe
echo ""
echo "[Step 2/5] Running Part 2: Benchmarking Three Indexes..."
python part2_benchmark.py --dataset both $QUICK_FLAG

# Step 3: Part 3 Scaling Experiment on SIFT
echo ""
echo "[Step 3/5] Running Part 3: Standard Data Experiment (Scaling on SIFT)..."
python part3_scaling.py $QUICK_FLAG

# Step 4: Part 4 Hard Geometry Experiment on Synthetic Data
echo ""
echo "[Step 4/5] Running Part 4: Custom Data Experiment (Hard Geometry & Distance Concentration)..."
python part4_geometry.py $QUICK_FLAG

# Step 5: Generate Final PDF Report from Results and Figures
echo ""
echo "[Step 5/5] Compiling Part 5 Final Report (report.pdf)..."
python generate_report.py

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS AND DELIVERABLES COMPLETED SUCCESSFULLY!"
echo "Generated outputs:"
echo "  - results/results.csv"
echo "  - results/figures/part2_sift_tradeoff.png"
echo "  - results/figures/part2_glove_tradeoff.png"
echo "  - results/figures/part3_scaling.png"
echo "  - results/figures/part4_geometry.png"
echo "  - report.pdf"
echo "======================================================================"
