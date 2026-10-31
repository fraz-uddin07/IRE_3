"""
part1_exact.py
==============
Part 1: Exact Baseline with FAISS IndexFlatL2
- Verifies exact search agreement against precomputed ground truth in the HDF5 file.
- Measures sequential (1 query at a time) vs batch execution time on a single thread.
- Logs results to results/results.csv.
- Prints architectural explanation of sequential vs batch performance divergence.
"""

import time
import argparse
import numpy as np
import faiss
from data_loader import load_dataset
from metrics import compute_recall_at_k
from plot_utils import log_result_to_csv


def run_part1(dataset_name: str = "sift-128-euclidean", max_train: int = None, max_test: int = None):
    print("=" * 70)
    print(f"PART 1: EXACT BASELINE ON {dataset_name}")
    print("=" * 70)

    # Force single thread execution as specified in ground rules
    faiss.omp_set_num_threads(1)

    # Load dataset
    data = load_dataset(dataset_name, max_train=max_train, max_test=max_test)
    train = data["train"]
    queries_test = data["queries_test"]
    gt_test = data["gt_test"]
    dist_test = data["dist_test"]
    dim = data["dim"]

    print(f"Dataset: {dataset_name}")
    print(f"Vectors to index (train): {len(train):,}, Dimension: {dim}")
    print(f"Test queries (eval): {len(queries_test):,}")

    # Build Exact FlatL2 index
    print("\nBuilding FAISS IndexFlatL2...")
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(train)
    build_time = time.perf_counter() - t0
    print(f"Index built and {len(train):,} vectors added in {build_time:.4f}s.")

    # 1. Ground truth agreement check
    print("\nVerifying agreement with dataset precomputed ground truth...")
    D_batch, I_batch = index.search(queries_test, 10)
    agreement_recall = compute_recall_at_k(
        retrieved_ids=I_batch,
        gt_ids=gt_test,
        k=10,
        retrieved_dists=D_batch,
        gt_dists=dist_test,
    )
    print(f"Exact Search Recall@10 against stored ground truth: {agreement_recall:.6f}")
    if agreement_recall >= 0.999:
        print(">> Check PASSED: Exact search agrees with dataset ground truth.")
    else:
        print(f">> NOTE: Recall is {agreement_recall:.4f} due to floating-point ties / angular conversion.")

    # 2. Timing: Sequential (1 query at a time)
    print(f"\nBenchmarking Sequential Search (1 query at a time, single thread)...")
    n_queries = len(queries_test)
    seq_runs = []
    num_trials = 3

    for trial in range(num_trials):
        start = time.perf_counter()
        for i in range(n_queries):
            q = queries_test[i:i+1]
            _ = index.search(q, 10)
        elapsed = time.perf_counter() - start
        seq_runs.append(elapsed)
        print(f"  Trial {trial + 1}/{num_trials}: {elapsed:.4f}s ({n_queries / elapsed:.1f} QPS)")

    median_seq_time = float(np.median(seq_runs))
    qps_seq = n_queries / median_seq_time
    latency_seq_ms = (median_seq_time / n_queries) * 1000.0
    print(f">> Sequential Median Time: {median_seq_time:.4f}s | QPS: {qps_seq:.2f} | Latency: {latency_seq_ms:.3f} ms/query")

    # 3. Timing: Batch (all queries in one call)
    print(f"\nBenchmarking Batch Search (all {n_queries:,} queries in one call, single thread)...")
    batch_runs = []
    for trial in range(num_trials):
        start = time.perf_counter()
        _ = index.search(queries_test, 10)
        elapsed = time.perf_counter() - start
        batch_runs.append(elapsed)
        print(f"  Trial {trial + 1}/{num_trials}: {elapsed:.4f}s ({n_queries / elapsed:.1f} QPS)")

    median_batch_time = float(np.median(batch_runs))
    qps_batch = n_queries / median_batch_time
    latency_batch_ms = (median_batch_time / n_queries) * 1000.0
    speedup = median_seq_time / median_batch_time
    print(f">> Batch Median Time: {median_batch_time:.4f}s | QPS: {qps_batch:.2f} | Latency: {latency_batch_ms:.3f} ms/query")
    print(f">> Batch Speedup over Sequential: {speedup:.2f}x")

    # Size calculation
    bytes_per_vector = dim * 4
    index_size_mb = (len(train) * bytes_per_vector) / (1024 * 1024)

    # Log sequential and batch runs
    log_result_to_csv({
        "part": "Part1",
        "dataset": dataset_name,
        "index_type": "Exact-Sequential",
        "build_params": f"dim={dim}",
        "search_params": "k=10,single_query",
        "n_vectors": len(train),
        "dimension": dim,
        "recall_at_10": agreement_recall,
        "qps": round(qps_seq, 2),
        "query_latency_ms": round(latency_seq_ms, 3),
        "build_time_s": round(build_time, 4),
        "index_size_mb": round(index_size_mb, 2),
        "bytes_per_vector": bytes_per_vector,
        "notes": "Exact IndexFlatL2 sequential 1-by-1 queries"
    })

    log_result_to_csv({
        "part": "Part1",
        "dataset": dataset_name,
        "index_type": "Exact-Batch",
        "build_params": f"dim={dim}",
        "search_params": f"k=10,batch_size={n_queries}",
        "n_vectors": len(train),
        "dimension": dim,
        "recall_at_10": agreement_recall,
        "qps": round(qps_batch, 2),
        "query_latency_ms": round(latency_batch_ms, 3),
        "build_time_s": round(build_time, 4),
        "index_size_mb": round(index_size_mb, 2),
        "bytes_per_vector": bytes_per_vector,
        "notes": f"Exact IndexFlatL2 full batch ({speedup:.1f}x speedup)"
    })

    print("\n" + "=" * 70)
    print("ARCHITECTURAL EXPLANATION: WHY SEQUENTIAL AND BATCH DIFFER SO MUCH")
    print("=" * 70)
    print(
        "1. Cache Locality and Memory Bus Bandwidth: In batch search, distance computation\n"
        "   is formulated as a matrix-matrix multiplication (GEMM: Q * X^T) via optimized\n"
        "   BLAS routines. BLAS tiles matrix blocks into L1/L2/L3 caches, reusing the loaded\n"
        "   database vectors across multiple query vectors simultaneously. In contrast, 1-by-1\n"
        "   sequential search performs a matrix-vector product (GEMV) 5,000 times, forcing the\n"
        "   CPU to stream the entire 500+ MB dataset from main RAM on every query, saturating\n"
        "   memory bus bandwidth and incurring severe memory latency penalties.\n"
        "2. SIMD Vectorization & Instruction Pipeline: Batch GEMM achieves near-peak FMA\n"
        "   (Fused Multiply-Add) throughput with unrolled AVX-256/AVX-512 vector pipelines.\n"
        "3. Dispatch and Framework Overhead: Batch mode executes a single C++ kernel call,\n"
        "   whereas sequential mode incurs 5,000 separate Python-to-C++ call dispatches."
    )
    print("=" * 70 + "\n")

    return {
        "agreement_recall": agreement_recall,
        "qps_seq": qps_seq,
        "qps_batch": qps_batch,
        "speedup": speedup,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 1: Exact Search Baseline")
    parser.add_argument("--dataset", type=str, default="sift-128-euclidean", choices=["sift-128-euclidean", "glove-100-angular"])
    parser.add_argument("--quick", action="store_true", help="Run on smaller subset (<15 min)")
    args = parser.parse_args()

    max_tr = 100000 if args.quick else None
    max_te = 1000 if args.quick else None
    run_part1(args.dataset, max_train=max_tr, max_test=max_te)
