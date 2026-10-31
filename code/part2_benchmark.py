"""
part2_benchmark.py
==================
Part 2: Benchmark HNSW, IVF-Flat, and IVF-PQ on SIFT and GloVe-100 datasets.
- Sweeps build settings (M, nlist, M_pq) and search parameters (efSearch, nprobe).
- Measures Recall@10 vs Queries Per Second (QPS) one query at a time on a single thread.
- Identifies Pareto-optimal frontiers and plots trade-off curves against Exact baseline.
- Computes summary tables:
  1. QPS at Recall 0.90 and 0.95.
  2. Build time, index size (MB), and bytes per vector at Recall 0.95.
- Logs every benchmark run to results/results.csv.
"""

import os
import time
import tempfile
import argparse
import numpy as np
import faiss
from tabulate import tabulate

from data_loader import load_dataset
from metrics import compute_recall_at_k
from plot_utils import log_result_to_csv, plot_tradeoff_curve, compute_pareto_frontier


def measure_single_query_search(index, queries, num_trials=3):
    """
    Measures search latency and QPS querying one vector at a time on a single thread.
    Repeats num_trials times and takes the median elapsed time.
    """
    n_queries = len(queries)
    trial_times = []
    retrieved_indices = None
    retrieved_distances = None

    for _ in range(num_trials):
        start = time.perf_counter()
        all_I = []
        all_D = []
        for i in range(n_queries):
            D, I = index.search(queries[i:i+1], 10)
            all_I.append(I[0])
            all_D.append(D[0])
        elapsed = time.perf_counter() - start
        trial_times.append(elapsed)
        if retrieved_indices is None:
            retrieved_indices = np.array(all_I, dtype=np.int64)
            retrieved_distances = np.array(all_D, dtype=np.float32)

    median_time = float(np.median(trial_times))
    qps = n_queries / median_time
    latency_ms = (median_time / n_queries) * 1000.0
    return qps, latency_ms, retrieved_indices, retrieved_distances


def get_index_size_bytes(index):
    """Measures serialized index size in bytes using a temporary file."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_name = tmp.name
    try:
        faiss.write_index(index, tmp_name)
        size_bytes = os.path.getsize(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return size_bytes


def interpolate_qps_at_recall(runs, target_recall):
    """
    Finds or linearly interpolates QPS at target_recall on the Pareto frontier.
    Returns None if target_recall is not reached.
    """
    frontier = compute_pareto_frontier([r["recall_at_10"] for r in runs], [r["qps"] for r in runs])
    if not frontier:
        return None

    # Check if target recall was achieved
    max_recall = max(r for r, q in frontier)
    if max_recall < target_recall:
        return None

    # Sort frontier by recall ascending
    frontier = sorted(frontier, key=lambda x: x[0])

    # If first point already exceeds target, return its QPS
    if frontier[0][0] >= target_recall:
        return frontier[0][1]

    # Linear interpolation along the frontier
    for i in range(len(frontier) - 1):
        r1, q1 = frontier[i]
        r2, q2 = frontier[i + 1]
        if r1 <= target_recall <= r2:
            alpha = (target_recall - r1) / (r2 - r1 + 1e-12)
            # Log-space QPS interpolation for smooth rate scaling
            log_q = (1 - alpha) * np.log(max(q1, 1e-5)) + alpha * np.log(max(q2, 1e-5))
            return float(np.exp(log_q))

    return frontier[-1][1]


def run_part2_for_dataset(dataset_name: str, quick: bool = False):
    print("\n" + "=" * 75)
    print(f"BENCHMARKING THREE INDEXES (HNSW, IVF-Flat, IVF-PQ) ON: {dataset_name.upper()}")
    print("=" * 75)

    faiss.omp_set_num_threads(1)

    max_train = 100000 if quick else None
    max_test = 1000 if quick else None

    data = load_dataset(dataset_name, max_train=max_train, max_test=max_test)
    train = data["train"]
    queries_test = data["queries_test"]
    gt_test = data["gt_test"]
    dist_test = data["dist_test"]
    dim = data["dim"]
    n_train = len(train)

    print(f"Loaded {n_train:,} base vectors, dimension {dim}, evaluation queries: {len(queries_test):,}")

    # Step 0: Measure exact baseline for the plot
    print("\n[Exact Baseline] Building IndexFlatL2...")
    exact_index = faiss.IndexFlatL2(dim)
    exact_index.add(train)
    exact_qps, _, _, _ = measure_single_query_search(exact_index, queries_test, num_trials=2 if quick else 3)
    print(f"Exact baseline 1-by-1 QPS: {exact_qps:.1f}")

    results_by_index = {"HNSW": [], "IVF-Flat": [], "IVF-PQ": []}

    # ---------------------------------------------------------
    # 1. HNSW Benchmark
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("1. BENCHMARKING HNSW (IndexHNSWFlat)")
    print("-" * 60)
    hnsw_m_values = [16, 32] if quick else [16, 32, 64]
    hnsw_ef_search_values = [10, 20, 40, 80, 160] if quick else [10, 16, 24, 32, 48, 64, 96, 128, 256, 512]
    ef_construction = 64 if quick else 200

    for M in hnsw_m_values:
        print(f"\nBuilding HNSW index with M={M}, efConstruction={ef_construction}...")
        t0 = time.perf_counter()
        hnsw_index = faiss.IndexHNSWFlat(dim, M)
        hnsw_index.hnsw.efConstruction = ef_construction
        hnsw_index.add(train)
        build_time = time.perf_counter() - t0

        size_bytes = get_index_size_bytes(hnsw_index)
        size_mb = size_bytes / (1024 * 1024)
        bytes_per_vec = size_bytes / n_train
        print(f"Built HNSW (M={M}) in {build_time:.2f}s | Size: {size_mb:.2f} MB ({bytes_per_vec:.1f} B/vec)")

        for ef in hnsw_ef_search_values:
            hnsw_index.hnsw.efSearch = ef
            qps, lat_ms, I_ret, D_ret = measure_single_query_search(hnsw_index, queries_test, num_trials=2 if quick else 3)
            rec = compute_recall_at_k(I_ret, gt_test, k=10, retrieved_dists=D_ret, gt_dists=dist_test)
            print(f"  HNSW (M={M}, efSearch={ef:3d}) -> Recall@10: {rec:.4f} | QPS: {qps:7.1f} | Latency: {lat_ms:6.2f} ms")

            run_entry = {
                "part": "Part2",
                "dataset": dataset_name,
                "index_type": "HNSW",
                "build_params": f"M={M},efConst={ef_construction}",
                "search_params": f"efSearch={ef}",
                "n_vectors": n_train,
                "dimension": dim,
                "recall_at_10": round(rec, 4),
                "qps": round(qps, 2),
                "query_latency_ms": round(lat_ms, 3),
                "build_time_s": round(build_time, 2),
                "index_size_mb": round(size_mb, 2),
                "bytes_per_vector": round(bytes_per_vec, 1),
                "notes": f"HNSW build M={M}"
            }
            results_by_index["HNSW"].append(run_entry)
            log_result_to_csv(run_entry)

    # ---------------------------------------------------------
    # 2. IVF-Flat Benchmark
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("2. BENCHMARKING IVF-Flat (IndexIVFFlat)")
    print("-" * 60)
    ivf_nlist_values = [256, 512] if quick else [512, 1024, 2048]
    ivf_nprobe_values = [1, 2, 4, 8, 16, 32, 64] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    for nlist in ivf_nlist_values:
        print(f"\nBuilding IVF-Flat index with nlist={nlist}...")
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatL2(dim)
        ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        # Train quantizer on sample of vectors if needed or full
        train_sample = train if len(train) <= 200000 else train[np.random.choice(len(train), 200000, replace=False)]
        ivf_index.train(train_sample)
        ivf_index.add(train)
        build_time = time.perf_counter() - t0

        size_bytes = get_index_size_bytes(ivf_index)
        size_mb = size_bytes / (1024 * 1024)
        bytes_per_vec = size_bytes / n_train
        print(f"Built IVF-Flat (nlist={nlist}) in {build_time:.2f}s | Size: {size_mb:.2f} MB ({bytes_per_vec:.1f} B/vec)")

        for nprobe in ivf_nprobe_values:
            if nprobe > nlist:
                continue
            ivf_index.nprobe = nprobe
            qps, lat_ms, I_ret, D_ret = measure_single_query_search(ivf_index, queries_test, num_trials=2 if quick else 3)
            rec = compute_recall_at_k(I_ret, gt_test, k=10, retrieved_dists=D_ret, gt_dists=dist_test)
            print(f"  IVF-Flat (nlist={nlist:4d}, nprobe={nprobe:3d}) -> Recall@10: {rec:.4f} | QPS: {qps:7.1f} | Latency: {lat_ms:6.2f} ms")

            run_entry = {
                "part": "Part2",
                "dataset": dataset_name,
                "index_type": "IVF-Flat",
                "build_params": f"nlist={nlist}",
                "search_params": f"nprobe={nprobe}",
                "n_vectors": n_train,
                "dimension": dim,
                "recall_at_10": round(rec, 4),
                "qps": round(qps, 2),
                "query_latency_ms": round(lat_ms, 3),
                "build_time_s": round(build_time, 2),
                "index_size_mb": round(size_mb, 2),
                "bytes_per_vector": round(bytes_per_vec, 1),
                "notes": f"IVF-Flat nlist={nlist}"
            }
            results_by_index["IVF-Flat"].append(run_entry)
            log_result_to_csv(run_entry)

    # ---------------------------------------------------------
    # 3. IVF-PQ Benchmark
    # ---------------------------------------------------------
    print("\n" + "-" * 60)
    print("3. BENCHMARKING IVF-PQ (IndexIVFPQ)")
    print("-" * 60)
    # Determine valid M_pq for dataset dimension
    if dim == 128:
        pq_m_candidates = [16] if quick else [16, 32]
    elif dim == 100:
        pq_m_candidates = [20] if quick else [10, 20]
    else:
        pq_m_candidates = [dim // 8]

    pq_nlist_values = [256] if quick else [512, 1024]
    pq_nprobe_values = [1, 2, 4, 8, 16, 32, 64] if quick else [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    nbits = 8

    for M_pq in pq_m_candidates:
        for nlist in pq_nlist_values:
            print(f"\nBuilding IVF-PQ index with nlist={nlist}, M_pq={M_pq}, nbits={nbits}...")
            t0 = time.perf_counter()
            quantizer = faiss.IndexFlatL2(dim)
            ivf_pq = faiss.IndexIVFPQ(quantizer, dim, nlist, M_pq, nbits)
            train_sample = train if len(train) <= 200000 else train[np.random.choice(len(train), 200000, replace=False)]
            ivf_pq.train(train_sample)
            ivf_pq.add(train)
            build_time = time.perf_counter() - t0

            size_bytes = get_index_size_bytes(ivf_pq)
            size_mb = size_bytes / (1024 * 1024)
            bytes_per_vec = size_bytes / n_train
            print(f"Built IVF-PQ (M={M_pq}, nlist={nlist}) in {build_time:.2f}s | Size: {size_mb:.2f} MB ({bytes_per_vec:.1f} B/vec)")

            for nprobe in pq_nprobe_values:
                if nprobe > nlist:
                    continue
                ivf_pq.nprobe = nprobe
                qps, lat_ms, I_ret, D_ret = measure_single_query_search(ivf_pq, queries_test, num_trials=2 if quick else 3)
                rec = compute_recall_at_k(I_ret, gt_test, k=10, retrieved_dists=D_ret, gt_dists=dist_test)
                print(f"  IVF-PQ (M={M_pq}, nlist={nlist}, nprobe={nprobe:3d}) -> Recall@10: {rec:.4f} | QPS: {qps:7.1f} | Latency: {lat_ms:6.2f} ms")

                run_entry = {
                    "part": "Part2",
                    "dataset": dataset_name,
                    "index_type": "IVF-PQ",
                    "build_params": f"nlist={nlist},M_pq={M_pq}",
                    "search_params": f"nprobe={nprobe}",
                    "n_vectors": n_train,
                    "dimension": dim,
                    "recall_at_10": round(rec, 4),
                    "qps": round(qps, 2),
                    "query_latency_ms": round(lat_ms, 3),
                    "build_time_s": round(build_time, 2),
                    "index_size_mb": round(size_mb, 2),
                    "bytes_per_vector": round(bytes_per_vec, 1),
                    "notes": f"IVF-PQ M={M_pq}, nlist={nlist}"
                }
                results_by_index["IVF-PQ"].append(run_entry)
                log_result_to_csv(run_entry)

    # ---------------------------------------------------------
    # Plots and Summary Tables
    # ---------------------------------------------------------
    plot_path = f"results/figures/part2_{'sift' if 'sift' in dataset_name else 'glove'}_tradeoff.png"
    plot_tradeoff_curve(
        results_by_index=results_by_index,
        exact_qps=exact_qps,
        dataset_title=dataset_name.upper(),
        output_path=plot_path
    )

    # Table 1: QPS at Recall 0.90 and 0.95
    print("\n" + "=" * 75)
    print(f"TABLE 1: QUERIES PER SECOND (QPS) AT RECALL 0.90 AND 0.95 ({dataset_name})")
    print("=" * 75)
    table1_rows = []
    for idx_name in ["HNSW", "IVF-Flat", "IVF-PQ"]:
        runs = results_by_index[idx_name]
        qps_90 = interpolate_qps_at_recall(runs, 0.90)
        qps_95 = interpolate_qps_at_recall(runs, 0.95)

        s_90 = f"{qps_90:.1f}" if qps_90 is not None else "Failed / N/A"
        s_95 = f"{qps_95:.1f}" if qps_95 is not None else "Failed / N/A"
        table1_rows.append([idx_name, s_90, s_95])

    print(tabulate(table1_rows, headers=["Index Type", "QPS @ Recall 0.90", "QPS @ Recall 0.95"], tablefmt="github"))

    # Table 2: Build time, Index Size, Bytes per vector at Recall 0.95
    print("\n" + "=" * 75)
    print(f"TABLE 2: BUILD TIME, INDEX SIZE, AND BYTES PER VECTOR AT RECALL 0.95 ({dataset_name})")
    print("=" * 75)
    table2_rows = []
    for idx_name in ["HNSW", "IVF-Flat", "IVF-PQ"]:
        runs = [r for r in results_by_index[idx_name] if r["recall_at_10"] >= 0.95]
        if runs:
            # Pick the run with highest QPS among those reaching recall >= 0.95
            best_run = max(runs, key=lambda x: x["qps"])
            table2_rows.append([
                idx_name,
                f"{best_run['build_time_s']:.1f} s",
                f"{best_run['index_size_mb']:.1f} MB",
                f"{best_run['bytes_per_vector']:.1f} B/vec",
                f"{best_run['recall_at_10']:.4f} (QPS: {best_run['qps']:.1f})"
            ])
        else:
            table2_rows.append([idx_name, "N/A", "N/A", "N/A", "Failed to reach Recall 0.95"])

    print(tabulate(table2_rows, headers=["Index Type", "Build Time", "Index Size", "Bytes/Vector", "Operating Point"], tablefmt="github"))

    print("\n" + "=" * 75)
    print("ANALYSIS: WHY IVF-PQ FAILS OR STRUGGLES TO REACH HIGH RECALL (0.95+)")
    print("=" * 75)
    print(
        "IVF-PQ struggles to achieve recall >= 0.95 due to an intrinsic quantization error floor.\n"
        "Product Quantization maps high-dimensional vectors to a coarse codebook (e.g. 16 or 20\n"
        "sub-vectors with 256 centroids each). This lossy compression introduces substantial\n"
        "asymmetric distance computation (ADC) distortion. Even if nprobe is set to examine every\n"
        "list (nprobe = nlist), the distance ranking computed using quantized lookup tables\n"
        "permutes the order of the true top-10 neighbors, preventing plain IVF-PQ from reaching\n"
        "0.95 recall without a second-stage exact re-ranking pass."
    )
    print("=" * 75 + "\n")

    return results_by_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2: Benchmark Three Indexes")
    parser.add_argument("--dataset", type=str, default="both", choices=["sift-128-euclidean", "glove-100-angular", "both"])
    parser.add_argument("--quick", action="store_true", help="Run on smaller subset (<15 min)")
    args = parser.parse_args()

    if args.dataset in ["sift-128-euclidean", "both"]:
        run_part2_for_dataset("sift-128-euclidean", quick=args.quick)
    if args.dataset in ["glove-100-angular", "both"]:
        run_part2_for_dataset("glove-100-angular", quick=args.quick)
