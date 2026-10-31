"""
plot_utils.py
=============
Publication-grade visualization and CSV logging utilities for ANN benchmarking:
- Generates high-resolution PNG figures (Part 2, Part 3, Part 4).
- Maintains results/results.csv with complete benchmark run details.
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling configuration for clean, publication-ready figures
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

INDEX_COLORS = {
    "HNSW": "#1f77b4",        # Blue
    "IVF-Flat": "#ff7f0e",    # Orange
    "IVF-PQ": "#2ca02c",      # Green
    "Exact": "#d62728",       # Red
}

INDEX_MARKERS = {
    "HNSW": "o",
    "IVF-Flat": "s",
    "IVF-PQ": "^",
    "Exact": "D",
}


def log_result_to_csv(row_dict: dict, csv_path: str = "results/results.csv"):
    """Appends a benchmark execution row to results.csv."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "part",
        "dataset",
        "index_type",
        "build_params",
        "search_params",
        "n_vectors",
        "dimension",
        "recall_at_10",
        "qps",
        "query_latency_ms",
        "build_time_s",
        "index_size_mb",
        "bytes_per_vector",
        "notes"
    ]

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def compute_pareto_frontier(recalls, qps_list):
    """Computes non-dominated Pareto-optimal points (maximizing recall and QPS)."""
    points = sorted(zip(recalls, qps_list), key=lambda p: p[0])
    frontier = []
    max_qps_so_far = -1
    # Iterate backwards from highest recall
    for r, q in reversed(points):
        if q > max_qps_so_far:
            frontier.append((r, q))
            max_qps_so_far = q
    frontier.reverse()
    return frontier


def plot_tradeoff_curve(
    results_by_index: dict,
    exact_qps: float,
    dataset_title: str,
    output_path: str,
):
    """
    Plots Recall@10 vs Queries Per Second (QPS) trade-off curves for Part 2.
    Shows Pareto-optimal frontiers for each index along with the exact baseline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    # Plot exact baseline horizontal line
    if exact_qps is not None and exact_qps > 0:
        ax.axhline(
            y=exact_qps,
            color=INDEX_COLORS["Exact"],
            linestyle="--",
            alpha=0.85,
            label=f"Exact (FAISS FlatL2) [{exact_qps:.1f} QPS]",
        )

    for index_name, runs in results_by_index.items():
        if not runs:
            continue
        recalls = [r["recall_at_10"] for r in runs]
        qps_vals = [r["qps"] for r in runs]

        color = INDEX_COLORS.get(index_name, "#333333")
        marker = INDEX_MARKERS.get(index_name, "o")

        # Scatter all configurations lightly
        ax.scatter(recalls, qps_vals, color=color, alpha=0.35, s=28, edgecolors="none")

        # Compute and plot Pareto frontier
        frontier = compute_pareto_frontier(recalls, qps_vals)
        if frontier:
            f_r, f_q = zip(*frontier)
            ax.plot(
                f_r, f_q,
                color=color,
                marker=marker,
                label=f"{index_name} (Pareto)",
                linewidth=2.2,
                markersize=6.5
            )

    ax.set_yscale("log")
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries Per Second (QPS) [Log Scale]")
    ax.set_title(f"Recall@10 vs. QPS Trade-Off on {dataset_title}")
    ax.set_xlim(left=0.0, right=1.02)
    ax.grid(True, which="both", axis="both")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved trade-off plot to {output_path}")


def plot_part3_scaling(scaling_data: dict, output_path: str):
    """
    Plots Part 3 Scaling results:
    1. QPS at Recall >= 0.95 vs N
    2. Build Time (s) vs N
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    n_labels = [str(n) for n in scaling_data["N"]]
    x_indices = np.arange(len(n_labels))
    bar_width = 0.25

    # Subplot 1: Query Throughput (QPS) at Recall 0.95
    for idx, (name, qps_list) in enumerate(scaling_data["qps"].items()):
        color = INDEX_COLORS.get(name, "#555")
        offset = (idx - 1) * bar_width
        ax1.bar(x_indices + offset, qps_list, width=bar_width, label=name, color=color, alpha=0.9)

    ax1.set_yscale("log")
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(n_labels)
    ax1.set_xlabel("Dataset Size N (SIFT Vectors)")
    ax1.set_ylabel("QPS at Recall $\\geq 0.95$ [Log Scale]")
    ax1.set_title("Search Throughput Scaling (Recall $\\geq 0.95$)")
    ax1.grid(True, axis="y")
    ax1.legend()

    # Subplot 2: Build Time
    for idx, (name, build_times) in enumerate(scaling_data["build_times"].items()):
        color = INDEX_COLORS.get(name, "#555")
        offset = (idx - 1) * bar_width
        ax2.bar(x_indices + offset, build_times, width=bar_width, label=name, color=color, alpha=0.9)

    ax2.set_yscale("log")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(n_labels)
    ax2.set_xlabel("Dataset Size N (SIFT Vectors)")
    ax2.set_ylabel("Build Time (Seconds) [Log Scale]")
    ax2.set_title("Index Build Time Scaling")
    ax2.grid(True, axis="y")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved Part 3 scaling plot to {output_path}")


def plot_part4_geometry(geometry_data: dict, output_path: str):
    """
    Plots Part 4 Hard Geometry (Curse of Dimensionality):
    1. Distance Contrast ((d10 - d1) / d1) vs Dimension d
    2. Recall@10 at Fixed Budget vs Dimension d for HNSW and IVF-Flat
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    dims = geometry_data["dimensions"]

    # Subplot 1: Metric Contrast Collapse
    ax1.plot(dims, geometry_data["contrast"], marker="o", color="#8c564b", linewidth=2.5)
    ax1.set_xlabel("Ambient Dimension $d$")
    ax1.set_ylabel("Relative Contrast: $(d_{10} - d_1) / d_1$")
    ax1.set_title("Metric Contrast Vanishing with Dimension")
    ax1.grid(True)
    ax1.set_xticks(dims)

    # Subplot 2: Recall Degradation
    ax2.plot(
        dims,
        geometry_data["hnsw_recall"],
        marker="o",
        color=INDEX_COLORS["HNSW"],
        linewidth=2.5,
        label="HNSW ($efSearch=32$)",
    )
    ax2.plot(
        dims,
        geometry_data["ivf_recall"],
        marker="s",
        color=INDEX_COLORS["IVF-Flat"],
        linewidth=2.5,
        label="IVF-Flat ($nprobe=8$)",
    )

    # Mark control baseline
    ax2.axvline(x=64, color="gray", linestyle=":", label="Control Dimension ($d=64$)")

    ax2.set_xlabel("Ambient Dimension $d$")
    ax2.set_ylabel("Recall@10 at Fixed Search Budget")
    ax2.set_title("Recall Collapse under Distance Concentration")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True)
    ax2.set_xticks(dims)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved Part 4 geometry plot to {output_path}")
