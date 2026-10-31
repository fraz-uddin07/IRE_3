"""
data_generator.py
=================
Synthetic Data Generator for Part 4: Hard Geometry (Curse of Dimensionality)
Generates uniformly distributed vectors on the unit hypersphere S^(d-1)
using Gaussian sampling followed by L2 normalization.

Ensures strict determinism via random seeding.
"""

import os
import argparse
import numpy as np
import faiss


def generate_hypersphere_dataset(
    n_vectors: int = 100000,
    n_queries: int = 1000,
    dimension: int = 64,
    seed: int = 42
):
    """
    Generates synthetic datasets on the unit hypersphere S^(d-1).

    Args:
        n_vectors: Number of base vectors to index.
        n_queries: Number of query vectors.
        dimension: Ambient dimension d.
        seed: Random seed for 100% reproducible generation.

    Returns:
        dict: {
            'train': (n_vectors, d) float32,
            'queries': (n_queries, d) float32,
            'gt_ids': (n_queries, 10) int64,
            'gt_dists': (n_queries, 10) float32,
            'contrast': float (mean (d10 - d1) / d1),
            'dimension': int
        }
    """
    rng = np.random.RandomState(seed)

    # Sample from standard isotropic Gaussian
    raw_train = rng.randn(n_vectors, dimension).astype(np.float32)
    raw_queries = rng.randn(n_queries, dimension).astype(np.float32)

    # Normalize onto unit hypersphere S^(d-1)
    train_norms = np.linalg.norm(raw_train, axis=1, keepdims=True)
    train = raw_train / np.maximum(train_norms, 1e-12)

    query_norms = np.linalg.norm(raw_queries, axis=1, keepdims=True)
    queries = raw_queries / np.maximum(query_norms, 1e-12)

    # Compute exact ground truth using FAISS FlatL2
    faiss.omp_set_num_threads(1)
    exact_index = faiss.IndexFlatL2(dimension)
    exact_index.add(train)
    gt_dists, gt_ids = exact_index.search(queries, 10)

    # Compute metric contrast: (d_10 - d_1) / d_1
    d1 = gt_dists[:, 0]
    d10 = gt_dists[:, 9]
    valid_mask = d1 > 1e-8
    contrast = float(np.mean((d10[valid_mask] - d1[valid_mask]) / d1[valid_mask]))

    return {
        "train": train,
        "queries": queries,
        "gt_ids": gt_ids,
        "gt_dists": gt_dists,
        "contrast": contrast,
        "dimension": dimension,
        "n_vectors": n_vectors,
        "n_queries": n_queries,
        "seed": seed,
    }


def save_synthetic_dataset(data_dict: dict, output_path: str):
    """Saves generated dataset to a compressed npz file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        train=data_dict["train"],
        queries=data_dict["queries"],
        gt_ids=data_dict["gt_ids"],
        gt_dists=data_dict["gt_dists"],
        contrast=data_dict["contrast"],
        dimension=data_dict["dimension"],
    )
    print(f"Saved synthetic dataset (d={data_dict['dimension']}) to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 4 Synthetic Data Generator")
    parser.add_argument("--dim", type=int, default=64, help="Ambient dimension d")
    parser.add_argument("--n", type=int, default=100000, help="Number of base vectors")
    parser.add_argument("--queries", type=int, default=1000, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="", help="Optional output npz file")
    args = parser.parse_args()

    data = generate_hypersphere_dataset(
        n_vectors=args.n,
        n_queries=args.queries,
        dimension=args.dim,
        seed=args.seed
    )
    print(f"Generated {data['n_vectors']:,} vectors in d={data['dimension']}.")
    print(f"Empirical Relative Contrast ((d10 - d1) / d1): {data['contrast']:.6f}")

    if args.output:
        save_synthetic_dataset(data, args.output)
