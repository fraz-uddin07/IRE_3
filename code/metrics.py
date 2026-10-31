"""
metrics.py
==========
Custom metrics module for evaluating nearest neighbour search accuracy.
Implements recall@k with rigorous handling of distance ties at the k-th boundary.
"""

import numpy as np


def compute_recall_at_k(
    retrieved_ids: np.ndarray,
    gt_ids: np.ndarray,
    k: int = 10,
    retrieved_dists: np.ndarray = None,
    gt_dists: np.ndarray = None,
    queries: np.ndarray = None,
    train_vectors: np.ndarray = None,
    tol: float = 1e-5,
) -> float:
    """
    Computes average Recall@k across queries, handling ties at the k-th neighbor.

    A retrieved item is counted as a true positive if:
      1) It is explicitly listed in the top-k ground truth IDs, OR
      2) Its distance to the query is within `tol` of the k-th ground truth neighbor's distance
         (i.e. dist(q, item) <= dist(q, gt[k-1]) + tol).

    Args:
        retrieved_ids: (num_queries, k) array of retrieved vector indices.
        gt_ids: (num_queries, >=k) array of ground truth nearest neighbor indices.
        k: Number of nearest neighbors to evaluate (default 10).
        retrieved_dists: (num_queries, k) optional array of retrieved distances.
        gt_dists: (num_queries, >=k) optional array of ground truth distances.
        queries: (num_queries, dim) optional array of query vectors.
        train_vectors: (N, dim) optional array of indexed base vectors.
        tol: Floating point tolerance for tie-breaking.

    Returns:
        float: Mean recall@k across all queries, in [0.0, 1.0].
    """
    n_queries = retrieved_ids.shape[0]
    if n_queries == 0:
        return 0.0

    total_recall = 0.0

    for i in range(n_queries):
        top_k_retrieved = list(dict.fromkeys(retrieved_ids[i, :k]))  # deduplicate preserving order
        gt_top_k_set = set(gt_ids[i, :k])

        # Baseline intersection
        matches = set()
        for r_idx in top_k_retrieved:
            if r_idx in gt_top_k_set:
                matches.add(r_idx)

        # Handle ties for items not directly matching gt_ids[:k]
        unmatched = [r for r in top_k_retrieved if r not in matches]

        if unmatched and gt_dists is not None:
            max_gt_dist = gt_dists[i, k - 1]
            for r_pos, r_idx in enumerate(top_k_retrieved):
                if r_idx in matches:
                    continue
                # Check distance via retrieved_dists if provided
                if retrieved_dists is not None:
                    d = retrieved_dists[i, r_pos]
                    if d <= max_gt_dist + tol:
                        matches.add(r_idx)
                elif queries is not None and train_vectors is not None:
                    # Recompute distance directly
                    q = queries[i]
                    v = train_vectors[r_idx]
                    d = np.linalg.norm(q - v)
                    if d <= max_gt_dist + tol:
                        matches.add(r_idx)

        query_recall = min(len(matches), k) / float(k)
        total_recall += query_recall

    return float(total_recall / n_queries)
