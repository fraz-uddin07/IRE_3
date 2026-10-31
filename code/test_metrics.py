"""
test_metrics.py
===============
Unit tests for the custom recall@k function, explicitly testing:
1. Perfect match (1.0)
2. Disjoint retrieval (0.0)
3. Partial match (e.g. 0.5)
4. Distance ties at the 10th neighbor (must count as valid matches)
5. Distances exceeding tolerance (must NOT count as matches)
6. Deduplication safety against repeated IDs
"""

import unittest
import numpy as np
from metrics import compute_recall_at_k


class TestRecallAtK(unittest.TestCase):

    def test_perfect_recall(self):
        """Retrieved IDs exactly match ground truth top-10."""
        gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        retrieved_ids = np.array([[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])  # Permuted
        recall = compute_recall_at_k(retrieved_ids, gt_ids, k=10)
        self.assertAlmostEqual(recall, 1.0, places=6)

    def test_zero_recall(self):
        """Retrieved IDs are completely disjoint from ground truth."""
        gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        retrieved_ids = np.array([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        recall = compute_recall_at_k(retrieved_ids, gt_ids, k=10)
        self.assertAlmostEqual(recall, 0.0, places=6)

    def test_partial_recall(self):
        """5 out of 10 items match ground truth."""
        gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        retrieved_ids = np.array([[0, 1, 2, 3, 4, 99, 98, 97, 96, 95]])
        recall = compute_recall_at_k(retrieved_ids, gt_ids, k=10)
        self.assertAlmostEqual(recall, 0.5, places=6)

    def test_tie_at_10th_neighbor(self):
        """
        Candidate #99 is NOT in gt_ids[:10], but its distance is exactly identical
        to the 10th ground truth neighbor. It should be counted as valid, yielding 1.0 recall.
        """
        gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 99]])
        gt_dists = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0]])

        # Retrieved list contains 99 instead of 9, with identical distance 1.0
        retrieved_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 99]])
        retrieved_dists = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        recall = compute_recall_at_k(
            retrieved_ids=retrieved_ids,
            gt_ids=gt_ids,
            k=10,
            retrieved_dists=retrieved_dists,
            gt_dists=gt_dists,
        )
        self.assertAlmostEqual(recall, 1.0, places=6)

    def test_worse_distance_not_tied(self):
        """Candidate #99 has distance 1.005 which exceeds 1.0 + tol (1e-5). Should NOT match."""
        gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        gt_dists = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        retrieved_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 99]])
        retrieved_dists = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.005]])

        recall = compute_recall_at_k(
            retrieved_ids=retrieved_ids,
            gt_ids=gt_ids,
            k=10,
            retrieved_dists=retrieved_dists,
            gt_dists=gt_dists,
        )
        self.assertAlmostEqual(recall, 0.9, places=6)

    def test_deduplication(self):
        """Duplicate retrieved IDs should not inflate recall."""
        gt_ids = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        # Contains repeated 0's
        retrieved_ids = np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        recall = compute_recall_at_k(retrieved_ids, gt_ids, k=10)
        # Unique matches are {0, 1} -> 2/10 = 0.2
        self.assertAlmostEqual(recall, 0.2, places=6)


if __name__ == "__main__":
    unittest.main()
