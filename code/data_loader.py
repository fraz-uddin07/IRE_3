"""
data_loader.py
==============
Utilities for downloading, caching, and loading ANN benchmark datasets:
- SIFT (1M descriptors, 128d, Euclidean)
- GloVe-100 (1.18M word vectors, 100d, Angular -> L2 normalized to Euclidean)

Splits queries in half:
- First 5,000 queries: dev / parameter tuning
- Last 5,000 queries: final evaluation reporting
"""

import os
import urllib.request
import h5py
import numpy as np
from tqdm import tqdm


DATASET_URLS = {
    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset(dataset_name: str, data_dir: str = "data") -> str:
    """Downloads dataset HDF5 file from ann-benchmarks repository if not present."""
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{dataset_name}.hdf5"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"Dataset '{filename}' already exists at {filepath}.")
        return filepath

    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_URLS.keys())}")

    url = DATASET_URLS[dataset_name]
    print(f"Downloading {dataset_name} from {url} to {filepath}...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to)
    print("Download completed successfully.")
    return filepath


def normalize_l2(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalizes vectors along axis 1 to have unit L2 norm."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (vectors / norms).astype(np.float32)


def load_dataset(
    dataset_name: str,
    data_dir: str = "data",
    max_train: int = None,
    max_test: int = None,
):
    """
    Loads dataset from HDF5 file, normalizes GloVe vectors, and splits queries in half.

    Returns:
        dict: {
            'train': (N, d) float32,
            'queries_dev': (5000, d) float32,
            'queries_test': (5000, d) float32,
            'gt_dev': (5000, 100) int32,
            'gt_test': (5000, 100) int32,
            'dist_dev': (5000, 100) float32,
            'dist_test': (5000, 100) float32,
            'dim': int,
            'metric': str
        }
    """
    filepath = download_dataset(dataset_name, data_dir=data_dir)

    print(f"Reading {filepath}...")
    with h5py.File(filepath, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
        distances = np.array(f["distances"], dtype=np.float32)

    # Subsample if quick mode requested
    if max_train is not None and max_train < len(train):
        print(f"Subsampling train set from {len(train)} to {max_train} vectors.")
        train = train[:max_train]

    # GloVe normalisation requirement:
    # "Normalise the GloVe vectors and use Euclidean distance for everything."
    if "glove" in dataset_name.lower() or "angular" in dataset_name.lower():
        print("Normalizing GloVe vectors to unit L2 norm (for Euclidean distance equivalence)...")
        train = normalize_l2(train)
        test = normalize_l2(test)

    # Split queries in half (5,000 dev, 5,000 test)
    # "Split the queries in half: tune anything you need on the first 5,000,
    # and report your final numbers on the last 5,000."
    half = len(test) // 2
    queries_dev = test[:half]
    queries_test = test[half:]

    gt_dev = neighbors[:half]
    gt_test = neighbors[half:]

    dist_dev = distances[:half]
    dist_test = distances[half:]

    if max_test is not None:
        queries_test = queries_test[:max_test]
        gt_test = gt_test[:max_test]
        dist_test = dist_test[:max_test]

    return {
        "dataset_name": dataset_name,
        "train": train,
        "queries_dev": queries_dev,
        "queries_test": queries_test,
        "gt_dev": gt_dev,
        "gt_test": gt_test,
        "dist_dev": dist_dev,
        "dist_test": dist_test,
        "dim": train.shape[1],
        "metric": "l2",
    }
