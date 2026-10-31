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

# Browser User-Agent header required because ann-benchmarks.com blocks default Python-urllib user-agent with 403 Forbidden
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def download_dataset(dataset_name: str, data_dir: str = "data") -> str:
    """
    Downloads dataset HDF5 file from ann-benchmarks repository if not present.
    Uses browser User-Agent to avoid 403 Forbidden errors from the host server.
    """
    os.makedirs(data_dir, exist_ok=True)
    filename = f"{dataset_name}.hdf5"
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"Dataset '{filename}' already exists at {filepath} ({os.path.getsize(filepath) / (1024*1024):.1f} MB).")
        return filepath

    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_URLS.keys())}")

    url = DATASET_URLS[dataset_name]
    tmp_filepath = filepath + ".part"
    print(f"Downloading {dataset_name} from {url} to {filepath}...")

    # Attempt download via urllib with custom User-Agent
    try:
        req = urllib.request.Request(url, headers=BROWSER_HEADERS)
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_filepath, "wb") as f_out:
            total_size = int(resp.headers.get("Content-Length", 0))
            with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=filename, miniters=1) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)  # 1MB buffer
                    if not chunk:
                        break
                    f_out.write(chunk)
                    pbar.update(len(chunk))

        if os.path.exists(tmp_filepath) and os.path.getsize(tmp_filepath) > 1000:
            os.replace(tmp_filepath, filepath)
            print(f"Successfully downloaded {filename} ({os.path.getsize(filepath) / (1024*1024):.1f} MB).")
            return filepath
    except Exception as e:
        print(f"urllib download failed: {e}. Trying fallback download...")
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)

    # Fallback via requests if available
    try:
        import requests
        resp = requests.get(url, headers=BROWSER_HEADERS, stream=True, timeout=60)
        resp.raise_for_status()
        total_size = int(resp.headers.get("content-length", 0))
        with open(tmp_filepath, "wb") as f_out, tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f_out.write(chunk)
                    pbar.update(len(chunk))
        os.replace(tmp_filepath, filepath)
        print(f"Successfully downloaded {filename} via requests fallback.")
        return filepath
    except Exception as e:
        print(f"requests download fallback failed: {e}. Trying curl...")
        if os.path.exists(tmp_filepath):
            os.remove(tmp_filepath)

    # Fallback via curl command line (standard on Linux/Colab)
    curl_cmd = f'curl -L -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -o "{filepath}" "{url}"'
    ret = os.system(curl_cmd)
    if ret == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"Successfully downloaded {filename} via curl fallback.")
        return filepath

    raise RuntimeError(f"Failed to download {dataset_name} from {url}. Please download manually to {filepath}.")


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
