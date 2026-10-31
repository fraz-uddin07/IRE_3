# CS4.406 IRE Assignment 3: Benchmarking Approximate Nearest Neighbour Search

This directory contains the complete, modular codebase for benchmarking Approximate Nearest Neighbour (ANN) search algorithms (HNSW, IVF-Flat, IVF-PQ) against an exact FAISS FlatL2 baseline on **SIFT-1M** and **GloVe-100**, as well as experimental investigations into dataset scaling and high-dimensional distance concentration.

---

## 1. Directory Structure

```text
code/
├── README.md               # This documentation and reproduction guide
├── requirements.txt        # Python package dependencies
├── hypothesis.md           # Pre-committed hypotheses for Parts 3 and 4
├── run_all.sh              # One-command reproduction script (supports --quick)
├── metrics.py              # Custom recall@10 implementation with tie-handling
├── test_metrics.py         # Unit tests for recall metric and tie resolution
├── data_loader.py          # Dataset download, L2 normalization, query splitting
├── part1_exact.py          # Exact baseline verification and sequential vs batch timing
├── part2_benchmark.py      # Sweeps for HNSW, IVF-Flat, IVF-PQ on SIFT & GloVe
├── part3_scaling.py        # Part 3: Dataset scaling on SIFT (100K, 300K, 1M)
├── data_generator.py       # Part 4: Standalone seeded synthetic hypersphere generator
├── part4_geometry.py       # Part 4: Hard geometry (Curse of Dimensionality) benchmark
└── plot_utils.py           # Publication plotting utilities and CSV logger

```

Associated Deliverable Folders (at repository root):
```text
├── report.pdf              # Final 5-page report answering all 5 questions
├── results/
│   ├── results.csv         # Full log of every benchmark configuration run
│   └── figures/            # High-resolution PNG plots for Parts 2, 3, and 4
└── ai_log/
    ├── ai_statement.md     # Declaration categorizing every source file
    └── prompts_and_responses.md  # Record of AI prompts and technical exchanges
```

---

## 2. Environment Setup

### Local / Linux / Laptop
```bash
# Recommended Python 3.10, 3.11, or 3.12
pip install -r requirements.txt
```

### Google Colab
To run on Google Colab, open a notebook (CPU or GPU runtime) and run:
```python
# Cell 1: Install dependencies
!pip install -q faiss-cpu h5py tabulate fpdf2

# Cell 2: Run quick reproduction (<15 minutes)
!cd code && bash run_all.sh --quick

# Cell 3: Or run full benchmark (~45-60 minutes)
!cd code && bash run_all.sh
```

---

## 3. One-Command Reproduction

The assignment requires one-command reproduction from inside `code/`:

### Quick Reproduction
Runs the full pipeline on a 100,000-vector subset with 1,000 evaluation queries:
```bash
cd code
bash run_all.sh --quick
```
This executes:
1. `test_metrics.py`: Runs all 6 unit tests for `recall@10` with tie-breaking.
2. `part1_exact.py`: Verifies exact search agreement and measures sequential vs batch throughput.
3. `part2_benchmark.py`: Runs sweeps for HNSW, IVF-Flat, IVF-PQ on SIFT & GloVe, generating Pareto curves.
4. `part3_scaling.py`: Runs dataset scaling ($N \in \{25\text{K}, 50\text{K}, 100\text{K}\}$).
5. `part4_geometry.py`: Runs synthetic hypersphere geometry tests ($d \in \{16, 64, 128, 256\}$).

### Full Benchmark
Runs the complete benchmark on all 1,000,000 SIFT vectors, 1,183,514 GloVe vectors, and 5,000 test queries:
```bash
cd code
bash run_all.sh
```

---

## 4. Running Individual Components

Each script can also be executed independently:

```bash
# Run unit tests
python test_metrics.py

# Run Part 1 Exact Baseline
python part1_exact.py --dataset sift-128-euclidean
python part1_exact.py --dataset glove-100-angular

# Run Part 2 Benchmark Sweeps
python part2_benchmark.py --dataset both

# Run Part 3 Scaling Experiment
python part3_scaling.py

# Run Part 4 Hard Geometry Experiment
python part4_geometry.py

```

---

## 5. Summary of Experimental Results

- **Part 1**: Exact search achieves 100% agreement with precomputed ground truth. Batch search achieves **31x higher throughput** than sequential 1-by-1 queries due to BLAS GEMM cache-blocking and amortized DRAM bandwidth.
- **Part 2**: HNSW dominates the Pareto frontier on both SIFT (2,250 QPS @ 0.95 recall) and GloVe (890 QPS @ 0.95 recall). IVF-PQ fails to reach 0.95 recall due to asymmetric distance quantization noise.
- **Part 3**: Exact search latency scales linearly $O(N)$, IVF-Flat scales as $O(N/nlist)$, while HNSW exhibits logarithmic $O(\log N)$ latency scaling at the expense of higher build time and memory.
- **Part 4**: As ambient dimension $d$ scales from 16 to 256 on $\mathbb{S}^{d-1}$, relative distance contrast collapses by 84%, causing HNSW recall under fixed search budget to collapse from 0.99 to 0.44 due to vanishing directional gradients.
