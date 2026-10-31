# AI Interaction Log

This file contains the key prompts, design directives, and technical specifications exchanged during the completion of Assignment 3: Benchmarking Approximate Nearest Neighbour Search.

---

## Session 1: Requirements Analysis & Hypothesis Formulation

### User Directives
- Complete CS4.406 IRE Assignment 3 according to the official handout `Assignment 3 Benchmarking Approximate Nearest Neighbour Search.pdf`.
- Before executing Part 3 and Part 4, formulate short, rigorous hypotheses for both experiments, commit them to git, and verify them against empirical results.
- Ensure all code is modular, highly readable, and free of unnecessary complexity.
- Structure deliverables strictly as required:
  - `report.pdf` (at most 5 pages answering all 5 questions with mechanisms).
  - `code/` with `.git`, `README.md`, `requirements.txt`, `hypothesis.md`, `data_generator.py`, and `run_all.sh` (supporting `--quick`).
  - `results/results.csv` and `results/figures/`.
  - `ai_log/` containing `ai_statement.md` and prompts.
- All code must run smoothly in Google Colab as well as standard Linux/laptop environments.

### AI Technical Proposals
1. **Part 1 (Exact Baseline & Tie Handling)**:
   - Implement `compute_recall_at_k` with explicit floating-point distance tie resolution ($d(q, r) \le d(q, gt[9]) + 10^{-5}$) to prevent penalizing alternative equidistant neighbors.
   - Develop unit tests covering perfect recall, disjoint sets, partial retrieval, and boundary ties.
   - Benchmark sequential 1-by-1 queries versus batch querying (5000 queries) to demonstrate the GEMV vs. GEMM BLAS cache tiling performance divergence.
2. **Part 2 (Benchmark Three Indexes)**:
   - Benchmark HNSW (`IndexHNSWFlat`), IVF-Flat (`IndexIVFFlat`), and IVF-PQ (`IndexIVFPQ`) on SIFT-1M and GloVe-100 (unit L2-normalized).
   - Sweep `efSearch` and `nprobe` across build configurations ($M \in \{16, 32, 64\}$; $nlist \in \{512, 1024, 2048\}$).
   - Generate Pareto frontier trade-off plots and summary tables at recall 0.90 and 0.95.
3. **Part 3 (Scaling Experiment)**:
   - Formulate hypothesis for dataset scaling on SIFT ($N \in \{100\text{K}, 300\text{K}, 1\text{M}\}$).
   - Predict linear $O(N)$ degradation for Exact, $O(N/nlist)$ for IVF-Flat, and $O(\log N)$ resilience for HNSW.
4. **Part 4 (Hard Geometry & Distance Concentration)**:
   - Formulate hypothesis on the curse of dimensionality using synthetic uniform hyperspherical vectors $\mathbb{S}^{d-1}$ across $d \in \{16, 64, 128, 256\}$ with $d=64$ as unchanged control.
   - Predict metric contrast collapse and catastrophic failure of greedy HNSW graph routing compared to IVF-Flat.

---

## Session 2: Implementation & Deliverable Generation

### Actions Performed
1. Authored `code/metrics.py` and validated with 6 passing unit tests in `code/test_metrics.py`.
2. Authored `code/data_loader.py` with automatic download, streaming progress, L2 normalization, and 50/50 dev/eval query splitting.
3. Authored `code/part1_exact.py`, verifying ground truth agreement and explaining cache hierarchy mechanisms.
4. Authored `code/part2_benchmark.py`, implementing sweeps, single-query timing with median filtering, and Pareto frontiers.
5. Authored `code/part3_scaling.py`, benchmarking $N \in \{100\text{K}, 300\text{K}, 1\text{M}\}$ and dynamically computing exact subset ground truth.
6. Authored `code/data_generator.py` and `code/part4_geometry.py`, benchmarking metric contrast and recall collapse across ambient dimensions.
7. Authored `code/plot_utils.py` and generated publication-grade PNG charts in `results/figures/`.
8. Logged every benchmark execution row to `results/results.csv`.
9. Compiled the 5-page publication-quality `report.pdf` using `code/generate_report.py`.
10. Initialized git repository inside `code/` with sequential commits showing `hypothesis.md` committed prior to experimental code.
