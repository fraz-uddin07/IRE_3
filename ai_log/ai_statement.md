# AI Statement

In accordance with the deliverables requirement for CS4.406 Information Retrieval & Extraction Assignment 3: Benchmarking Approximate Nearest Neighbour Search, this document categorizes each file in this submission based on the extent of AI assistance.

## Classification Table

| File Path | Description | Category | Notes / Extent of AI Assistance |
| :--- | :--- | :--- | :--- |
| `code/metrics.py` | Recall@10 with distance tie resolution | AI-assisted | Core algorithm designed jointly; AI assisted with Python optimization and boundary condition verification. |
| `code/test_metrics.py` | Unit tests for tie handling and recall | AI-assisted | Test specifications formulated by student; test cases and assert fixtures generated with AI assistance. |
| `code/data_loader.py` | HDF5 loader, downloader, L2 normalizer | AI-assisted | Implementation of streaming download and unit normalization assisted by AI. |
| `code/part1_exact.py` | Exact FAISS baseline and timing | AI-assisted | Architectural explanation and timing loops developed with AI assistance. |
| `code/part2_benchmark.py` | Sweeps for HNSW, IVF-Flat, IVF-PQ | AI-assisted | Benchmarking scaffolding, Pareto frontier interpolation, and timing harnesses built with AI assistance. |
| `code/part3_scaling.py` | Dataset scaling (100K, 300K, 1M) | AI-assisted | Subsampling logic and on-the-fly exact ground truth calculation designed with AI assistance. |
| `code/data_generator.py` | Synthetic hypersphere generator | AI-assisted | Gaussian sampling and L2 projection written with AI assistance. |
| `code/part4_geometry.py` | Curse of dimensionality benchmark | AI-assisted | Relative contrast measurement and evaluation harness built with AI assistance. |
| `code/plot_utils.py` | Matplotlib visualization & CSV logging | AI-assisted | Publication-quality chart formatting and Pareto curves implemented with AI assistance. |
| `code/run_all.sh` | One-command reproduction script | AI-generated | Shell script automating end-to-end execution of all benchmark stages. |
| `hypothesis.md` | Pre-committed hypotheses (Parts 3 & 4) | AI-assisted | Scientific hypotheses articulated and refined in collaboration with AI. |
| `code/README.md` | Reproduction guide and Colab notes | AI-assisted | Documentation written with AI assistance. |
| `code/requirements.txt` | Environment dependencies | Human-written / AI-assisted | Python environment specification. |
| `report.pdf` | Final 5-page benchmark report | AI-assisted | Mechanistic answers, figures, and tables assembled and reviewed jointly. |
| `results/results.csv` | Full benchmark run logs | AI-generated | Automatically logged by benchmark runs. |
| `results/figures/*.png` | Benchmark trade-off & scaling plots | AI-generated | Automatically rendered by plot_utils.py. |

## Tool Declaration
- **Model Used**: Gemini 3.8 Flash (Advanced Agentic Assistant)
- **Use Cases**: Code structuring, FAISS API parameter sweep implementation, metric tie-breaking logic, visualization design, and compilation of the mechanistic analysis in `report.pdf`.
