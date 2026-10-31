# Hypotheses for Assignment 3: Benchmarking Approximate Nearest Neighbour Search

This document presents the hypotheses prepared before running the experiments for **Part 3** (Experiments on standard data) and **Part 4** (Experiments on custom synthetic data), as mentioned in the question.

---

## Part 3 Hypothesis: Dataset Scaling on SIFT-1M

### Selected Experiment

**Scaling:** Compare HNSW, IVF-Flat, and Exact Search using SIFT datasets containing \(N \in \{100K, 300K, 1M\}\) vectors.

The goal is to study how query latency, QPS, index build time, and memory usage change as the dataset grows, while maintaining a recall@10 of at least 0.95.

### Background and Motivation

Exact search calculates the distance between the query and all \(N\) vectors, requiring \(O(N \cdot d)\) operations.

Approximate search methods reduce the number of vectors that need to be checked:

- **IVF-Flat:** Divides vectors into `nlist` clusters and searches only the `nprobe` closest clusters.
- **HNSW:** Builds a multi-layer graph and uses greedy search to quickly find nearby vectors.

### Formal Hypothesis

1. **Exact Search Latency:**
   - Query latency should increase linearly (\(O(N)\)) as the dataset grows from 100K to 1M vectors.
   - As a result, QPS is expected to decrease by approximately 10 times.

2. **IVF-Flat Latency:**
   - If the number of clusters (`nlist`) remains fixed, each cluster will contain more vectors as the dataset grows.
   - To maintain recall of at least 0.95, more vectors will need to be checked for each query.
   - Therefore, QPS will decrease significantly unless `nlist` is also increased.

3. **HNSW Query Performance:**
   - HNSW query latency is expected to grow slowly, approximately as \(O(\log N)\).
   - It should maintain higher QPS than IVF-Flat and remain much faster than Exact Search.
   - This is because its graph structure allows it to quickly move between distant regions and reach the area containing the nearest neighbours.

4. **Build Time and Memory Usage:**
   - HNSW build time is expected to grow approximately as \(O(N \log N)\).
   - Building the graph takes more time because it involves adding connections and selecting suitable neighbours.
   - HNSW is also expected to use around 1.5–2 times more memory than IVF-Flat because it stores graph connections.

---

## Part 4 Hypothesis: Hard Geometry and Curse of Dimensionality on Synthetic Data

### Selected Experiment

**Hard Geometry (Dimensionality and Distance Concentration):**

Study how increasing the number of dimensions affects search performance and recall@10.

We will generate synthetic datasets containing \(N = 100,000\) points, uniformly distributed on a unit hypersphere \(\mathbb{S}^{d-1}\).

- **Test dimensions:** \(d \in \{16, 64, 128, 256\}\)
- **Control setting:** \(d = 64\) (baseline)
- **Indices compared:** HNSW (`IndexHNSWFlat`) and IVF-Flat (`IndexIVFFlat`)

### Background and Mechanism

As the number of dimensions (\(d\)) increases, distances between points become more similar, also known as **distance concentration**.

\[
\lim_{d \to \infty}
\frac{\text{dist}_{\max}(q)-\text{dist}_{\min}(q)}
{\text{dist}_{\min}(q)} \to 0
\]

In simple terms, points become almost equally distant from the query. This makes it harder for search algorithms to identify the nearest neighbours.

This affects the two indexing methods differently:

- **HNSW:** Uses greedy search, where each step moves to a neighbour that is closer to the query. When distances become similar, it becomes difficult to identify a better neighbour. As a result, the search may get stuck in local minima and miss the actual nearest neighbours.

- **IVF-Flat:** Divides the space into clusters using Voronoi cells. With higher dimensions, the nearest neighbours may be spread across several clusters instead of being concentrated in the closest cluster. This makes it harder to find them without searching more clusters.

### Formal Hypothesis

1. **HNSW Recall Degradation:**
   - With a fixed search budget (e.g., `efSearch = 32`), HNSW recall@10 is expected to decrease as dimensions increase from 16 to 256.
   - At \(d = 16\), recall should be above 0.95.
   - At \(d = 256\), recall may fall below 0.60 because greedy search struggles to find better neighbours when distances become similar.

2. **IVF-Flat Performance:**
   - IVF-Flat is expected to show a smaller drop in recall than HNSW when using low search settings.
   - This is because it performs exact distance calculations within the selected clusters instead of relying on greedy graph traversal.
   - However, to achieve high recall (at least 0.90), it will need to search a much larger fraction of the `nlist` clusters, which will reduce QPS.

3. **Comparison of Both Methods:**
   - HNSW is expected to be more affected by increasing dimensionality than IVF-Flat.
   - As dimensions increase, the performance gap between HNSW and IVF-Flat is expected to become smaller.
   - In particular, HNSW's advantage in terms of QPS for a given recall level is expected to decrease.
