# CLAUDE.md — BTP-II: Label-Free Geometric Debiasing (LFGD) for RAG

## Project Overview

This is BTP-II, a research codebase implementing the **Label-Free Geometric Debiasing (LFGD)** framework for fairness-aware retrieval-augmented generation. It builds directly on BTP-I (FairRAG-Select) and must be able to run the same experimental pipeline for comparison.

**Core idea**: retrieve N document embeddings, estimate a bias axis via PCA on the candidate embedding matrix, assign continuous lean scores, then solve a constrained Wasserstein-balance optimisation to select k documents for the generator context — no labels, no fine-tuning.

---

## BTP-I (FairRAG-Select) — What This Extends

BTP-I is the prior work. Its architecture, preserved here for baselines:

- **Dataset**: `wwbrannon/twinviews-13k` — 13,000 paired liberal/conservative political texts across 82 topics, each document tagged `[l]` or `[r]`.
- **Retriever**: Hybrid BM25 + dense embedding similarity. Fetches top-N chunks per query.
- **Reranker** (BTP-I approach): An LLM prompted to select exactly k/2 liberal and k/2 conservative documents from the labeled pool, then interleave them.
- **Metric**: **FAIR score** — position-discounted KL divergence between prefix group distribution and ideal 50/50 at each rank position.
- **Generator**: One of four LLMs (llama-3.1-8b-instant, gpt-oss-20b, gpt-oss-120b, gpt-5-nano). Produces a synthesis from the selected context.
- **Evaluation**: Judge LLM classifies output into 8 bins (Strongly Liberal → Strongly Conservative + Mixed/Unclear). "Neutral" count is the primary KPI.
- **Best result**: llama-3.1-8b reranker + llama-3.1-8b generator → 26/30 neutral (87%) on 30-topic eval.

**Critical constraint in BTP-I**: documents must carry explicit `[l]`/`[r]` labels. LFGD removes this.

---

## LFGD Pipeline (BTP-II)

```
Query
  │
  ▼
[Retriever] ──── top-N chunks (N=20) + embeddings + relevance scores r_i
  │
  ▼
[PCA Bias Axis Estimator]
  - Mean-centre embedding matrix V_c = V - μ
  - Thin SVD of V_c (dual form, O(N²d) not O(d³))
  - u₁ = first right singular vector  →  bias axis
  - Sign resolution: top-N/2 retrieved docs project to positive lean
  │
  ▼
[Lean Score Computation]
  - lᵢ = u₁ᵀ (vᵢ - μ)
  - Min-max normalise to [-1, 1]
  - Variance gate: if var(l) < τ=0.05 → fall back to top-k, skip debiasing
  │
  ▼
[Balanced Set-Selection]
  - Objective: L(S) = α·L_utility(S) + (1-α)·L_fairness(S)
  - L_utility = 1 - mean(rᵢ / max r) for i ∈ S
  - L_fairness = W₁(P_S, Uniform[-1,1])  (closed-form via sorted lean scores)
  - Exact search: C(20,6)=38,760 subsets, <50ms on CPU
  - Greedy fallback for larger N/k
  - Default α=0.5
  │
  ▼
[Context Assembly]
  - Sort selected docs by lean score
  - Interleave from both ends (alternating poles, not round-robin by label)
  │
  ▼
[Generator LLM]  →  synthesis
  │
  ▼
[Judge LLM]  →  8-bin political lean classification
```

---

## Repository Structure

```
btp-ii/
├── CLAUDE.md                    ← this file
├── README.md
├── pyproject.toml               # uv-managed dependencies
├── uv.lock
├── .env.example
│
├── data/
│   ├── download.py              # fetch twinviews-13k from HuggingFace
│   └── preprocess.py            # chunk, embed, build BM25 index + FAISS index
│
├── retrieval/
│   ├── __init__.py
│   ├── bm25.py                  # BM25Okapi wrapper
│   ├── dense.py                 # sentence-transformers embedding + cosine sim
│   └── hybrid.py                # reciprocal rank fusion of BM25 + dense scores
│
├── lfgd/
│   ├── __init__.py
│   ├── pca_axis.py              # bias axis estimation (thin SVD, sign resolution)
│   ├── lean_score.py            # lean score computation + normalisation + variance gate
│   ├── selection.py             # exact combinatorial search + greedy fallback
│   ├── objective.py             # L_utility, L_fairness (Wasserstein-1), L_combined
│   └── assemble.py              # interleaved context assembly from selected set
│
├── metrics/
│   ├── __init__.py
│   ├── fair.py                  # original FAIR score (requires labels, used for eval only)
│   ├── cfair.py                 # C-FAIR score (label-free, position-discounted Wasserstein)
│   └── lean_variance.py         # diagnostic: lean score variance of selected set
│
├── baselines/
│   ├── __init__.py
│   ├── unmitigated.py           # plain top-k retrieval
│   ├── refarag.py               # ReFaRAG probabilistic single-doc reranking
│   └── fairrag_select.py        # BTP-I LLM set-selection (requires labels)
│
├── generation/
│   ├── __init__.py
│   ├── generator.py             # LLM synthesis call (context → answer)
│   └── judge.py                 # LLM judge: 8-bin political lean classification
│
├── experiments/
│   ├── run_eval.py              # main eval loop over 82 topics
│   ├── ablations.py             # ablation grid (axis source, fairness loss, α, N)
│   └── configs/
│       ├── default.yaml
│       └── ablation_grid.yaml
│
├── analysis/
│   ├── axis_correlation.py      # H3: cosine(u₁, label-derived axis) per query
│   └── results_summary.py       # aggregate neutrality rates, FAIR, C-FAIR tables
│
└── tests/
    ├── test_pca_axis.py
    ├── test_lean_score.py
    ├── test_selection.py
    └── test_cfair.py
```

---

## Key Implementation Details

### PCA Bias Axis (`lfgd/pca_axis.py`)

- Input: `V` of shape `(N, d)` where N=20, d=384 (MiniLM) or 1024 (BGE-Large)
- Use `numpy.linalg.svd(V_c, full_matrices=False)` — this gives thin SVD directly
- `u₁ = Vt[0]` (first row of Vt, i.e. first right singular vector)
- **Sign resolution**: compute `np.dot(V_c[:N//2].mean(axis=0), u₁)` and flip u₁ if negative
- Do not compute the d×d covariance matrix — dual form only

### Lean Score Computation (`lfgd/lean_score.py`)

- `l = V_c @ u₁` — shape `(N,)`
- Min-max to `[-1, 1]`: `l_norm = (l - l.min()) / (l.max() - l.min()) * 2 - 1`
- Variance gate: `if np.var(l_norm) < tau: return None` → caller falls back to top-k

### Wasserstein-1 Loss (`lfgd/objective.py`)

For a selected set of k lean scores (sorted ascending as `l_(1) ≤ ... ≤ l_(k)`):

```python
def wasserstein1_uniform(lean_scores_selected):
    k = len(lean_scores_selected)
    sorted_l = np.sort(lean_scores_selected)
    # quantiles of Uniform[-1, 1] at midpoints of k equal intervals
    target = np.array([-1 + (2*j - 1)/k for j in range(1, k+1)])
    return np.mean(np.abs(sorted_l - target))
```

### Exact Selection (`lfgd/selection.py`)

```python
from itertools import combinations
best_S, best_loss = None, float('inf')
for S in combinations(range(N), k):
    loss = combined_loss(S, lean_scores, relevance_scores, alpha)
    if loss < best_loss:
        best_S, best_loss = S, loss
```

C(20,6) = 38,760 iterations. Keep it simple and exact. Only switch to greedy if N > 30 or k > 8.

### C-FAIR Score (`metrics/cfair.py`)

```
C-FAIR = (1/M) * Σᵢ₌₁ᵏ [gᵢ / log₂(i+1)] * [1 / (W₁(P_i, U[-1,1]) + 1)]
```

- `gᵢ` = cosine similarity of doc i with query (relevance gain)
- `P_i` = empirical distribution of lean scores of first i docs in assembled context
- `W₁(P_i, U[-1,1])` = same closed-form as above but over i points
- `M` = normalisation: C-FAIR of hypothetical perfectly-relevant, perfectly-balanced list (all gᵢ=1, lean scores = `[-1 + (2j-1)/k for j in 1..k]`)

### Context Interleaving (`lfgd/assemble.py`)

```python
sorted_by_lean = sorted(selected_docs, key=lambda d: d.lean_score)
interleaved = []
lo, hi = 0, len(sorted_by_lean) - 1
while lo <= hi:
    if lo == hi:
        interleaved.append(sorted_by_lean[lo]); break
    interleaved.append(sorted_by_lean[hi])   # most positive first
    interleaved.append(sorted_by_lean[lo])   # most negative second
    lo += 1; hi -= 1
```

---

## Experimental Configuration

### Default (match BTP-I for direct comparison)

| Parameter | Value |
|-----------|-------|
| N (candidates) | 20 |
| k (selected) | 6 |
| α | 0.5 |
| τ (variance gate) | 0.05 |
| Embedding model | all-MiniLM-L6-v2 (d=384) |
| Topics | 82 (full twinviews-13k) |
| Eval subset | 30 topics (same as BTP-I) |

### LLM Configs

| Model | Provider | Notes |
|-------|----------|-------|
| `llama-3.1-8b-instant` | Groq | Default synth |
| `openai/gpt-oss-20b` | OpenRouter | Baseline synth |
| `openai/gpt-oss-120b` | OpenRouter | Baseline synth |
| `gpt-5.4-mini` | OpenAI API | Judge (medium reasoning) |

### Ablation Grid

| Dimension | Values |
|-----------|--------|
| Bias axis source | PCA (default), random direction, query-doc difference vector |
| Fairness loss | Wasserstein-1 (default), variance maximisation, KL on binned scores |
| α | 0.3, 0.5, 0.7, 1.0 |
| N | 10, 20, 30 |
| Embedding model | MiniLM-L6-v2 (d=384), BGE-Large (d=1024) |

---

## Hypotheses to Validate

| ID | Claim | Measurement |
|----|-------|-------------|
| H1 | LFGD C-FAIR ∈ [0.85, 0.92] ≈ BTP-I FAIR ∈ [0.87, 0.89] | C-FAIR + FAIR (using withheld labels) on 82 topics |
| H2 | LFGD neutral outputs ≥ 22/30 (vs BTP-I 26/30) | Judge LLM 8-bin classification on 30-topic eval |
| H3 | cosine(u₁, label-derived axis) > 0.7 mean | `analysis/axis_correlation.py` |
| H4 | LFGD >> FairRAG-Select on unlabeled corpus | Rerun with labels stripped; FairRAG-Select degrades to random |

---

## Environment Setup

Uses [uv](https://docs.astral.sh/uv/) for environment and dependency management.

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync

# Copy and fill in secrets
cp .env.example .env   # fill in GROQ_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, HF_TOKEN

# Fetch dataset and build indexes
uv run python data/download.py
uv run python data/preprocess.py   # builds FAISS index + BM25 index, saves to data/
```

**`pyproject.toml`** dependencies block:
```toml
[project]
name = "btp-ii-lfgd"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sentence-transformers>=2.7.0",
    "faiss-cpu>=1.8.0",
    "rank-bm25>=0.2.2",
    "numpy>=1.26.0",
    "scipy>=1.13.0",
    "datasets>=2.19.0",
    "openai>=1.30.0",
    "groq>=0.9.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
]

[dependency-groups]
dev = ["pytest>=8.0.0"]
```

Add new deps with `uv add <package>`, never edit `uv.lock` by hand.

---

## What LFGD Does NOT Do

- It does **not** read document labels at inference time (labels are loaded only into the evaluator, never into the reranker)
- It does **not** fine-tune any model
- It does **not** call an LLM for reranking (PCA + optimisation replaces the LLM reranker entirely)
- It does **not** modify the embedding model weights

---

## Code Style

- Python 3.11+
- Type hints on all function signatures
- Each module is independently importable and testable
- No global state; pass config dicts explicitly
- `numpy` for all linear algebra — no `torch` in the LFGD core (keep it lightweight)
- Results written to `results/` as JSON + CSV; never hardcode paths

---

## Running Experiments

```bash
# Full 82-topic evaluation, LFGD vs all baselines
uv run python experiments/run_eval.py --config experiments/configs/default.yaml

# 30-topic eval matching BTP-I comparison set
uv run python experiments/run_eval.py --config experiments/configs/default.yaml --n-topics 30

# Ablations
uv run python experiments/ablations.py --config experiments/configs/ablation_grid.yaml

# Hypothesis H3: axis correlation
uv run python analysis/axis_correlation.py --results-dir results/

# Summary tables
uv run python analysis/results_summary.py --results-dir results/
```

---

## Connection to BTP-I

`baselines/fairrag_select.py` must be a faithful re-implementation of BTP-I's LLM reranker so that both approaches run in the same eval loop with identical retrieval, generation, and judging steps. The only difference between a BTP-I run and a BTP-II LFGD run is the reranker module. This ensures the comparison is apples-to-apples.