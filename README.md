# BTP-II: Label-Free Geometric Debiasing (LFGD)

This repository implements the BTP-II label-free reranking pipeline for fairness-aware RAG on `wwbrannon/twinviews-13k`.

## Setup

```bash
uv sync
cp .env.example .env
```

Fill `.env` with credentials/runtime toggles (model/provider choices live in config YAML).  
Main variables:

- Provider keys: `GROQ_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
- Preprocess toggle: `USE_MODAL_EMBEDDING`
- Modal preprocess batching: `MODAL_EMBED_BATCH_SIZE`
- Qdrant setup: `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `QDRANT_PATH`
- Qdrant upload tuning: `QDRANT_PREFER_GRPC`, `QDRANT_TIMEOUT_SEC`, `QDRANT_UPSERT_BATCH_SIZE`, `QDRANT_UPSERT_TIMEOUT_SEC`

Set synth/judge/fairrag models in `experiments/configs/default.yaml` under `llm`.
Default config now runs **LFGD only** (`systems: [lfgd]`) for cross-LLM synth comparisons.

## Data preparation

```bash
uv run python data/download.py
uv run python data/preprocess.py
```

`data/preprocess.py` will use `data/twinviews-13k.csv` if present (from `data/download.py`), and only falls back to HuggingFace when the local CSV is missing.
It writes BM25 artifacts to `data/` and indexes dense vectors into Qdrant (`QDRANT_URL`/`QDRANT_API_KEY` or local `QDRANT_PATH`).

`run_eval.py` supports embedding caches via config:
- `retrieval.use_preprocessed_corpus: true`
- `retrieval.use_embedding_cache: true`
- `retrieval.embedding_cache_dir: data/cache`
- `retrieval.reindex_qdrant_on_eval: false` (reuse existing Qdrant index; do not upsert every run)

With `use_preprocessed_corpus`, retrieval runs against the large preprocessed corpus (`data/documents.json` + `data/embeddings.npy`) instead of a tiny per-run subset.
Caching avoids re-encoding docs/queries (and repeated MiniLM initialization) on repeated runs with the same model+topic set.

## Run evaluation

```bash
# full run
uv run python experiments/run_eval.py --config experiments/configs/default.yaml

# 30-topic subset
uv run python experiments/run_eval.py --config experiments/configs/default.yaml --n-topics 30
```

## Run ablations

```bash
uv run python experiments/ablations.py --config experiments/configs/ablation_grid.yaml
```

## Analysis

```bash
uv run python analysis/axis_correlation.py --results-dir results/
uv run python analysis/results_summary.py --results-dir results/
```

For faster axis correlation retrieval on large topic sets, tune batched retrieval:

```bash
uv run python analysis/axis_correlation.py --results-dir results/ --retrieval-batch-size 256
```

To offload embedding generation to Modal GPU:

```bash
uv run python analysis/axis_correlation.py --results-dir results/ --use-modal-embedding --modal-batch-size 512
```

## Tests

```bash
uv run pytest -q
```
