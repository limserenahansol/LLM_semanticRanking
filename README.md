# Research Aggregator (group3)

Colab-export script: **`research_aggregator_group3.py`** — multi-source scraping (Scholar, X, Threads, web), **BGE + BM25 + keyword** fusion (**RRF** or **weighted min–max**), optional **top-K cross-encoder** rerank, reliability scoring, dedup, abstract summarization, SQLite cache.

## Documentation

| Doc | Language | Contents |
|-----|----------|----------|
| [research_aggregator_RANKING_UPDATE_KR.md](./research_aggregator_RANKING_UPDATE_KR.md) | Korean | BGE/BM25/RRF/CE, fallbacks, tuning |
| [research_aggregator_WORKFLOW_README_KR.md](./research_aggregator_WORKFLOW_README_KR.md) | Korean | Pipeline, Mermaid ranking subflow |

## Ranking stack (quick)

| Component | Default |
|-----------|---------|
| Bi-encoder | `BAAI/bge-base-en-v1.5` — query: `Represent this sentence for searching relevant passages: …` |
| Lexical | `rank-bm25` / `BM25Okapi` on the current result batch |
| Extra signal | Heuristic English keyword overlap |
| Merge | `RANK_MERGE_MODE = "rrf"` (or `"weighted_norm"`) |
| Rerank | `RANK_CROSS_ENCODER_TOP_K = 40` with `cross-encoder/ms-marco-MiniLM-L-6-v2`; set **`0`** to disable |
| Final | `raw_score` = min-max of fused scores (after optional CE blend) |

**GTE:** You can point `RANK_BI_ENCODER_MODEL_NAME` to e.g. `Alibaba-NLP/gte-base-en-v1.5` and adjust/remove `BGE_QUERY_PREFIX` per that model’s card.

**Cache** query similarity still uses **multilingual** `paraphrase-multilingual-MiniLM-L12-v2` (separate from the ranker).

## Dependencies

First cell installs `rank-bm25` in addition to `sentence-transformers`, `torch`, etc.

Run in Colab as intended, or set `DB_PATH` / API keys for local use.
