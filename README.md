# Research Aggregator (group3)

Colab-export script: **`research_aggregator_group3.py`** — multi-source scraping (Scholar, X, Threads, web), **hybrid semantic ranking** (dense + BM25 + keyword, with optional cross-encoder reranking), reliability scoring, dedup, abstract summarization, and SQLite cache.

## Documentation

| Doc | Language | Contents |
|-----|----------|----------|
| [research_aggregator_RANKING_UPDATE_KR.md](./research_aggregator_RANKING_UPDATE_KR.md) | Korean | BGE/BM25/RRF/CE, fallbacks, tuning |
| [research_aggregator_WORKFLOW_README_KR.md](./research_aggregator_WORKFLOW_README_KR.md) | Korean | Full pipeline, Mermaid diagram |
| [research_aggregator_RANKING_EVAL_KR.md](./research_aggregator_RANKING_EVAL_KR.md) | Korean | 벤치마크로 구버전 vs 하이브리드 랭킹 평가 |

---

## Ranking benchmark & evaluation (legacy vs hybrid)

### Where this lives on GitHub (repo root)

| Path | Purpose |
|------|---------|
| [`ranking_core.py`](./ranking_core.py) | Standalone rankers: **legacy** (MiniLM, title+snippet) vs **hybrid** (BGE + BM25 + keyword, optional CE) |
| [`ranking_metrics.py`](./ranking_metrics.py) | IR metrics: P@k, R@k, NDCG@k, MRR, MAP |
| [`ranking_benchmark.py`](./ranking_benchmark.py) | CLI: run both rankers on the same fixture, print per-query and mean metrics |
| [`fixtures/ranking_eval_sample.json`](./fixtures/ranking_eval_sample.json) | Small **toy** benchmark: 2 queries, graded relevance labels |
| [`requirements-benchmark.txt`](./requirements-benchmark.txt) | Python deps for evaluation only (not the Colab installer cell) |
| [`results/benchmark_results.json`](./results/benchmark_results.json) | Last checked-in machine-readable summary |
| [`results/benchmark_run_log.txt`](./results/benchmark_run_log.txt) | Console log for that run |
| [`research_aggregator_RANKING_EVAL_KR.md`](./research_aggregator_RANKING_EVAL_KR.md) | Korean walkthrough of the benchmark |

These files sit alongside [`research_aggregator_group3.py`](./research_aggregator_group3.py); they **do not** auto-run when you open the Colab script.

---

### English — What the benchmark does

1. **Same candidate pool** for each query: `fixtures/ranking_eval_sample.json` lists `topic`, `results[]`, and `relevance` (URL → grade 0/1/2…).
2. **Two rankers** score and sort that pool independently:
   - **Legacy:** `all-MiniLM-L6-v2`, document text = title + snippet (README baseline).
   - **Hybrid:** `BGE` family (see env vars below) + BM25 + keyword + RRF; cross-encoder off by default in the checked-in run.
3. **Metrics** answer “are relevant documents near the top?” **P@k / R@k** (binary hits in top‑k), **NDCG@k** (graded order quality), **MRR** (rank of first relevant), **MAP**.

**How to read Δ (hybrid − legacy):** positive Δ on `NDCG@k` or `P@k` means hybrid ranked labeled relevant items higher for that query. The script also prints how many queries hybrid “wins” on `NDCG@k`.

**Checked-in example result (honest read):** the saved run used **`BAAI/bge-small-en-v1.5`** for speed (`RANKING_EVAL_HYBRID_MODEL`), **not** the full pipeline default `bge-base`, and only **two toy queries**. On that run, **P@5, R@5, and MRR were identical** between legacy and hybrid (both already put all relevant URLs in the top‑5), while **mean NDCG@5** was slightly **lower** for hybrid (~**0.97** vs **1.0**) because graded relevance cares about **ordering among highly relevant items**—a small reordering lowers NDCG even when P@k is unchanged. **`ndcg_hybrid_wins`: 0 / 2** in [`results/benchmark_results.json`](./results/benchmark_results.json).

**Takeaway:** this commit proves the **evaluation harness works** and documents a **reproducible** comparison; it does **not** claim hybrid is universally “much better” on this tiny set. For a stronger claim, add more labeled queries from real scrapes, use **`BAAI/bge-base-en-v1.5`** (and optionally `--ce`), and report mean ± variance or paired tests.

**Quick start**

```bash
pip install -r requirements-benchmark.txt
python ranking_benchmark.py
python ranking_benchmark.py --k 10 --json-out results/benchmark_results.json
# Faster smoke test (smaller hybrid model):
RANKING_EVAL_HYBRID_MODEL=BAAI/bge-small-en-v1.5 python ranking_benchmark.py --json-out results/benchmark_results.json
```

---

### 한국어 — 벤치마크와 평가 결과 설명

**GitHub 저장소 루트에 추가된 위치**는 위 표와 같습니다. 요약하면: 랭킹 비교 코드(`ranking_*.py`), 예시 라벨(`fixtures/`), 실행 결과(`results/`), 한국어 상세 문서(`research_aggregator_RANKING_EVAL_KR.md`)입니다.

**벤치마크가 하는 일**

1. 같은 후보 목록(`results` 배열)에 대해 **구(legacy)** 와 **신(hybrid)** 랭커를 각각 돌립니다.  
2. **Legacy:** `all-MiniLM-L6-v2`, 문서 = 제목+스니펫.  
3. **Hybrid:** BGE 계열 + BM25 + 키워드 + RRF (저장된 결과에서는 CE 끔, 하이브리드 모델은 속도용 **`bge-small`**).  
4. **지표:** 상위 k개에 정답이 얼마나 모였는지(**P@k, R@k**), 등급까지 고려한 순위 품질(**NDCG@k**), 첫 정답 위치(**MRR**), **MAP**.

**체크인된 `results/benchmark_results.json` 결과를 어떻게 볼까**

- **짧은 예시(쿼리 2개)** 이라서 “실전 전체 성능”을 대표하지는 않습니다.  
- 그 실행에서는 **P@5, R@5, MRR은 legacy와 hybrid가 동일**했습니다(둘 다 관련 URL을 상위 5안에 넣음).  
- **NDCG@5 평균**만 보면 hybrid가 약 **0.97**, legacy가 **1.0**으로, hybrid가 **살짝 낮게** 나왔습니다. 이유는 NDCG가 **등급 2 vs 1** 문서의 **상대 순서**까지 반영하기 때문입니다. 상위권 안에서 순서만 바뀌어도 P@k는 같아도 NDCG는 줄 수 있습니다. **`ndcg_hybrid_wins`: 0 / 2** 입니다.

**정리:** 이 커밋의 목적은 “**평가 파이프라인이 돌아가고, 결과가 파일로 남는다**”는 것을 보여 주는 것이지, **작은 샘플 + small 모델**만으로 hybrid가 항상 압도한다고 주장하는 것은 아닙니다. 더 설득력 있게 하려면 **실제 스크랩 풀**을 JSON으로 저장해 라벨을 늘리고, 파이프라인과 동일하게 **`bge-base`**(필요 시 `--ce`)로 다시 측정하면 됩니다.

---

## Semantic ranking: what we improved (details)

This section explains **how** the ranker works today, **what changed** versus a simpler baseline, and **trade-offs** (pros / cons).

### Baseline (original idea)

Originally the pipeline used a **single bi-encoder** (`all-MiniLM-L6-v2`), encoded the **topic** and each result as `title + snippet`, computed **cosine similarity**, sorted by that score, and fell back to a **crude keyword overlap** (split on spaces) if `sentence-transformers` was missing.

**Limits of that baseline**

- Smaller English models miss nuanced relevance vs stronger retrieval encoders.
- **Title + snippet only** ignores authors and venue strings that often carry topic signal (e.g. “NeurIPS”, lab names).
- **Pure dense similarity** can underweight exact **entity names, acronyms, and rare terms** the user typed.
- **No lexical IDF** within the result set: frequent words in snippets were not distinguished the way classical search (BM25) does.
- **No late reranking**: bi-encoders trade accuracy for speed; the most precise relevance models compare query and document jointly (cross-encoder).

---

### What we changed (overview)

| Area | Before (conceptually) | After (current code) |
|------|------------------------|----------------------|
| **Dense model** | `all-MiniLM-L6-v2` | **`BAAI/bge-base-en-v1.5`** with the official **query prefix**; passages **unprefixed** ([BGE model card](https://huggingface.co/BAAI/bge-base-en-v1.5)). |
| **Document text** | Title + snippet | **Title, snippet, authors, venue** concatenated (`_result_document_text`). |
| **Lexical signal** | None (beyond simple keyword heuristic) | **BM25** on the **current batch** of results via `rank-bm25` / `BM25Okapi`. |
| **Keyword heuristic** | Raw word split overlap | **Token regex**, **English stopwords**, recall + density + **phrase bonus** for long queries. |
| **Combining signals** | Fixed blend of dense + keyword only | **RRF** (default) over **three rankings** (dense, BM25, keyword) **or** **`weighted_norm`** (min–max each score, then 0.55 / 0.35 / 0.10). |
| **Optional precision pass** | None | **Top-K cross-encoder** (`ms-marco-MiniLM-L-6-v2`) reranks the fused top **K** (default 40), blended with fused scores. |
| **Model loading** | Loaded every call (wasteful) | **Lazy singleton** for bi-encoder and cross-encoder. |
| **Fallback** | ImportError only | Bi-encoder failure → **keyword-only**; BM25 failure → BM25 zeros / skip; CE failure → fused list unchanged. |
| **Transparency** | `raw_score` only | **`semantic_score`**, **`bm25_score`**, **`keyword_score`**, optional **`cross_encoder_score`**, then **`raw_score`** (min–max of final fused vector). |

**Note:** **Cache / “similar past query”** matching still uses **`paraphrase-multilingual-MiniLM-L12-v2`** so non-English or paraphrased queries can still hit the cache. That is **intentionally separate** from the English-focused ranker.

---

### How each stage works (mechanics)

1. **Dense (bi-encoder)**  
   - Query string: `BGE_QUERY_PREFIX + topic` (BGE retrieval instruction).  
   - Each document: concatenated fields, truncated for encoding.  
   - Embeddings are **L2-normalized**; **cosine similarity** → `semantic_score` (clipped to [0, 1] for storage).

2. **BM25**  
   - Tokenize query and each document (lowercase alphanumeric).  
   - Build `BM25Okapi` on **this scrape’s documents only** (not the whole web).  
   - Scores reflect **term frequency + batch-level IDF**: rare query terms in a long snippet help; common boilerplate is damped relative to the batch.

3. **Keyword heuristic**  
   - Complements BM25 with a **lightweight** overlap score (stopword filtering, recall/density, substring bonus for long topics).  
   - Cheap and stable when the batch is tiny or BM25 is flat.

4. **Fusion — RRF (default)**  
   - Sort indices by dense, by BM25, and by keyword (three **ordinal** lists).  
   - **Reciprocal Rank Fusion:** each list contributes `1 / (k + rank_position)` with `RRF_K = 60`.  
   - **Why RRF:** scores from dense vs BM25 live on **different scales**; RRF avoids fragile hand-tuned scaling.

5. **Fusion — `weighted_norm` (optional)**  
   - Min–max normalize dense and BM25 to [0, 1] **within the batch**; keyword already in [0, 1].  
   - Linear mix: **0.55·dense + 0.35·BM25 + 0.10·keyword** (tunable via `_WEIGHTED_NORM_*`).  
   - **When useful:** you want an explicit interpretable weighting instead of rank fusion.

6. **Cross-encoder (optional)**  
   - Take the **top K** items by fused score (default K = 40).  
   - Score pairs `(topic, document_text)` with a **cross-encoder** (joint attention over both texts).  
   - Replace fused values for those indices with a blend: **`_CROSS_ENCODER_BLEND`·CE_norm + (1−blend)·fused_norm** on that subset, then **global min–max** → `raw_score`.  
   - **`cross_encoder_score`** stored only for items that were in the top-K CE batch.

7. **Final `raw_score`**  
   - Min–max normalization of the **post-CE** fused vector so displayed relevance stays in **[0, 1]** across the batch.

---

### Pros and cons

#### Stronger bi-encoder (BGE base)

| Pros | Cons |
|------|------|
| Better **semantic** match for paraphrases and topical similarity | **Larger download** and **slower** encoding than MiniLM |
| Official **query/passage** convention improves **retrieval-style** matching | Tuned for **English**; other languages should use another model or multilingual ranker |
| Still **fast enough** for small batches (typical scrape size) | GPU RAM can bite on **Colab free** if combined with CE + large base model |

#### BM25 on the batch

| Pros | Cons |
|------|------|
| Strong on **exact tokens**, acronyms, rare names | Statistics are **only within this result set** — not a global corpus IDF |
| Complements dense models when embeddings **smooth over** critical terms | If every document is generic, BM25 can be **flat** (little separation) |
| Standard IR baseline, easy to reason about | Extra dependency (`rank-bm25`) |

#### RRF fusion

| Pros | Cons |
|------|------|
| **No need** to calibrate dense vs BM25 onto the same numeric scale | **Hyperparameter `RRF_K`** matters a little; defaults (e.g. 60) are common |
| Robust when one signal is **noisy or tied** | Less direct than a single weighted formula if you want strict interpretability |

#### Weighted normalized sum

| Pros | Cons |
|------|------|
| **Interpretable** weights | **Min–max per batch** makes absolute scores **not comparable** across different searches |
| Easy to tune `_WEIGHTED_NORM_*` | If one signal has tiny variance in a batch, normalization can **amplify noise** |

#### Cross-encoder rerank (top-K)

| Pros | Cons |
|------|------|
| **Highest relevance quality** among the components (full cross-attention) | **K times** forward passes — **latency** and **memory** |
| Only applied to **top K**, so cost is bounded | Items **below K** never get CE; order below K is only from fusion |
| Big win when top of list must be **precise** | Risk of **OOM** on small GPUs if K or document length is large |

#### Richer document string (authors + venue)

| Pros | Cons |
|------|------|
| Venue and author names carry **strong topical cues** | Noisy handles on social posts can add **distracting** tokens |
| Aligns ranking with what users see in the UI | Slightly **longer** texts → marginally more encode time |

#### Keyword heuristic (retained)

| Pros | Cons |
|------|------|
| Very **cheap**; good **fallback** | English-centric token/stopword rules |
| Helps when dense+BM25 disagree | Not a substitute for full **multilingual** NLP |

---

### Tunables (where to look in code)

| Constant | Role |
|----------|------|
| `RANK_BI_ENCODER_MODEL_NAME` | e.g. `BAAI/bge-small-en-v1.5` for speed |
| `BGE_QUERY_PREFIX` | Must match the bi-encoder family (GTE uses different instructions) |
| `RANK_MERGE_MODE` | `"rrf"` or `"weighted_norm"` |
| `RRF_K` | RRF smoothing (typical 30–80) |
| `_WEIGHTED_NORM_*` | Weights in weighted mode |
| `RANK_CROSS_ENCODER_TOP_K` | Set **`0`** to **disable** cross-encoder entirely |
| `RANK_CROSS_ENCODER_MODEL` | Alternate rerankers possible (size vs quality) |
| `_CROSS_ENCODER_BLEND` | How much CE overrides fused score for top-K |

---

## Ranking stack (quick reference)

| Component | Default |
|-----------|---------|
| Bi-encoder | `BAAI/bge-base-en-v1.5` — query: `Represent this sentence for searching relevant passages: …` |
| Lexical | `rank-bm25` / `BM25Okapi` on the current result batch |
| Extra signal | Heuristic English keyword overlap |
| Merge | `RANK_MERGE_MODE = "rrf"` (or `"weighted_norm"`) |
| Rerank | `RANK_CROSS_ENCODER_TOP_K = 40` with `cross-encoder/ms-marco-MiniLM-L-6-v2`; set **`0`** to disable |
| Final | `raw_score` = min-max of fused scores (after optional CE blend) |

**GTE:** You can point `RANK_BI_ENCODER_MODEL_NAME` to e.g. `Alibaba-NLP/gte-base-en-v1.5` and adjust/remove `BGE_QUERY_PREFIX` per that model’s card.

---

## Dependencies

First cell installs `rank-bm25` in addition to `sentence-transformers`, `torch`, etc.

Set **`SERPER_API_KEY`** in the environment (do not commit real keys). Run in Colab as intended, or set `DB_PATH` for local use.
