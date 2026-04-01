# Research Aggregator — 랭킹 로직 설명 (한국어)

대상: `research_aggregator_group3.py`의 **`rank_by_relevance`** 및 관련 상수·헬퍼.

영어 검색·스니펫 기준으로 **밀집(dense) 의미 + 희소(lexical) BM25 + 가벼운 키워드**를 한데 묶고, 필요 시 **크로스 인코더**로 상위 후보만 다시 정렬합니다.

---

## 1. 구성 요약

| 단계 | 설명 |
|------|------|
| **Bi-encoder** | `BAAI/bge-base-en-v1.5` — 쿼리에만 `Represent this sentence for searching relevant passages: ` 프리픽스, 문서(제목·스니펫·저자·venue 합친 텍스트)는 접두사 없음. 코사인 유사도 → `semantic_score`. |
| **BM25** | `rank-bm25`의 `BM25Okapi`로 **이번 검색 결과 배치 안에서** IDF·빈도 기반 점수 → `bm25_score`. |
| **키워드** | 기존과 같이 불용어 제거·토큰 겹침·짧은 구문 보너스 → `keyword_score`. |
| **융합** | `RANK_MERGE_MODE` 가 **`rrf`**(기본) 또는 **`weighted_norm`**. |
| **(선택) CE** | 융합 점수 상위 `K`개에 `cross-encoder/ms-marco-MiniLM-L-6-v2` 적용 후 점수 보정 → `cross_encoder_score`; 최종 `raw_score`는 전체에 대해 min–max 정규화. |

---

## 2. RRF vs 정규화 가중합

### `rrf` (기본)

세 가지 **순위 리스트**(dense / BM25 / keyword 각각 점수 내림차순 인덱스)에 대해

\[
\text{RRF}(d) = \sum_{\ell} \frac{1}{k + \text{rank}_\ell(d)}
\]

처럼 합산합니다. `RRF_K`=`k`(코드에선 60)이며, **서로 다른 척도**의 점수를 맞출 필요가 적습니다.

### `weighted_norm`

현재 배치에서 dense·BM25·keyword 점수를 각각 **min–max**로 0–1에 맞춘 뒤

`0.55 * dense + 0.35 * BM25 + 0.10 * keyword` 로 합칩니다. 계수는 코드 상 `_WEIGHTED_NORM_*` 로 조절.

---

## 3. 크로스 인코더 (선택)

- 상수 `RANK_CROSS_ENCODER_TOP_K`(기본 40)가 **0이면** 크로스 인코더는 **로드하지 않음**.
- 0보다 크면 융합 후 상위 `min(K, n)`개 쌍 `(query, document)`에 대해 재점수를 매기고, `_CROSS_ENCODER_BLEND` 비율로 융합 점수와 섞은 뒤 전체를 다시 min–max 하여 `raw_score`로 씁니다.
- **느리고 VRAM을 씁니다.** Colab에서 OOM이면 `RANK_CROSS_ENCODER_TOP_K = 0` 또는 더 작은 `bge-small-en-v1.5` 등으로 바꾸세요.

---

## 4. 바이 인코더를 GTE로 바꾸려면

코드 기본은 **BGE**. [Alibaba-NLP/gte-base-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5) 등으로 바꿀 때는 **모델 카드의 query/document 포맷**을 꼭 맞추세요. BGE용 `BGE_QUERY_PREFIX`는 GTE에 그대로 쓰이지 않을 수 있습니다(프리픽스 제거 또는 `query: ` 등 문서 확인).

---

## 5. 폴백 동작

| 상황 | 동작 |
|------|------|
| `sentence_transformers` 없음 | 키워드 점수만으로 `raw_score`, `semantic_score`·`bm25_score`는 없음 |
| Bi-encoder 실패 | 동일 |
| `rank-bm25` 없음 | BM25는 0에 가깝게 스킵되며, RRF는 dense·keyword 두 리스트만으로 동작하지 않고 **BM25 순위는 여전히 생성**됨 — 실제로는 BM25가 전부 동일하면 순위가 동점 처리되어 RRF에 노이즈가 될 수 있음. **패키지 설치 권장.** |

---

## 6. 튜닝용 상수 (파일 내)

| 이름 | 의미 |
|------|------|
| `RANK_BI_ENCODER_MODEL_NAME` | 기본 `BAAI/bge-base-en-v1.5` (가벼운 대안: `BAAI/bge-small-en-v1.5`) |
| `RANK_MERGE_MODE` | `"rrf"` / `"weighted_norm"` |
| `RRF_K` | RRF의 `k` |
| `_WEIGHTED_NORM_*` | 가중합 모드 가중치 |
| `RANK_CROSS_ENCODER_TOP_K` | 0이면 CE 비활성 |
| `RANK_CROSS_ENCODER_MODEL` | 재순위 모델 ID |
| `_CROSS_ENCODER_BLEND` | CE 점수 vs 융합 점수 혼합 비율 |

---

## 7. 관련 문서

- 전체 파이프라인: **`research_aggregator_WORKFLOW_README_KR.md`**
- 영문 인덱스: **`research_aggregator_README.md`**
