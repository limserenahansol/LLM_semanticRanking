# 랭킹 벤치마크 — 구버전 vs 하이브리드 성능 평가

Colab 메인 스크립트(`research_aggregator_group3.py`)와 **분리된** 평가 전용 모듈로, README에 적힌 **원래(legacy) 랭커**와 **현재 하이브리드 랭커**를 **같은 후보 목록·같은 정답 라벨** 위에서 비교합니다.

## 무엇을 비교하나

| 랭커 | 구현 (`ranking_core.py`) |
|------|---------------------------|
| **Legacy** | `all-MiniLM-L6-v2`, 문서 텍스트 = 제목+스니펫, 코사인 유사도 정렬 (실패 시 단순 키워드) |
| **Hybrid** | `BAAI/bge-base-en-v1.5` (+ 쿼리 프리픽스), 제목·스니펫·저자·venue, BM25 + 키워드, RRF 융합, (선택) 크로스 인코더 |

## 평가 지표 (관련 문서가 위에 오는가?)

- **P@k / R@k**: 상위 k개에 정답이 얼마나 포함되는지  
- **NDCG@k**: 관련도 **등급**(0,1,2…)을 반영한 순위 품질  
- **MRR**: 첫 정답이 몇 번째인지  
- **MAP**: 정답 문서들에 대한 평균 정밀도  

ROUGE/BERTScore는 **생성·요약** 평가에 가깝고, **순위 품질**의 주 지표로는 부적합합니다. 여기서는 **정보 검색에서 쓰는 순위 지표**를 사용합니다.

## 설치 및 실행

```bash
pip install -r requirements-benchmark.txt
python ranking_benchmark.py
python ranking_benchmark.py --k 10 --json-out benchmark_results.json
```

- **`--ce`**: 하이브리드에 크로스 인코더 상위 K 재순위 켜기 (느리고 VRAM 사용).  
- **빠른 스모크 테스트**: `RANKING_EVAL_HYBRID_MODEL=BAAI/bge-small-en-v1.5 python ranking_benchmark.py`

## 픽스처(JSON) 형식

`fixtures/ranking_eval_sample.json` 참고:

- `queries`: 배열  
- 각 요소: `topic`, `results[]`, `relevance` (url → 0 이상 정수 등급)  
- `results` 항목은 스크래퍼와 동일한 키(`url`, `title`, `snippet`, …)를 쓰면 됩니다.

실제 파이프라인에서 저장한 후보 풀을 JSON으로 내보내 같은 형식에 맞추면 **실데이터**로도 평가할 수 있습니다.

## 결과 해석

- 출력의 **delta**가 양수이면 해당 지표에서 **hybrid가 legacy보다 유리**합니다.  
- **NDCG@k**에서 hybrid가 이긴 쿼리 수를 스크립트가 요약합니다.  
- 샘플 픽스처는 **소규모 데모**입니다. 논문/보고서용으로는 **주제 수를 늘리고 사람이 라벨링**하거나 약한 감독(예: DOI 화이트리스트)을 쓰는 것이 좋습니다.

## 관련 문서

- [research_aggregator_RANKING_UPDATE_KR.md](./research_aggregator_RANKING_UPDATE_KR.md) — 하이브리드 랭킹 구성  
- [README.md](./README.md) — 벤치마크 영문 요약
