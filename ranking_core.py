# -*- coding: utf-8 -*-
"""
Standalone semantic rankers for evaluation (no pip install on import).

- ``rank_by_relevance_legacy``: README baseline — ``all-MiniLM-L6-v2``, title+snippet, cosine sort.
- ``rank_by_relevance_hybrid``: current pipeline — BGE + BM25 + keyword (+ optional CE).

``research_aggregator_group3.py``는 Colab용 단일 스크립트라 import 시 부작용이 있습니다.
평가·A/B 비교는 이 모듈을 사용하세요.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np


def _env_str(key: str, default: str) -> str:
    v = (os.environ.get(key) or "").strip()
    return v if v else default


# Override via env for faster benchmark runs, e.g. RANKING_EVAL_HYBRID_MODEL=BAAI/bge-small-en-v1.5
LEGACY_BI_ENCODER_MODEL = _env_str("RANKING_EVAL_LEGACY_MODEL", "all-MiniLM-L6-v2")
_LEGACY_MODEL = None

# --- Hybrid (new) ------------------------------------------------------------
HYBRID_BI_ENCODER_MODEL = _env_str("RANKING_EVAL_HYBRID_MODEL", "BAAI/bge-base-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_HYBRID_MODEL = None
_CROSS_ENCODER = None

RANK_MERGE_MODE = "rrf"
RRF_K = 60
_WEIGHTED_NORM_DENSE = 0.55
_WEIGHTED_NORM_BM25 = 0.35
_WEIGHTED_NORM_KEYWORD = 0.10
RANK_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_CROSS_ENCODER_BLEND = 0.55

_RANK_KEYWORD_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "where",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into",
    "is", "was", "are", "were", "been", "be", "being", "am",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "could", "should", "may", "might", "must", "shall", "can",
    "not", "no", "nor", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "both", "each", "few", "more", "most", "other", "some", "such",
    "this", "that", "these", "those", "here", "there", "what", "which", "who", "whom",
    "whose", "why", "how", "all", "any", "every", "about", "above", "after", "again",
    "against", "before", "below", "between", "through", "during", "without", "within",
})


def stable_doc_id(r: dict) -> str:
    u = (r.get("url") or "").strip()
    if u:
        return u
    t = str(r.get("title") or "")
    s = str(r.get("snippet") or "")[:200]
    return f"hash:{hash(t + s)}"


def _legacy_document_text(r: dict) -> str:
    return f"{r.get('title') or ''} {r.get('snippet') or ''}".strip()


def _hybrid_document_text(r: dict) -> str:
    parts = [
        str(r.get("title") or ""),
        str(r.get("snippet") or ""),
        str(r.get("authors") or ""),
        str(r.get("venue") or ""),
    ]
    return " ".join(p for p in parts if p).strip()


def _tokenize_keywords_en(text: str) -> set:
    if not text:
        return set()
    words = re.findall(r"[a-z][a-z0-9]*", text.lower())
    return {w for w in words if len(w) >= 2 and w not in _RANK_KEYWORD_STOPWORDS}


def _keyword_relevance_score(topic: str, doc: str) -> float:
    tw = _tokenize_keywords_en(topic)
    if not tw:
        return 0.0
    dw = _tokenize_keywords_en(doc)
    if not dw:
        return 0.0
    overlap = len(tw & dw)
    recall = overlap / len(tw)
    denom = min(len(dw), max(len(tw) * 4, 12))
    density = overlap / denom if denom else 0.0
    score = 0.68 * recall + 0.32 * min(1.0, density)
    tl, dl = topic.lower().strip(), doc.lower()
    if len(tl) >= 10 and tl in dl:
        score = min(1.0, score + 0.14)
    elif overlap == len(tw) and len(tw) >= 2:
        score = min(1.0, score + 0.06)
    return round(min(1.0, score), 4)


def _legacy_crude_keyword(topic: str, doc: str) -> float:
    """README baseline: split-on-spaces overlap."""
    t = set(topic.lower().split())
    d = set(doc.lower().split())
    t = {w for w in t if len(w) > 1}
    d = {w for w in d if len(w) > 1}
    if not t:
        return 0.0
    return len(t & d) / len(t)


def _bm25_tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", (text or "").lower()) or ["_"]


def _min_max_norm(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.ones_like(v) * 0.5
    return (v - lo) / (hi - lo)


def _scores_to_order(scores: np.ndarray) -> list:
    return [i for i, _ in sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))]


def _rrf_fuse(orders: list, k: int) -> np.ndarray:
    n = max((max(o, default=-1) for o in orders), default=-1) + 1
    if n <= 0:
        return np.array([])
    acc = np.zeros(n, dtype=np.float64)
    for order in orders:
        for pos, doc_i in enumerate(order):
            acc[doc_i] += 1.0 / (k + pos + 1)
    return acc


def _get_legacy_model():
    global _LEGACY_MODEL
    if _LEGACY_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _LEGACY_MODEL = SentenceTransformer(LEGACY_BI_ENCODER_MODEL)
    return _LEGACY_MODEL


def _get_hybrid_model():
    global _HYBRID_MODEL
    if _HYBRID_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _HYBRID_MODEL = SentenceTransformer(HYBRID_BI_ENCODER_MODEL)
    return _HYBRID_MODEL


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        from sentence_transformers import CrossEncoder
        _CROSS_ENCODER = CrossEncoder(RANK_CROSS_ENCODER_MODEL)
    return _CROSS_ENCODER


def rank_by_relevance_legacy(
    topic: str,
    results: List[dict],
    *,
    verbose: bool = False,
) -> List[dict]:
    """
    Original-style ranker: MiniLM-L6, title+snippet only, cosine similarity.
    If sentence-transformers fails, falls back to crude keyword-only (README behavior).
    """
    if not results:
        return []
    topic_clean = (topic or "").strip()
    if not topic_clean:
        for r in results:
            r["raw_score"] = 0.0
            r["semantic_score"] = None
            r["keyword_score"] = 0.0
            r["ranker"] = "legacy"
        return results

    n = len(results)
    docs = [_legacy_document_text(r) for r in results]
    kw = np.array(
        [_legacy_crude_keyword(topic_clean, d) for d in docs], dtype=np.float64
    )

    dense = np.zeros(n, dtype=np.float64)
    try:
        from sentence_transformers import util
        model = _get_legacy_model()
        topic_emb = model.encode(
            topic_clean,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        texts = [d[:8192] if d else " " for d in docs]
        result_emb = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        row = util.cos_sim(topic_emb, result_emb)[0]
        dense = np.clip(np.ravel(row.detach().cpu().numpy()), 0.0, 1.0)
    except Exception as e:
        if verbose:
            print(f"  [legacy] ST failed ({e}) → keyword-only")
        fused = kw
        fused_n = _min_max_norm(fused)
        for i, r in enumerate(results):
            r["semantic_score"] = None
            r["keyword_score"] = round(float(kw[i]), 4)
            r["bm25_score"] = None
            r["cross_encoder_score"] = None
            r["raw_score"] = round(float(fused_n[i]), 4)
            r["ranker"] = "legacy"
        return sorted(results, key=lambda x: x["raw_score"], reverse=True)

    fused_n = _min_max_norm(dense)
    for i, r in enumerate(results):
        r["semantic_score"] = round(float(dense[i]), 4)
        r["keyword_score"] = round(float(kw[i]), 4)
        r["bm25_score"] = None
        r["cross_encoder_score"] = None
        r["raw_score"] = round(float(fused_n[i]), 4)
        r["ranker"] = "legacy"
    if verbose:
        print(f"  [legacy] MiniLM cosine only (title+snippet; {n} items)")
    return sorted(results, key=lambda x: x["raw_score"], reverse=True)


def rank_by_relevance_hybrid(
    topic: str,
    results: List[dict],
    *,
    cross_encoder_top_k: int = 0,
    merge_mode: Optional[str] = None,
    verbose: bool = False,
) -> List[dict]:
    """
    New hybrid ranker (BGE + BM25 + keyword + optional CE).

    ``cross_encoder_top_k=0`` disables cross-encoder (faster evaluation).
    """
    if not results:
        return []
    topic_clean = (topic or "").strip()
    mode = merge_mode or RANK_MERGE_MODE

    if not topic_clean:
        for r in results:
            r["raw_score"] = 0.0
            r["semantic_score"] = None
            r["keyword_score"] = 0.0
            r["bm25_score"] = None
            r["cross_encoder_score"] = None
            r["ranker"] = "hybrid"
        return results

    n = len(results)
    docs = [_hybrid_document_text(r) for r in results]
    keyword_scores = np.array(
        [_keyword_relevance_score(topic_clean, d) for d in docs], dtype=np.float64
    )

    bm25_scores = np.zeros(n, dtype=np.float64)
    try:
        from rank_bm25 import BM25Okapi
        corpus_tokens = [_bm25_tokenize(d) for d in docs]
        bm25 = BM25Okapi(corpus_tokens)
        bm25_scores = np.array(
            bm25.get_scores(_bm25_tokenize(topic_clean)), dtype=np.float64
        )
    except Exception as e:
        if verbose:
            print(f"  [hybrid] BM25 skipped ({type(e).__name__})")

    dense_scores = np.zeros(n, dtype=np.float64)
    try:
        from sentence_transformers import util
        model = _get_hybrid_model()
        q_text = BGE_QUERY_PREFIX + topic_clean
        topic_emb = model.encode(
            q_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        texts = [d[:8192] if d else " " for d in docs]
        result_emb = model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        row = util.cos_sim(topic_emb, result_emb)[0]
        dense_scores = np.clip(np.ravel(row.detach().cpu().numpy()), 0.0, 1.0)
    except Exception as e:
        if verbose:
            print(f"  [hybrid] bi-encoder failed → keyword-only ({e})")
        fused_n = _min_max_norm(keyword_scores)
        for i, r in enumerate(results):
            r["semantic_score"] = None
            r["bm25_score"] = None
            r["keyword_score"] = round(float(keyword_scores[i]), 4)
            r["cross_encoder_score"] = None
            r["raw_score"] = round(float(fused_n[i]), 4)
            r["ranker"] = "hybrid"
        return sorted(results, key=lambda x: x["raw_score"], reverse=True)

    orders = [
        _scores_to_order(dense_scores),
        _scores_to_order(bm25_scores),
        _scores_to_order(keyword_scores),
    ]
    if mode == "rrf":
        fused = _rrf_fuse(orders, RRF_K)
    elif mode == "weighted_norm":
        nd = _min_max_norm(dense_scores)
        nb = _min_max_norm(bm25_scores)
        nk = np.clip(keyword_scores, 0.0, 1.0)
        fused = (
            _WEIGHTED_NORM_DENSE * nd
            + _WEIGHTED_NORM_BM25 * nb
            + _WEIGHTED_NORM_KEYWORD * nk
        )
    else:
        fused = _rrf_fuse(orders, RRF_K)

    prelim_order = _scores_to_order(fused)
    K = min(cross_encoder_top_k, n) if cross_encoder_top_k > 0 else 0
    ce_by_idx: Dict[int, float] = {}
    if K > 0:
        try:
            ce = _get_cross_encoder()
            top_idx = prelim_order[:K]
            pairs = [(topic_clean, (docs[i] or " ")[:2000]) for i in top_idx]
            ce_raw = np.array(
                ce.predict(pairs, show_progress_bar=False), dtype=np.float64
            )
            ce_n = _min_max_norm(ce_raw)
            fus_sub = np.array([fused[i] for i in top_idx], dtype=np.float64)
            fus_n = _min_max_norm(fus_sub)
            for j, idx in enumerate(top_idx):
                fused[idx] = float(
                    _CROSS_ENCODER_BLEND * ce_n[j]
                    + (1.0 - _CROSS_ENCODER_BLEND) * fus_n[j]
                )
                ce_by_idx[idx] = round(float(ce_raw[j]), 4)
            if verbose:
                print(f"  [hybrid] CE rerank top-{K}")
        except Exception as e:
            if verbose:
                print(f"  [hybrid] CE skipped ({type(e).__name__})")

    fused_norm = _min_max_norm(fused)
    for i, r in enumerate(results):
        r["semantic_score"] = round(float(dense_scores[i]), 4)
        r["bm25_score"] = round(float(bm25_scores[i]), 4)
        r["keyword_score"] = round(float(keyword_scores[i]), 4)
        r["cross_encoder_score"] = ce_by_idx.get(i)
        r["raw_score"] = round(float(fused_norm[i]), 4)
        r["ranker"] = "hybrid"
    if verbose:
        print(f"  [hybrid] BGE+BM25+kw → {mode} ({n} items, CE_K={cross_encoder_top_k})")
    return sorted(results, key=lambda x: x["raw_score"], reverse=True)


def reset_ranking_models() -> None:
    """Unload cached SentenceTransformer / CrossEncoder weights (e.g. after changing env)."""
    global _LEGACY_MODEL, _HYBRID_MODEL, _CROSS_ENCODER
    _LEGACY_MODEL = None
    _HYBRID_MODEL = None
    _CROSS_ENCODER = None
