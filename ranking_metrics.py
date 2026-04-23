# -*- coding: utf-8 -*-
"""IR-style ranking metrics: Precision@k, Recall@k, NDCG@k, MRR, MAP.

Used to answer: "Are relevant items high in the list?" given binary or graded labels.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def precision_at_k(ranked_ids: Sequence[str], relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(ranked_ids[:k])
    if not top:
        return 0.0
    hits = sum(1 for d in top if d in relevant)
    return hits / min(k, len(top))


def recall_at_k(ranked_ids: Sequence[str], relevant: set, k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    top = set(ranked_ids[:k])
    return len(top & relevant) / len(relevant)


def _dcg(gains: Sequence[float], k: int) -> float:
    s = 0.0
    for i, g in enumerate(gains[:k]):
        if g <= 0:
            continue
        s += (2.0**g - 1.0) / math.log2(i + 2)
    return s


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    """NDCG@k with graded relevance (0 = irrelevant). Binary {0,1} is fine."""
    gains = [float(relevance.get(d, 0.0)) for d in ranked_ids]
    actual = _dcg(gains, k)
    ideal_gains = sorted((relevance.get(d, 0.0) for d in ranked_ids), reverse=True)
    ideal_gains = [float(g) for g in ideal_gains[:k]]
    ideal = _dcg(ideal_gains, k)
    return (actual / ideal) if ideal > 0 else 0.0


def mrr(ranked_ids: Sequence[str], relevant: set) -> float:
    for i, d in enumerate(ranked_ids):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranked_ids: Sequence[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = 0
    prec_sum = 0.0
    for i, d in enumerate(ranked_ids):
        if d in relevant:
            hits += 1
            prec_sum += hits / (i + 1)
    return prec_sum / len(relevant)


def summarize_query(
    ranked_ids: Sequence[str],
    relevance: Mapping[str, float],
    relevant_set: set,
    k: int = 10,
) -> Dict[str, float]:
    return {
        f"P@{k}": precision_at_k(ranked_ids, relevant_set, k),
        f"R@{k}": recall_at_k(ranked_ids, relevant_set, k),
        f"NDCG@{k}": ndcg_at_k(ranked_ids, relevance, k),
        "MRR": mrr(ranked_ids, relevant_set),
        "MAP": average_precision(ranked_ids, relevant_set),
    }


def aggregate(
    per_query: Iterable[Tuple[str, Dict[str, float]]],
) -> Dict[str, float]:
    rows = list(per_query)
    if not rows:
        return {}
    keys: set = set()
    for _label, m in rows:
        keys.update(m.keys())
    out: Dict[str, float] = {}
    for key in sorted(keys):
        vals = [r[1][key] for r in rows if key in r[1]]
        out[f"mean_{key}"] = sum(vals) / len(vals) if vals else 0.0
    out["n_queries"] = float(len(rows))
    return out
