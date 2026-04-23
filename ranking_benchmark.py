#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare **legacy** (README baseline: MiniLM, title+snippet) vs **hybrid** (BGE + BM25 + keyword [+ CE])
on a labeled JSON fixture.

Answers: “Are relevant documents ranked near the top?” using P@k, R@k, NDCG@k, MRR, MAP.

Usage:
  pip install -r requirements-benchmark.txt
  python ranking_benchmark.py
  python ranking_benchmark.py --fixture fixtures/ranking_eval_sample.json --k 5 --json-out results.json
  RANKING_EVAL_HYBRID_MODEL=BAAI/bge-small-en-v1.5 python ranking_benchmark.py   # faster smoke test

See README § Ranking benchmark and research_aggregator_RANKING_EVAL_KR.md (Korean).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from ranking_core import (
    HYBRID_BI_ENCODER_MODEL,
    LEGACY_BI_ENCODER_MODEL,
    rank_by_relevance_hybrid,
    rank_by_relevance_legacy,
    stable_doc_id,
)
from ranking_metrics import aggregate, summarize_query


def _load_fixture(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _relevant_set(rel: Dict[str, Any]) -> set:
    return {doc_id for doc_id, g in rel.items() if int(g) >= 1}


def run_one_query(
    topic: str,
    results: List[dict],
    relevance: Dict[str, float],
    *,
    k: int,
    cross_encoder_top_k: int,
    verbose: bool,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    rel_float = {doc_id: float(g) for doc_id, g in relevance.items()}
    relevant = _relevant_set(relevance)

    ranked_leg = rank_by_relevance_legacy(topic, copy.deepcopy(results), verbose=verbose)
    leg_ids = [stable_doc_id(r) for r in ranked_leg]

    ranked_hyb = rank_by_relevance_hybrid(
        topic,
        copy.deepcopy(results),
        cross_encoder_top_k=cross_encoder_top_k,
        verbose=verbose,
    )
    hyb_ids = [stable_doc_id(r) for r in ranked_hyb]

    return (
        summarize_query(leg_ids, rel_float, relevant, k=k),
        summarize_query(hyb_ids, rel_float, relevant, k=k),
    )


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_fixture = os.path.join(here, "fixtures", "ranking_eval_sample.json")

    p = argparse.ArgumentParser(
        description="Benchmark: legacy vs hybrid ranking (P@k, NDCG@k, MRR, MAP)."
    )
    p.add_argument("--fixture", default=default_fixture, help="JSON: queries[].topic, results, relevance")
    p.add_argument("--k", type=int, default=5, help="Cutoff for P/R/NDCG")
    p.add_argument("--ce", action="store_true", help="Enable cross-encoder top-40 for hybrid (slow, more VRAM)")
    p.add_argument("--json-out", default="", help="Write aggregated metrics + per-query rows to this path")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not os.path.isfile(args.fixture):
        print(f"Fixture not found: {args.fixture}", file=sys.stderr)
        return 1

    data = _load_fixture(args.fixture)
    queries = data.get("queries") or []
    if not queries:
        print("No queries in fixture.", file=sys.stderr)
        return 1

    ce_k = 40 if args.ce else 0
    rows_leg: List[Tuple[str, Dict[str, float]]] = []
    rows_hyb: List[Tuple[str, Dict[str, float]]] = []
    per_query_out: List[dict] = []

    print(f"Fixture: {args.fixture}")
    print(f"Legacy model: {LEGACY_BI_ENCODER_MODEL}")
    print(f"Hybrid model: {HYBRID_BI_ENCODER_MODEL}")
    print(f"k={args.k}, hybrid cross_encoder_top_k={ce_k}\n")

    ndcg_key = f"NDCG@{args.k}"
    hybrid_wins = 0
    ties = 0

    for i, q in enumerate(queries):
        topic = q["topic"]
        results = q["results"]
        relevance = q["relevance"]
        leg_m, hyb_m = run_one_query(
            topic,
            results,
            relevance,
            k=args.k,
            cross_encoder_top_k=ce_k,
            verbose=args.verbose,
        )
        rows_leg.append((topic[:48], leg_m))
        rows_hyb.append((topic[:48], hyb_m))

        h_ndcg = hyb_m.get(ndcg_key, 0.0)
        l_ndcg = leg_m.get(ndcg_key, 0.0)
        if h_ndcg > l_ndcg + 1e-9:
            hybrid_wins += 1
        elif abs(h_ndcg - l_ndcg) <= 1e-9:
            ties += 1

        row = {
            "query_index": i,
            "topic": topic,
            "legacy": leg_m,
            "hybrid": hyb_m,
            "delta": {mk: hyb_m[mk] - leg_m[mk] for mk in leg_m},
        }
        per_query_out.append(row)

        print(f"--- Query {i + 1}: {topic[:70]}{'...' if len(topic) > 70 else ''} ---")
        print("  legacy:  " + "  ".join(f"{mk}={vv:.4f}" for mk, vv in leg_m.items()))
        print("  hybrid:  " + "  ".join(f"{mk}={vv:.4f}" for mk, vv in hyb_m.items()))
        print(
            "  delta:   "
            + "  ".join(f"{mk}={hyb_m[mk] - leg_m[mk]:+.4f}" for mk in leg_m)
        )
        print()

    agg_leg = aggregate(rows_leg)
    agg_hyb = aggregate(rows_hyb)

    print("=== Mean over queries ===")
    for key in sorted(agg_leg.keys()):
        if key.startswith("mean_"):
            a, b = agg_leg[key], agg_hyb[key]
            name = key.replace("mean_", "")
            print(f"  {name}: legacy={a:.4f}  hybrid={b:.4f}  (Δ={b - a:+.4f})")

    nq = len(queries)
    print()
    print(
        f"=== Hybrid vs legacy on {ndcg_key} ===\n"
        f"  hybrid better: {hybrid_wins}/{nq}  |  ties: {ties}/{nq}  |  legacy better: {nq - hybrid_wins - ties}/{nq}"
    )

    if args.json_out:
        payload = {
            "fixture": os.path.abspath(args.fixture),
            "k": args.k,
            "cross_encoder_top_k": ce_k,
            "legacy_model": LEGACY_BI_ENCODER_MODEL,
            "hybrid_model": HYBRID_BI_ENCODER_MODEL,
            "mean_legacy": {k.replace("mean_", ""): v for k, v in agg_leg.items() if k.startswith("mean_")},
            "mean_hybrid": {k.replace("mean_", ""): v for k, v in agg_hyb.items() if k.startswith("mean_")},
            "ndcg_hybrid_wins": hybrid_wins,
            "ndcg_ties": ties,
            "per_query": per_query_out,
        }
        out_path = os.path.abspath(args.json_out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
