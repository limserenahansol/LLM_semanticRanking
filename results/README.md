# Benchmark run artifacts

| File | Description |
|------|-------------|
| `benchmark_results.json` | Machine-readable metrics from `ranking_benchmark.py --json-out` |
| `benchmark_run_log.txt` | Console capture of the same run |

**Last run (committed):** hybrid model `BAAI/bge-small-en-v1.5` (env `RANKING_EVAL_HYBRID_MODEL`), legacy `all-MiniLM-L6-v2`, fixture `fixtures/ranking_eval_sample.json`, `k=5`, no cross-encoder.

This is a **small toy fixture** for CI/demo; do not over-interpret. Regenerate after changing models or adding labeled queries:

```bash
RANKING_EVAL_HYBRID_MODEL=BAAI/bge-small-en-v1.5 python ranking_benchmark.py --k 5 --json-out results/benchmark_results.json
```
