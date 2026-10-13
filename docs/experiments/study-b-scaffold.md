# Study B — Explore / agents incremental value (scaffold)

**Status:** minimal runner shipped — see [`study-b-report.md`](study-b-report.md).  
**Protocol:** `catml-benchmark-v1` 1.0.0 frozen · `holdout_policy: never_seen_during_search`

## Configurations

| Config | Meaning |
| --- | --- |
| `sklearn_baseline` | Ridge / LogisticRegression baseline |
| `catml_base` | `catml.AutoML` without Explore/agent loop |
| `catml_explore` | Explore study → optional recipe/hypothesis → verified experiment |
| `catml_agent` | Agent session (Planner/Critic); optional Ollama/Qwen later |

## Rules

1. Fix shared train/test once per (dataset, seed); never score test until final.
2. Explore / HPO / agent proposals only see train folds.
3. Propose ≠ Accept: agent hypotheses become accepted only after measured experiment lift.
4. Record wall time, RSS, failure/fallback rates (esp. LLM → deterministic fallback).

## Entrypoint

```bash
.venv/bin/python benchmarks/run.py --study B --dry-run
.venv/bin/python benchmarks/run.py --study B --datasets sklearn_breast_cancer --seeds 42
```

Report: [`study-b-report.md`](study-b-report.md).
