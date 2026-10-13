# Study B — Explore incremental value (minimal runner)

**Status:** executable smoke (agent/LLM deferred)  
**Protocol:** `catml-benchmark-v1` · `holdout_policy: never_seen_during_search`

## Configurations

| Config | Status |
| --- | --- |
| `sklearn_baseline` | Executed |
| `catml_base` | Executed (`AutoML.fit` on train) |
| `catml_explore` | Executed (Explore study on **train only**, then fit) |
| `catml_agent` | Skipped (FakeLLM/Ollama phase later) |

## Rules enforced

1. Shared holdout per (dataset, seed); Explore never registers the test split.
2. `primary_score` from `score_external` on holdout only.
3. Propose ≠ Accept: Explore findings are recorded (`findings_count`); they do **not** auto-mutate the training pipeline in this phase.

## Entrypoint

```bash
.venv/bin/python benchmarks/run.py --study B --dry-run
.venv/bin/python benchmarks/run.py --study B \
  --datasets sklearn_breast_cancer,sklearn_wine \
  --seeds 42 --time-budget-minutes 1
```

Artifacts: `benchmarks/results/v1/study-b-raw.json` (gitignored).

## Next

- Apply verified Explore recipes into `catml_explore` fit path (real lift).
- Wire `catml_agent` with FakeLLM then optional Ollama.
