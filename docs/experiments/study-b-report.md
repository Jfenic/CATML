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

## Smoke (breast_cancer + wine × seeds 42/43/44, budget 1 min)

Local run under `benchmarks/results/v1/study-b-smoke/` (gitignored).

| dataset | config | metric | mean (3 seeds) |
| --- | --- | --- | --- |
| breast_cancer | sklearn_baseline | roc_auc ↑ | 0.9918 |
| breast_cancer | catml_base | roc_auc ↑ | 0.9917 |
| breast_cancer | catml_explore | roc_auc ↑ | 0.9917 |
| wine | sklearn_baseline | f1_macro ↑ | 0.9717 |
| wine | catml_base | f1_macro ↑ | 0.9809 |
| wine | catml_explore | f1_macro ↑ | (re-run after task_type alias fix) |

Notes:
- Explore produced ~150–170 findings on breast_cancer train folds; scores match `catml_base` because this phase does **not** auto-apply recipes (Propose ≠ Accept).
- `catml_agent`: 6/6 skipped by design.
- Fix: manifest `multiclass` → `multiclass_classification` for Explore registration.

## Next

- Apply verified Explore recipes into `catml_explore` fit path (real lift).
- Wire `catml_agent` with FakeLLM then optional Ollama.
- Re-run smoke after alias fix for full `comparison_complete=true`.
