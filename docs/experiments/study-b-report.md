# Study B — Explore incremental value (lift path)

**Status:** lift wired on `feat/study-b-lift` (verify → recipe → fit; FakeLLM agent)  
**Protocol:** `catml-benchmark-v1` · `holdout_policy: never_seen_during_search`

## Configurations

| Config | Status |
| --- | --- |
| `sklearn_baseline` | Executed |
| `catml_base` | Executed (`AutoML.fit` on train) |
| `catml_explore` | Explore on **train only** → verify ≤2 hyps → recipe from evidence → fit/predict |
| `catml_agent` | FakeLLM + one `AgentSessionManager` step, then same verify/recipe lift |

## Rules enforced

1. Shared holdout per (dataset, seed); Explore never registers the test split.
2. `primary_score` from `score_external` on holdout only.
3. Propose ≠ Accept: recipes are created only from verified evidence (`accepted` or `improved_in_experiment`); otherwise fall back to base fit (`recipes_applied=0`).

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
| wine | catml_explore | f1_macro ↑ | 0.9809 |

**Result (pre-lift smoke):** `comparison_complete=true` · 18 ok · 6 skipped (agent) · 0 failed.

Notes (pre-lift):
- Explore produced ~150–170 findings; scores matched `catml_base` because recipes were not applied.
- Fix: manifest `multiclass` → `multiclass_classification` for Explore registration.

## Lift phase smoke (breast_cancer + wine × seeds 42/43/44, budget 1 min)

Artifacts: `benchmarks/results/v1/study-b-smoke-lift/` (gitignored).

| dataset | config | metric | mean (3 seeds) | recipes_applied |
| --- | --- | --- | --- | --- |
| breast_cancer | sklearn_baseline | roc_auc ↑ | 0.9918 | — |
| breast_cancer | catml_base | roc_auc ↑ | 0.9917 | 0 |
| breast_cancer | catml_explore | roc_auc ↑ | 0.9917 | 0 |
| breast_cancer | catml_agent | roc_auc ↑ | 0.9917 | 0 |
| wine | sklearn_baseline | f1_macro ↑ | 0.9717 | — |
| wine | catml_base | f1_macro ↑ | 0.9809 | 0 |
| wine | catml_explore | f1_macro ↑ | 0.9717 | 1 |
| wine | catml_agent | f1_macro ↑ | 0.9717 | 1 |

**Result:** `comparison_complete=true` · **24 ok / 0 skipped / 0 failed**.

Notes:
- Path: verify → recipe from evidence → fit/predict; fallback base when no usable evidence.
- breast_cancer: no accepted/improved evidence in the verify budget → `recipes_applied=0`, ≈ base.
- wine: recipes applied from `improved_in_experiment` (not always statistically `accepted`); holdout mean **below** `catml_base` — honest Propose ≠ Accept outcome, not a free win.
- Agent: FakeLLM + one session step; same lift path as explore on this smoke.

## Next

- Optional: Ollama provider behind env flag.
- Optional: tighten recipe apply to `accepted` only and compare.
