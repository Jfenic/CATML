# Pilot 1A — Shared external holdout (3 seeds × 5 min)

**Status:** complete 4-framework comparison on the public trio (still not a product-superiority claim)  
**Protocol:** `catml-benchmark-v1` · `evaluation=shared_external_holdout`  
**Created (UTC):** `2026-10-12T17:07:26Z` (see `pilot-raw.json`)  
**Budget:** **300 s** per framework×dataset×seed  
**Seeds:** `42, 43, 44`  
**Runs:** **36/36 `ok`** (sklearn, catml, flaml, autogluon × 3 datasets × 3 seeds)  
**California Housing:** n=5000 subsample (`random_state=0`) for wall-clock fairness

## Design (review-aligned)

| Requirement | How |
| --- | --- |
| Binary + multiclass + regression | breast_cancer / wine / california_housing |
| Identical external test | `make_holdout` once per (dataset, seed); fit on train only |
| Identical evaluator | `score_external()` (sklearn metrics) on shared test preds |
| Resources | `wall_time_s`, `peak_rss_mb`, `cpu_user_s` |
| No cherry-picking | all seeds and all datasets reported, including losses |

Primary metrics: **roc_auc** ↑ · **f1_macro** ↑ · **rmse** ↓.

## Aggregate results (mean ± population std over 3 seeds)

| dataset | metric | sklearn | catml | flaml | autogluon |
| --- | --- | --- | --- | --- | --- |
| breast_cancer | roc_auc ↑ | **0.9918 ± 0.0085** | **0.9918 ± 0.0085** | 0.9830 ± 0.0140 | 0.9888 ± 0.0112 |
| wine | f1_macro ↑ | 0.9717 ± 0.0228 | 0.9717 ± 0.0228 | **0.9820 ± 0.0127** | 0.9727 ± 0.0229 |
| california_housing | rmse ↓ | 3.91 ± 4.51† | 0.523 ± 0.008 | 0.469 ± 0.003 | **0.465 ± 0.006** |

† sklearn Ridge produced one catastrophic seed (`44`, RMSE ≈ 10.3 from a ~−323 prediction). Seeds 42/43 alone are ≈ **0.72**. Treat the sklearn housing mean as unstable; medians: sklearn ≈ 0.72, catml ≈ 0.52, flaml ≈ 0.47, AG ≈ 0.47.

### Mean wall time (s)

| dataset | sklearn | catml | flaml | autogluon |
| --- | --- | --- | --- | --- |
| breast_cancer | ~0 | ~2.0 | ~300 | ~4.8 |
| wine | ~0 | ~1.6 | ~301 | ~7.0 |
| california_housing | ~0 | ~10.8 | ~308 | ~8.1 |

AutoGluon often finished early under `medium_quality` despite the 300 s cap. FLAML typically consumed the full budget.

## Per-seed winners (external score)

| seed | breast_cancer | wine | california_housing |
| --- | --- | --- | --- |
| 42 | sklearn / catml (tie 0.9954) | flaml | autogluon (tie flaml ≈ 0.472) |
| 43 | sklearn / catml (tie 0.9802) | flaml | autogluon |
| 44 | sklearn / catml / flaml (1.000) | sklearn / catml / AG (1.000) | autogluon |

## Honest reading

- **CATML matches a strong sklearn baseline** on the two classification problems under this budget and model set (`logistic_regression`/`random_forest` or `ridge`/`random_forest`).
- **On housing, FLAML and AutoGluon beat CATML** on external RMSE (~0.47 vs ~0.52). That is a real loss — publish it.
- **Speed:** CATML finishes in seconds where FLAML burns the full 5 minutes; AG was also fast here with early stop.
- This pilot does **not** prove CATML is the best AutoML. It does show a reproducible, fair protocol and a usable local engine.

## Artifacts

- Raw: [`benchmarks/results/v1/pilot-raw.json`](../../benchmarks/results/v1/pilot-raw.json)
- Summary: [`benchmarks/results/v1/summary.csv`](../../benchmarks/results/v1/summary.csv)
- Log: `benchmarks/results/v1/pilot-run-5min.log`

## Reproduce

```bash
.venv/bin/pip install flaml 'autogluon.tabular'
.venv/bin/python benchmarks/run.py --study pilot --seeds 42,43,44 --time-budget-minutes 5
.venv/bin/python benchmarks/analyze.py
.venv/bin/pytest tests/test_benchmark_pilot_1a.py tests/test_benchmark_study_c.py -q
```

## Suggested next steps

1. Smoke-install CATML from TestPyPI / publish PyPI RC if installation smoke is green.
2. Optional: fix sklearn Ridge housing instability (clip preds / stronger preprocessing) — baseline hygiene, not CATML.
3. LinkedIn / README: lead with protocol honesty + local/MCP/export story; cite this table including losses.
