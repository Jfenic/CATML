# CATML Benchmark v1

Official, reproducible evaluation protocol. **`protocol-v1.yaml` is frozen at 1.0.0.**

Public product name is **catml**; adapters import the public API. Scores are **always**
from the shared external holdout evaluator. Do not claim predictive superiority without
the full Study A table (including losses and failures).

## Frozen means

- Metrics, seeds (`42,43,44`), and `shared_external_holdout` are locked for v1.0.0.
- Mid-campaign changes require **protocol 1.1+** and a new results directory.
- A framework with `status=skipped` does **not** count as a completed comparison.

## Studies

| Study | Goal | Status |
| --- | --- | --- |
| **C — Robustness & trust** | Adversarial cases with known expected outcomes | Gate: `tests/test_benchmark_study_c.py` |
| **A — Predictive performance** | vs sklearn / FLAML / AutoGluon on curated OpenML/sklearn | **Done** — 20 min, 132/132 ok |
| **B — Explore / agents** | Base vs Explore vs agent under hidden holdout | **Lift wired** — verify→recipe; FakeLLM agent |

## Results summary

### Study A — predictive (formal 20 min × seeds 42/43/44)

- **11 datasets** · frameworks: `sklearn_baseline`, `catml`, `flaml`, `autogluon`.
- **132/132 ok** · `comparison_complete=true`.
- Extra budget mainly helped **FLAML**; CATML holdout means were largely flat vs 5 min on this suite (short fixed fit path in the adapter).
- CATML still shows a large lift vs sklearn on **phoneme**; often ties sklearn elsewhere; trails FLAML/AG on several regression/search-heavy sets.
- Full table + wins: [`docs/experiments/study-a-report.md`](../docs/experiments/study-a-report.md).

### Study B — Explore incremental value

- Configs: `sklearn_baseline` / `catml_base` / `catml_explore` / `catml_agent` (FakeLLM + session step).
- Explore on **train only**; Propose ≠ Accept — verify hyps → recipe from evidence → fit; else base fallback (`recipes_applied=0`).
- Lift smoke (`breast_cancer` + `wine` × 3 seeds, 1 min): **24 ok / 0 skipped / 0 failed**.
- Detail: [`docs/experiments/study-b-report.md`](../docs/experiments/study-b-report.md).

Also: [`docs/experiments/pilot-1a-report.md`](../docs/experiments/pilot-1a-report.md).

## Layout

```
benchmarks/
  README.md
  protocol-v1.yaml          # FROZEN 1.0.0
  datasets-manifest.csv
  frameworks/               # adapters + eval (shared holdout)
  run.py
  analyze.py
  results/v1/               # gitignored raw artifacts
```

## Quick start

```bash
# Gate (always before claiming robustness)
.venv/bin/pytest tests/test_benchmark_study_c.py -q

# Study A
.venv/bin/python benchmarks/run.py --study A --dry-run
.venv/bin/pip install flaml 'autogluon.tabular'
.venv/bin/python benchmarks/run.py --study A --seeds 42,43,44 --time-budget-minutes 5
.venv/bin/python benchmarks/analyze.py --study A
# Formal: --time-budget-minutes 20

# Study B (train-only Explore; FakeLLM agent)
.venv/bin/python benchmarks/run.py --study B --dry-run
.venv/bin/python benchmarks/run.py --study B \
  --datasets sklearn_breast_cancer,sklearn_wine \
  --seeds 42,43,44 --time-budget-minutes 1
```

Pilot trio remains available via `--study pilot` (same evaluator).

## Rules

1. Hypotheses / HPO / Explore decisions stay inside development folds; external test stays hidden until final score.
2. Failed runs → `status=failed`, score **absent** (never `0.0` as a fake win/loss).
3. Record CATML commit, dependency versions, CPU/RAM, OS, seeds, wall time, peak RSS, failure rates.
4. Publish all datasets in the campaign, including where CATML loses.
