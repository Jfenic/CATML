# CATML Benchmark v1

Official, reproducible evaluation protocol. **`protocol-v1.yaml` is frozen at 1.0.0.**

Public product name is **catml**; adapters import the public API. Do not claim predictive
superiority until Study A publishes full results (including losses and failures).

## Frozen means

- Metrics, seeds (`42,43,44`), and `shared_external_holdout` are locked for v1.0.0.
- Mid-campaign changes require **protocol 1.1+** and a new results directory.
- A framework with `status=skipped` does **not** count as a completed comparison.

## Studies

| Study | Goal | Status |
| --- | --- | --- |
| **C — Robustness & trust** | Adversarial cases with known expected outcomes | Gate: `tests/test_benchmark_study_c.py` |
| **A — Predictive performance** | vs sklearn / FLAML / AutoGluon on curated OpenML/sklearn | `run.py --study A` |
| **B — Explore / agents** | Base vs Explore vs agent under hidden holdout | Lift wiring (`feat/study-b-lift`): verify→recipe; FakeLLM agent |

## Layout

```
benchmarks/
  README.md
  protocol-v1.yaml          # FROZEN 1.0.0
  datasets-manifest.csv
  frameworks/               # adapters + eval (shared holdout)
  run.py
  analyze.py
  results/v1/
```

## Quick start

```bash
# Gate (always before claiming robustness)
.venv/bin/pytest tests/test_benchmark_study_c.py -q

# Study A dry-run
.venv/bin/python benchmarks/run.py --study A --dry-run

# Study A smoke (5 min × 3 seeds)
.venv/bin/pip install flaml 'autogluon.tabular'
.venv/bin/python benchmarks/run.py --study A --seeds 42,43,44 --time-budget-minutes 5
.venv/bin/python benchmarks/analyze.py --study A

# Formal campaign (20 min)
.venv/bin/python benchmarks/run.py --study A --seeds 42,43,44 --time-budget-minutes 20
```

Pilot trio remains available via `--study pilot` (same evaluator).

Scores are **always** from the shared external holdout evaluator.
Reports: `docs/experiments/pilot-1a-report.md`, `docs/experiments/study-a-report.md`, `docs/experiments/study-b-report.md`.

## Rules

1. Hypotheses / HPO / Explore decisions stay inside development folds; external test stays hidden until final score.
2. Failed runs → `status=failed`, score **absent** (never `0.0` as a fake win/loss).
3. Record CATML commit, dependency versions, CPU/RAM, OS, seeds, wall time, peak RSS, failure rates.
4. Publish all datasets in the campaign, including where CATML loses.
