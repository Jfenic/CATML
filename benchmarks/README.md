# CATML Benchmark v1

Official, reproducible evaluation protocol. **Freeze `protocol-v1.yaml` before measuring.**

Public product name is **catml**; adapters import the public API. Do not claim predictive superiority until Studies A/B publish full results (including losses and failures).

## Studies

| Study | Goal | Status |
| --- | --- | --- |
| **C — Robustness & trust** | Adversarial cases with known expected outcomes | Harness in-repo (`tests/test_benchmark_study_c.py`) |
| **A — Predictive performance** | vs Dummy / sklearn / FLAML / AutoGluon on OpenML/AMLB | Pilot scaffolding only |
| **B — Explore / agents incremental value** | Base vs Explore vs agent under hidden holdout | After pilot |

## Layout

```
benchmarks/
  README.md
  protocol-v1.yaml
  datasets-manifest.csv
  frameworks/          # adapters (catml, sklearn, flaml, autogluon)
  run.py               # entrypoint for pilots
  analyze.py           # aggregate raw → summary
  results/v1/          # raw-results, failures, environment, checksums (gitignored raw optional)
```

## Quick start (pilot)

```bash
# Study C (always run before claiming robustness)
.venv/bin/pytest tests/test_benchmark_study_c.py -q

# Dry-run pilot (no external AutoML deps required)
.venv/bin/python benchmarks/run.py --study pilot --dry-run
```

## Rules

1. Hypotheses / HPO / Explore decisions must stay inside development folds; external test rows stay hidden until final score.
2. Failed runs → `status=failed`, score **absent** (never `0.0` as a fake win/loss).
3. Record CATML commit, dependency versions, CPU/RAM, OS, seeds, wall time, and failure rates.
4. Publish all datasets in the campaign, including where CATML loses.
