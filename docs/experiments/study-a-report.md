# Study A — Predictive performance (Benchmark v1.0.0 frozen)

**Protocol:** `catml-benchmark-v1` **1.0.0 frozen** · `evaluation=shared_external_holdout`  
**Smoke (5 min):** **300 s** × seeds `[42,43,44]` · **132/132 ok** · artifacts `benchmarks/results/v1/study-a-5min/`  
**Formal (20 min):** **1200 s** × seeds `[42,43,44]` · **132/132 ok** · `comparison_complete=true` · artifacts `benchmarks/results/v1/study-a-20min/` (+ current `study-a-raw.json`)  
**Machine:** 13th Gen Intel i7-13620H · ~7.6 GB RAM · Python 3.12 · commit `05a004f8`  
**Wall clock (20 min campaign):** ~10.5 h sequential

## Design

- 11 curated datasets (binary / multiclass / regression); OpenML + sklearn.
- One holdout per (dataset, seed); all frameworks fit train only; scores from shared `score_external()`.
- Frameworks: sklearn_baseline, catml, flaml, autogluon (all required for claims).

## Mean ± std (3 seeds, **20 min** budget) — primary

| dataset | metric | sklearn | catml | flaml | autogluon |
| --- | --- | --- | --- | --- | --- |
| breast_cancer | roc_auc ↑ | **0.9918±0.0085** | **0.9918±0.0085** | 0.9823±0.0148 | 0.9888±0.0112 |
| credit_g | roc_auc ↑ | 0.7867±0.0379 | 0.7838±0.0338 | 0.7926±0.0210 | **0.8006±0.0181** |
| blood | roc_auc ↑ | 0.7480±0.0327 | 0.7480±0.0327 | **0.7608±0.0213** | 0.7401±0.0353 |
| phoneme | roc_auc ↑ | 0.8216±0.0164 | 0.9568±0.0048 | **0.9662±0.0039** | 0.9590±0.0018 |
| wine | f1_macro ↑ | 0.9717±0.0228 | 0.9717±0.0228 | **0.9820±0.0127** | 0.9727±0.0229 |
| iris | f1_macro ↑ | 0.9555±0.0157 | 0.9443±0.0315 | **0.9777±0.0158** | 0.9554±0.0417 |
| vehicle | f1_macro ↑ | **0.7896±0.0169** | **0.7896±0.0169** | 0.7559±0.0111 | 0.7332±0.0372 |
| california_housing | rmse ↓ | 3.91±4.51† | 0.5231±0.0083 | **0.4550±0.0029** | 0.4648±0.0060 |
| diabetes | rmse ↓ | **52.72±1.28** | **52.72±1.28** | 53.72±2.31 | 52.99±2.60 |
| kin8nm | rmse ↓ | 0.2057±0.0058 | 0.1506±0.0049 | 0.1099±0.0012 | **0.1013±0.0026** |
| elevators | rmse ↓ | 0.0027±0.0001 | 0.0027±0.0001 | **0.0022±0.0000** | 0.0022±0.0001 |

† sklearn Ridge unstable on housing seed 44 (same pilot anomaly); treat mean with caution.

## Wins by mean (20 min)

| Winner | Datasets |
| --- | --- |
| sklearn / catml (tie) | breast_cancer, vehicle, diabetes |
| flaml | blood, phoneme, wine, iris, california_housing, elevators |
| autogluon | credit_g, kin8nm |
| catml vs sklearn | phoneme still large lift (0.957 vs 0.822); kin8nm/housing still behind FLAML/AG |

## 5 min → 20 min (what moved)

- **CATML / sklearn / AutoGluon:** holdout means **unchanged** vs 5 min on this suite (CATML does not burn the full 20 min the way search-heavy Autogluon/FLAML do; AG often early-stops to the same quality on these sizes).
- **FLAML** used the extra budget: phoneme +0.006 roc_auc, wine +0.018 f1, vehicle +0.007 f1, housing −0.015 rmse, kin8nm −0.010 rmse (diabetes slightly worse).
- Extra wall clock bought **FLAML search gains**, not a CATML leap — consistent with CATML’s short fixed fit path under this adapter.

## Mean ± std (3 seeds, **5 min** smoke) — archive

| dataset | metric | sklearn | catml | flaml | autogluon |
| --- | --- | --- | --- | --- | --- |
| breast_cancer | roc_auc ↑ | **0.9918±0.0085** | **0.9918±0.0085** | 0.9827±0.0137 | 0.9888±0.0112 |
| credit_g | roc_auc ↑ | 0.7867±0.0379 | 0.7838±0.0338 | 0.7909±0.0289 | **0.8006±0.0181** |
| blood | roc_auc ↑ | 0.7480±0.0327 | 0.7480±0.0327 | **0.7608±0.0213** | 0.7401±0.0353 |
| phoneme | roc_auc ↑ | 0.8216±0.0164 | 0.9568±0.0048 | **0.9604±0.0107** | 0.9590±0.0018 |
| wine | f1_macro ↑ | 0.9717±0.0228 | 0.9717±0.0228 | 0.9638±0.0132 | **0.9727±0.0229** |
| iris | f1_macro ↑ | 0.9555±0.0157 | 0.9443±0.0315 | **0.9777±0.0158** | 0.9554±0.0417 |
| vehicle | f1_macro ↑ | **0.7896±0.0169** | **0.7896±0.0169** | 0.7487±0.0123 | 0.7332±0.0372 |
| california_housing | rmse ↓ | 3.91±4.51† | 0.523±0.008 | 0.470±0.003 | **0.465±0.006** |
| diabetes | rmse ↓ | **52.72±1.28** | **52.72±1.28** | 53.14±1.65 | 52.99±2.60 |
| kin8nm | rmse ↓ | 0.206±0.006 | 0.151±0.005 | 0.120±0.005 | **0.101±0.003** |
| elevators | rmse ↓ | 0.0027±0.0001 | 0.0027±0.0001 | **0.0022±0.0001** | **0.0022±0.0001** |

## Honest reading

- CATML stays competitive with a strong sklearn baseline and much stronger when non-linear structure matters (phoneme, kin8nm vs Ridge).
- Under a **search** budget, FLAML/AutoGluon still take most of the suite; at 20 min FLAML widens several leads (housing, wine, phoneme).
- No global superiority claim. Longer CATML budgets only help if the adapter actually spends them (HPO / multi-model search) — see wiring backlog, not this protocol rerun alone.
- Publish losses honestly; CATML wall times remain seconds vs hundreds for FLAML.

## Reproduce

```bash
.venv/bin/pytest tests/test_benchmark_study_c.py tests/test_benchmark_pilot_1a.py -q
.venv/bin/python benchmarks/run.py --study A --seeds 42,43,44 --time-budget-minutes 5
.venv/bin/python benchmarks/run.py --study A --seeds 42,43,44 --time-budget-minutes 20
.venv/bin/python benchmarks/analyze.py --study A
```

## Next

- Study B (Explore/agent lift): see [`study-b-scaffold.md`](study-b-scaffold.md).
