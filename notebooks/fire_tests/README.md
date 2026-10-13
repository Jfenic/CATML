# Fire tests (post-rc1)

Ordered after Benchmark Pilot 1A. Ten smoke scenarios covering common Kaggle-like tabular tasks via OpenML/sklearn (no Kaggle API).

## Run all

```bash
# from repo root
.venv/bin/python notebooks/fire_tests/run_all.py
```

Summary: `notebooks/fire_tests/results/fire_tests_summary.json`.

## Notebooks

Companion `.ipynb` files wrap each script:

```bash
.venv/bin/python notebooks/fire_tests/generate_notebooks.py
jupyter notebook notebooks/fire_tests/01_titanic_like.ipynb
```

## Individual scripts

| Script | Task |
| --- | --- |
| `01_titanic_like.py` | binary mixed types |
| `02_house_prices_like.py` | regression |
| `03_credit_risk.py` | binary + categoricals |
| `04_multiclass.py` | multiclass log_loss |
| `05_high_cardinality.py` | high-card cats / id-like |
| `06_ordered_time_split.py` | chronological holdout |
| `07_hpo_short.py` | `time_budget` |
| `08_export_predict.py` | save → load → predict |
| `09_explore_smoke.py` | Explore CQRS study |
| `10_ensemble_compare.py` | multi-model leaderboard |
