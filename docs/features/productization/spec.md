# Feature Specification: Ergonomic AutoML Facade & Standalone Model Artifacts

> **Status:** Draft / Active • **Author:** Agent Antigravity • **Date:** 2026-10-02  
> **Sprint:** Phase 0 (Productization P0) • **Target Release:** v0.8.0

---

## 1. Goal & Business Need

CATML possess a robust Hexagonal architecture, CQRS buses, persistent job workers, and an MCP agent server. However, external data scientists and ML engineers expect an ergonomic, one-liner Python API:

```python
from catml import AutoML

model = AutoML()
result = model.fit(df, target="churn")
result.leaderboard()
result.best_model.save("model.pkl")
```

Currently, training requires interacting with `AutoMLWorkspace`, building CQRS command buses, or invoking the CLI. Furthermore, inference requires the workspace SQLite database and re-fitting the pipeline on demand.

**This feature delivers:**
1. A scikit-learn compatible facade (`AutoML`) providing `fit(X, y)` and `fit(df, target="col")`.
2. An `AutoMLResult` object with `.leaderboard()`, `.best_model`, `.best_score`, and `.predict()`.
3. A self-contained `ModelArtifact` capable of `.save("model.pkl")` and standalone `load("model.pkl")` for production inference without database or workspace dependencies.
4. Top-level library imports (`from automl import AutoML` and `from catml import AutoML`).

---

## 2. Requirements & Capabilities

### 2.1 Ergonomic Facade (`AutoML`)
- **Constructor:**
  ```python
  AutoML(
      task: str | None = None,            # "classification", "regression", or auto-detect
      metric: str | None = None,          # default: "roc_auc" (binary), "f1_macro" (multi), "rmse" (reg)
      cv_folds: int = 5,
      time_budget: int | None = None,     # max seconds
      models: list[str] | None = None,    # specific candidate model plugins to evaluate
      random_state: int = 42,
      workspace_dir: str | Path | None = None, # optional persistent path; defaults to managed temp dir
  )
  ```
- **`fit()` Method:**
  - Signature: `fit(data: pd.DataFrame | np.ndarray, target: str | pd.Series | np.ndarray | None = None) -> AutoMLResult`
  - Supports both `fit(df, target="column_name")` and `fit(X, y)`.
  - Automatically identifies task type (binary classification, multiclass, regression) if not explicitly set.
  - Automatically provisions an `AutoMLWorkspace`, registers the dataset, runs problem planning, and executes priority experiments.
- **Inference Delegation:**
  - `predict(data)` delegates to `best_model.predict(data)`.
  - `predict_proba(data)` delegates to `best_model.predict_proba(data)`.

### 2.2 Result Container (`AutoMLResult`)
- Attributes:
  - `best_model: ModelArtifact`
  - `best_score: float`
  - `best_model_id: str`
  - `task_type: str`
  - `metric: str`
- Methods:
  - `leaderboard() -> pd.DataFrame`: Returns a formatted DataFrame with model ranks, primary scores, cross-validation std, duration, and trial parameters.
  - `predict(data)` -> numpy array.
  - `predict_proba(data)` -> numpy array.
  - `summary() -> dict[str, Any]`.

### 2.3 Standalone Model Artifact (`ModelArtifact`)
- Encapsulates:
  - Fitted preprocessor / feature pipeline.
  - Fitted underlying estimator (scikit-learn / LightGBM / XGBoost / CatBoost).
  - Metadata: feature names, target column name, task type, validation score, random seed.
- Methods:
  - `save(path: str | Path) -> Path`: Serializes the entire artifact using `joblib.dump`.
  - `load(path: str | Path) -> ModelArtifact` (classmethod): Loads the artifact for immediate inference anywhere.
  - `predict(X: pd.DataFrame | np.ndarray) -> np.ndarray`.
  - `predict_proba(X: pd.DataFrame | np.ndarray) -> np.ndarray`.

---

## 3. Constraints & Architecture Alignment

1. **Hexagonal Integrity:** The `AutoML` facade acts as a client adapter on top of `AutoMLWorkspace` and `CommandBus` / `QueryBus`. Core domain models in `src/automl/domain/` remain pure standard library.
2. **Backward Compatibility:** All existing CLI commands, Workbench HTTP handlers, and MCP tools remain 100% functional and uncompromised.
3. **No Workspace Leakage in Production:** `ModelArtifact` must run standalone with only `joblib`, `numpy`, and `scikit-learn` installed.

---

## 4. Acceptance Criteria

- [ ] `from automl import AutoML` and `from catml import AutoML` work.
- [ ] `automl.fit(df, target="col")` runs end-to-end and returns `AutoMLResult`.
- [ ] `automl.fit(X, y)` runs end-to-end and returns `AutoMLResult`.
- [ ] `result.leaderboard()` returns a populated pandas DataFrame.
- [ ] `result.best_model.save(tmp_path / "model.pkl")` creates a valid file.
- [ ] `ModelArtifact.load(tmp_path / "model.pkl")` successfully predicts without workspace or database.
- [ ] Full existing test suite passes (421+ tests), with test coverage $\ge 85\%$.
