# Feature Specification: V0.6 Plugin Architecture & Extensible Ecosystem

> Formalizing extensibility so that adding a model, metric, preprocessor, or optimizer does not modify core domain or engine code.

## Goal

Provide a unified, contract-based plugin system (`PluginRegistry`, `ModelPlugin`, `MetricPlugin`, `OptimizerPlugin`, `PreprocessorPlugin`) with capability discovery and compatibility validation, allowing third-party and custom components (e.g. LightGBM, XGBoost, custom business metrics) to be plugged into CATML seamlessly.

---

## User & Business Need

1. **Seamless Extensibility:** Adding a new ML library (e.g. LightGBM, XGBoost, CatBoost) or domain-specific loss function should not require editing `trainer.py`, `planner.py`, or `domain/`.
2. **Safety & Early Failure:** Running an incompatible combination (e.g. regression model on multiclass task, or an image preprocessor on tabular data) must be detected and rejected before spending time and compute on trial execution.
3. **Reproducibility & Auditing:** The exact plugin version and configuration must be captured in `TrialResult` and persisted in SQLite for compliance and reproducibility.

---

## Requirements

### 1. Unified Plugin Contract
- All plugins implement a common base protocol `PluginPort`:
  - `plugin_id: str`
  - `name: str`
  - `version: str`
  - `plugin_type: PluginType` ("model", "metric", "optimizer", "preprocessor")
  - `capabilities: PluginCapability` (supported task types, supported data modalities, GPU requirement)

### 2. Specialized Plugin Interfaces
- **`ModelPluginPort`:**
  - Instantiates the underlying estimator with parameters.
  - Exposes default `SearchSpace` for hyperparameter optimization.
  - Provides fit / predict / predict_proba capabilities.
- **`MetricPluginPort`:**
  - Evaluates predictions vs ground truth.
  - Defines optimization direction (`greater_is_better: bool`).
- **`PreprocessorPluginPort`:**
  - Fits and transforms input data representations.

### 3. Registry & Compatibility Validation
- `PluginRegistry`: Registers, discovers, and queries plugins by capability and type.
- `CompatibilityValidator`: Validates plugin compatibility against `TaskType` and run configuration before trial dispatch.

### 4. Gradient Boosting & Custom Metric Plugins
- Migration/adaptation of existing sklearn models to `ModelPluginPort`.
- Adapters for `LightGBM` and `XGBoost` (with optional graceful fallback if packages are not installed).
- Support for custom business metrics (e.g. profit/cost curve, custom weighted F1).

---

## Constraints

- Pure domain boundary: `PluginPort`, `PluginCapability`, and `PluginType` in `domain/` must remain pure Python.
- Concrete ML plugins (LightGBM, XGBoost, Scikit-learn) reside strictly in `src/automl/plugins/`.
- Must not break existing 52 tests or existing CLI commands.

---

## Acceptance Criteria

1. **Contracts & Registry:** `PluginPort`, `ModelPluginPort`, `MetricPluginPort`, `PluginRegistry`, and `CompatibilityValidator` fully implemented and tested.
2. **Gradient Boosting Models:** Adaptable support for LightGBM and XGBoost via plugins with declarative search spaces.
3. **Custom Metrics:** Ability to register and execute custom metrics with explicit `greater_is_better` direction.
4. **Validation:** Incompatible model/task assignments raise explicit `ValueError` before trial execution.
5. **Testing & Coverage:** Test suite `tests/test_v06_plugins.py` passes with global test coverage maintained $\ge 85\%$.

---

## Non-Goals / Out of Scope

- Distributed plugin execution across multiple remote clusters (deferred to later versions).
- Dynamic hot-reloading from arbitrary remote URLs at runtime without local installation.
