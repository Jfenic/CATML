# Implementation Plan: V0.6 Plugin Architecture & Extensible Ecosystem

> Sequence and roadmap for implementing the modular plugin architecture.

## 1. Implementation Approach

1. **Domain Contracts (`src/automl/domain/`):**
   - Create `src/automl/domain/plugins/plugin.py`: Define `PluginType` (enum), `PluginCapability` (dataclass), and protocols in `src/automl/domain/ports.py`: `PluginPort`, `ModelPluginPort`, `MetricPluginPort`, `PreprocessorPluginPort`.
2. **Engine Validation & Registry (`src/automl/engine/` & `application/`):**
   - Create `src/automl/engine/plugins/validator.py`: `CompatibilityValidator` ensuring model/task/metric alignment.
   - Create `src/automl/application/plugins/registry.py`: `PluginRegistry` managing plugin registration, discovery, and filtering by task.
3. **Plugins Layer (`src/automl/plugins/`):**
   - Create `src/automl/plugins/models/sklearn_plugin.py`: Wrap existing sklearn models into `ModelPluginPort`.
   - Create `src/automl/plugins/models/gradient_boosting.py`: Implement `LightGBMPlugin` and `XGBoostPlugin` with declarative `SearchSpace` and fallback handling.
   - Create `src/automl/plugins/metrics/business_metrics.py`: Custom metrics with cost matrices and directional optimization.
4. **Execution Engine Integration (`src/automl/engine/training/`):**
   - Enhance `SklearnTrainer` to check `PluginRegistry` for external model plugins when a model is not standard sklearn.
5. **Application CQRS & Workspace:**
   - Integrate `PluginRegistry` into `AutoMLWorkspace` and `bootstrap.py`.
   - Add queries: `ListPluginsQuery(plugin_type=None)`.
6. **Tests & Verification:**
   - Suite `tests/test_v06_plugins.py` validating registration, compatibility rejection, custom metrics, and gradient boosting execution.

---

## 2. Affected Modules & Files

- `src/automl/domain/plugins/plugin.py` (new)
- `src/automl/domain/ports.py` (add plugin protocols)
- `src/automl/application/plugins/registry.py` (new)
- `src/automl/engine/plugins/validator.py` (new)
- `src/automl/plugins/models/sklearn_plugin.py` (new)
- `src/automl/plugins/models/gradient_boosting.py` (new)
- `src/automl/plugins/metrics/business_metrics.py` (new)
- `src/automl/engine/training/sklearn_trainer.py` (dispatch to ModelPlugin)
- `src/automl/application/services/workspace.py` (integrate PluginRegistry)
- `src/automl/application/bootstrap.py` (register ListPluginsQuery)
- `tests/test_v06_plugins.py` (new test suite)

---

## 3. Implementation Sequence

1. **Step 1:** Domain abstractions (`PluginCapability`, `PluginType`, `PluginPort`, `ModelPluginPort`, `MetricPluginPort`).
2. **Step 2:** Compatibility validator and `PluginRegistry`.
3. **Step 3:** Sklearn model adapter plugin and custom business metric plugins.
4. **Step 4:** Gradient boosting model plugins (LightGBM/XGBoost).
5. **Step 5:** Trainer and workspace integration.
6. **Step 6:** Unit tests, regression verification, and progress updates.

---

## 4. Risks & Mitigations

- **Optional ML Dependencies:** `lightgbm` or `xgboost` might not be preinstalled in all environments.
  - *Mitigation:* Implement graceful `is_available()` check and mock/fallback estimators so tests and platform function with or without optional packages installed.

---

## 5. Validation Strategy

- Run tests: `.venv/bin/pytest tests/test_v06_plugins.py`
- Full test suite: `.venv/bin/pytest --cov=src/automl` (ensuring 0 regressions and >= 85% coverage).
