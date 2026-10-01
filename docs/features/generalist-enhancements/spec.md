# Technical Specification: Generalist AutoML Engine, Temporal Dynamics & Super-Ensembles

> Status: **Proposed / Architectural Design**. Track progress in [TASKS.md](../../../TASKS.md).

---

## 1. Context & Motivation

Following empirical validation across competitive and tabular benchmarks (Kaggle S6E9 and the Rainfall Probability Prediction Challenge), four key architectural bottlenecks were identified for generalist AutoML adoption:

1. **Temporal and Physical Inertia:** Continuous tabular datasets (meteorological, financial, industrial telemetry) contain consecutive chronological observations where predictive power relies heavily on rate-of-change deltas ($X_t - X_{t-1}$), sensor lags, and cyclic seasons. Previously, these required manual feature scripts outside the automated pipeline.
2. **OOF Ensemble Scalability Bottleneck:** [`GenerateOOFSubmissionCommand`](../../../src/automl/application/services/oof_submission.py) rigidly capped ensembles to at most 2 individual models (`if len(models) > 2: raise ValueError`) with equal 50/50 weighting. This prevented blending 4 to 5 orthogonal algorithm families or applying Level-2 Stacking (Super Learner).
3. **Operational UI Friction:** The Workbench UI allowed single-model execution but lacked 1-click visual controls for $K$-fold cross-validation strategies, multi-trial ensemble creation, and drag-and-drop submission generation.
4. **Data Leakage Guardian:** Lack of automated warnings during profiling for spurious perfect correlations ($|r| \ge 0.999$) or sequential target contamination.

---

## 2. Scope & Architectural Boundaries

### A. In Scope

- **`TemporalDynamicsGenerator` Engine:**
  - Automated detection of temporal and sequential structures in `DatasetProfiler`.
  - Deterministic generation of:
    - Sensor lags: $X_{t-1}, X_{t-2}$.
    - Trend deltas: $\Delta X = X_t - X_{t-1}$.
    - Cyclical trigonometric projections: $\sin(2\pi \cdot t / T)$ and $\cos(2\pi \cdot t / T)$ for detected periodicities ($T \in \{365.25, 12, 7, 24\}$).
  - Packaged as candidate [`FeatureSet`](../../../src/automl/domain/features/feature_set.py) instances obeying *"Propose ≠ Accept"*.
- **Generalized $N$-Model Out-Of-Fold (OOF) Engine:**
  - Removal of the 2-model limitation in [`evaluate_oof()`](../../../src/automl/engine/ensemble/oof.py) to accept arbitrary lists $M = [m_1, \dots, m_N]$ ($N \ge 1$).
  - First-class Rank-Averaging (`method="rank"`) and Nelder-Mead simplex weight optimization in OOF commands.
  - Level-2 Stacking with a regularized linear meta-estimator (`RidgeClassifier` / `Ridge`).
- **Workbench Visual ML Lab (UI Enhancements):**
  - Validation Strategy Selector in New Experiment modal: *Holdout (80/20)*, *5-Fold CV*, *10-Fold CV*.
  - Visual Ensemble Studio: Trial multi-select checkboxes on the Leaderboard and `[🔨 Assemble Selected Models]` action.
  - Kaggle Submission Center: Drag-and-drop file upload with pre-download integrity checklist.
- **Anti-Leakage Guardian:**
  - Automated diagnostic in `DatasetProfiler` for suspicious target correlations ($|r| \ge 0.999$) and sequential ordering leaks.

### B. Out of Scope & Explicit Non-Claims

- No guaranteed score improvements or platform-specific rankings; gains must be verified empirically via local cross-validation.
- Does not replace independent holdout verification for formal production candidate promotion.
- Does not train heavy deep learning recurrent neural networks (LSTMs/Transformers); focuses on high-efficiency tabular lag/delta engineering for GBDTs and linear estimators.

---

## 3. Application Layer Contracts (CQRS)

### Mutating Commands
- `GenerateTemporalFeaturesCommand`:
  - `run_id: str`
  - `dataset_id: str`
  - `max_lags: int = 1`
  - `include_deltas: bool = True`
  - `include_cyclical: bool = True`
  - Returns: `list[FeatureSet]` (registered candidate sets).
- `GenerateOOFSubmissionCommand` (extended):
  - `model_ids: list[str]` (supports $N \ge 1$ heterogeneous models).
  - `method: str = "average" | "rank" | "simplex" | "stacked"`.
  - `meta_model: str | None = None` (e.g. `"ridge"` for Level-2 Stacking).
  - `folds: int = 5` (or 10).
- `CreateEnsembleExperimentCommand`:
  - `run_id: str`
  - `trial_ids: list[str]`
  - `method: str`
  - Returns: `experiment_id: str`.

### Read-Only Queries
- `GetOOFResultQuery`: Returns report with individual metrics for all $N$ models, per-fold scores, dispersion, artifact hashes, and blend diagnostics.
- `DetectTemporalStructureQuery`: Inspects dataset and returns detected periodicities and sequential columns.

---

## 4. File Traceability Matrix

| Component | Layer | Target Files |
|---|---|---|
| **Domain Contracts** | `domain/` | [`src/automl/domain/features/temporal.py`](../../../src/automl/domain/features/), [`src/automl/domain/ports.py`](../../../src/automl/domain/ports.py) |
| **Temporal Dynamics Engine** | `engine/` | `src/automl/engine/features/generation/temporal_generator.py` |
| **OOF & Stacking Engine** | `engine/` | [`src/automl/engine/ensemble/oof.py`](../../../src/automl/engine/ensemble/oof.py), [`src/automl/engine/ensemble/blender.py`](../../../src/automl/engine/ensemble/blender.py) |
| **Application Services & CQRS** | `application/` | [`src/automl/application/services/oof_submission.py`](../../../src/automl/application/services/oof_submission.py), [`src/automl/application/bootstrap.py`](../../../src/automl/application/bootstrap.py) |
| **Interfaces (CLI & Web UI)** | `interfaces/` | [`src/automl/interfaces/cli/main.py`](../../../src/automl/interfaces/cli/main.py), [`src/automl/interfaces/web/server.py`](../../../src/automl/interfaces/web/server.py), `static/js/views/` |
| **Test Suites** | `tests/` | `tests/test_temporal_dynamics.py`, `tests/test_oof_multi_model.py`, `tests/test_web_dashboard.py` |
