# Feature Specification: V0.5 Feature Discovery & Selection

> Specification for intelligent feature selection, interaction discovery, and experimental validation in CATML.

## Goal

Provide an explicit, modular subsystem for feature analysis, selection, and discovery that generates ranked feature subsets and interaction candidates, validating every feature change through controlled experiments rather than heuristic acceptance.

---

## User & Business Need

1. **Curse of Dimensionality & Noise:** Large tabular datasets often contain redundant, collinear, or noisy features that degrade model generalization and increase training cost.
2. **Explainability & Auditing:** Data scientists and regulators need to understand *why* features were retained or excluded (e.g. mutual information score, tree importance, ablation impact).
3. **Hypothesis Verification ("Propose ≠ Accept"):** Statistical importance does not guarantee improved generalization on a test split. The system must treat feature sets as testable hypotheses.

---

## Requirements

### 1. Conceptual Separation
- **Feature Selection:** Identifying which original columns provide predictive signal (Filter: Mutual Information, Variance, Correlation; Embedded/Post-hoc: Tree Importance, L1, SHAP; Wrapper: RFE).
- **Dimensionality Reduction:** Projecting features into lower-dimensional components (e.g. PCA) treated as distinct feature pipelines.
- **Experimental Validation:** Evaluating candidate feature sets using cross-validation trials against baseline models.

### 2. Core Capabilities
- **FeatureSelectorPort Contract:** A unified protocol requiring `fit(context)`, `rank_features()`, and `select(k)`.
- **Declarative Selection Strategy:** `FeatureSelectionStrategy` defining methods to run, top-k limits, thresholding, and ensemble ranking combination (e.g. Borda count or weighted rank).
- **Ablation Studies:** Automated generation of leave-one-out experiments to measure the marginal contribution of individual features.
- **Evidence Tracking:** Accumulation of `FeatureEvidence` records (method scores, ablation delta, confidence) linked to runs and datasets.
- **Candidate FeatureSets:** Generation of candidate subsets that feed directly into the `ExperimentPlanner` and `PriorityQueue`.

---

## Constraints

- Pure domain boundary: `FeatureSelectorPort`, `FeatureSelectionStrategy`, and `FeatureEvidence` must remain pure Python without direct scikit-learn or external library imports in `domain/`.
- Concrete implementations (scikit-learn selectors, stats) reside strictly in `engine/features/` or `plugins/`.
- Feature sets must only be promoted to active/curated status after experimental validation confirms metric parity or improvement.

---

## Acceptance Criteria

1. **Protocol & Implementations:** `FeatureSelectorPort` implemented with at least `MutualInfoSelector` (Filter) and `ImportanceFeatureSelector` (Embedded/Tree).
2. **Ablation Planner:** Planner capable of proposing leave-one-out experiments for top features.
3. **CQRS Integration:** `SelectFeaturesCommand` and `GetFeatureEvidenceQuery` exposed and registered in `bootstrap.py`.
4. **Testing:** Complete test suite covering selectors, ranking combination, and ablation generation with >= 85% coverage.
5. **Benchmark Validation:** Benchmark scenario `feature_selection_v05` showing competitive or superior performance using selected subsets vs all features.

---

## Non-Goals / Out of Scope

- Deep representation learning (autoencoders) or neural embeddings (deferred to V0.7).
- Full exhaustive search of all $2^N$ feature combinations (prohibited; must use prioritized heuristic subsets).
- LLM-based semantic feature engineering (deferred to V0.9/V1.0).
