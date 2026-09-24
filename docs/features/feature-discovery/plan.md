# Implementation Plan: V0.5 Feature Discovery & Selection

> Detailed roadmap and implementation sequence for phase V0.5.

## 1. Implementation Approach

V0.5 introduces an explicit feature analysis and selection pipeline that connects dataset profiling with experiment planning:

1. **Domain Abstraction:** Define `FeatureSelectorPort`, `FeatureRank`, `FeatureSelectionStrategy`, and `FeatureEvidence` in `domain/features/` and `domain/ports.py`.
2. **Engine Implementation:** Build filter (`MutualInfoSelector`) and embedded selectors (`ImportanceFeatureSelector`) in `engine/features/`.
3. **Ablation Generation:** Build `AblationPlanner` in `engine/planning/` to propose leave-one-out feature candidate experiments.
4. **Application CQRS:** Wire `SelectFeaturesCommand` and `GetFeatureEvidenceQuery` through `CommandBus`/`QueryBus` into `AutoMLWorkspace`.
5. **CLI Integration:** Provide `automl features select` and `automl features ablation` subcommands.
6. **Benchmark Validation:** Add a quality regression benchmark scenario comparing full feature baseline against selected feature subsets.

---

## 2. Affected Modules & Files

### Domain
- `src/automl/domain/ports.py` — Add `FeatureSelectorPort` protocol.
- `src/automl/domain/features/selection_strategy.py` — Add `FeatureSelectionStrategy`, `FeatureRank`, and combination rules.
- `src/automl/domain/features/evidence.py` — Add `FeatureEvidence` and `FeatureInteractionEvidence`.

### Engine
- `src/automl/engine/features/selectors.py` — Implement `MutualInfoSelector` and `ImportanceFeatureSelector`.
- `src/automl/engine/features/ranking.py` — Implement ranking aggregators (`Borda`, weighted rank).
- `src/automl/engine/planning/ablation_planner.py` — Implement leave-one-out candidate generation.

### Application Layer
- `src/automl/application/commands/workspace_commands.py` — `SelectFeaturesCommand`, `PlanAblationExperimentsCommand`.
- `src/automl/application/queries/workspace_queries.py` — `GetFeatureEvidenceQuery`, `ListCandidateFeatureSetsQuery`.
- `src/automl/application/services/workspace.py` — Orchestration methods for feature selection and ablation.
- `src/automl/application/bootstrap.py` — Handlers registration.

### Persistence & Benchmarks
- `src/automl/infrastructure/database/sqlite_repository.py` — Save and retrieve feature evidence and candidate sets.
- `src/automl/benchmarks/runner.py` — New scenario `feature_selection_v05`.
- `tests/test_v05_features.py` — Unit and integration tests.

---

## 3. Implementation Sequence

1. **Step 1: Domain Entities & Protocols**
   - Create `FeatureSelectorPort` in `ports.py`.
   - Create `FeatureSelectionStrategy` and `FeatureEvidence` dataclasses.
2. **Step 2: Concrete Selectors (Engine)**
   - Implement `MutualInfoSelector` using `scikit-learn`'s `mutual_info_classif` / `mutual_info_regression`.
   - Implement `ImportanceFeatureSelector` extracting Gini / tree feature importances.
3. **Step 3: Ranking Combiner & Candidate FeatureSet Generation**
   - Support `top_k` extraction and combined ensemble ranks.
4. **Step 4: Ablation Planner**
   - Implement leave-one-out experiment generator for top features.
5. **Step 5: Application Bus Wiring**
   - Add Commands and Queries, register in `bootstrap.py`, implement methods in `workspace.py`.
6. **Step 6: CLI & Benchmark Verification**
   - Add CLI subcommands.
   - Add benchmark scenario `feature_selection_v05` and verify score & runtime improvement.

---

## 4. Risks & Mitigations

- **Risk: High computational overhead for wrapper/ablation methods.**
  - *Mitigation:* Restrict ablation candidate generation to top-$K$ prioritized features (default $K \le 5$) and respect `BudgetPolicy`.
- **Risk: Domain pollution with sklearn estimators.**
  - *Mitigation:* Strict adherence to ports; domain entities only hold string column names, weights, and scores.

---

## 5. Migrations

- Extend SQLite schema in `sqlite_repository.py` to add `feature_evidence` table if persisting between runs, or store evidence as structured JSON within existing event tables.

---

## 6. Validation Strategy

- Run unit tests: `.venv/bin/pytest tests/test_v05_features.py`
- Run regression suite: `.venv/bin/pytest tests/` (all 44+ tests passing, coverage >= 85%)
- Run benchmark: `.venv/bin/automl benchmark run` ensuring `feature_selection_v05` records positive gain or reduced trial runtime.
