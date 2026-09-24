# Decision: Hypothesis-Driven Experiments and "Propose ≠ Accept" Principle

## Context

In traditional AutoML systems, feature selection or model exploration algorithms directly modify the active training pipeline whenever a statistical score or heuristic suggests an improvement. 

In CATML, we need:
1. Complete traceability of why a feature or model was selected or discarded.
2. The ability to prioritize, review, or override candidate configurations before spending computational budget.
3. Separation between proposal generation and empirical validation.

---

## Decision

We model **`Experiment`** and **`Trial`** as first-class domain entities and enforce the **"Propose ≠ Accept"** principle:
- Planners, optimizers, and feature selectors output uncommitted **Candidates** (e.g. `ExperimentCandidate`, feature subset rankings).
- Candidates are evaluated and ranked through an explainable **Priority Engine** (`system_score`, `user_score`, `effective_score`).
- A candidate is only "accepted" or promoted to a curated set after being executed as an `Experiment` with measurable `TrialResult` metrics compared against a baseline.

---

## Alternatives Considered

- **Immediate in-place mutation:** Let selectors and optimizers modify the active dataset and model directly. Rejected because it eliminates explainability, prevents user overrides, and risks silent degradation on holdout data.
- **Purely manual experiment definitions:** Require human operators to define every experiment. Rejected because it eliminates the benefits of automated ML search.

---

## Consequences

- **Easier:** Auditing decisions; comparing baseline vs experiments in benchmarks; resuming interrupted runs; explaining priority calculations to users and agents.
- **Harder:** Requires candidate generation, priority queueing, and promotion logic.
- **Required:** Every proposed feature subset or model candidate must produce a quantifiable delta ($\Delta$) against baseline before being promoted.
