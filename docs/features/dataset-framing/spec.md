# Technical Specification: Dataset Framing & Problem Context Questionnaire

> Status: **Implementation / Feature Branch (`feat/dataset-framing-questionnaire`)**. Track progress in [TASKS.md](../../../TASKS.md).

---

## 1. Context & Motivation

Traditional AutoML platforms operate blindly on numbers, ignoring the underlying business and operational context of the machine learning problem:
- **Asymmetric Cost of Errors:** In fraud detection, customer churn, or medical diagnostics, failing to detect a positive case (False Negative) is exponentially more expensive than a False Positive. Standard accuracy or unweighted metrics optimize for majority classes, leading to models that look statistically good but fail catastrophically in business impact.
- **Temporal & Sequential Realities:** If a dataset represents a customer lifecycle or sensor timeline, random K-Fold cross-validation leaks future information into past predictions.
- **Operational Serving Constraints:** Models for mobile or edge inference require sub-10ms response times (favoring shallow LightGBM or regularized linear models), whereas batch scoring pipelines can afford complex stacked super-ensembles.
- **Semantic Understanding via LLMs:** A human data scientist immediately knows that `customers_churn.csv` with columns `monthly_charges`, `tenure`, `contract`, `churn` represents a churn problem in subscription services. An LLM or heuristic advisor can inspect these metadata tokens and pre-fill an optimal problem framing questionnaire automatically.

---

## 2. Scope & Architectural Boundaries

### A. In Scope
- **Pure Domain Entities (`src/automl/domain/tasks/questionnaire.py`):**
  - `DatasetQuestionnaire`: Immutable, pure Python dataclass capturing framing dimensions.
  - Enums for `ErrorCostPriority`, `TemporalStructure`, `LatencyConstraint`, and `ExplainabilityLevel`.
- **Inference Engine (`src/automl/engine/planning/questionnaire_advisor.py`):**
  - `QuestionnaireAdvisor`: Evaluates dataset name, column names, target column, cardinality, and profile statistics to synthesize a structured questionnaire.
  - **Dual Engine:**
    1. Deterministic heuristic rules (zero dependencies, instant, domain pattern matching).
    2. Optional LLM provider mode (invoking `LLMProviderPort` for deep semantic reasoning).
- **Workspace & Service Integration (`src/automl/application/services/workspace.py`):**
  - First-class workspace methods: `generate_dataset_questionnaire`, `get_dataset_questionnaire`, `save_dataset_questionnaire`.
  - Persisted in dataset metadata in SQLite repository.
- **Planner & Metric Alignment:**
  - Automated translation of questionnaire answers into recommended primary metrics (e.g. `pr_auc` or `weighted_f1` for imbalanced churn/fraud) and split strategies (`stratified_kfold`, `time_series_split`, `group_kfold`).
- **Workbench Web & CLI Inspection:**
  - API endpoints `/api/dataset/questionnaire` (GET and POST).
  - CLI subcommand `automl dataset questionnaire --dataset-id <ID>`.

### B. Out of Scope
- Does not modify existing agent orchestrator loop (maintains zero coupling with Persona B / Milestone B4).
- Does not mandate cloud API keys; runs with deterministic heuristic rules by default.

---

## 3. Domain Model Specifications

```python
class ErrorCostPriority(str, Enum):
    BALANCED = "balanced"
    AVOID_FALSE_NEGATIVES = "avoid_false_negatives"
    AVOID_FALSE_POSITIVES = "avoid_false_positives"
    CUSTOM_COST_MATRIX = "custom_cost_matrix"

class TemporalStructure(str, Enum):
    CROSS_SECTIONAL = "cross_sectional"
    SEQUENTIAL_TIME_SERIES = "sequential_time_series"
    GROUPED_COHORTS = "grouped_cohorts"
    UNKNOWN = "unknown"

class LatencyConstraint(str, Enum):
    ULTRA_LOW_REALTIME = "ultra_low_realtime"      # < 10ms (Linear, shallow tree)
    STANDARD_INTERACTIVE = "standard_interactive"  # < 200ms (GBDT)
    BATCH_OFFLINE = "batch_offline"                # No limit (Stacking, Super-Learner)

class ExplainabilityLevel(str, Enum):
    HIGHLY_REGULATED = "highly_regulated"          # Linear, decision trees, strict monotonicity
    MODERATE = "moderate"                          # Tree SHAP, feature importances
    PERFORMANCE_FIRST = "performance_first"        # Deep ensembles, unrestricted

@dataclass
class DatasetQuestionnaire:
    dataset_id: str
    domain_hint: str
    error_cost: ErrorCostPriority
    temporal_structure: TemporalStructure
    latency_constraint: LatencyConstraint
    explainability: ExplainabilityLevel
    imbalance_strategy: str
    primary_metric_override: str | None
    split_strategy_recommendation: str
    auto_generated: bool = True
    confidence: float = 1.0
    reasoning: str = ""
```

---

## 4. Acceptance Criteria & Validation

1. **Pure Domain:** `questionnaire.py` has 0 third-party imports (pure dataclasses and enums).
2. **Deterministic Heuristic Inference:** Correctly identifies telecom churn, fraud, and medical datasets based on column names and class balance.
3. **LLM Provider Compatibility:** Accurately formats schemas and handles LLM output via `LLMProviderPort`.
4. **Persistence & Serialization:** Full JSON roundtrip testing and persistence in `SqliteRepository`.
5. **Coverage & Tests:** Unit test suite in `tests/test_dataset_questionnaire.py` maintaining repository coverage >= 85%.
