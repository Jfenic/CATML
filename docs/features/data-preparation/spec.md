# Feature Specification: Data Preparation Subsystem

**Status:** Draft / Accepted as product direction (post-`0.9.0rc1`)  
**Milestone:** Post-RC tabular evolution (not a Labs scaffold)  
**Related:** [ADR-012](../../decisions/012-data-preparation-subsystem.md), [ADR-001](../../decisions/001-hexagonal-architecture-and-cqrs.md), [ADR-002](../../decisions/002-hypothesis-driven-experiments.md), [ADR-009](../../decisions/009-experiment-recipes-and-interface-parity.md), [Explore](../catml-explore/spec.md), [Derived features](../derived-features/spec.md), [Experiment Recipes](../experiment-recipes/spec.md), [plan](plan.md)

---

## 1. Executive summary

Data preparation is repetitive and a common source of silent evaluation errors. CATML already **diagnoses** (Explore / profiler / anti-leakage) and **preprocesses inside training folds** (`SimpleImputer`, encoders, scalers). What is missing is a reproducible bridge:

> Diagnose → Propose plan → Preview → Approve → Apply (new dataset version **or** training-pipeline step)

**Product positioning:** ML-oriented preparation — not a general ETL competitor (OpenRefine / Alteryx).

**Parity rule:** one `PreparationService` in the application layer; Python facade, CLI, MCP, and Workbench are adapters only (CQRS).

**Calendar:** design and document now; implement after tabular RC ship. Do **not** force a long cleaning wizard before `fit`.

---

## 2. Core distinction (non-negotiable)

| Dataset preparation (`scope=dataset`) | Model preprocessing (`scope=training_pipeline`) |
| --- | --- |
| Fix import types, column names, date formats | Impute with statistics learned on train |
| Unify categorical labels via **fixed rules** | Scale / OneHot / ordinal encode |
| Dedupe under an **explicit policy** | Feature selection / PCA fit on train |
| Schema constraint validation | Anything that **learns parameters from data** |
| Produces a **new immutable dataset version** | Lives inside CV folds / pipeline |

**Leakage rule:** any step that learns from column values (median, mean, min/max for scaling, target encoding, etc.) **must** declare `scope=training_pipeline` and fit only on training folds. Computing those stats on the full table before split is forbidden.

---

## 3. Reuse existing CATML pieces

| Existing | Role |
| --- | --- |
| `DatasetProfiler` / Explore | Evidence for diagnostics (nulls, types, cardinality, IDs, associations) |
| Anti-leakage guardian | Continues to flag/exclude contaminated columns |
| Derived feature apply path | Precedent for transform → re-profile; evolve toward **versions**, not silent overwrite |
| `ExperimentRecipe` | Persist preparation evidence + `dataset_content_hash` for replay |
| Trainer `ColumnTransformer` | Keep trainable preprocessing inside pipelines |

**Do not duplicate** Explore’s statistical engine or the trainer’s fold-safe preprocess. Data Prep **orchestrates** typed operations and versions.

---

## 4. Domain contracts (conceptual)

### 4.1 `PreparationPlan`

Declarative list of `PreparationStep`s:

- `operation` (typed enum; no arbitrary Python from agents)
- `columns` / selectors
- `params`
- `scope`: `dataset` | `training_pipeline`
- `requires_approval` (destructive / semantic changes)
- `rationale` / link to Explore finding ids (optional)
- expected effect metadata for preview

### 4.2 Lifecycle

1. **Analyze** — quality report (actionable issues + severity + evidence)  
2. **Suggest** — plan from report heuristics (and optional LLM via existing ports)  
3. **Preview** — before/after stats: rows/cols touched, value changes, warnings, information loss  
4. **Validate** — schema / policy checks; reject illegal scopes  
5. **Apply** — only after approval when required → **new dataset version** + audit log; pipeline steps attached to recipe/run config  

### 4.3 Dataset versions

- Never overwrite the user’s original file in place as the sole outcome.  
- Store prepared artifact under workspace with content hash + lineage (`parent_dataset_id`, plan id, step digests).  
- Export / predict must replay the same preparation lineage (with Recipes).

---

## 5. Interface parity

| Surface | Behavior |
| --- | --- |
| Python | e.g. `automl.prepare.analyze / suggest / preview / apply` (names TBD) |
| CLI | `catml data inspect`, `catml data prepare suggest|preview|apply` |
| MCP | `analyze_data_quality`, `suggest_preparation`, `preview_preparation`, `apply_preparation` (apply gated) |
| Workbench | “Preparación” tab: OpenRefine-inspired **controlled** actions; client of the same commands |

Agents may analyze and propose freely; **apply** for destructive or meaning-changing ops requires human/policy approval. No free-form cleaning scripts via MCP.

---

## 6. v0 scope (deliberately small)

**In**

- Diagnostics: nulls, invalid types, inconsistent categoricals, duplicate candidates, constants, identifiers, target issues  
- Operations: normalize whitespace, replace-value maps, cast types, parse dates, drop/keep columns, explicit dedupe policy, mark pipeline impute/scale  
- Preview + approve + versioned apply  
- CLI + application service first  

**Out of v0**

- Automatic outlier deletion  
- Relabeling using the target  
- Class balancing outside folds (SMOTE etc.)  
- Free-form pandas/Polars scripts from UI/agents  
- Full graph ETL (KNIME-style)  
- Multimodal media cleaning (Vision/NLP labs later)

---

## 7. Risks and care points (read before implementing)

1. **Scope leakage** — easiest footgun; validate every step’s `scope` in domain.  
2. **Duplicates** — not all repeats are errors; policy must be explicit (keep first / drop / flag only).  
3. **Target-aware “cleaning”** — can encode the label into features; ban in v0 except clear non-learning rules.  
4. **Overwriting source data** — breaks audit and agent trust; versions only.  
5. **Blocking the happy path** — prep must remain optional; default fit keeps fold-safe preprocess.  
6. **Explore vs Prep confusion** — Explore proposes hypotheses; Prep changes data/pipeline contracts.  
7. **Recipe / export drift** — prepared dataset used at train must be reconstructible at predict.  
8. **Honest UX copy** — “no leakage detected” ≠ “dataset fully validated”.  
9. **Disk growth** — many versions; need retention/GC policy later.  
10. **Agent autonomy** — typed ops + approval; never `exec` user/LLM code for cleaning.  
11. **Dependency bloat** — start with pandas + existing sklearn; optional profiling/GE later.  
12. **Labs timing** — Prep service is **core application**; UI can live under Tabular Workbench first, not blocked on ADR-011 Labs extraction.

---

## 8. Acceptance criteria (when implemented)

- [ ] Domain plan/steps have zero sklearn/pandas imports in `domain/`.  
- [ ] Illegal `training_pipeline` stats computed on full frame are rejected by tests.  
- [ ] Apply creates a new dataset id/version; original remains readable.  
- [ ] Preview never mutates stored datasets.  
- [ ] Same plan runnable via CommandBus from CLI and a unit test (parity stub for MCP/UI).  
- [ ] Coverage gate ≥ 85% maintained.  
- [ ] Workbench badge/copy does not claim full validation from leakage-only checks.

---

## 9. References (inspiration, not dependencies)

- OpenRefine — interactive, reproducible transforms  
- ydata-profiling — diagnostic report depth  
- Great Expectations / Soda — future quality contracts  
- Existing CATML Explore + Recipes + trainer pipelines  
