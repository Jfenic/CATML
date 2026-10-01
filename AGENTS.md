# Agent Instructions — CATML (AutoML Platform)

> Working rules, validation commands, and documentation navigation for AI coding agents.

## Navigation

- **System Architecture:** See [`ARCHITECTURE.md`](ARCHITECTURE.md). For deep technical specifications and roadmap, see [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md).
- **Current Work & Backlog:** See [`TASKS.md`](TASKS.md).
- **Short-Term Memory & Progress:** See [`.agent/progress.md`](.agent/progress.md).
- **Developer Extension Guide:** See [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).
- **Active Feature Specs & Plans:** See [`docs/features/`](docs/features/) (e.g. [`docs/features/feature-discovery/spec.md`](docs/features/feature-discovery/spec.md) and [`plan.md`](docs/features/feature-discovery/plan.md)).
- **UI Design System (Neo-Industrial):** See [`docs/design/neo-industrial-ui-spec.md`](docs/design/neo-industrial-ui-spec.md) and [`docs/decisions/005-neo-industrial-visual-ml-lab-ui.md`](docs/decisions/005-neo-industrial-visual-ml-lab-ui.md).
- **Architectural Decisions (ADRs):** See [`docs/decisions/`](docs/decisions/).
- **Original Architectural Notes:** See [`planning.txt`](planning.txt).

---

## Validation Commands

Run from repository root with the project virtual environment:

```bash
# Run all unit and integration tests
.venv/bin/pytest

# Run tests with coverage
.venv/bin/pytest --cov=src/automl --cov-fail-under=85

# Run a specific test suite
.venv/bin/pytest tests/test_v06_plugins.py

# Verify CLI entry point
.venv/bin/automl --help
.venv/bin/automl task list
```

The entire current test suite must pass before completing any task. Test coverage must remain >= 85%.

---

## Core Rules for Working in CATML

1. **Strict Dependency Direction (Hexagonal Architecture):**
   - `domain/` is pure Python (dataclasses, protocols, standard library).
   - **NEVER** import `scikit-learn`, `optuna`, `sqlite3`, `pandas`, or framework code inside `src/automl/domain/`.
   - External dependencies belong exclusively in `plugins/`, `infrastructure/`, or `engine/`.
2. **CQRS Separation:**
   - Commands mutate state and return IDs or void; they never return open queries.
   - Queries read state and return DTOs or primitives; they must have zero side effects.
   - Every Command and Query must be registered in `src/automl/application/bootstrap.py`.
3. **Parity Principle:**
   - The CLI (`src/automl/interfaces/cli/`), API, and future LLM agent tools operate exclusively via `CommandBus`, `QueryBus`, and `AutoMLWorkspace`. Never bypass the application layer to call models directly.
4. **Hypothesis-Driven Experimentation ("Propose ≠ Accept"):**
   - Feature selectors, planners, and agents propose candidates; they are only accepted into production or promoted to curated sets after verification by an `Experiment` producing measurable metrics.
5. **Scoped Changes:**
   - Keep changes tightly scoped to the assigned task.
   - Do not modify unrelated files.
   - Reuse existing abstractions in `src/automl/domain/ports.py` before creating new ones.
6. **Task & Progress Updates:**
   - Check `TASKS.md` and `.agent/progress.md` before starting.
   - Update `TASKS.md` and `.agent/progress.md` upon completion or when pausing work.
7. **Cross-Agent Communication & Out-of-Scope Blackboard (GitHub Issues):**
   - **Pre-flight review:** Before starting any new task, run `gh issue list --label blackboard --state open` to inspect recent warnings, discovered edge cases, or alerts left by other developers or agents.
   - **Out-of-Scope Reporting Protocol:** If during your work you identify a bug, code smell, design limitation, performance bottleneck, or potential improvement that is **outside your assigned task**:
     - **NEVER** edit unrelated files on the spot (strictly obey Rule 5: Scoped Changes).
     - **Report it immediately to the blackboard** using the GitHub CLI:
       ```bash
       gh issue create --title "[FINDING/IMPROVEMENT] Short title" --body "### Context\n...\n### Finding / Limitation\n...\n### Affected Modules\n...\n### Suggested Resolution\n..." --label "blackboard"
       ```
   - When an issue on the blackboard is addressed, close it via `gh issue close <id> --comment "Addressed in commit/PR <ref>"`.
8. **Neo-Industrial UI Standards:**
   - Any work on web interfaces (`src/automl/interfaces/web/`) must strictly comply with [`docs/design/neo-industrial-ui-spec.md`](docs/design/neo-industrial-ui-spec.md).
   - Use the visual ML lab paradigm: palette (`#111111`, `#D8D6CF`, `#F1EFE9`, `#E5512D`), typography (`Space Grotesk` + `IBM Plex Mono`), modular 1px visible borders, sharp corners (`0px`-`4px`), and signal orange strictly reserved for primary execution CTAs.
9. **Multi-Agent / Two-Person Concurrency & Conflict Prevention:**
   - When collaborating concurrently across Persona A and Persona B (or multiple developers/agents), strictly adhere to the isolation protocol in [`docs/features/agentic-system/two-person-plan.md`](docs/features/agentic-system/two-person-plan.md):
     - **Subdirectory Ownership:** Work exclusively within assigned subdirectories (e.g. `specialists/` vs `orchestrator/`).
     - **Shared Contract Stability:** Never edit monolithic shared files (`executor.py`, `sqlite_agent_ledger.py`, `ports.py`) concurrently on feature branches without a prior pre-agreed contract PR merged into `main`.
     - **Separated Documentation Tracks:** In `TASKS.md` and `.agent/progress.md`, update only your designated Persona track (`### Track Persona A` vs `### Track Persona B`).
     - **No Rebase:** Always merge (`git merge origin/main`); never rebase.
