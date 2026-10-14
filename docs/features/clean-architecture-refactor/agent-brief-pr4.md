# Brief — PR4 Persistencia, estados y Unit of Work

> **Spec/plan:** [spec.md](spec.md) · [plan.md](plan.md) · [ADR-010](../../decisions/010-progressive-clean-architecture-refactor.md)  
> **Rama:** `refactor/clean-arch-pr4-persistence-uow`  
> **Base:** `origin/main` (merge `#128` when available; never rebase)

## Misión

Modularizar persistencia SQLite y formalizar transiciones de estado sin romper la fachada `SQLiteExperimentRepository` ni CQRS.

## Entregables

1. **Domain transitions** — `RunStatus` / `ExperimentStatus` con mapas explícitos; rechazar `CANCELLED`/`COMPLETED` → estados vivos.
2. **Unit of Work** — conexión compartida + commit/rollback para operaciones multi-tabla (trial + result).
3. **Repos por agregado** — `datasets` / `runs` / `experiments` (+ evidence) como módulos; fachada delega.
4. **Tests** — `tests/test_clean_arch_pr4_persistence_uow.py`.

## Fuera de alcance

- Ruff/mypy/ESLint (PR5)
- Reescritura de `workspace.py`
- Cambiar schema SQLite salvo migraciones aditivas mínimas
- Microservicios / otro motor de BD

## Criterio done

- Suite verde; cobertura ≥ 85 %
- Callers existentes (`Workspace`, jobs, recipes) sin cambio de firma pública
- `TASKS.md` track Clean Arch PR4 → Done
