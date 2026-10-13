# Documentación de CATML

La versión del paquete se define en [`automl.__version__`](../src/automl/__init__.py). Las fases V0.1–V1.0 representan hitos del diseño. El estado operativo y las tareas pendientes se mantienen en [TASKS.md](../TASKS.md).

## Dónde empezar

| Necesidad | Documento |
| --- | --- |
| Instalar y ejecutar CLI o Workbench | [README](../README.md) |
| Entender capas y dependencias | [Arquitectura actual](../ARCHITECTURE.md) |
| Añadir modelos y casos de uso | [Guía de desarrollo](../DEVELOPER_GUIDE.md) |
| Contribuir y validar cambios | [Contribución](../CONTRIBUTING.md) y [reglas de agentes](../AGENTS.md) |
| Sistema agéntico, MCP y roles A/B | [Sistema agéntico](features/agentic-system/README.md) |
| Gestionar trabajos, controles y recuperación | [Cola persistente](features/persistent-jobs/spec.md) |
| Comprobar backend y reproducibilidad | [Backends](backends.md) |
| Consultar diseño objetivo y fases futuras | [Especificación técnica](AutoML_Arquitectura_Tecnica.md) y [ADR 007: Roadmap con puertas de decisión](decisions/007-decision-gated-multimodal-governance-roadmap.md) |
| Publicar TestPyPI / PyPI (`0.9.0rc1`) | [Guía de publicación](publishing.md) y [Release workflow](../.github/workflows/release.yml) |
| Plan Maestro de Producto, Arquitectura y Mercado | [Plan Maestro](MASTER_PLAN.md) |
| Sistema de diseño UI (Tech Minimalista) | [Diseño UI](design/tech-minimalist-brand-system.md), [ADR 006](decisions/006-tech-minimalist-premium-brand-system.md) y [ADR 005](decisions/005-neo-industrial-visual-ml-lab-ui.md) |
| Entender decisiones | [ADRs](decisions/) |
| Consultar última ejecución y contexto | [Progreso](../.agent/progress.md) |

## Capacidades y evidencia

Esta tabla describe el alcance comprobable; los criterios ampliados de una especificación pueden seguir pendientes.

| Capacidad | Alcance actual y limitaciones | Implementación / evidencia |
| --- | --- | --- |
| Experimentos tabulares y CQRS | Operativos vía `CommandBus` / `QueryBus`; guardas anti-fuga a repositorio privado en interfaces | [Bootstrap](../src/automl/application/bootstrap.py), [guard CQRS](../tests/test_cqrs_architecture_guard.py) |
| Selección de features | Selectores, ranking, candidatos y ablación; propuestas requieren evaluación (*Propose ≠ Accept*) | [Spec V0.5](features/feature-discovery/spec.md), [tests](../tests/test_v05_features.py) |
| Plugins | Registro en memoria, compatibilidad y ejecución; re-registrar plugins personalizados en cada proceso; sin descubrimiento por entry points | [Spec V0.6](features/plugin-ecosystem/spec.md), [tests de registro](../tests/test_plugin_registration.py), [ejemplo](../examples/plugins/custom_model.py) |
| LightGBM / XGBoost / CatBoost | Backend nativo opcional, con alternativa sklearn por defecto | [Adaptadores](../src/automl/plugins/models/gradient_boosting.py), [backends](backends.md), [tests](../tests/test_v06_plugins.py) |
| Ensemble | Voting, blending ponderado, rank-averaging, simplex y stacking L2 (Ridge/Logistic/Lasso) | [Plugin](../src/automl/plugins/models/ensemble.py), [tests](../tests/test_ensemble_plugin.py), [Workbench Phase 3](../tests/test_workbench_phase3.py) |
| Interacciones | Generador de ratios, productos, diferencias y target encoding OOF | [Generador](../src/automl/engine/features/generation/interaction_generator.py), [tests](../tests/test_feature_interactions.py) |
| Multimodal | DAG, NLP ligero (TF-IDF), visión spike (`ImageEncoderNode` / feature head); fine-tuning end-to-end es roadmap v1.0 | [Spec V0.7](features/multimodal/spec.md), [tests E2E](../tests/test_v07_multimodal_e2e.py), [vision spike](../tests/test_v09_vision_spike.py) |
| CATML Explore | E0–E6 shipped: estudios, motor estadístico, UI, MCP/CLI, verify + `EvidenceLink`, benchmarks | [Plan](features/catml-explore/plan.md), [ADR-008](decisions/008-catml-explore-modular-monolith.md), [tests E1–E6](../tests/) |
| Experiment Recipes | R0–R4 shipped: verify transforms reales, dominio+CQRS, fit-in-fold, paridad CLI/MCP/REST/Workbench | [Spec](features/experiment-recipes/spec.md), [ADR-009](decisions/009-experiment-recipes-and-interface-parity.md), [tests R0–R4](../tests/) |
| Clean Architecture refactor | Progresivo PR1–PR5: contratos métrica/CV/ejecución → fachada Workspace → API/UI → UoW → tooling | [Spec](features/clean-architecture-refactor/spec.md), [Plan](features/clean-architecture-refactor/plan.md), [ADR-010](decisions/010-progressive-clean-architecture-refactor.md), código PR1 `#102` |
| Workbench Labs (futuro) | App Shell + labs por modalidad; motor CQRS único; **tabular first** — no implementar Labs hasta cerrar flujo tabular | [Plan](features/workbench-labs/plan.md), [ADR-011](decisions/011-workbench-labs-modular-frontend.md) |
| Data Preparation (futuro) | Plan declarativo dataset vs training_pipeline; preview/approve/versiones; paridad CLI/MCP/UI — post-rc1 | [Spec](features/data-preparation/spec.md), [Plan](features/data-preparation/plan.md), [ADR-012](decisions/012-data-preparation-subsystem.md) |
| Workbench HTTP | Loopback sin token; bind remoto fail-closed con Bearer / `#token=`; confinamiento de rutas | [Servidor](../src/automl/interfaces/web/server.py), [trusted deployment](trusted-deployment.md), [tests auth](../tests/test_workbench_auth.py) |
| Trabajos persistentes | Experimentos, OOF y submissions con worker, progreso real y controles pause/cancel | [Spec](features/persistent-jobs/spec.md), [tests](../tests/test_jobs.py), [ADR 003](decisions/003-persistent-local-jobs.md) |
| Knowledge / agente | **H0–H5 completados:** tools, aprobación, leases, MCP dual, especialistas, orquestador determinista + LangGraph durable | [Application agents](../src/automl/application/agents/), [CLI / MCP](../src/automl/interfaces/), [guía agéntica](features/agentic-system/README.md) |
| Predicciones / submissions | Plantilla Kaggle + OOF multi-modelo; **validación en holdout independiente aún pendiente** antes de promover candidatos | [Tests de plantilla](../tests/test_submission_template.py), [protocolo OOF](features/oof-blending/spec.md), [TASKS.md](../TASKS.md) |
| CATML Benchmark v1 | Protocolo **frozen** 1.0.0; Pilot 1A + Study A smoke; campaña 20 min / Study B scaffold | [`benchmarks/README.md`](../benchmarks/README.md), [Study A report](experiments/study-a-report.md) |

## Mantener la documentación

Usar enlaces relativos, ejemplos ejecutables y resultados de validación fechados. Mantener el backlog en `TASKS.md`, los contratos y criterios en las specs, y la secuencia histórica en sus planes. Los ADRs registran decisiones aceptadas y sus consecuencias; las modificaciones sustanciales deben indicar si reemplazan otra decisión. No trasladar resultados ilustrativos a afirmaciones de rendimiento medido.
