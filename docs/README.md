# Documentación de CATML

La versión del paquete se define en [`automl.__version__`](../src/automl/__init__.py). Las fases V0.1–V1.0 representan hitos del diseño. El estado operativo y las tareas pendientes se mantienen en [TASKS.md](../TASKS.md).

## Dónde empezar

| Necesidad | Documento |
| --- | --- |
| Instalar y ejecutar CLI o Workbench | [README](../README.md) |
| Entender capas y dependencias | [Arquitectura actual](../ARCHITECTURE.md) |
| Añadir modelos y casos de uso | [Guía de desarrollo](../DEVELOPER_GUIDE.md) |
| Contribuir y validar cambios | [Contribución](../CONTRIBUTING.md) y [reglas de agentes](../AGENTS.md) |
| Comprobar backend y reproducibilidad | [Backends](backends.md) |
| Consultar diseño objetivo y fases futuras | [Especificación técnica](../AutoML_Arquitectura_Tecnica.md) |
| Entender decisiones | [ADRs](decisions/) |
| Consultar última ejecución y contexto | [Progreso](../.agent/progress.md) |

## Capacidades y evidencia

Esta tabla describe el alcance comprobable; los criterios ampliados de una especificación pueden seguir pendientes.

| Capacidad | Alcance actual y limitaciones | Implementación / evidencia |
| --- | --- | --- |
| Experimentos tabulares y CQRS | Operativos; algunos handlers devuelven entidades o resultados, aunque el contrato objetivo pide IDs o `None` | [Bootstrap](../src/automl/application/bootstrap.py), [tests V0.2](../tests/test_v02.py) |
| Selección de features | Selectores, ranking, candidatos y ablación; propuestas requieren evaluación | [Spec V0.5](features/feature-discovery/spec.md), [tests](../tests/test_v05_features.py) |
| Plugins | Registro en memoria, compatibilidad y ejecución; re-registrar plugins personalizados en cada proceso; sin descubrimiento por entry points | [Spec V0.6](features/plugin-ecosystem/spec.md), [tests de registro](../tests/test_plugin_registration.py), [ejemplo](../examples/plugins/custom_model.py) |
| LightGBM / XGBoost | Backend nativo opcional, con alternativa sklearn por defecto | [Adaptadores](../src/automl/plugins/models/gradient_boosting.py), [backends](backends.md), [tests](../tests/test_v06_plugins.py) |
| Ensemble | Voting y blending ponderado disponibles; stacking con meta-estimador pendiente | [Plugin](../src/automl/plugins/models/ensemble.py), [tests](../tests/test_ensemble_plugin.py) |
| Interacciones | Generador de ratios, productos y target encoding OOF; su existencia no implica incorporación automática al planner o CLI | [Generador](../src/automl/engine/features/generation/interaction_generator.py), [tests](../tests/test_feature_interactions.py) |
| Multimodal | DAG, imágenes y fusión tabular; encoder determinista predeterminado; texto/audio y late fusion no acreditados como implementación completa | [Spec V0.7](features/multimodal/spec.md), [tests E2E](../tests/test_v07_multimodal_e2e.py) |
| Workbench HTTP | Servicio local, ejecución y consulta de experimentos, datasets y submissions; sin autenticación | [Servidor](../src/automl/interfaces/web/server.py), [tests](../tests/test_web_dashboard.py) |
| Knowledge / agente | Vistas y respuestas ilustrativas de preview; no equivalen a meta-learning o agente autónomo operativo | [Servidor](../src/automl/interfaces/web/server.py), [backlog](../TASKS.md) |
| Predicciones / submissions | Predicción con plantilla disponible; blending OOF y opción de predicción `--folds` pendientes | [Tests de plantilla](../tests/test_submission_template.py), [protocolo OOF](features/oof-blending/spec.md) |

## Mantener la documentación

Usar enlaces relativos, ejemplos ejecutables y resultados de validación fechados. Mantener el backlog en `TASKS.md`, los contratos y criterios en las specs, y la secuencia histórica en sus planes. Los ADRs registran decisiones aceptadas y sus consecuencias; las modificaciones sustanciales deben indicar si reemplazan otra decisión. No trasladar resultados ilustrativos a afirmaciones de rendimiento medido.
