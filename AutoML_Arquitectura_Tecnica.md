# AutoML — Arquitectura Técnica

> Especificación técnica del proyecto AutoML modular, extensible, multimodal y preparado para integración futura con agentes LLM.

AutoML Platform

Documento técnico de arquitectura y roadmap

Motor extensible de experimentación automática de Machine Learning con control humano, arquitectura preparada para multimodalidad y futura orquestación mediante agentes LLM.

| Estado | Diseño inicial / arquitectura objetivo |
| --- | --- |
| Versión del documento | 0.1 |
| Alcance inicial | Datos tabulares · clasificación y regresión |
| Evolución prevista | Imágenes · multimodalidad · knowledge layer · agente LLM |

# 1. Resumen ejecutivo

El proyecto consiste en construir una plataforma AutoML modular cuyo objetivo no sea únicamente encontrar el modelo con mejor métrica, sino gestionar de forma reproducible la búsqueda de una estrategia completa de datos, features, pipelines, modelos y experimentos bajo restricciones de tiempo y cómputo.

La primera versión estará orientada a un usuario humano. El usuario podrá añadir o quitar modelos, incluir o excluir columnas, marcar features como prioritarias, definir experimentos concretos, comparar modelos y reanudar ejecuciones. Posteriormente, un agente LLM podrá operar exactamente las mismas capacidades mediante una capa de comandos y consultas, sin acceder directamente a las librerías de ML.

| Principio arquitectónico principal. La interfaz humana, la API, el CLI y el futuro agente LLM deben consumir las mismas capacidades de aplicación. El dominio AutoML no debe depender de FastAPI, Optuna, XGBoost, PyTorch ni de un proveedor LLM concreto. |
| --- |

# 2. Objetivos del proyecto

- Construir un motor AutoML reproducible, extensible y controlable por el usuario.
- Permitir experimentación manual y automática sobre modelos, hiperparámetros, columnas, grupos de features, combinaciones y pipelines.
- Priorizar experimentos mediante reglas automáticas y prioridades explícitas del usuario.
- Registrar el historial de experimentos, resultados, decisiones y artefactos para permitir auditoría, comparación y reanudación.
- Facilitar la incorporación de nuevos modelos, métricas, optimizadores y modalidades mediante plugins y contratos estables.
- Preparar desde el inicio la evolución hacia imágenes, texto, series temporales y escenarios multimodales.
- Incorporar más adelante un agente LLM como capa de razonamiento y planificación, manteniendo la ejecución numérica y la validación dentro del motor determinista.
## 2.1 No objetivos de la primera versión

- No construir desde el inicio entrenamiento distribuido sobre Kubernetes.
- No soportar todas las modalidades en V0.1; la primera implementación será tabular.
- No introducir un LLM antes de disponer de un motor AutoML utilizable sin IA generativa.
- No crear un framework de deep learning propio; los modelos externos se integrarán mediante adaptadores/plugins.
- No optimizar todas las combinaciones posibles de features por fuerza bruta; se utilizarán estrategias de priorización y búsqueda jerárquica.
# 3. Ejemplo de uso objetivo

El usuario dispone de un dataset de clientes y quiere predecir churn. Desea excluir identificadores, priorizar variables financieras, evitar Random Forest, incluir CatBoost y ejecutar un experimento específico antes de dejar actuar al planificador automático.

```text
from automl import AutoMLWorkspace

workspace = AutoMLWorkspace.create("churn-demo")
run = workspace.create_run(dataset="customers.parquet", target="churn")

run.models.exclude("random_forest")
run.models.include("catboost")

run.features.exclude("customer_id")
run.features.prioritize("salary")
run.features.prioritize("debt")

experiment = run.experiments.create(
    name="financial_interaction",
    features=["salary", "debt", "account_balance"],
    models=["lightgbm", "catboost"],
    metric="roc_auc",
    priority="high"
)

run.experiments.execute(experiment.id)
run.automl.continue_run()
print(run.leaderboard())
```

En una fase posterior, el mismo flujo podrá iniciarse mediante lenguaje natural. El LLM traducirá una intención del usuario a comandos como PrioritizeFeature, ExcludeModel, CreateExperiment o RunExperiment. El agente no invocará directamente model.fit().

# 4. Principios de diseño

| Principio | Aplicación al proyecto |
| --- | --- |
| Hexagonal / Ports & Adapters | El dominio expone contratos; infraestructura y librerías concretas los implementan. |
| Command / Query separation | Las modificaciones se realizan mediante comandos y la lectura mediante consultas. |
| Experiment first | Experiment es una entidad de primer nivel; Trial representa una ejecución concreta del experimento. |
| Declarative configuration | Las decisiones de un Run se almacenan como configuración y no como ramas hardcodeadas. |
| Plugin architecture | Modelos, métricas, optimizadores, preprocessors y modalidades son extensibles. |
| Event driven | Los cambios relevantes generan eventos internos que pueden consumir logging, UI o futuros agentes. |
| Reproducibility | Dataset, configuración, versión de plugin, seeds, trials, métricas y artefactos quedan versionados. |
| Human + Agent parity | Un usuario y un agente LLM utilizan los mismos comandos/queries. |

# 5. Arquitectura de alto nivel

INTERFACES
             +------------+------------+
             |            |            |
            Web          CLI          REST
             |            |            |
             +------------+------------+
                          |
                    APPLICATION
                 Commands / Queries
                          |
                        DOMAIN
      +-------------------+-------------------+
      |                   |                   |
   Features           Experiments           Models
      |                   |                   |
      +-------------------+-------------------+
                          |
                        ENGINE
      +-------------------+-------------------+
      |                   |                   |
   Planner            Priority            Optimizer
      |                   |                   |
      +-------------------+-------------------+
                          |
                       PLUGINS
     models / metrics / preprocessing / modalities
                          |
                INFRASTRUCTURE + COMPUTE

FUTURO: LLM Agent -> Tools -> Commands / Queries

## 5.1 Regla de dependencias

El dominio debe permanecer como Python prácticamente puro. No debe importar frameworks de infraestructura o librerías concretas de ML. Las dependencias deben apuntar hacia el dominio y los contratos, no al revés.

interfaces  ---> application ---> domain
                          ^
                          |
infrastructure -----------+
plugins ------------------+

agents (futuro) ---> application commands / queries

# 6. Modelo de dominio base

| Entidad | Responsabilidad |
| --- | --- |
| Workspace | Contenedor lógico de datasets, ejecuciones y artefactos relacionados. |
| AutoMLRun | Representa una ejecución completa, su estado, presupuesto, configuración y fase actual. |
| RunConfig | Configuración declarativa de target, métricas, modelos, features, presupuesto y políticas. |
| Dataset | Referencia lógica a las fuentes de datos y al target. |
| DatasetProfile | Resumen estructurado del dataset que consume el motor y, posteriormente, el LLM. |
| Feature | Unidad de información con modalidad, origen, tipo semántico, estado y prioridad. |
| FeatureSet | Conjunto versionado de features utilizado por uno o más experimentos. |
| ModelSpec | Definición lógica de un modelo disponible y sus capacidades. |
| Experiment | Hipótesis o pregunta que se desea comprobar. |
| Trial | Ejecución concreta: pipeline, modelo, parámetros, dataset split y seed. |
| TrialResult | Resultado normalizado de un Trial: métricas, coste, tiempos y artefactos. |
| PipelineGraph | Representación de un pipeline como grafo para facilitar multimodalidad futura. |

## 6.1 Estados principales

AutoMLRun:
CREATED -> PROFILING -> PLANNING -> EXPERIMENTING
        -> OPTIMIZING -> FINALIZING -> COMPLETED
                         |              |
                       PAUSED         FAILED

Experiment:
CREATED -> QUEUED -> RUNNING -> EVALUATING -> COMPLETED
                    |                       |
                  PAUSED                 FAILED/CANCELLED

# 7. Organización recomendada del repositorio

automl-platform/
|-- src/automl/
|   |-- domain/
|   |   |-- runs/
|   |   |-- datasets/
|   |   |-- features/
|   |   |-- models/
|   |   |-- experiments/
|   |   `-- pipelines/
|   |-- application/
|   |   |-- commands/
|   |   |-- queries/
|   |   |-- services/
|   |   `-- events/
|   |-- engine/
|   |   |-- profiling/
|   |   |-- planning/
|   |   |-- priority/
|   |   |-- optimization/
|   |   |-- training/
|   |   |-- evaluation/
|   |   `-- ensemble/
|   |-- plugins/
|   |   |-- models/
|   |   |-- metrics/
|   |   |-- preprocessing/
|   |   |-- optimizers/
|   |   `-- modalities/
|   |-- infrastructure/
|   |   |-- database/
|   |   |-- storage/
|   |   |-- workers/
|   |   `-- event_bus/
|   |-- interfaces/
|   |   |-- api/
|   |   |-- cli/
|   |   `-- web/
|   `-- agents/              # se activa en fases futuras
|       |-- tools/
|       |-- planner/
|       `-- orchestrator/
|-- tests/
|-- examples/
|-- pyproject.toml
`-- README.md

# 8. Roadmap por fases

| Fase | Objetivo | Resultado utilizable |
| --- | --- | --- |
| V0.1 | Dominio y ejecución manual básica | Run, dataset, modelos, features, Experiment/Trial y almacenamiento local. |
| V0.2 | Control humano de experimentos | Añadir/quitar modelos, prioridades, experimentos directos, pausa/reanudación. |
| V0.3 | Planificación y priorización automática | Planner rule-based, queue y scoring de experimentos/features. |
| V0.4 | Optimización automática | Optuna, budgets, early stopping y leaderboard reproducible. |
| V0.5 | Experimentación avanzada de features | Ablation, interacciones, grupos, búsqueda jerárquica y knowledge inicial. |
| V0.6 | Sistema de plugins estable | Modelos/métricas/optimizers/preprocessors instalables sin tocar el core. |
| V0.7 | Imágenes y multimodalidad inicial | Image modality, embeddings, PipelineGraph y experimentos tabular+image. |
| V0.8 | Knowledge y meta-learning | Historial reusable, similitud de datasets y warm-start del planner. |
| V0.9 | LLM tools | El LLM consulta y ejecuta Commands/Queries con permisos controlados. |
| V1.0 | Agente LLM híbrido | Planner/Critic/Orchestrator que propone hipótesis y colabora con el AutoML. |

# V0.1 — Dominio estable y ejecución manual mínima

Crear el núcleo que permita ejecutar y registrar experimentos tabulares sin depender de una UI ni de un LLM. El objetivo es validar el modelo de dominio y los contratos antes de añadir automatización.

## Alcance

- Clasificación binaria/multiclase y regresión tabular.
- Registro de dataset y target.
- Registro básico de features y modelos.
- Creación manual de Experiment y generación de Trial.
- Ejecución síncrona local y evaluación mediante holdout o cross-validation.
- Persistencia inicial en SQLite y artefactos en filesystem local.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| AutoMLRun | Estado y ciclo de vida de una ejecución. | Domain |
| RunConfig | Configuración declarativa del run. | Domain |
| Dataset | Referencia al dataset y target. | Domain |
| DatasetProfile | Perfil estructurado de columnas y target. | Domain |
| Feature | Metadata y estado lógico de una feature. | Domain |
| FeatureRegistry | Alta, consulta y estados de features. | Domain |
| ModelSpec | Identidad y capacidades de un modelo. | Domain |
| ModelRegistry | Registro de modelos disponibles. | Domain |
| Experiment | Definición de una pregunta experimental. | Domain |
| Trial | Ejecución concreta de un experimento. | Domain |
| TrialResult | Resultado normalizado y métricas. | Domain |
| ExperimentRepository | Contrato de persistencia. | Port |
| TrainerPort | Contrato de ejecución de trials. | Port |
| EvaluatorPort | Contrato de evaluación. | Port |

## Flujo principal

CreateRun
   -> RegisterDataset
   -> ProfileDataset
   -> RegisterFeatures / RegisterModels
   -> CreateExperiment
   -> CreateTrial
   -> TrainerPort.run(trial)
   -> EvaluatorPort.evaluate(...)
   -> TrialResult
   -> ExperimentRepository.save(...)

## Criterios de finalización

- Un dataset tabular puede registrarse y perfilarse.
- Un usuario puede crear manualmente un experimento y ejecutarlo.
- Los resultados quedan persistidos y son reproducibles mediante seed/configuración.
- El core funciona sin FastAPI, Optuna ni LLM.
## Preparación para fases futuras

Los contratos TrainerPort, EvaluatorPort y Repository permiten cambiar las implementaciones sin modificar el dominio.

# V0.2 — Control humano y experimentos directos

Convertir el núcleo en una herramienta realmente controlable por el usuario. Esta fase implementa la paridad conceptual que utilizará después el agente LLM.

## Alcance

- Añadir y quitar modelos de un Run sin modificar código del core.
- Incluir, excluir o priorizar features y feature sets.
- Crear experimentos manuales sobre modelos y columnas concretos.
- Definir prioridad manual: low, normal, high y pinned.
- Pausar, reanudar, cancelar y clonar ejecuciones.
- Exponer comandos y consultas primero por CLI/API mínima.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| CommandBus | Ejecuta comandos de aplicación. | Application |
| QueryBus | Ejecuta consultas de lectura. | Application |
| AddModelCommand | Añade un modelo al Run. | Command |
| ExcludeModelCommand | Excluye un modelo del Run. | Command |
| PrioritizeFeatureCommand | Asigna prioridad explícita a una feature. | Command |
| ExcludeFeatureCommand | Excluye lógicamente una feature. | Command |
| CreateFeatureSetCommand | Crea un conjunto de features versionado. | Command |
| CreateExperimentCommand | Define un experimento directo. | Command |
| RunExperimentCommand | Lanza un experimento seleccionado. | Command |
| PauseRunCommand | Pausa la ejecución. | Command |
| ResumeRunCommand | Reanuda desde estado persistido. | Command |
| ListModelsQuery | Lista modelos activos/disponibles. | Query |
| GetDatasetProfileQuery | Obtiene el perfil para UI o CLI. | Query |
| GetLeaderboardQuery | Devuelve resultados ordenados. | Query |
| CompareExperimentsQuery | Compara experimentos/trials. | Query |

## Flujo principal

Human / CLI / API
       |
       v
Command / Query
       |
       v
Application Service
       |
       v
Domain + Engine
       |
       v
Events + Persistence

## Criterios de finalización

- El usuario puede controlar modelos y columnas sin editar archivos de configuración internos.
- Se puede lanzar un experimento directo sin ejecutar el AutoML completo.
- La pausa y reanudación conserva historial y estado.
- Todos los cambios relevantes generan un evento y quedan auditables.
## Preparación para fases futuras

Los futuros LLM Tools invocarán exactamente estos mismos comandos y queries.

# V0.3 — Planificador automático y Priority Engine

Añadir automatización controlada: generar experimentos candidatos y decidir qué ejecutar primero teniendo en cuenta evidencia, coste computacional y prioridades del usuario.

## Alcance

- Planificador rule-based basado en DatasetProfile, FeatureRegistry y ModelRegistry.
- Generación de experimentos de baseline, modelos alternativos y subsets de features.
- Priority Engine con system_score, user_score y effective_score.
- ExperimentQueue con soporte para pinned y cancelación.
- Budget awareness básico para evitar experimentos caros de baja prioridad.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| ExperimentPlannerPort | Contrato para proponer experimentos. | Port |
| RuleBasedExperimentPlanner | Planner inicial determinista. | Engine |
| ExperimentCandidate | Propuesta antes de persistir/ejecutar. | Domain |
| Priority | Valor y origen de prioridades. | Domain |
| PriorityScorerPort | Contrato de scoring. | Port |
| RuleBasedPriorityScorer | Combina gain, uncertainty, cost y user boost. | Engine |
| ExperimentQueue | Cola ordenada por prioridad efectiva. | Engine |
| BudgetPolicy | Decide elegibilidad según presupuesto. | Domain |
| Scheduler | Selecciona siguiente experimento ejecutable. | Engine |

## Flujo principal

DatasetProfile + RunConfig + History
              |
              v
      ExperimentPlanner
              |
              v
     ExperimentCandidates
              |
              v
       PriorityScorer
              |
              v
       ExperimentQueue
              |
              v
          Scheduler
              |
              v
         Execution

## Criterios de finalización

- El sistema genera un conjunto razonable de experimentos sin intervención manual.
- Una prioridad manual puede alterar de forma explícita el orden automático.
- Los experimentos pinned se ejecutan antes que los sugeridos automáticamente.
- El score de prioridad conserva componentes explicables, no solo un número final.
## Preparación para fases futuras

Más adelante, LLMExperimentPlanner implementará el mismo ExperimentPlannerPort y podrá convivir con el planner rule-based.

# V0.4 — Optimización de modelos e hiperparámetros

Incorporar búsqueda automática eficiente dentro de un experimento manteniendo separada la decisión estratégica de qué experimentar y la optimización numérica de un modelo.

## Alcance

- Random Search como referencia y Optuna como optimizer principal.
- SearchSpace declarativo por modelo/plugin.
- Budgets de tiempo, número de trials y recursos.
- Early stopping y pruning cuando el modelo lo soporte.
- Leaderboard por experimento y run.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| OptimizerPort | suggest/observe/should_stop sin acoplar a Optuna. | Port |
| RandomSearchOptimizer | Implementación simple de referencia. | Engine |
| OptunaOptimizer | Adaptador a Optuna. | Plugin |
| SearchSpace | Espacio declarativo y restricciones. | Domain |
| ParameterSpec | Int/float/categorical/log-scale. | Domain |
| SearchSpaceBuilder | Construye espacio según modelo/contexto. | Engine |
| TrialFactory | Materializa un Trial a partir de una suggestion. | Engine |
| EarlyStoppingPolicy | Criterios para detener un trial. | Engine |
| LeaderboardService | Normaliza y ordena resultados. | Application |

## Flujo principal

Experiment
   -> SearchSpaceBuilder
   -> Optimizer.suggest()
   -> TrialFactory
   -> Trainer
   -> Evaluator
   -> TrialResult
   -> Optimizer.observe()
   -> repeat until budget exhausted

## Criterios de finalización

- Se pueden optimizar al menos LightGBM/XGBoost/CatBoost y modelos sklearn seleccionados.
- El engine no conoce Optuna directamente; solo OptimizerPort.
- Cada trial almacena parámetros, seed, duración y métricas.
- La ejecución finaliza correctamente al agotar presupuesto o criterio de stop.
## Preparación para fases futuras

Permite reemplazar Optuna por BayesianOptimizer, Hyperband o un optimizer propio sin reestructurar el proyecto.

# V0.5 — Experimentación avanzada de features

Hacer que el AutoML aprenda qué columnas, grupos y combinaciones aportan valor, evitando una explosión combinatoria mediante priorización y búsqueda jerárquica.

## Alcance

- Feature ablation y feature addition.
- FeatureGroup y FeatureSet versionados.
- Pairwise interactions priorizadas.
- Búsqueda jerárquica/beam search sobre combinaciones prometedoras.
- Importance mediante permutation importance, SHAP u otras técnicas disponibles.
- Feature history para reutilizar evidencia dentro del mismo run/workspace.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| FeatureGroup | Agrupa features por semántica/origen. | Domain |
| FeatureSet | Conjunto versionado de features. | Domain |
| FeatureExperimentFactory | Genera ablation/addition/interaction experiments. | Engine |
| FeatureSelectorPort | Contrato de selección/pruning. | Port |
| BeamFeatureSelector | Explora combinaciones por niveles. | Engine |
| FeatureImportancePort | Contrato para calcular importancia. | Port |
| FeatureInteractionScorer | Score de posibles interacciones. | Engine |
| FeatureKnowledge | Historial agregado por feature/interacción. | Domain |
| FeatureKnowledgeRepository | Persistencia de conocimiento. | Port |

## Flujo principal

Baseline
   -> individual / group tests
   -> rank features
   -> select top-K
   -> pairwise candidate generation
   -> priority scoring
   -> execute promising combinations
   -> update FeatureKnowledge
   -> next level if budget allows

## Criterios de finalización

- El sistema puede responder si una feature o grupo mejora significativamente el baseline.
- No genera todas las combinaciones posibles; aplica top-K/beam/budget.
- El historial identifica combinaciones útiles y no útiles.
- El usuario puede forzar una combinación aunque el score automático sea bajo.
## Preparación para fases futuras

El futuro LLM podrá generar hipótesis semánticas de interacción, mientras que esta fase seguirá validándolas experimentalmente.

# V0.6 — Arquitectura de plugins estable

Formalizar la extensibilidad para que añadir un modelo, métrica, transformador u optimizer no requiera modificar el core ni introducir condicionales por implementación.

## Alcance

- Plugin registry y capability discovery.
- ModelPlugin, MetricPlugin, OptimizerPlugin y PreprocessorPlugin.
- Validación de compatibilidad con TaskType y Modality.
- Versionado de plugin dentro de cada Trial para reproducibilidad.
- Carga por configuración/entry points de Python en una iteración posterior.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| Plugin | Contrato común de identidad, versión y capabilities. | Port |
| PluginRegistry | Registro y descubrimiento. | Application |
| ModelPlugin | Construcción de modelo y search space. | Plugin API |
| MetricPlugin | Cálculo y orientación min/max. | Plugin API |
| OptimizerPlugin | Implementación de OptimizerPort. | Plugin API |
| PreprocessorPlugin | Transformación reusable dentro de pipelines. | Plugin API |
| Capability | Task/modalities/resources compatibles. | Domain |
| CompatibilityValidator | Impide combinaciones inválidas. | Engine |

## Flujo principal

Plugin installed
     -> PluginRegistry.register()
     -> capabilities indexed
     -> Planner/SearchSpace query registry
     -> CompatibilityValidator
     -> Experiment/Trial uses plugin reference

## Criterios de finalización

- Añadir un modelo nuevo no exige editar Planner, Trainer ni Domain.
- Un plugin incompatible se rechaza antes de ejecutar el trial.
- La versión del plugin queda registrada en artefactos/resultados.
## Preparación para fases futuras

Esta capa es la base para introducir posteriormente plugins de imagen, texto, audio o modelos propios del usuario.

# V0.7 — Imágenes y multimodalidad inicial

Generalizar Dataset/Feature/Pipeline para trabajar con modalidades diferentes a columnas tabulares sin romper los experimentos existentes.

## Alcance

- Introducir Modality y DataSource como conceptos del dominio.
- ImageFeature e ImageModalityPlugin.
- Pipelines de embeddings con encoders preentrenados como primer caso de visión.
- Feature fusion entre tabular e image embeddings.
- PipelineGraph como DAG de nodos en lugar de una secuencia rígida.
- Experimentos tabular-only, image-only y tabular+image.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| Modality | TABULAR, IMAGE, TEXT, AUDIO, TIMESERIES. | Domain |
| DataSource | Referencia abstracta a una fuente de datos. | Domain |
| ModalityPlugin | Contrato para análisis y construcción de pipelines. | Plugin API |
| ImageModalityPlugin | Implementación inicial para imágenes. | Plugin |
| PipelineNode | Nodo tipado de procesamiento/modelo. | Domain |
| PipelineGraph | DAG versionado de nodos y conexiones. | Domain |
| ImageEncoderNode | Extrae embeddings de imágenes. | Plugin |
| FeatureFusionNode | Fusiona representaciones compatibles. | Engine/Plugin |
| GraphValidator | Valida tipos y compatibilidad del DAG. | Engine |

## Flujo principal

TabularSource -> tabular features ---------+
                                             FeatureFusion -> Model
ImageSource -> resize -> encoder -> embedding ----+

Experiment A: tabular only
Experiment B: image only
Experiment C: tabular + image embedding

## Criterios de finalización

- Los runs tabulares existentes siguen funcionando sin cambios de API.
- Una imagen puede convertirse en embedding mediante plugin.
- El sistema puede comparar el valor incremental de la modalidad imagen.
- PipelineGraph detecta conexiones inválidas antes de entrenar.
## Preparación para fases futuras

La misma abstracción permitirá incorporar TextModalityPlugin, AudioModalityPlugin y TimeSeriesModalityPlugin.

# V0.8 — Knowledge layer y meta-learning

Transformar el historial de ejecuciones en conocimiento reutilizable para priorizar modelos, features y experimentos en datasets nuevos o similares.

## Alcance

- Persistir meta-features de datasets.
- Resumir qué modelos/pipelines/features funcionaron y a qué coste.
- Dataset similarity y recuperación de experiencias relevantes.
- Warm-start del Experiment Planner y Optimizer.
- Separar hechos observados de recomendaciones derivadas.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| DatasetMetaFeatures | Descripción compacta de un dataset. | Domain |
| ExperimentKnowledge | Resumen reusable de evidencia. | Domain |
| KnowledgeRepository | Persistencia/consulta de conocimiento. | Port |
| DatasetSimilarityPort | Contrato de similitud. | Port |
| MetaLearningService | Recupera experiencias y genera priors. | Engine |
| WarmStartPolicy | Convierte conocimiento en prioridades/suggestions. | Engine |

## Flujo principal

New Dataset
    -> DatasetMetaFeatures
    -> similarity search
    -> relevant ExperimentKnowledge
    -> priors for models/features/experiments
    -> Planner + Priority Engine
    -> new observed evidence
    -> Knowledge update

## Criterios de finalización

- El sistema puede explicar de qué runs históricos procede un prior.
- Los priors no sustituyen la evaluación del dataset actual.
- Puede hacerse warm-start sin LLM.
## Preparación para fases futuras

Será la principal fuente de contexto estructurado para el futuro agente LLM, evitando enviar datasets completos al modelo.

# V0.9 — Tools para LLM

Exponer las capacidades existentes como herramientas seguras y estructuradas para un LLM, manteniendo permisos, validaciones y trazabilidad.

## Alcance

- Tools de solo lectura para perfiles, leaderboard, historial y estado.
- Tools mutables que envuelven Commands existentes.
- Schemas estructurados de entrada/salida.
- Autorización de operaciones y límites de presupuesto.
- Audit trail que distingue cambios de usuario, sistema y agente.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| AgentTool | Contrato de herramienta invocable. | Agent API |
| ToolRegistry | Lista tools permitidas y sus schemas. | Agent |
| GetDatasetProfileTool | Envuelve GetDatasetProfileQuery. | Tool |
| ListExperimentsTool | Consulta historial/estado. | Tool |
| CreateExperimentTool | Envuelve CreateExperimentCommand. | Tool |
| PrioritizeFeatureTool | Envuelve PrioritizeFeatureCommand. | Tool |
| RunExperimentTool | Envuelve RunExperimentCommand. | Tool |
| AgentPolicy | Permisos, budgets y acciones que requieren aprobación. | Domain/App |
| AgentActionLog | Audita intención, comando y resultado. | Domain |

## Flujo principal

LLM
  -> Tool call
  -> schema validation
  -> AgentPolicy
  -> Command / Query
  -> Application
  -> Domain / Engine
  -> structured result
  -> LLM

## Criterios de finalización

- El LLM no importa ni invoca directamente librerías ML.
- Toda acción mutante pasa por CommandBus y políticas.
- El mismo comando funciona desde UI, CLI y LLM.
- Las tools pueden probarse sin conectar un proveedor LLM concreto.
## Preparación para fases futuras

La capa de tools desacopla el motor del proveedor de modelos generativos y permite cambiar de LLM sin afectar el AutoML.

# V1.0 — Agente LLM híbrido de Machine Learning

Añadir razonamiento semántico y planificación adaptativa sobre un AutoML ya estable. El agente propone hipótesis y estrategias; el motor determinista ejecuta y valida.

## Alcance

- Planner Agent para interpretar objetivo y proponer experimentos.
- Feature Advisor para detectar semántica, leakage y posibles interacciones.
- Critic Agent para revisar resultados y proponer la siguiente hipótesis.
- Orchestrator para coordinar herramientas y límites.
- Human-in-the-loop para acciones de alto coste o riesgo.
- Modo híbrido: planner rule-based + LLM suggestions + Priority Engine.
## Clases e interfaces principales

| Clase / interfaz | Responsabilidad | Tipo |
| --- | --- | --- |
| AgentContextBuilder | Construye contexto compacto desde profile/history/knowledge. | Agent |
| ExperimentPlanningAgent | Propone Hypothesis/ExperimentCandidate. | Agent |
| FeatureAdvisorAgent | Propone exclusiones, semántica e interacciones. | Agent |
| ExperimentCriticAgent | Analiza resultados y propone iteraciones. | Agent |
| AgentOrchestrator | Gestiona ciclo observe-plan-act. | Agent |
| HybridExperimentPlanner | Fusiona reglas, knowledge y propuestas LLM. | Engine/Agent |
| ApprovalPolicy | Decide cuándo solicitar confirmación humana. | Domain/App |

## Flujo principal

User goal
   -> AgentContextBuilder
   -> Planning Agent
   -> structured hypotheses
   -> HybridExperimentPlanner
   -> Priority Engine
   -> deterministic AutoML execution
   -> results + knowledge
   -> Critic Agent
   -> next hypothesis / stop

## Criterios de finalización

- El AutoML sigue siendo completamente utilizable con el LLM deshabilitado.
- Las propuestas del agente son objetos estructurados, no código ejecutado arbitrariamente.
- Toda recomendación se valida mediante experimentos y métricas.
- Las acciones importantes son trazables y reproducibles.
## Preparación para fases futuras

A partir de esta versión el producto puede evolucionar hacia un “AI ML Engineer”, sin haber convertido el LLM en una dependencia estructural del motor.

# 9. Interfaces y clases transversales críticas

Las siguientes interfaces deben diseñarse temprano porque determinan si el sistema podrá crecer sin reestructuraciones importantes.

| Interfaz | Métodos conceptuales | Motivo |
| --- | --- | --- |
| TrainerPort | run(trial) -> TrialExecution | Desacopla ejecución local, workers y futuros backends distribuidos. |
| EvaluatorPort | evaluate(execution, strategy) -> TrialResult | Permite cambiar validación/métricas sin tocar modelos. |
| OptimizerPort | suggest(), observe(), should_stop() | Separa optimización numérica de planificación. |
| ExperimentPlannerPort | propose(context) -> candidates | Permite planner rule-based, meta-learning o LLM. |
| PriorityScorerPort | score(candidate, context) | Hace intercambiable la estrategia de priorización. |
| ModelPlugin | capabilities(), search_space(), build() | Permite añadir modelos sin ramas hardcodeadas. |
| ModalityPlugin | profile(), build_nodes(), capabilities() | Base de multimodalidad. |
| Repository ports | save/get/list/version | Permite SQLite/Postgres/object storage sin contaminar dominio. |
| EventBusPort | publish(event) | Aísla event bus in-memory, Redis, Kafka u otros. |

# 10. Modelo conceptual de prioridades

El sistema debe combinar evidencia automática y preferencias explícitas del usuario. La prioridad efectiva debe ser explicable y auditable.

```text
effective_priority =
    w_gain        * expected_gain
  + w_info        * expected_information_gain
  + w_uncertainty * uncertainty
  + w_novelty     * novelty
  - w_cost        * normalized_cost
  + user_boost

PINNED -> bypass normal ordering (subject to safety/budget policies)
```

| Componente | Ejemplo de fuente |
| --- | --- |
| expected_gain | Resultados de baselines, importances, modelos similares. |
| information_gain | Cuánto reduce incertidumbre una comparación/ablation. |
| uncertainty | Poca evidencia histórica o resultados inconsistentes. |
| novelty | Combinación aún no probada o poco representada. |
| normalized_cost | Tiempo estimado, RAM, GPU y volumen de datos. |
| user_boost | Low/Normal/High/Pinned o score numérico explícito. |

# 11. Diseño de Experiment y Trial

Experiment
- id
- run_id
- name
- hypothesis
- type
- feature_set_id
- model_ids[]
- metric
- validation_strategy
- priority
- budget
- status
- created_by: user | system | agent

Trial
- id
- experiment_id
- pipeline_graph_id
- model_plugin + version
- parameters
- seed
- split_strategy
- resource_request
- status

TrialResult
- trial_id
- primary_metric
- secondary_metrics
- training_time
- inference_time
- memory_peak
- artifacts
- failure_reason

# 12. Diseño de Feature

Feature
- id
- dataset_id
- name
- modality
- semantic_type
- physical_dtype
- source
- status: ACTIVE | EXCLUDED | PRIORITY | EXPERIMENTAL | GENERATED
- user_priority
- system_priority
- metadata

FeatureSet
- id
- version
- feature_ids[]
- generated_feature_ids[]
- created_by
- lineage

No se debe modificar físicamente el dataset cada vez que se excluye o prioriza una columna. La selección debe ser lógica y versionada mediante Feature/FeatureSet.

# 13. Eventos recomendados

| Evento | Uso principal |
| --- | --- |
| RunCreated | Inicialización/auditoría. |
| DatasetRegistered | Disparar profiling o UI updates. |
| DatasetProfiled | Permitir planificación. |
| FeaturePrioritized / FeatureExcluded | Actualizar plan/cola y audit log. |
| ModelAdded / ModelExcluded | Recalcular compatibilidad/search space. |
| ExperimentCreated | Persistir y priorizar. |
| ExperimentQueued / Started / Completed | Estado y observabilidad. |
| TrialStarted / TrialCompleted / TrialFailed | Optimizer, leaderboard, métricas. |
| KnowledgeUpdated | Meta-learning y futuro contexto del agente. |
| AgentActionRequested / Executed | Trazabilidad de automatización generativa. |

# 14. Persistencia y reproducibilidad

- SQLite en V0.1-V0.3 para simplificar; PostgreSQL cuando concurrencia y multiusuario lo justifiquen.
- Filesystem local para artefactos al inicio; object storage mediante StoragePort en fases posteriores.
- Cada Trial debe registrar versiones de plugins/librerías relevantes, seed, parámetros, hash de dataset/config y pipeline.
- Los cambios sobre RunConfig deben versionarse o almacenarse como eventos para poder reconstruir el estado.
- Los experimentos y resultados no deben sobrescribirse silenciosamente; las nuevas ejecuciones deben generar nuevas versiones/ids.
# 15. Estrategia de testing

| Nivel | Qué probar |
| --- | --- |
| Unit | Domain invariants, scoring, estados, commands, graph validation. |
| Contract | Cada ModelPlugin/Optimizer/Repository cumple su Port. |
| Integration | Dataset -> Experiment -> Trial -> Result con plugins reales. |
| Regression | Datasets pequeños fijos para detectar degradaciones de comportamiento. |
| Reproducibility | Mismo seed/config produce resultados equivalentes dentro de tolerancia. |
| Agent safety (futuro) | Tools, permissions, budgets y validación estructurada. |

# 16. Orden recomendado de implementación

1. Definir entidades de dominio y estados: Run, Dataset, Feature, ModelSpec, Experiment y Trial.
1. Definir Ports antes de integrar librerías concretas.
1. Implementar un happy path manual con un único modelo sklearn y SQLite.
1. Añadir ModelRegistry, FeatureRegistry y Commands/Queries.
1. Integrar LightGBM/XGBoost/CatBoost mediante plugins/adapters.
1. Añadir ExperimentPlanner y Priority Engine rule-based.
1. Añadir Optuna detrás de OptimizerPort.
1. Construir feature experiments y knowledge local.
1. Estabilizar plugin API antes de multimodalidad.
1. Añadir PipelineGraph y ImageModalityPlugin.
1. Crear knowledge/meta-learning.
1. Exponer Commands/Queries como LLM Tools.
1. Añadir agentes solamente cuando el flujo determinista ya sea completo y observable.
# 17. Decisiones que deben evitarse

- Un AutoML.fit() monolítico con todas las fases hardcodeadas.
- Condicionales if/elif por nombre de modelo repartidos por el código.
- Usar pandas.DataFrame como definición universal de Dataset en todo el dominio.
- Hacer que el agente LLM genere y ejecute código arbitrario como mecanismo principal.
- Acoplar ExperimentPlanner a Optuna; son problemas distintos.
- Mezclar Experiment y Trial como una misma entidad.
- Perder la procedencia de quién realizó una modificación: user/system/agent.
- Guardar únicamente el “best model” y descartar el historial experimental.
- Añadir imágenes mediante excepciones especiales en el pipeline tabular en vez de una abstracción de modalidad.
# 18. Definición de éxito de V1.0

El proyecto habrá alcanzado su arquitectura objetivo inicial cuando un mismo Run pueda ser gestionado de manera equivalente por una interfaz humana o por un agente, pudiendo seleccionar modelos y features, crear y priorizar experimentos, ejecutar optimización, comparar resultados y utilizar conocimiento histórico, manteniendo trazabilidad y reproducibilidad.

| Regla de oro. El LLM propone; el motor AutoML valida. La semántica y estrategia pueden ser asistidas por un agente, mientras que el entrenamiento, la evaluación, las métricas, los budgets y el estado del sistema permanecen deterministas y verificables. |
| --- |

# 19. Próximo incremento recomendado

El primer incremento de implementación debería limitarse a V0.1 y parte de V0.2: crear el dominio, los Ports, un backend local, ModelRegistry/FeatureRegistry y comandos básicos. La primera demo debería permitir registrar un dataset, excluir/priorizar columnas, elegir dos modelos y ejecutar un experimento directo con leaderboard. Solo después conviene añadir el planner automático.

# Anexo A. Checklist resumido de clases por fase

| Fase | Clases principales |
| --- | --- |
| V0.1 | AutoMLRun, RunConfig, Dataset, DatasetProfile, Feature, FeatureRegistry, ModelSpec, ModelRegistry, Experiment, Trial, TrialResult, TrainerPort, EvaluatorPort, ExperimentRepository |
| V0.2 | CommandBus, QueryBus, Add/ExcludeModelCommand, Prioritize/ExcludeFeatureCommand, CreateFeatureSetCommand, Create/RunExperimentCommand, Pause/ResumeRunCommand, queries de profile/leaderboard/compare |
| V0.3 | ExperimentPlannerPort, RuleBasedExperimentPlanner, ExperimentCandidate, Priority, PriorityScorerPort, RuleBasedPriorityScorer, ExperimentQueue, BudgetPolicy, Scheduler |
| V0.4 | OptimizerPort, RandomSearchOptimizer, OptunaOptimizer, SearchSpace, ParameterSpec, SearchSpaceBuilder, TrialFactory, EarlyStoppingPolicy, LeaderboardService |
| V0.5 | FeatureGroup, FeatureExperimentFactory, FeatureSelectorPort, BeamFeatureSelector, FeatureImportancePort, FeatureInteractionScorer, FeatureKnowledge, FeatureKnowledgeRepository |
| V0.6 | Plugin, PluginRegistry, ModelPlugin, MetricPlugin, OptimizerPlugin, PreprocessorPlugin, Capability, CompatibilityValidator |
| V0.7 | Modality, DataSource, ModalityPlugin, ImageModalityPlugin, PipelineNode, PipelineGraph, ImageEncoderNode, FeatureFusionNode, GraphValidator |
| V0.8 | DatasetMetaFeatures, ExperimentKnowledge, KnowledgeRepository, DatasetSimilarityPort, MetaLearningService, WarmStartPolicy |
| V0.9 | AgentTool, ToolRegistry, GetDatasetProfileTool, ListExperimentsTool, CreateExperimentTool, PrioritizeFeatureTool, RunExperimentTool, AgentPolicy, AgentActionLog |
| V1.0 | AgentContextBuilder, ExperimentPlanningAgent, FeatureAdvisorAgent, ExperimentCriticAgent, AgentOrchestrator, HybridExperimentPlanner, ApprovalPolicy |


---

## Principio de evolución

La capa AutoML debe poder funcionar sin LLM. La interfaz humana, la API, el CLI y el futuro agente deben operar sobre los mismos `Commands`, `Queries`, servicios y contratos del dominio. Esto permite añadir nuevas modalidades, modelos, optimizadores y agentes sin reestructurar el núcleo.
