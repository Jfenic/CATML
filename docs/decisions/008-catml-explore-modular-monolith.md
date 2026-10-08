# ADR-008: CATML Explore como Módulo de Monolito Modular

> **Estado:** Aceptada  
> **Fecha:** 2026-10-08  
> **Alcance:** Arquitectura de plataforma y modelo de extensión  
> **Relacionado con:** [ADR-001 (Hexagonal/CQRS)](001-hexagonal-architecture-and-cqrs.md), [ADR-002 (Hipótesis)](002-hypothesis-driven-experiments.md), [ADR-006 (Brand System Tech Minimalista)](006-tech-minimalist-premium-brand-system.md), [Plan de Evolución Explore](../features/catml-explore/plan.md) y [Plan Maestro](../MASTER_PLAN.md).

---

## Contexto

CATML ha evolucionado exitosamente como una plataforma local-first de AutoML con arquitectura hexagonal, CQRS, agentes autónomos, persistencia SQLite y servidores MCP.

Para expandir las capacidades hacia el análisis de datos exploratorio (EDA), diagnóstico de calidad estadística, pruebas de hipótesis y descubrimiento científico guiado, se evaluaron dos caminos estratégicos:
1. Crear un producto o repositorio independiente (ej. `catml-explore` o un microservicio desacoplado).
2. Integrar CATML Explore como un nuevo contexto funcional delimitado (*bounded context*) dentro del repositorio existente de CATML estructurado como un **monolito modular**.

---

## Decisión

Adoptamos **CATML Explore como un contexto funcional dentro de un monolito modular**, manteniendo la separación hexagonal estricta:

1. **Compartición de Infraestructura:** Explore reutiliza los workspaces locales, el sistema de persistencia relacional SQLite (`automl.db`), el bus de comandos/consultas (`CommandBus`/`QueryBus`), el planificador de trabajos en segundo plano (`JobService`), el servidor MCP y el Workbench visual.
2. **Contextos Delimitados:**
   * `CATML Core`: Workspaces, datasets, políticas, presupuestos y orquestación.
   * `CATML Explore`: Estudios estadísticos, correlaciones, pruebas de hipótesis, anomalías y visualizaciones declarativas (`VisualizationSpec`).
   * `CATML AutoML`: Entrenamiento supervisado, optimización de hiperparámetros, ensamble y evaluación.
3. **Estructura de Directorios:**
   * `src/automl/domain/analysis/`: Entidades puras (`StudySpec`, `StatisticalFinding`, `AnalysisRun`, etc.) sin dependencias de pandas ni scipy.
   * `src/automl/application/analysis/`: Casos de uso CQRS y orquestación (`AnalysisStudyService`).
   * `src/automl/engine/analysis/`: Algoritmos numéricos y estadísticos encapsulados.
   * `src/automl/infrastructure/database/sqlite_studies.py`: Almacenamiento aditivo de estudios y hallazgos.
   * Interfaces: Paridad entre `catml explore` (CLI), `interfaces/mcp/analysis_tools.py` (MCP) y `views/explore.js` (Workbench).
4. **"La IA Interpreta, CATML Calcula":** Los cálculos y p-values son generados exclusivamente por el motor matemático determinista; los agentes LLM externos sólo consumen hallazgos estructurados para formular hipótesis.

---

## Alternativas Consideradas

* **Separar Explore en un microservicio o repositorio diferente:**
  * *Rechazada.* Implicaría duplicar persistencia, sincronización de esquemas, sistemas de autenticación y buses de mensajería, aumentando drásticamente la fricción operativa y dificultando la instalación sencilla mediante `pip install catml`.
* **Mezclar análisis exploratorio dentro de `AutoMLWorkspace`:**
  * *Rechazada.* `AutoMLWorkspace` ya era un God Object que recientemente descompusimos en servicios de aplicación especializados. Incorporar estadística descriptiva en el mismo servicio degradaría la mantenibilidad.

---

## Consecuencias

### Positivas
* **Experiencia de usuario unificada:** El usuario y los agentes gestionan datasets, estudios y modelos bajo un único comando `catml` y un único servidor Workbench.
* **Cierre del ciclo científico:** Un hallazgo estadístico (`StatisticalFinding`) puede vincularse directamente a una hipótesis (`AnalysisHypothesis`) y a un experimento en AutoML mediante `EvidenceLink` bajo el principio *"Propose ≠ Accept"*.
* **Cero sobrecoste de red:** Procesamiento 100% local sin latencia de microservicios ni puertos adicionales.

### Desafíos y Mitigaciones
* **Tamaño del paquete:** Las dependencias estadísticas avanzadas (`statsmodels`) se segregan bajo el extra opcional `catml[explore]`, mientras que `scipy` ya forma parte de las dependencias base tabulares.
* **Gobierno de contexto en LLMs:** Las herramientas MCP aplican filtros y paginación para no desbordar el context window con análisis masivos.
