# Decision: Neo-Industrial Visual ML Lab UI Design System

Estado: aceptada. Revisada: 2026-09-30. Alcance: interfaz visual del Workbench; el estándar de diseño completo se detalla en [docs/design/neo-industrial-ui-spec.md](../design/neo-industrial-ui-spec.md).

## Context

El Workbench web de CATML permite a desarrolladores, científicos de datos y agentes inspeccionar datasets, construir pipelines, orquestar experimentos y validar modelos. 

Las interfaces web típicas de SaaS de ML tienden a usar componentes genéricos ("tarjetas redondeadas flotantes", gradientes azulados/púrpuras difusos, spinners opacos y badges infantiles de trofeos) que transmiten la sensación de una "caja negra" poco rigurosa y desconectada del control técnico del ingeniero.

CATML requiere:
1. Una estética que transmita **sistema, precisión, ingeniería y control**, actuando como un auténtico **laboratorio visual de machine learning**.
2. Jerarquía visual de alta densidad con tipografía diferenciada (sans-serif arquitectónico para estructura + monospace para métricas, parámetros y logs).
3. Estandarización de componentes modulares con bordes visibles, radio de esquina mínimo (`0px`-`4px`) y códigos de color funcionales inmediatos (naranja señal `#E5512D` para acciones principales, verde `#63D49A` para modelos validados, amarillo `#E7C84B` para entrenamiento en curso, azul `#5C78FF` para datos).

---

## Decision

Adoptamos el diseño **Neo-Industrial (Visual ML Lab)** para todas las interfaces web y herramientas visuales de CATML:
- **Base cromática:** `#111111` (negro carbón base), `#17181D` (superficies técnicas), `#D8D6CF` (gris cemento) y `#F1EFE9` (blanco cálido).
- **Acento de acción:** `#E5512D` (naranja señal reservado para CTAs de ejecución de entrenamiento y nuevos experimentos).
- **Tipografía dual:** `Space Grotesk` (UI y títulos) + `IBM Plex Mono` (métricas, IDs, hiperparámetros, timers y logs).
- **Estructura modular:** Paneles delimitados por bordes técnicos visibles (`border: 1px solid #27272a`), bordes de esquina rectos o sutiles (`0px` a `4px`), y navegación lateral numerada (`01 Dashboard`, `02 Datasets`, `03 Experiments`, `04 Models`, `05 Pipelines`, `06 Deployments`).

---

## Consequences

- **Positivas:**
  - Identidad de marca e ingeniería contundente, visualmente diferencial frente a SaaS genéricos.
  - Mayor densidad de información y legibilidad técnica en dashboards complejos.
  - Consistencia garantizada entre agentes de IA y desarrolladores humanos al seguir tokens de diseño unificados.
- **Reglas para agentes y contribuidores:**
  - Prohibido reintroducir bordes redondeados tipo burbuja (`16px`, `24px`, etc.).
  - Prohibido el uso de naranja señal en fondos decorativos o textos genéricos; su uso queda reservado a acciones de ejecución y resaltado del modelo campeón.
  - Monospace obligatorio para todos los valores numéricos y métricas de experimentos.
