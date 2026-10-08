# Decision: Tech Minimalista Premium Brand System (CATML Unified Identity)

Estado: Aceptada. Fecha: 2026-10-02.  
Alcance: Identidad visual, tipografía, paleta de colores y componentes del proyecto CATML.  
Especificación técnica de referencia: [docs/design/tech-minimalist-brand-system.md](../design/tech-minimalist-brand-system.md).  
Evolución de: [ADR 005: Neo-Industrial Visual ML Lab UI](005-neo-industrial-visual-ml-lab-ui.md).

---

## 1. Context

CATML ha evolucionado de un prototipo de investigación hacia un proyecto abierto (Apache 2.0) con motor AutoML local, CLI, Workbench e integración MCP.

Para soportar este crecimiento y transmitir simultáneamente:
- **Seriedad técnica y de ingeniería** frente a científicos de datos y MLOps,
- **Capacidades agent-native de nueva generación** (interacción con LLMs mediante MCP),
- **Confianza y solidez técnica** para usuarios y equipos que trabajan con datos privados,

era indispensable unificar y elevar el sistema de diseño visual en toda la presencia del producto (Web, Workbench, Documentación, README de GitHub, CLI branding y presentaciones corporativas).

---

## 2. Decision

Adoptamos el sistema de diseño **CATML — Tech Minimalista Premium** como estándar de marca e interfaz para todo el ecosistema CATML:

### Principios Fundamentales
- **80% Sobriedad / 20% Impacto:** Fondos neutros, amplia respiración espacial, tipografía geométrica sólida y acentos eléctricos utilizados con propósito estricto (acción, estado, conexión, inteligencia, progreso).
- **Dualidad Luz/Sombra con Propósito:**
  - *Modo Claro (`#FFFFFF`, `#F7F8FA`):* Documentación, landing pages, marketing y lectura técnica continuada.
  - *Modo Oscuro Grafito (`#080A0F`, `#11141B`, `#171A22`):* Workbench visual, consolas CLI, visualización de métricas y terminales de agentes.
- **Color Identificativo:** **CATML Electric Blue (`#4F67FF`)** como tono primario de acción y marca. Gradiente controlado *Electric Gradient* (`#4F67FF` → `#6956E8` → `#53C8FF`) reservado exclusivamente para representar inteligencia artificial, agentes activos y flujo de datos.
- **Tipografía Oficial:**
  - Primaria (UI, encabezados, botones): **Geist** (o alternativamente *Inter* como fallback neutro).
  - Técnica (código, tablas, métricas, logs, JSON, hiperparámetros): **Geist Mono** / **IBM Plex Mono**.
- **Geometría y Componentes:** Bordes nítidos (`1px solid`), radios limpios (8px–12px en contenedores, 4px–6px en elementos densos de instrumentación), eliminando sombras excesivas o efectos glass desenfocados.

### Relación con ADR 005 (Workbench UI)
El diseño Neo-Industrial introducido en ADR 005 evoluciona de manera armoniosa: se preserva la alta densidad de instrumentación técnica, paneles modulares y tipografía mono para métricas, actualizando los tokens cromáticos (adoptando la base grafito `#0B0D12` / `#171A22` y acentos `#4F67FF` / `#53C8FF`) para mantener perfecta coherencia entre la web pública y el Workbench interactivo.

---

## 3. Consequences

- **Positivas:**
  - Percepción inmediata de producto tecnológico maduro y de nivel enterprise.
  - Coherencia visual entre la web, el Workbench y la documentación.
  - Tokens CSS claros (`--catml-*`) directamente reutilizables en web, FastAPI endpoints y dashboards.
  - Alineación de los cuatro mensajes clave de marca: *Local-First*, *Agent-Native*, *Production-Ready*, *Open-Source*.
- **Reglas de Implementación:**
  - Prohibido el uso de ilustraciones genéricas de IA (robots 3D, cerebros de neón flotantes, efectos de humo). Usar esquemas arquitectónicos limpios, código real y terminales de agentes.
  - El acento Electric Blue `#4F67FF` no debe emplearse en fondos de grandes bloques, sino en botones primarios, bordes de foco, estados de selección y progreso.
  - Mantener estricto soporte tipográfico para números tabulares (`font-feature-settings: "tnum"`) en leaderboards y tablas de métricas.
