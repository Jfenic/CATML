# CATML — Brand System: Tech Minimalista Premium

> Comprehensive visual and identity specification for the CATML open-source project.  
> Transmits technical seriousness, next-generation AI, and enterprise-grade reliability across Web, Workbench, Documentation, GitHub, and Presentations.

---

La identidad conceptual sería:

**CATML**  
*Agent-native AutoML for tabular data.*

Y la personalidad de marca:

**Precisa · técnica · limpia · potente · moderna · sobria · developer-first**

La sensación debería estar mucho más cerca de una herramienta profesional para ingenieros que de una startup que utiliza ilustraciones genéricas de inteligencia artificial.

---

## 1. Concepto visual

La regla principal sería:

> **Minimalismo estructural + pequeños momentos de impacto tecnológico.**

Aproximadamente:

**80% sobriedad / 20% impacto.**

Fondos muy limpios, mucho espacio, tipografía fuerte, componentes geométricos y una única familia de acentos eléctricos.

No utilizaría diez colores, efectos glass por todas partes ni gradientes continuamente.

El color se utiliza para comunicar:

```text
acción
estado
conexión
inteligencia
progreso
```

El resto permanece neutro.

---

# 2. Paleta principal

Esta sería mi propuesta para CATML.

| Token | Color | Uso |
|---|---|---|
| `--catml-black` | `#0B0D12` | Texto principal |
| `--catml-graphite` | `#171A22` | Fondos oscuros / código |
| `--catml-white` | `#FFFFFF` | Fondo principal |
| `--catml-offwhite` | `#F7F8FA` | Secciones alternas |
| `--catml-border` | `#E5E7EB` | Bordes |
| `--catml-muted` | `#667085` | Texto secundario |
| `--catml-blue` | `#4F67FF` | Color principal |
| `--catml-indigo` | `#6956E8` | Segundo tono |
| `--catml-cyan` | `#53C8FF` | Datos / agentes |
| `--catml-success` | `#22C55E` | Ejecución correcta |
| `--catml-warning` | `#F59E0B` | Advertencias |
| `--catml-danger` | `#EF4444` | Errores |

El color verdaderamente identificativo sería:

**CATML Electric Blue — `#4F67FF`**

No utilizaría el azul como fondo de media página. Lo utilizaría estratégicamente para botones, estados activos, diagramas y pequeños elementos.

---

# 3. Gradiente de marca

Puedes tener un gradiente identificativo, pero usarlo poco:

```css
background:
  linear-gradient(
    135deg,
    #4F67FF 0%,
    #6956E8 55%,
    #53C8FF 100%
  );
```

Utilízalo en:

```text
formas decorativas del hero
hover muy puntual
líneas agentic
iconografía especial
gráficos
CTA premium
```

No como fondo de todas las tarjetas.

Eso es lo que mantiene la sensación premium.

---

# 4. Tipografía

Yo utilizaría tres niveles.

### UI y cuerpo — Geist

**Geist** funciona especialmente bien para una herramienta técnica.

Alternativa:

**Inter**

La web puede utilizar:

```css
font-family: "Geist", "Inter", sans-serif;
```

### Headlines

También puedes utilizar Geist, pero en pesos 600–700.

Por ejemplo:

```text
Agent-native AutoML
for tabular data.
```

Con:

```text
font-weight: 650–700
letter-spacing: -0.04em
```

### Código

Aquí usaría:

**Geist Mono**  
o  
**IBM Plex Mono**

Código:

```css
font-family: "Geist Mono", monospace;
```

Eso permite que toda la identidad tenga una coherencia fuerte.

---

# 5. Escala tipográfica

Hero desktop:

```text
64–72 px
line-height: 0.98–1.05
```

H2:

```text
42–48 px
```

H3:

```text
24–28 px
```

Body:

```text
16–18 px
line-height: 1.6
```

Labels:

```text
12–13 px
uppercase opcional
letter-spacing: .06em
```

El hero debería tener pocas palabras y mucha presencia.

---

# 6. Logo CATML

Yo simplificaría muchísimo el logo.

No haría robot, cerebro, gato, chip o red neuronal en el isotipo.

CATML debería poder funcionar simplemente como:

```text
CATML
```

Tipografía fuerte y personalizada.

Puedes introducir un pequeño detalle propio:

```text
CΛTML
CATML_
[CATML]
CATML/
```

Pero mi favorita seguiría siendo:

**CATML**

limpio.

Luego crearía un símbolo independiente basado en:

```text
C
+
nodo
+
pipeline
```

o una geometría abstracta basada en las letras `C` y `M`.

El objetivo es que dentro de unos años el logo siga pareciendo serio.

---

# 7. Hero de la web

La composición debería ser aproximadamente:

```text
┌────────────────────────────────────────────────────────┐

 CATML                       Docs GitHub Community Platform


 OPEN SOURCE · LOCAL-FIRST · AGENT-NATIVE

 Agent-native AutoML
 for tabular data.

 Train, compare and export production-ready
 models locally — and let AI agents operate
 experiments through MCP.

 [ Get Started ]   [ GitHub ]   Watch demo →

                                  ┌───────────────┐
                                  │ CATML         │
                                  │ Workbench     │
                                  │               │
                                  │ Experiment 28 │
                                  │ LightGBM .914 │
                                  └───────────────┘

└────────────────────────────────────────────────────────┘
```

Yo pondría un screenshot real del Workbench.

No una ilustración.

El propio producto tiene que ser la ilustración.

---

# 8. Botones

Los botones también deben ser muy sobrios.

Primario:

```text
████████████
 Get Started →
████████████
```

Color:

```css
background: #0B0D12;
color: white;
```

O azul CATML:

```css
background: #4F67FF;
```

Secundario:

```css
background: white;
border: 1px solid #E5E7EB;
```

Radio:

```text
8–10 px
```

No utilizaría botones extremadamente redondos tipo píldora para todo.

CATML es técnico.

Un radio moderado transmite mejor esa personalidad.

---

# 9. Tarjetas

Las tarjetas no deberían parecer neobrutalistas.

Usaría:

```css
background: #FFFFFF;
border: 1px solid #E5E7EB;
border-radius: 14px;
box-shadow:
  0 1px 2px rgba(16,24,40,.04),
  0 8px 24px rgba(16,24,40,.04);
```

Muy poco shadow.

En hover:

```text
border → ligeramente azul
transform → translateY(-2px)
shadow → aumenta suavemente
```

Nada espectacular.

El movimiento tiene que transmitir calidad.

---

# 10. Código

Esta es una parte crucial de CATML.

Los snippets deberían convertirse en uno de los elementos principales de identidad visual.

Ejemplo:

```python
from catml import AutoML

automl = AutoML(task="classification")

result = automl.fit(
    df,
    target="churn"
)

result.leaderboard()
```

Panel:

```text
#10141B / #0F1720
```

Con syntax highlighting discreto:

```text
keywords       violet
strings        cyan
functions      blue
comments       gray
```

Y arriba:

```text
● ● ●                Python
```

Puedes añadir:

**Copy**

y:

**Run example →**

---

# 11. Workbench

Aquí mantendría exactamente el mismo design system.

Sidebar oscura:

```text
#11141B
```

Zona principal:

```text
#F7F8FA
```

Sidebar:

```text
CATML

Overview
Experiments
Datasets
Models
Artifacts
Agents

──────────

Community
Settings
```

Una selección activa puede tener:

```text
fondo #222638
texto blanco
pequeño indicador azul
```

No llenaría la interfaz de gradientes.

---

# 12. Leaderboard

Una de las pantallas más importantes visualmente:

```text
Experiment #128

MODEL          ROC-AUC      TIME       STATUS
────────────────────────────────────────────
LightGBM       0.914        32s         ✓
XGBoost        0.907        43s         ✓
CatBoost       0.901        36s         ✓
RandomForest   0.876        55s         ✓
```

El modelo ganador:

```text
BEST
LightGBM
ROC-AUC 0.914
```

con una pequeña banda azul.

No convertir todo en colores.

---

# 13. El lenguaje visual de agentes

Esta puede convertirse en una de las partes más reconocibles de CATML.

Tu flujo:

```text
Prompt
 ↓
Explore
 ↓
Read
 ↓
Reason
 ↓
Modify
 ↓
Test
 ↓
Evaluate
```

Yo lo mostraría horizontalmente:

```text
01          02          03         04
Prompt  →  Explore  →  Read  →  Reason

             ↓

07          06          05
Evaluate ← Test ← Modify
```

Cada etapa puede tener un símbolo lineal.

Las conexiones:

```text
#4F67FF → #53C8FF
```

con una animación extremadamente sutil de movimiento.

Eso puede convertirse en un elemento identificativo de CATML.

---

# 14. Iconografía

Usaría exclusivamente iconos:

```text
lineales
1.5–2 px
geométricos
monocromos
```

Nada de emojis en la web comercial.

Para:

```text
Python
Workbench
Agent
Models
Artifacts
Deploy
Cloud
```

Todos deben compartir la misma familia visual.

---

# 15. Las tres entradas a CATML

Esta sección debería ser extremadamente importante:

```text
One engine.
Three ways to work.

┌────────────────┐
│ </>            │
│ Python         │
│                │
│ Simple API for │
│ data scientists│
└────────────────┘

┌────────────────┐
│ ▣              │
│ Workbench      │
│                │
│ Visual AutoML  │
│ environment    │
└────────────────┘

┌────────────────┐
│ ◇              │
│ AI Agents      │
│                │
│ Operate CATML  │
│ through MCP    │
└────────────────┘
```

Esto debería convertirse casi en un elemento central de la marca.

---

# 16. Presentación del producto de código abierto

La web pública debe documentar únicamente las capacidades disponibles: Python, CLI, Workbench, MCP y ejecución local.

El diseño visual mantiene una identidad coherente entre documentación, web y Workbench, sin segmentar funciones por planes comerciales.

---

# 17. Fotografía e ilustración

Prácticamente eliminaría las fotos de stock.

CATML debería utilizar como imágenes:

```text
producto real
gráficos
diagramas
terminal
código
datasets
workflows
```

Si necesitas decoración:

formas geométricas abstractas.

Por ejemplo:

```text
rectángulos 3D
stacks
grid
nodes
pipelines
cubes
```

No:

```text
cerebros brillantes
robots humanoides
caras generadas por IA
redes neuronales genéricas
```

Esto es importante para no parecer otra web genérica de AI.

---

# 18. Motion design

Movimiento muy sutil.

Duraciones:

```text
150–250 ms UI
400–700 ms elementos de landing
```

Curvas:

```css
cubic-bezier(.22,1,.36,1)
```

Animaciones interesantes:

```text
workflow del agente avanzando
líneas que conectan nodos
leaderboard que aparece
code → result
dataset → model
```

Nada de elementos flotando continuamente sin propósito.

---

# 19. Dark mode

CATML debería tenerlo.

Light:

```text
Background       #FFFFFF
Surface          #F7F8FA
Text             #0B0D12
```

Dark:

```text
Background       #080A0F
Surface          #11141B
Surface elevated #171A22
Text             #F7F8FA
Border           #252A35
```

El mismo:

```text
#4F67FF
```

puede funcionar como accent principal.

Para CATML Workbench incluso podría hacer **dark como apariencia por defecto**, mientras la web comercial es principalmente clara.

Eso diferencia:

```text
Marketing → luminoso / premium
Product   → oscuro / technical
```

y queda muy bien.

---

# 20. Grid

Desktop:

```text
max-width: 1200–1280px
12 columns
24px gutter
```

Espaciado de secciones:

```text
120–160px desktop
80–100px tablet
64–80px mobile
```

Sistema base de spacing:

```text
4
8
12
16
24
32
48
64
96
128
```

No uses valores aleatorios continuamente.

---

# 21. Border radius

Crearía tokens:

```css
--radius-sm: 6px;
--radius-md: 10px;
--radius-lg: 16px;
--radius-xl: 24px;
```

Código:

```text
10 px
```

Cards:

```text
14–16 px
```

Grand screenshots:

```text
18–20 px
```

Nada de 40 px en todas partes.

---

# 22. CSS tokens iniciales

Podrías literalmente empezar tu sistema con:

```css
:root {
  --bg: #FFFFFF;
  --surface: #F7F8FA;

  --text: #0B0D12;
  --text-muted: #667085;

  --border: #E5E7EB;

  --brand: #4F67FF;
  --brand-secondary: #6956E8;
  --brand-cyan: #53C8FF;

  --success: #22C55E;
  --warning: #F59E0B;
  --danger: #EF4444;

  --code-bg: #10141B;

  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-xl: 24px;
}
```

Ya tienes prácticamente el ADN visual inicial.

---

# 23. Voz de marca

CATML no debería hablar como:

> “Revolutionize your AI journey with our groundbreaking intelligent solution.”

Evítalo completamente.

La voz CATML debería ser:

**corta, técnica y segura.**

Ejemplo malo:

> Unlock the future of machine learning with a revolutionary platform powered by next-generation artificial intelligence.

CATML:

> **Train better models. Keep control.**

O:

> **AutoML for humans and AI agents.**

O:

> **From dataset to production-ready model.**

O:

> **Local-first. Agent-native. Production-ready.**

Muchísimo mejor.

---

# 24. Mensajes principales

Yo empezaría a construir la marca alrededor de cuatro conceptos:

```text
LOCAL-FIRST
Your data stays under your control.

AGENT-NATIVE
Let AI agents operate AutoML through MCP.

PRODUCTION-READY
Export autonomous model artifacts.

OPEN-CORE
Open source locally. Scale when you need it.
```

Estas cuatro palabras deberían repetirse estratégicamente.

---

# 25. El sistema completo

La identidad final sería:

```text
CATML
│
├── Brand
│   ├── Electric Blue
│   ├── Graphite
│   └── White
│
├── Typography
│   ├── Geist
│   └── Geist Mono
│
├── Product
│   ├── Python
│   ├── Workbench
│   └── Agents
│
├── Principles
│   ├── Local-first
│   ├── Agent-native
│   ├── Production-ready
│   └── Open-source
│
└── Product
    └── CATML
```

La identidad CATML debe ser coherente en documentación, web y Workbench.

### Mi dirección final

Si fuera CATML, adoptaría como identidad:

> **CATML — Technical Minimalism**

con **Geist + Geist Mono**, blanco/grafito como base, **Electric Blue `#4F67FF`** como color identificativo y pequeños gradientes azul–violeta–cyan únicamente para representar inteligencia, agentes y procesamiento.

Eso conservaría el impacto que buscas, pero elevaría muchísimo la percepción desde **“proyecto open-source interesante”** hacia **“producto tecnológico que podría convertirse en una empresa seria”**.
