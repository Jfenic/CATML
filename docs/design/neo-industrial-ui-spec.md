# Neo-Industrial UI & Visual ML Lab Specification

> **Aesthetic & Technical Design System for CATML AutoML Workbench**  
> *Target audience:* Developers and AI Coding Agents contributing to CATML web interfaces and visualizations.

---

## 1. Vision & Design Philosophy

CATML is an engineering and scientific platform for hypothesis-driven Automated Machine Learning ("Propose ≠ Accept"). The user interface must reflect this identity:

> **Visual Reference:** Software de ingeniería + dashboard científico + terminal moderna + herramienta creativa.  
> **Core Feeling:** *Técnica + modular + precisa + visual + experimental.*

### What to Avoid (Anti-Patterns)
- ❌ **Generic SaaS Look:** Floating bubble cards, soft pastel gradients, large border radii (`16px`, `24px`, `32px`), childish badges (trophies, medals, stars).
- ❌ **Opaque "Black-Box" Spinners:** Vague loading spinners with no progress, missing ETA, or hidden metrics.
- ❌ **Cluttered Navigation:** Deep nested dropdowns or unnumbered generic menus.

### What to Build (Neo-Industrial Standard)
- ✔ **Architectural Modules:** Panels defined by crisp, visible lines (`border: 1px solid #27272a` or `rgba(216, 214, 207, 0.15)`).
- ✔ **Minimal Radii:** `0px`, `2px`, or maximum `4px` for modular blocks (maximum `8px` for global modals/drawers).
- ✔ **High Information Density & Visual Control:** Large monospaced numbers, technical state chips (`SYS / ONLINE`, `● READY`, `● TRAINING`), and dense diagnostic tables.
- ✔ **Signal Accent Color:** Signal orange (`#E5512D`) reserved strictly for primary execution and training CTAs.

---

## 2. Color Palette & Functional Token System

```text
Base Palette:
  #111111  — Carbon Black (Primary Background base)
  #17181D  — Technical Panel / Surface Background
  #1C1D24  — Card / Modular Block Background
  #D8D6CF  — Cement Gray (Borders, structural dividers, secondary labels)
  #F1EFE9  — Warm White (Primary high-contrast text and active values)
  #E5512D  — Signal Orange (Primary Action CTA: Train, Start, New Experiment)

Functional / Semantic Accents:
  #63D49A  — Success / Model Completed / Metric Validation
  #E7C84B  — Training in Progress / Warning / Attention
  #D85454  — Error / Failed Trial / Destructive Action
  #5C78FF  — Dataset / Data Profiling / Information
  #9C7CFF  — Model Space / Intelligence / Agent / Knowledge
```

### Color Usage Rules
1. **Signal Orange (`#E5512D`) is an ACTION color, not a decorative color.**
   - Use for: `[ ▶ START TRAINING ]`, `[ + NEW EXPERIMENT ]`, `[ RUN FEATURE SELECTION ]`, active champion highlight bar (`▌01`).
   - Do NOT use orange for backgrounds of static cards, generic text, or incidental icons.
2. **Status Color Mapping:**
   - **Completed / Ready:** Green (`#63D49A`)
   - **Training / Searching / Warning:** Yellow/Amber (`#E7C84B`)
   - **Failed / Error:** Red (`#D85454`)
   - **Dataset / Explorer:** Blue (`#5C78FF`)
   - **Model / DAG / Intelligence:** Violet (`#9C7CFF`)

---

## 3. Typography System

The interface pairs an architectural sans-serif for UI layout with a high-legibility monospace font for all data, identifiers, and metrics.

| Role | Font Family | Typical Weights | Usage Examples |
|---|---|---|---|
| **Headings & UI** | `Space Grotesk`, `Inter Tight`, sans-serif | `500`, `600`, `700` | Navigation, Section Headers, Modal Titles |
| **Data & Metrics** | `IBM Plex Mono`, `JetBrains Mono`, monospace | `400`, `500`, `600`, `700` | Metric numbers (`94.82%`), Model IDs (`XGBOOST_04`), Timers (`03:42`), Feature names, Code/Logs |

### Metric Display Format
Metrics must be displayed with high visual impact: large number in monospace, followed by a crisp uppercase technical label.

```text
┌──────────────────────────────┐
│ 94.82 %                      │
│ ACCURACY (ROC-AUC: 0.961)    │
│ +2.41% vs baseline           │
└──────────────────────────────┘
```

---

## 4. Navigation Architecture

Navigation uses a numbered sidebar reflecting a scientific machine learning laboratory:

```text
AUTO/ML LAB

01  Dashboard       (Overview, recent runs, system health)
02  Datasets        (Schema, profiling, box plots, correlation matrix, feature selection)
03  Experiments     (Studio, trial execution, live progress, logs)
04  Models          (Leaderboard, comparison, ROC curves, confusion matrix)
05  Pipelines       (DAG visualizer, feature transformations, ensemble graphs)
06  Deployments     (Kaggle submissions, artifacts, export)

────────────────────
SYS / ONLINE
Workspace: /data/churn_exp
Core: V0.7.0 (CQRS + Hexagonal)
```

The numbered prefixes (`01`, `02`, `03`…) enforce precision and industrial aesthetic. Small, linear icons are allowed, but oversized emojis must be avoided.

---

## 5. Component Construction Rules

### 5.1 Modules (Not Cards)
Cards should not appear as floating clouds. They must look like modular equipment racks:
- **Border:** `1px solid #27272a` (or `rgba(216, 214, 207, 0.18)`).
- **Border Radius:** `2px` or `4px` (never `16px` or `24px`).
- **Header:** Delimited by a `1px` horizontal separator line.

```text
┌────────────────────────────────────────────────────────┐
│ MOD / 02  DATASET PROFILING              STATUS: READY │
├────────────────────────────────────────────────────────┤
│ ROWS: 12,482    FEATURES: 18    TARGET: churn (binary) │
│ ...                                                    │
└────────────────────────────────────────────────────────┘
```

### 5.2 Training Visualizer (Live Industrial Dashboard)
During training, never use a static spinner. Provide active metrics and progress:

```text
EXPERIMENT / EXP-024
SEARCHING MODEL SPACE — 14 / 20 MODELS

01  XGBOOST_04      0.9482   [BEST]
02  LIGHTGBM_02     0.9441   +1.98%
03  RANDOMFOREST    0.9324   +0.73%

████████████████████░░░░░░░░ 72%
ELAPSED: 03:42     ETA: 01:21
```

### 5.3 Model Leaderboard
The leaderboard highlights the champion model with a signal orange indicator line:

```text
MODEL LEADERBOARD
▌01  XGBOOST_04      0.9482   +2.41% vs baseline   [INSPECT]
 02  LIGHTGBM_02     0.9441   +1.98%               [INSPECT]
 03  XGBOOST_03      0.9418   +1.72%               [INSPECT]
```

### 5.4 Buttons & Controls
- **Primary Execution CTA (`.btn-signal`):**
  - Background: `#E5512D`
  - Text: `#FFFFFF`
  - Font: `IBM Plex Mono`, `font-bold`, `uppercase`, tracking `0.05em`
  - Radius: `2px`
- **Secondary Action (`.btn-technical`):**
  - Background: `#17181D`
  - Border: `1px solid #333333`
  - Text: `#F1EFE9`
  - Hover: Border color `#D8D6CF`
- **Ghost / Outline (`.btn-ghost`):**
  - Background: Transparent
  - Border: `1px solid rgba(216, 214, 207, 0.25)`
  - Text: `#D8D6CF`

---

## 6. Instructions for AI Coding Agents

When working on any frontend template (`index.html`), stylesheet (`workbench.css`), or JavaScript view (`src/automl/interfaces/web/static/js/`):

1. **Strictly adhere to the Neo-Industrial palette:** Use CSS variables `--catml-signal`, `--catml-bg-base`, `--catml-border`, etc.
2. **Never add rounded balloon borders:** Always use `rounded-sm` (`2px`) or `rounded` (`4px`). Avoid `rounded-xl`, `rounded-2xl`, `rounded-3xl`.
3. **Use Monospace for all numbers, identifiers, parameters, and timestamps:** Wrap them in `.font-mono` (`IBM Plex Mono`).
4. **Preserve Hexagonal Parity:** The frontend must never call ML engines directly. It only communicates with `/api/...` endpoints backed by CQRS commands and queries.
5. **No bulky external heavy UI frameworks:** Keep lightweight, responsive, native SVGs for charts and diagrams.
