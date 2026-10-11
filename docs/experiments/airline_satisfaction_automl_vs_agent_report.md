# Informe experimental: AutoML solo vs potencial con Explore/agente

**Dataset:** Kaggle airline passenger satisfaction (`data/kagle/train.csv`, ~699 635 filas)  
**Muestra de trabajo:** 25 000 filas estratificadas 50/50 sobre `satisfaction` (`random_state=42`)  
**Métrica:** ROC-AUC · **CV:** 5-fold · **Modelos:** `logistic_regression`, `random_forest`  
**Fecha:** 2026-10-11  
**Artefactos:** `docs/experiments/airline_automl_vs_explore_results.json`, `airline_automl_vs_explore_decomposition.json`

> Objetivo: medir qué aporta CATML “solo AutoML” frente a un flujo asistido por Explore/agente, y listar **mejoras transferibles** (sin reglas hardcodeadas a aerolíneas).

---

## 1. Resultados (muestra)

| Configuración | Mejor ROC-AUC | Δ vs naive |
|---------------|---------------|------------|
| **A. AutoML naive** (todas las columnas, incl. `id`) | **0.817** | — |
| **B. AutoML + Anti-Leakage Guardian** (solo features “safe”) | **0.952** | **+0.135** |
| **C. B + drops de colinealidad propuestos por Explore** | 0.952 | +0.135 (−0.0006 vs B) |
| **D. C + log1p** | n/d | Explore pidió `yeo_johnson` (transform con estado); no materializado |

**Conclusión empírica en esta muestra:** casi todo el salto viene del **guardian genérico anti-leakage / anti-identificadores**, no de drops de colinealidad ni de transforms aún no aplicables. Explore **sí detectó** sesgo en delays (`yeo_johnson`) y colinealidades; el sistema **aún no puede promover esas recetas de forma segura** (gap documentado en Experiment Recipes).

Leaderboard B (referencia fuerte):

| Modelo | ROC-AUC | CV std |
|--------|---------|--------|
| random_forest | 0.952 | 0.003 |
| logistic_regression | 0.919 | 0.002 |

---

## 2. Qué hizo “AutoML solo”

- Entrenar candidatos con la fachada `AutoML.fit`.
- Si el usuario (o un script) pasa el CSV **crudo con `id`**, el rendimiento cae (~0.82) y el CV es más inestable (`cv_std` ~0.036 vs ~0.003).
- Cuando el **profiler + `resolve_safe_feature_names()`** actúa (Workbench / registro correcto), AutoML ya opera cerca del techo de esta muestra (~0.95) **sin agente**.

Interpretación: en datasets “sucios” tipo Kaggle, **la potencia base de CATML está más en el guardian que en el LLM**.

---

## 3. Qué aportaría un agente / Explore (potencial real)

Explore sobre la misma muestra generó **55 hallazgos** y **37 hipótesis**, entre ellos (tipos, no columnas hardcodeadas):

| Tipo de hallazgo | # | Valor para un agente |
|------------------|---|----------------------|
| `target_class_separation` | 17 | Priorizar features con señal |
| `normality_violation` / `multimodal_distribution` | muchos | Contexto; no implica transform automática |
| `high_collinearity` | 5 | Candidatos a ablación |
| `skewness` + `power_transform` | 2 | Recetas `yeo_johnson` / `log1p` |
| `high_cardinality_identifier` | 1 | Confirma exclusión de IDs |
| Group leakage (`Customer Type` en profiler) | 1 | Exige `GroupKFold` en validación real |

**Potencial del agente (cuando exista Experiment Recipes R0–R4):**

1. Leer hallazgos → proponer hipótesis (ya existe).  
2. Verificar empíricamente (*Propose ≠ Accept*) con transforms **reales** fit-in-fold.  
3. Guardar **ExperimentRecipe** y re-probar en un solo modelo o grid.  
4. Promover solo lo que generaliza — no confundir `accepted` con “aplicar siempre”.

En **este** experimento, los drops de colinealidad **no mejoraron** RF; un agente disciplinado debería **rechazar** esa receta (aprendizaje negativo), no forzarla.

---

## 4. Elementos que facilitan / mejoran el modelo (transferibles a otros datasets)

Reglas **genéricas**; ninguna depende de nombres de columnas de aerolínea:

### P0 — Ya disponibles y de alto impacto

1. **Excluir identificadores de alta cardinalidad** (`id`-like) antes de entrenar.  
2. **Detectar group leakage** y no validar con split IID si una entidad se repite entre folds (aquí el profiler marcó `Customer Type`).  
3. **Usar siempre el feature set “safe” del profiler** como default de AutoML/Workbench.  
4. **Semilla + CV fijos** para comparar configs de forma justa.

### P1 — Explore/agente (parcialmente listo)

5. **Inventario de colinealidad** → ablación controlada (probar; a veces no ayuda a árboles).  
6. **Sesgo / colas pesadas** → hipótesis `log1p` (sin estado) vs `yeo_johnson` / `RobustScaler` (con estado, fit-in-fold).  
7. **Separación respecto al target** → priorizar presupuesto de HPO en features con efecto.  
8. **EvidenceLink + recetas** → reutilizar solo mejoras verificadas (spec ADR-009).

### P2 — Plataforma (no específicos del dataset)

9. **Paridad CLI / MCP / REST / UI** para el mismo flujo de recetas (agentes = mismos permisos).  
10. **Presupuesto y aprobación** antes de lanzar grids caros.  
11. **Informes exportables** (Markdown/JSON) para auditar por qué se eligió una config.

### Lo que NO recomendar como “solución”

- Hardcodear “siempre dropear wifi / delays / Class”.  
- Materializar `RobustScaler`/`Yeo-Johnson` sobre todo el CSV antes del split.  
- Tratar `EvidenceLink.accepted` como promoción automática a producción.  
- Optimizar a ciegas en 700k filas sin muestra estratificada + luego confirmar en holdout/test Kaggle.

---

## 5. Cómo escalar al dataset completo (sin hardcode)

1. Fijar pipeline “safe features” + política de grouping si aplica.  
2. Validar en muestra (como aquí) → promover receta.  
3. Re-entrenar en train completo con la **misma receta**.  
4. Generar submission con el artefacto ganador.  
5. Opcional: HPO solo sobre el modelo ganador (`random_forest` aquí).

---

## 6. Respuesta directa a la pregunta

| Pregunta | Respuesta |
|----------|-----------|
| ¿Cuánto saca **solo AutoML**? | ~**0.82** ROC-AUC si se traga el CSV con `id`; ~**0.95** si el guardian limpia features. |
| ¿Cuánta potencia añade el **agente/Explore hoy**? | Gran detección y propuestas; **poco delta extra** en esta muestra porque el gran salto ya lo da el anti-leakage, y los transforms con estado **aún no se aplican** en verify. |
| ¿Qué faltaría para que el agente brille? | **Experiment Recipes** (R0–R4): apply real + fit-in-fold + Selected/Promoted + paridad MCP/CLI. |
| ¿Sirve para otros datasets? | Sí: identificadores, group leakage, colinealidad, sesgo, priorización por separación al target — todo genérico. |

---

## 7. Limitaciones del informe

- Muestra de 25k, no el train completo ni el `test.csv` de Kaggle (no hay score público de leaderboard aquí).  
- Solo dos familias de modelos, sin HPO.  
- `Customer Type` excluido por heurística de group leakage del profiler; en producción podría usarse como **columna de agrupación** en `GroupKFold` en lugar de dropearla — eso también es genérico y conviene probarlo como receta alternativa.

---

## 8. Push de rendimiento (sesión posterior, sin soluciones externas)

Protocolo: dropear solo `id`; **mantener** `Customer Type` (solo 2 niveles → es categoría, no entidad para GroupKFold); StratifiedKFold(5); LightGBM/XGBoost + Optuna; sin mirar notebooks ajenos.

| Config | ROC-AUC CV |
|--------|------------|
| LightGBM default (50k) | 0.9565 |
| LightGBM HPO 25 trials (50k) | 0.9576 |
| Blend LGBM+XGB (50k) | 0.9577 |
| **LightGBM HPO confirmado (150k)** | **0.9586** |
| **LightGBM HPO en train completo (699 635)** | **0.9586** |

Artefactos: `docs/experiments/airline_push_scores.json`, `docs/experiments/push_airline_score.py`.

Delta vs naive con `id` (~0.817): **≈ +0.14 ROC-AUC**. Casi todo el salto = quitar ID + GBDT; HPO aporta ~+0.001–0.002.

**Leaderboard Kaggle (test ciego):** submission `submission_lgbm_hpo.csv` → **0.9578** (Δ ≈ −0.0008 vs CV full-train). Buena calibración del protocolo de validación.

---

## 9. Conclusiones útiles para otros datasets y aprendizaje

### Lo que más importó (transferible)

1. **Quitar identificadores antes de modelar** — Incluir `id` hundió el CV (~0.82). El Anti-Leakage Guardian / exclusiones de alta cardinalidad no son “detalle”: son el mayor salto.
2. **No confundir categoría binaria con entity leakage** — `Customer Type` (2 niveles) no es un `patient_id`. GroupKFold ahí no tiene sentido; dropearla por heurística puede **quitar señal**. Regla: cardinality baja + semántica de atributo → feature; cardinality ~n_rows → ID.
3. **GBDT > RF/logreg en tabular denso con interacciones** — LightGBM/XGBoost subieron el techo; el stack “solo sklearn básico” deja puntos en la mesa cuando hay extras `models`.
4. **HPO ayuda poco si el default ya es bueno** — Optuna ~+0.001 AUC. Primero pipeline limpio + buen modelo; luego afinar.
5. **CV honesto predice el leaderboard** — 0.9586 CV → 0.9578 público. StratifiedKFold + mismas features train/test + sin peek al test = protocolo fiable.
6. **Explore/agente = brújula, no piloto automático** — Detectó sesgo, colinealidad e IDs; los drops de colinealidad **no** mejoraron árboles. *Propose ≠ Accept* y aprendizaje negativo importan.
7. **Transforms con estado aún no están en el producto** — `yeo_johnson`/`RobustScaler` propuestos pero no aplicables de forma segura → justifica Experiment Recipes (fit-in-fold + recetas + paridad MCP/CLI).
8. **Muestra estratificada acelera** — Afinar en 50k y confirmar en full train dio el mismo orden de magnitud; útil en datasets grandes.

### Aprendizajes de proceso (CATML / agentes)

| Hacer | Evitar |
|-------|--------|
| Default = features “safe” del profiler | Entrenar el CSV crudo “por si acaso” |
| Verificar hipótesis con Δ de métrica | Aplicar todo lo que Explore sugiere |
| Guardar receta reproducible (params, seed, CV, features) | Solo un score en un notebook suelto |
| Paridad CLI/MCP/UI para el mismo flujo | Lógica solo en el Workbench |
| Reportar CV **y** score externo | Optimizar solo al public LB |

### Mensaje único

> En datos tabulares tipo competición, **higiene de features + modelo adecuado + validación alineada** aportan más que EDA decorativa o HPO agresivo. El agente brilla cuando puede **proponer, medir y conservar recetas**, no cuando aplica transforms a ciegas.
