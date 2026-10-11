/**
 * PipelineView — Visual Pipeline DAG
 * Interactive directed acyclic graph of the complete CATML end-to-end flow.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { isRunActive } from "../utils.js";

export class PipelineView {
  constructor() {
    this.container = null;
    this.selectedNode = "FeatureGenerator";
    this.activeRun = null;
    this.profile = null;
    this.leaderboard = [];
  }

  async mount(container) {
    this.container = container;
    this.renderLoading();
    await this.fetchData();
    this.render();
  }

  renderLoading() {
    this.container.innerHTML = `
      <div class="workbench-card p-12 text-center text-[#8B95A7] space-y-3">
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-medium font-sans">Construyendo grafo visual del pipeline de ejecución...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const dsRuns = store.getRunsForActiveDataset();
      this.activeRun = store.getActiveRun()
        || dsRuns.find(r => isRunActive(r))
        || dsRuns[0]
        || null;

      if (this.activeRun) {
        const [profile, leaderboard] = await Promise.all([
          api.getDatasetProfile(this.activeRun.dataset_id).catch(() => null),
          api.getLeaderboard(this.activeRun.id).catch(() => []),
        ]);
        this.profile = profile;
        this.leaderboard = leaderboard || [];
      }
    } catch (e) {
      console.warn("PipelineView fetchData error:", e);
    }
  }

  render() {
    const p = this.profile;
    const run = this.activeRun;
    if (!p && !run) {
      this.container.innerHTML = `
        <div class="workbench-card p-12 text-center space-y-4">
          <div class="text-[#8B95A7]/40 flex justify-center">${icon("workflow", "icon-xl", 36)}</div>
          <h3 class="text-base font-bold text-[#F7F8FA]">Sin Pipeline Generado</h3>
          <p class="text-xs text-[#8B95A7] max-w-sm mx-auto">Registra un dataset y ejecuta un experimento para generar el grafo DAG de transformaciones, ingeniería de features y ensamble.</p>
          <button id="btnNewExpPipelineEmpty" class="btn-signal" aria-label="Crear nuevo experimento">
            <span class="inline-flex items-center gap-1.5">${icon("plus", "icon-sm")} <span>Nuevo Experimento</span></span>
          </button>
        </div>
      `;
      this.container.querySelector("#btnNewExpPipelineEmpty")?.addEventListener("click", () => {
        bus.emit("modal:new-experiment");
      });
      return;
    }
    const columns = p ? (p.columns || []) : [];
    const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float"))).length;
    const imgCols = columns.filter(c => c.is_image).length;
    const catCols = columns.filter(c => c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string") && !c.is_image).length;
    const idCols = columns.filter(c => c.is_identifier || c.catml_action === "Exclude");

    const topModels = this.leaderboard.slice(0, 3);
    const nodeDetails = this._getNodeDetails(this.selectedNode);

    const dsDisplayName = run ? (run.dataset_name || "Dataset") : (p ? p.name : "Dataset");
    const rowCountText = p && p.row_count ? `${Number(p.row_count).toLocaleString()} filas` : "Datos Brutos";
    const colCountText = p && p.column_count ? `${p.column_count} columnas` : "Características";

    const hasModels = topModels.length > 0;
    const dagStatusText = hasModels
      ? "Pipeline Entrenado y Alineado"
      : (p ? "Esquema Inferido" : "Ingesta Pendiente");
    const dagStatusBadgeClass = hasModels ? "badge-gain" : (p ? "badge-sys" : "badge-warn");

    this.container.innerHTML = `
      <div class="space-y-6">
        <div class="workbench-card p-4 flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <span class="text-[#4F67FF]">${icon("workflow", "icon-lg")}</span>
            <div>
              <h2 class="text-base font-semibold text-[#F7F8FA] font-sans">Grafo Visual del Pipeline DAG — ${dsDisplayName}</h2>
              <p class="text-xs text-[#8B95A7] font-sans">Grafo acíclico dirigido (DAG) reproducible para ingesta de datos, preprocesamiento, generación de características y orquestación de ensambles.</p>
            </div>
          </div>
          <span class="${dagStatusBadgeClass} text-xs px-2.5 py-0.5 rounded-md font-mono font-medium inline-flex items-center gap-1">${icon(hasModels ? "check" : "circle", "icon-sm")} <span>${dagStatusText}</span></span>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Graph Canvas (Left) -->
          <div class="lg:col-span-8 workbench-card p-6 flex flex-col items-center justify-center space-y-4 bg-slate-950/40 min-h-[480px]">
            <!-- Node: Dataset -->
            <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'Dataset' ? 'active' : ''}" data-node="Dataset" tabindex="0" role="button" aria-label="Nodo Dataset">
              ${icon("database", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
              <div class="text-[10px] text-slate-400 uppercase font-mono">Fuente de Entrada</div>
              <div class="text-xs font-bold text-slate-200">${dsDisplayName}</div>
              <div class="text-[10px] text-slate-500 font-mono">${rowCountText} × ${colCountText}</div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Node: SchemaDetector & ID Exclusion -->
            <div class="flex space-x-6">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'SchemaDetector' ? 'active' : ''}" data-node="SchemaDetector" tabindex="0" role="button" aria-label="Nodo SchemaDetector">
                ${icon("search", "icon-sm text-[#6956E8] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Inferencia</div>
                <div class="text-xs font-bold text-purple-300">SchemaDetector</div>
                <div class="text-[10px] text-slate-500 font-mono capitalize">${run ? (run.task_type || "Clasificación").replace("_", " ") : "Clasificación"}</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'IDExclusion' ? 'active' : ''}" data-node="IDExclusion" tabindex="0" role="button" aria-label="Nodo ID Exclusion">
                ${icon("shield-x", "icon-sm text-[#EF4444] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Poda</div>
                <div class="text-xs font-bold text-rose-300">ID Exclusion</div>
                <div class="text-[10px] text-slate-500 font-mono">${idCols.length > 0 ? `Descartar '${idCols[0].name}'` : 'Cero Fugas'}</div>
              </div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Preprocessing: Numerical, Categorical & Vision -->
            <div class="flex flex-wrap justify-center gap-4">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'NumericalImputer' ? 'active' : ''}" data-node="NumericalImputer" tabindex="0" role="button" aria-label="Nodo Imputador Numérico">
                ${icon("binary", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pipeline</div>
                <div class="text-xs font-bold text-indigo-300">Imputador Numérico</div>
                <div class="text-[10px] text-slate-500 font-mono">${numCols} Numéricas</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'CategoricalEncoder' ? 'active' : ''}" data-node="CategoricalEncoder" tabindex="0" role="button" aria-label="Nodo Codificador Categórico">
                ${icon("tags", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pipeline</div>
                <div class="text-xs font-bold text-indigo-300">Codificador Categórico</div>
                <div class="text-[10px] text-slate-500 font-mono">${catCols} Categóricas</div>
              </div>

              ${imgCols > 0 ? `
                <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'ImageEncoderNode' ? 'active' : ''}" data-node="ImageEncoderNode" tabindex="0" role="button" aria-label="Nodo Codificador de Imágenes">
                  ${icon("image", "icon-sm text-[#53C8FF] mb-1.5 block mx-auto")}
                  <div class="text-[10px] text-slate-400 uppercase font-mono">Visión</div>
                  <div class="text-xs font-bold text-cyan-300">ImageEncoderNode</div>
                  <div class="text-[10px] text-cyan-400 font-mono">${imgCols} Imágenes</div>
                </div>
              ` : ''}
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- FeatureGenerator (Key Node) -->
            <div class="dag-node text-center cursor-pointer border-purple-500/60 bg-purple-950/20 ${this.selectedNode === 'FeatureGenerator' ? 'active' : ''}" data-node="FeatureGenerator" tabindex="0" role="button" aria-label="Nodo FeatureGenerator">
              ${icon("sparkles", "icon-sm text-[#6956E8] mb-1.5 block mx-auto")}
              <div class="text-[10px] text-purple-400 uppercase font-mono font-bold">Guiado por Hipótesis</div>
              <div class="text-xs font-bold text-purple-200">FeatureGenerator</div>
              <div class="text-[10px] text-purple-300 font-mono">Propose ≠ Accept</div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Top Models -->
            <div class="flex space-x-4">
              ${topModels.length > 0 ? topModels.map(m => `
                <div class="dag-node text-center cursor-pointer ${this.selectedNode === m.model_id ? 'active' : ''}" data-node="${m.model_id}" tabindex="0" role="button" aria-label="Modelo ${m.model_id}">
                  ${icon("brain", "icon-sm text-[#22C55E] mb-1.5 block mx-auto")}
                  <div class="text-[10px] text-slate-400 uppercase font-mono">Modelo</div>
                  <div class="text-xs font-bold text-slate-200 capitalize">${m.model_id}</div>
                  <div class="text-[10px] text-emerald-400 font-mono">${Number(m.score).toFixed(5)}</div>
                </div>
              `).join("") : `
                <div class="dag-node text-center cursor-pointer" data-node="Models" tabindex="0" role="button" aria-label="Modelos de Ensamble">
                  ${icon("boxes", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
                  <div class="text-[10px] text-slate-400 uppercase font-mono">Modelos</div>
                  <div class="text-xs font-bold text-slate-200">Candidatos de Ensamble</div>
                  <div class="text-[10px] text-indigo-400 font-mono">Stratified CV</div>
                </div>
              `}
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Prediction / Submission Node -->
            <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'Prediction' ? 'active' : ''}" data-node="Prediction" tabindex="0" role="button" aria-label="Nodo Predicción y Envío">
              ${icon("package", "icon-sm text-[#22C55E] mb-1.5 block mx-auto")}
              <div class="text-[10px] text-slate-400 uppercase font-mono">Artefacto de Salida</div>
              <div class="text-xs font-bold text-emerald-300">submission.csv</div>
              <div class="text-[10px] text-slate-400 font-mono">Formato validado</div>
            </div>
          </div>

          <!-- Inspector Panel (Right) -->
          <div class="lg:col-span-4 workbench-card p-5 space-y-4">
            <div class="border-b border-slate-800 pb-3 flex items-center justify-between">
              <div>
                <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Inspector de Nodos</span>
                <h3 class="text-base font-bold text-slate-100" id="inspectorNodeName">${nodeDetails.name}</h3>
              </div>
              <span class="badge-gain text-xs px-2 py-0.5 rounded font-mono" id="inspectorNodeStatus">${nodeDetails.status}</span>
            </div>

            <p class="text-xs text-slate-300 leading-relaxed" id="inspectorNodeDesc">
              ${nodeDetails.description}
            </p>

            <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-2 text-xs font-mono">
              <div class="text-slate-400 font-sans font-semibold text-[11px] uppercase">Contrato del Nodo</div>
              <div class="flex justify-between">
                <span class="text-slate-400">Entrada:</span>
                <span class="text-slate-200" id="inspectorInput">${nodeDetails.input}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-400">Salida:</span>
                <span class="text-indigo-400" id="inspectorOutput">${nodeDetails.output}</span>
              </div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-2 text-xs font-mono" id="inspectorExtra">
              ${nodeDetails.extra}
            </div>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _getNodeDetails(node) {
    const p = this.profile;
    const run = this.activeRun;
    const columns = p ? (p.columns || []) : [];
    const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float"))).length;
    const catCols = columns.filter(c => c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string")).length;
    const imgCols = (p && p.image_columns ? p.image_columns.length : 0) || columns.filter(c => c.is_image).length;

    const details = {
      Dataset: {
        name: `Fuente de Entrada (${run ? run.dataset_name : 'Dataset'})`,
        status: "Ingestado",
        description: "Dataset tabular cargado directamente a través del puerto DataSource sin acoplar frameworks al dominio.",
        input: run ? (run.dataset_path || "data/train.csv") : "data/train.csv",
        output: `${p ? Number(p.row_count).toLocaleString() : '—'} filas × ${columns.length} columnas`,
        extra: `<div>Target: '${p ? p.target_column : (run ? run.target : 'target')}'</div><div>Modalidad: Tabular</div>`,
      },
      SchemaDetector: {
        name: "Detector de Esquema y Tarea",
        status: "Completado",
        description: "Inferencia autónoma de la modalidad del problema, tipo de tarea, tipos de datos y distribución del target.",
        input: `${columns.length} columnas`,
        output: run ? (run.task_type || "Clasificación Binaria") : "Clasificación",
        extra: `<div>Numéricas: ${numCols}</div><div>Categóricas: ${catCols}</div>`,
      },
      IDExclusion: {
        name: "Poda de Identificadores",
        status: "Completado",
        description: "Aislamiento automático de identificadores para prevenir fugas de datos (data leakage) y memorización.",
        input: `${columns.length} columnas`,
        output: `${columns.filter(c => !c.is_identifier).length} características predictivas`,
        extra: "<div>Regla: Descartar características con >99% de cardinalidad única</div>",
      },
      NumericalImputer: {
        name: "Imputador Numérico y StandardScaler",
        status: p ? "Configurado" : "Pendiente",
        description: "Imputación de valores nulos con mediana/media seguida de normalización estándar de características.",
        input: `${numCols} columnas numéricas`,
        output: `${numCols} características escaladas`,
        extra: "<div>Método: SimpleImputer + StandardScaler</div>",
      },
      CategoricalEncoder: {
        name: "Codificador Categórico y TargetAdapter",
        status: p ? "Configurado" : "Pendiente",
        description: "Codificación One-Hot y normalización alineada al target para columnas categóricas.",
        input: `${catCols} columnas categóricas`,
        output: `${catCols} características codificadas`,
        extra: "<div>Método: OneHotEncoder + Ordinal / TargetAdapter</div>",
      },
      ImageEncoderNode: {
        name: "ImageEncoderNode (Visión Multimodal)",
        status: imgCols > 0 ? "Activo" : "Omitido",
        description: "Extracción de embeddings de representación visual profunda usando timm / PyTorch, con fallback automático a mapas de características estándar.",
        input: `${imgCols} columnas de imagen / fuentes de imagen`,
        output: `${imgCols * 512 || 512} características densas de embedding`,
        extra: "<div>Motor: timm / PyTorch (Vision Transformer / ResNet / ConvNeXt)</div><div class='text-cyan-400'>División sin fugas garantizada</div>",
      },
      FeatureGenerator: {
        name: "FeatureGenerator (Propose ≠ Accept)",
        status: "Configurado",
        description: "Generador autónomo de hipótesis de interacción que produce características candidatas por pares.",
        input: `${columns.length} características base`,
        output: "Características empíricamente aceptadas",
        extra: "<div class='text-emerald-400'>Aceptadas: Validado por ganancia en CV</div><div class='text-rose-400'>Rechazadas: Degradan métrica o son colineales</div>",
      },
      Prediction: {
        name: "Predicción y Archivo de Envío",
        status: this.leaderboard.length > 0 ? "Listo" : "Pendiente de Entrenamiento",
        description: "Genera predicciones alineadas o artefacto de submission para registros de prueba / validación.",
        input: "Dataset / características de prueba",
        output: "submission.csv",
        extra: "<div>Formato: Predicciones alineadas de test</div>",
      },
    };

    if (details[node]) return details[node];

    // Check if it's a model in leaderboard
    const model = this.leaderboard.find(m => m.model_id === node);
    if (model) {
      return {
        name: `Modelo ${model.model_id.toUpperCase()}`,
        status: "Evaluado",
        description: `Modelo entrenado y evaluado mediante ${run ? (run.validation_strategy || '5-fold CV') : 'validación cruzada'}.`,
        input: "Conjunto de Características Activo",
        output: "Vector de probabilidades / predicciones Out-of-fold",
        extra: `<div>Score en CV: ${Number(model.score).toFixed(5)}</div><div>Tiempo: ${model.training_time_seconds ? Number(model.training_time_seconds).toFixed(2) + 's' : '—'}</div>`,
      };
    }

    return details["FeatureGenerator"];
  }

  _bindEvents() {
    this.container.querySelectorAll(".dag-node").forEach(nodeEl => {
      nodeEl.addEventListener("click", () => {
        const node = nodeEl.getAttribute("data-node");
        if (node) {
          this.selectedNode = node;
          this.render();
        }
      });
    });
  }

  destroy() {
    this.container = null;
  }
}
