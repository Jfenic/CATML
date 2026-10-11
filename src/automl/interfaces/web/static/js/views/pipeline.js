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
        <div class="text-sm font-medium font-sans">Building visual execution pipeline graph...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const state = store.getState();
      const runs = state.runs || [];
      this.activeRun = runs.find(r => r.id === state.activeRunId)
        || runs.find(r => isRunActive(r))
        || runs[0]
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
    const columns = p ? (p.columns || []) : [];
    const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float"))).length;
    const imgCols = columns.filter(c => c.is_image).length;
    const catCols = columns.filter(c => c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string") && !c.is_image).length;
    const idCols = columns.filter(c => c.is_identifier || c.catml_action === "Exclude");

    const topModels = this.leaderboard.slice(0, 3);
    const nodeDetails = this._getNodeDetails(this.selectedNode);

    const dsDisplayName = run ? (run.dataset_name || "Dataset") : (p ? p.name : "Dataset");
    const rowCountText = p && p.row_count ? `${Number(p.row_count).toLocaleString()} rows` : "Raw Data";
    const colCountText = p && p.column_count ? `${p.column_count} cols` : "Features";

    const hasModels = topModels.length > 0;
    const dagStatusText = hasModels
      ? "Pipeline Trained & Aligned"
      : (p ? "Schema Inferred" : "Pending Ingestion");
    const dagStatusBadgeClass = hasModels ? "badge-gain" : (p ? "badge-sys" : "badge-warn");

    this.container.innerHTML = `
      <div class="space-y-6">
        <div class="workbench-card p-4 flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <span class="text-[#4F67FF]">${icon("workflow", "icon-lg")}</span>
            <div>
              <h2 class="text-base font-semibold text-[#F7F8FA] font-sans">Visual Pipeline DAG — ${dsDisplayName}</h2>
              <p class="text-xs text-[#8B95A7] font-sans">Reproducible DAG for data ingestion, preprocessing, feature generation, and ensemble orchestration.</p>
            </div>
          </div>
          <span class="${dagStatusBadgeClass} text-xs px-2.5 py-0.5 rounded-md font-mono font-medium inline-flex items-center gap-1">${icon(hasModels ? "check" : "circle", "icon-sm")} <span>${dagStatusText}</span></span>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Graph Canvas (Left) -->
          <div class="lg:col-span-8 workbench-card p-6 flex flex-col items-center justify-center space-y-4 bg-slate-950/40 min-h-[480px]">
            <!-- Node: Dataset -->
            <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'Dataset' ? 'active' : ''}" data-node="Dataset">
              ${icon("database", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
              <div class="text-[10px] text-slate-400 uppercase font-mono">Input Source</div>
              <div class="text-xs font-bold text-slate-200">${dsDisplayName}</div>
              <div class="text-[10px] text-slate-500 font-mono">${rowCountText} × ${colCountText}</div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Node: SchemaDetector & ID Exclusion -->
            <div class="flex space-x-6">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'SchemaDetector' ? 'active' : ''}" data-node="SchemaDetector">
                ${icon("search", "icon-sm text-[#6956E8] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Inference</div>
                <div class="text-xs font-bold text-purple-300">SchemaDetector</div>
                <div class="text-[10px] text-slate-500 font-mono capitalize">${run ? (run.task_type || "Classification").replace("_", " ") : "Classification"}</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'IDExclusion' ? 'active' : ''}" data-node="IDExclusion">
                ${icon("shield-x", "icon-sm text-[#EF4444] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pruning</div>
                <div class="text-xs font-bold text-rose-300">ID Exclusion</div>
                <div class="text-[10px] text-slate-500 font-mono">${idCols.length > 0 ? `Drop '${idCols[0].name}'` : 'Zero Leaks'}</div>
              </div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Preprocessing: Numerical, Categorical & Vision -->
            <div class="flex flex-wrap justify-center gap-4">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'NumericalImputer' ? 'active' : ''}" data-node="NumericalImputer">
                ${icon("binary", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pipeline</div>
                <div class="text-xs font-bold text-indigo-300">Numerical Imputer</div>
                <div class="text-[10px] text-slate-500 font-mono">${numCols} Numerics</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'CategoricalEncoder' ? 'active' : ''}" data-node="CategoricalEncoder">
                ${icon("tags", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pipeline</div>
                <div class="text-xs font-bold text-indigo-300">Categorical Encoder</div>
                <div class="text-[10px] text-slate-500 font-mono">${catCols} Categoricals</div>
              </div>

              ${imgCols > 0 ? `
                <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'ImageEncoderNode' ? 'active' : ''}" data-node="ImageEncoderNode">
                  ${icon("image", "icon-sm text-[#53C8FF] mb-1.5 block mx-auto")}
                  <div class="text-[10px] text-slate-400 uppercase font-mono">Vision</div>
                  <div class="text-xs font-bold text-cyan-300">ImageEncoderNode</div>
                  <div class="text-[10px] text-cyan-400 font-mono">${imgCols} Image Features</div>
                </div>
              ` : ''}
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- FeatureGenerator (Key Node) -->
            <div class="dag-node text-center cursor-pointer border-purple-500/60 bg-purple-950/20 ${this.selectedNode === 'FeatureGenerator' ? 'active' : ''}" data-node="FeatureGenerator">
              ${icon("sparkles", "icon-sm text-[#6956E8] mb-1.5 block mx-auto")}
              <div class="text-[10px] text-purple-400 uppercase font-mono font-bold">Hypothesis-Driven</div>
              <div class="text-xs font-bold text-purple-200">FeatureGenerator</div>
              <div class="text-[10px] text-purple-300 font-mono">Propose ≠ Accept</div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Top Models -->
            <div class="flex space-x-4">
              ${topModels.length > 0 ? topModels.map(m => `
                <div class="dag-node text-center cursor-pointer ${this.selectedNode === m.model_id ? 'active' : ''}" data-node="${m.model_id}">
                  ${icon("brain", "icon-sm text-[#22C55E] mb-1.5 block mx-auto")}
                  <div class="text-[10px] text-slate-400 uppercase font-mono">Model</div>
                  <div class="text-xs font-bold text-slate-200 capitalize">${m.model_id}</div>
                  <div class="text-[10px] text-emerald-400 font-mono">${Number(m.score).toFixed(5)}</div>
                </div>
              `).join("") : `
                <div class="dag-node text-center cursor-pointer" data-node="Models">
                  ${icon("boxes", "icon-sm text-[#4F67FF] mb-1.5 block mx-auto")}
                  <div class="text-[10px] text-slate-400 uppercase font-mono">Models</div>
                  <div class="text-xs font-bold text-slate-200">Ensemble Candidates</div>
                  <div class="text-[10px] text-indigo-400 font-mono">Stratified CV</div>
                </div>
              `}
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Prediction / Submission Node -->
            <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'Prediction' ? 'active' : ''}" data-node="Prediction">
              ${icon("package", "icon-sm text-[#22C55E] mb-1.5 block mx-auto")}
              <div class="text-[10px] text-slate-400 uppercase font-mono">Output Artifact</div>
              <div class="text-xs font-bold text-emerald-300">submission.csv</div>
              <div class="text-[10px] text-slate-400 font-mono">Verified Format</div>
            </div>
          </div>

          <!-- Inspector Panel (Right) -->
          <div class="lg:col-span-4 workbench-card p-5 space-y-4">
            <div class="border-b border-slate-800 pb-3 flex items-center justify-between">
              <div>
                <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Node Inspector</span>
                <h3 class="text-base font-bold text-slate-100" id="inspectorNodeName">${nodeDetails.name}</h3>
              </div>
              <span class="badge-gain text-xs px-2 py-0.5 rounded font-mono" id="inspectorNodeStatus">${nodeDetails.status}</span>
            </div>

            <p class="text-xs text-slate-300 leading-relaxed" id="inspectorNodeDesc">
              ${nodeDetails.description}
            </p>

            <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-2 text-xs font-mono">
              <div class="text-slate-400 font-sans font-semibold text-[11px] uppercase">Node Contract</div>
              <div class="flex justify-between">
                <span class="text-slate-400">Input:</span>
                <span class="text-slate-200" id="inspectorInput">${nodeDetails.input}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-slate-400">Output:</span>
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
        name: `Input Source (${run ? run.dataset_name : 'Dataset'})`,
        status: "Ingested",
        description: "Raw tabular dataset loaded directly through DataSource port without framework pollution in domain.",
        input: run ? (run.dataset_path || "data/train.csv") : "data/train.csv",
        output: `${p ? Number(p.row_count).toLocaleString() : '—'} rows × ${columns.length} columns`,
        extra: `<div>Target: '${p ? p.target_column : (run ? run.target : 'target')}'</div><div>Modality: Tabular</div>`,
      },
      SchemaDetector: {
        name: "Schema & Task Detector",
        status: "Completed",
        description: "Autonomous inference of problem modality, task type, data types, and target distribution.",
        input: `${columns.length} columns`,
        output: run ? (run.task_type || "Binary Classification") : "Classification",
        extra: `<div>Numerical: ${numCols}</div><div>Categorical: ${catCols}</div>`,
      },
      IDExclusion: {
        name: "Identifier Pruning",
        status: "Completed",
        description: "Automatic identifier isolation to prevent data leakage and memorization.",
        input: `${columns.length} columns`,
        output: `${columns.filter(c => !c.is_identifier).length} predictive features`,
        extra: "<div>Rule: Drop features with >99% unique cardinality</div>",
      },
      NumericalImputer: {
        name: "Numerical Imputer & StandardScaler",
        status: p ? "Configured" : "Pending",
        description: "Missing value imputation with median/mean followed by standard feature normalization.",
        input: `${numCols} numeric columns`,
        output: `${numCols} scaled features`,
        extra: "<div>Method: SimpleImputer + StandardScaler</div>",
      },
      CategoricalEncoder: {
        name: "Categorical Encoder & TargetAdapter",
        status: p ? "Configured" : "Pending",
        description: "One-hot encoding and target-aligned normalization for categorical feature columns.",
        input: `${catCols} categorical columns`,
        output: `${catCols} encoded features`,
        extra: "<div>Method: OneHotEncoder + Ordinal / TargetAdapter</div>",
      },
      ImageEncoderNode: {
        name: "ImageEncoderNode (Multimodal Vision)",
        status: imgCols > 0 ? "Active" : "Bypassed",
        description: "Extracts deep visual representation embeddings using timm / PyTorch backends, with graceful fallback to standard feature maps.",
        input: `${imgCols} image path columns / raw image sources`,
        output: `${imgCols * 512 || 512} dense embedding features`,
        extra: "<div>Backend: timm / PyTorch (Vision Transformer / ResNet / ConvNeXt)</div><div class='text-cyan-400'>Zero-leakage split preserved</div>",
      },
      FeatureGenerator: {
        name: "FeatureGenerator (Propose ≠ Accept)",
        status: "Configured",
        description: "Autonomous interaction hypothesis generator producing candidate pairwise features.",
        input: `${columns.length} base features`,
        output: "Empirically accepted features",
        extra: "<div class='text-emerald-400'>Accepted: Verified by CV gain</div><div class='text-rose-400'>Rejected: Degrading or collinear</div>",
      },
      Prediction: {
        name: "Prediction & Submission",
        status: this.leaderboard.length > 0 ? "Ready" : "Pending Training",
        description: "Generates aligned predictions or submission artifact for holdout / test records.",
        input: "Test dataset / features",
        output: "submission.csv",
        extra: "<div>Format: Aligned test predictions</div>",
      },
    };

    if (details[node]) return details[node];

    // Check if it's a model in leaderboard
    const model = this.leaderboard.find(m => m.model_id === node);
    if (model) {
      return {
        name: `${model.model_id.toUpperCase()} Model`,
        status: "Evaluated",
        description: `Model trained and evaluated on ${run ? (run.validation_strategy || '5-fold CV') : 'cross-validation'}.`,
        input: "Active Feature Set",
        output: "Out-of-fold probability / prediction vector",
        extra: `<div>CV Score: ${Number(model.score).toFixed(5)}</div><div>Time: ${model.training_time_seconds ? Number(model.training_time_seconds).toFixed(2) + 's' : '—'}</div>`,
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
