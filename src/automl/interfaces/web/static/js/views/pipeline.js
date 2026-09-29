/**
 * PipelineView — Visual Pipeline DAG
 * Interactive directed acyclic graph of the complete CATML end-to-end flow.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";

export class PipelineView {
  constructor() {
    this.container = null;
    this.selectedNode = "FeatureGenerator";
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    const nodeDetails = this._getNodeDetails(this.selectedNode);

    this.container.innerHTML = `
      <div class="space-y-6">
        <div class="workbench-card p-4 bg-slate-900/80 border-slate-800 flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <span class="text-indigo-400 font-bold text-lg">◇</span>
            <div>
              <h2 class="text-base font-bold text-slate-100">Visual Pipeline DAG (Execution Graph)</h2>
              <p class="text-xs text-slate-400">Grafo reproducible de ingestión, preprocesamiento, generación de features y blending</p>
            </div>
          </div>
          <span class="badge-gain text-xs px-3 py-1 rounded-full font-mono font-bold">DAG Validated</span>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Graph Canvas (Left) -->
          <div class="lg:col-span-8 workbench-card p-6 flex flex-col items-center justify-center space-y-4 bg-slate-950/40 min-h-[480px]">
            <!-- Node: train.csv -->
            <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'Dataset' ? 'active' : ''}" data-node="Dataset">
              <div class="text-[10px] text-slate-400 uppercase font-mono">Input Source</div>
              <div class="text-xs font-bold text-slate-200">train.csv</div>
              <div class="text-[10px] text-slate-500 font-mono">668k rows × 14 cols</div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Node: SchemaDetector & ID Exclusion -->
            <div class="flex space-x-6">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'SchemaDetector' ? 'active' : ''}" data-node="SchemaDetector">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Inference</div>
                <div class="text-xs font-bold text-purple-300">SchemaDetector</div>
                <div class="text-[10px] text-slate-500 font-mono">Binary Clf</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'IDExclusion' ? 'active' : ''}" data-node="IDExclusion">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pruning</div>
                <div class="text-xs font-bold text-rose-300">ID Exclusion</div>
                <div class="text-[10px] text-slate-500 font-mono">Drop 'id' (99.9%)</div>
              </div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Preprocessing: Numerical & Categorical -->
            <div class="flex space-x-6">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'NumericalImputer' ? 'active' : ''}" data-node="NumericalImputer">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pipeline</div>
                <div class="text-xs font-bold text-indigo-300">Numerical Imputer</div>
                <div class="text-[10px] text-slate-500 font-mono">Median / Scaler</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'CategoricalEncoder' ? 'active' : ''}" data-node="CategoricalEncoder">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Pipeline</div>
                <div class="text-xs font-bold text-indigo-300">Target / Ordinal Enc</div>
                <div class="text-[10px] text-slate-500 font-mono">7 Categoricals</div>
              </div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- FeatureGenerator (Key Node) -->
            <div class="dag-node text-center cursor-pointer border-purple-500/60 bg-purple-950/20 ${this.selectedNode === 'FeatureGenerator' ? 'active' : ''}" data-node="FeatureGenerator">
              <div class="text-[10px] text-purple-400 uppercase font-mono font-bold">Hypothesis-Driven</div>
              <div class="text-xs font-bold text-purple-200">FeatureGenerator</div>
              <div class="text-[10px] text-purple-300 font-mono">Propose ≠ Accept</div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Model Trio -->
            <div class="flex space-x-4">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'LightGBM' ? 'active' : ''}" data-node="LightGBM">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Fold Models</div>
                <div class="text-xs font-bold text-slate-200">LightGBM</div>
                <div class="text-[10px] text-emerald-400 font-mono">0.94110</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'CatBoost' ? 'active' : ''}" data-node="CatBoost">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Fold Models</div>
                <div class="text-xs font-bold text-slate-200">CatBoost HPO</div>
                <div class="text-[10px] text-emerald-400 font-mono font-bold">0.94582</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'XGBoost' ? 'active' : ''}" data-node="XGBoost">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Fold Models</div>
                <div class="text-xs font-bold text-slate-200">XGBoost</div>
                <div class="text-[10px] text-slate-500 font-mono">Queued</div>
              </div>
            </div>

            <div class="dag-node-connector h-6"></div>

            <!-- Blend & Prediction -->
            <div class="flex space-x-6">
              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'Blend' ? 'active' : ''}" data-node="Blend">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Meta-Ensemble</div>
                <div class="text-xs font-bold text-indigo-300">Weighted Blender</div>
                <div class="text-[10px] text-emerald-400 font-mono font-bold">0.94621</div>
              </div>

              <div class="dag-node text-center cursor-pointer ${this.selectedNode === 'Prediction' ? 'active' : ''}" data-node="Prediction">
                <div class="text-[10px] text-slate-400 uppercase font-mono">Output Artifact</div>
                <div class="text-xs font-bold text-emerald-300">submission.csv</div>
                <div class="text-[10px] text-slate-500 font-mono">286,571 predictions</div>
              </div>
            </div>
          </div>

          <!-- Node Inspector (Right) -->
          <div class="lg:col-span-4 workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <span class="text-sm font-semibold text-slate-200">${nodeDetails.name}</span>
              <span class="badge-gain text-xs px-2 py-0.5 rounded font-mono">${nodeDetails.status}</span>
            </div>

            <div class="p-5 space-y-4 flex-1 text-xs">
              <div>
                <span class="text-slate-400 uppercase font-semibold tracking-wider text-[10px]">Description</span>
                <p class="text-slate-300 mt-1">${nodeDetails.description}</p>
              </div>

              <div class="grid grid-cols-2 gap-3 pt-2">
                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div class="text-slate-400 text-[10px]">Input Features</div>
                  <div class="text-base font-bold text-slate-200 font-mono mt-0.5">${nodeDetails.input}</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <div class="text-slate-400 text-[10px]">Output Features</div>
                  <div class="text-base font-bold text-indigo-400 font-mono mt-0.5">${nodeDetails.output}</div>
                </div>
              </div>

              ${
                nodeDetails.extra
                  ? `
                <div class="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-2 font-mono text-[11px]">
                  <div class="text-slate-400 font-sans font-semibold">Transformations Details:</div>
                  ${nodeDetails.extra}
                </div>
              `
                  : ""
              }

              <div class="pt-4 border-t border-slate-800 flex justify-end">
                <button class="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs px-3.5 py-1.5 rounded font-medium transition-colors">
                  Inspect Transformations Code
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _getNodeDetails(node) {
    const details = {
      Dataset: {
        name: "train.csv Ingestion",
        status: "Completed",
        description: "Raw competition dataset loaded into memory using streaming chunk verification.",
        input: "0 (Raw file)",
        output: "14 columns",
        extra: "<div>File: competitions/s6e9/train.csv</div><div>Rows: 668,665</div><div>Size: ~48 MB</div>",
      },
      SchemaDetector: {
        name: "SchemaDetector",
        status: "Completed",
        description: "Infers target column types, missingness heuristics, and problem task classification.",
        input: "14 columns",
        output: "1 Target + 13 Predictors",
        extra: "<div>Task: Binary Classification</div><div>Metric: ROC-AUC</div>",
      },
      IDExclusion: {
        name: "ID Exclusion Filter",
        status: "Completed",
        description: "Automatic identifier isolation to prevent data leakage.",
        input: "13 columns",
        output: "12 features (id dropped)",
        extra: "<div>Column: 'id'</div><div>Cardinality: 99.999%</div><div>Action: Excluded from active set</div>",
      },
      NumericalImputer: {
        name: "Numerical Imputer & Scaler",
        status: "Completed",
        description: "Median imputation for missing values followed by RobustScaler.",
        input: "6 numeric columns",
        output: "6 numeric columns",
        extra: "<div>Imputed: Income (0.3% missing)</div><div>Scaler: RobustScaler</div>",
      },
      CategoricalEncoder: {
        name: "Categorical Encoder",
        status: "Completed",
        description: "Multi-level encoding: Ordinal mapping for tree models and smoothed target encoding.",
        input: "7 categorical columns",
        output: "7 encoded columns",
        extra: "<div>TargetAdapter: ['No', 'Yes'] -> [0, 1]</div><div>Encoder: Ordinal & Target Encoding</div>",
      },
      FeatureGenerator: {
        name: "FeatureGenerator (Propose ≠ Accept)",
        status: "Completed",
        description: "Autonomous interaction hypothesis generator producing candidate pairwise features.",
        input: "13 features",
        output: "21 features (validated)",
        extra: "<div>Generated: 8 interaction candidates</div><div class='text-emerald-400'>Accepted: 5 (passed ROC-AUC gain threshold)</div><div class='text-rose-400'>Rejected: 3 (failed holdout CV)</div>",
      },
      LightGBM: {
        name: "LightGBM Gradient Boosting",
        status: "Completed",
        description: "Fast baseline gradient booster trained across 5 stratified folds.",
        input: "18 features",
        output: "Out-of-fold Probabilities",
        extra: "<div>CV Score: 0.94110 ROC-AUC</div><div>Training time: 5.03s</div>",
      },
      CatBoost: {
        name: "CatBoost HPO",
        status: "Completed",
        description: "Optimal tree model tuned with Optuna TPE over 60 trials.",
        input: "18 features",
        output: "Out-of-fold Probabilities",
        extra: "<div>CV Score: 0.94582 ROC-AUC</div><div>Best trial: #37</div><div>Depth: 8, LR: 0.031</div>",
      },
      XGBoost: {
        name: "XGBoost HPO",
        status: "Queued",
        description: "Extreme Gradient Boosting regularization exploration.",
        input: "18 features",
        output: "Out-of-fold Probabilities",
        extra: "<div>Status: Waiting for CatBoost HPO completion</div>",
      },
      Blend: {
        name: "Weighted Blender Ensemble",
        status: "Completed",
        description: "Meta-learner blending out-of-fold probability vectors to minimize variance.",
        input: "2 Model Preds",
        output: "1 Ensembled Vector",
        extra: "<div>Weights: LightGBM (0.35), CatBoost (0.65)</div><div>Ensemble CV: 0.94621 ROC-AUC</div>",
      },
      Prediction: {
        name: "Prediction & Kaggle Submission",
        status: "Ready",
        description: "Generates aligned submission.csv for 286,571 test records.",
        input: "test.csv (286,571)",
        output: "submission.csv",
        extra: "<div>Valid rows: 286,571</div><div>ID column: 'id'</div><div>Proba range: [0.0001, 0.9998]</div>",
      },
    };
    return details[node] || details["FeatureGenerator"];
  }

  _bindEvents() {
    this.container.querySelectorAll(".dag-node").forEach(nodeEl => {
      nodeEl.addEventListener("click", () => {
        this.selectedNode = nodeEl.getAttribute("data-node");
        this.render();
      });
    });
  }

  destroy() {
    this.container = null;
  }
}
