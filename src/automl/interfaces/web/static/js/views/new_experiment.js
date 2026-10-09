/**
 * NewExperimentModal — Clean Experiment Creator
 * Provides Auto, Guided, and Manual operating strategies with Compute Budget selection.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { isRunActive } from "../utils.js";

export class NewExperimentModal {
  constructor() {
    this.container = null;
    this.mode = "auto"; // 'auto', 'guided', 'manual'
    this.budget = "balanced"; // 'quick', 'balanced', 'thorough'
    this.validationStrategy = "stratified_kfold";
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    const state = store.getState();
    const runs = state.runs || [];
    const activeRun = runs.find(r => isRunActive(r)) || runs[0] || null;
    const recentDs = (state.overview && state.overview.recent_datasets && state.overview.recent_datasets[0]) || null;
    const customFeatures = state.customFeatures || null;

    const datasetName = activeRun ? (activeRun.dataset_name || activeRun.id) : (recentDs ? recentDs.name : "No Dataset Registered");
    const targetCol = activeRun ? (activeRun.target || "Target") : (recentDs ? (recentDs.target || recentDs.target_column || "Target") : "None");
    const metricName = activeRun ? (activeRun.metric || "ROC-AUC") : "ROC-AUC";
    const taskType = activeRun ? (activeRun.task_type || "binary_classification").replace("_", " ") : "Classification";

    const hasTarget = Boolean(activeRun || recentDs);

    this.container.innerHTML = `
      <div class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-xl w-full p-6 space-y-5 border-slate-700 shadow-2xl">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 class="text-base font-bold text-[#F7F8FA] font-sans">Create New Experiment</h3>
              <p class="text-xs text-[#8B95A7] font-sans">Evidence-guided rapid AutoML experiment configuration</p>
            </div>
            <button id="btnCloseModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors">${icon("x", "icon-sm")}</button>
          </div>

          <!-- Dataset Card -->
          <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-1">
            <div class="text-[10px] uppercase font-semibold text-[#8B95A7] tracking-wider font-sans">Target Dataset</div>
            <div class="text-sm font-bold text-[#F7F8FA] font-mono">${datasetName}</div>
            <div class="text-xs text-[#8B95A7] font-sans">Task: <span class="capitalize text-[#F7F8FA]">${taskType}</span> • Target: <span class="text-[#4F67FF] font-mono font-bold">${targetCol}</span> • Metric: <span class="text-[#22C55E] font-mono font-bold">${metricName}</span></div>
          </div>

          ${!hasTarget ? `
            <div class="p-3 rounded-lg bg-amber-950/30 border border-amber-800/60 text-xs text-amber-300 flex items-center justify-between font-sans">
              <span>No dataset is currently registered in this workspace.</span>
              <button id="btnGoToDatasets" class="font-bold underline text-amber-200 hover:text-white">Register Dataset →</button>
            </div>
          ` : ""}

          <!-- How should CATML operate? -->
          <div class="space-y-2">
            <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">How should CATML operate?</label>
            <div class="grid grid-cols-3 gap-3">
              <div class="mode-card p-3 rounded-xl border cursor-pointer text-xs ${this.mode === 'auto' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-mode="auto">
                <div class="flex items-center gap-1.5 font-bold text-[#F7F8FA] font-sans">${icon("zap", "icon-sm text-[#4F67FF]")} <span>Auto</span></div>
                <div class="text-[11px] text-[#8B95A7] mt-1 font-sans">Autonomous model selection, feature engineering, and Bayesian HPO.</div>
              </div>

              <div class="mode-card p-3 rounded-xl border cursor-pointer text-xs ${this.mode === 'guided' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-mode="guided">
                <div class="flex items-center gap-1.5 font-bold text-[#F7F8FA] font-sans">${icon("compass", "icon-sm text-[#6956E8]")} <span>Guided</span></div>
                <div class="text-[11px] text-[#8B95A7] mt-1 font-sans">CATML proposes candidates; you configure constraints and models.</div>
              </div>

              <div class="mode-card p-3 rounded-xl border cursor-pointer text-xs ${this.mode === 'manual' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-mode="manual">
                <div class="flex items-center gap-1.5 font-bold text-[#F7F8FA] font-sans">${icon("sliders-horizontal", "icon-sm text-[#8B95A7]")} <span>Manual</span></div>
                <div class="text-[11px] text-[#8B95A7] mt-1 font-sans">Full step-by-step pipeline control and manual selection.</div>
              </div>
            </div>
          </div>

          <!-- Guided Constraints (Only visible in Guided mode) -->
          ${
            this.mode === "guided"
              ? `
            <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-3 text-xs font-sans">
              <div class="font-semibold text-[#F7F8FA]">Algorithmic Model Families:</div>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[#8B95A7]">
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked value="lightgbm" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">LightGBM</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked value="xgboost" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">XGBoost</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked value="catboost" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">CatBoost</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" value="extra_trees" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Extra Trees</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" value="mlp" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">MLP Neural</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" value="random_forest" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Random Forest</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" value="logistic_regression" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Logistic/Ridge</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" value="voting_ensemble" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Ensemble Blender</span></label>
                <label class="flex items-center space-x-1.5" title="Vision Feature Extraction + MLP Head"><input type="checkbox" value="vision" class="model-check rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span class="text-[#F7F8FA]">Vision Feature MLP</span></label>
              </div>
              <div class="grid grid-cols-2 gap-2 pt-2 border-t border-[#252C38] text-[11px] text-[#8B95A7]">
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Feature engineering</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Optuna Bayesian HPO</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Stratified Cross-Validation</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]"> <span>Empirical Propose ≠ Accept</span></label>
              </div>
            </div>
          `
              : ""
          }

          <!-- Feature Set Selection -->
          <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-2 text-xs font-sans">
            <div class="flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-[#4F67FF]">${icon("sparkles", "icon-sm")}</span>
                <span class="font-semibold text-[#F7F8FA]">Feature Set Selection:</span>
              </div>
              <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono font-bold">${customFeatures && customFeatures.length > 0 ? `${customFeatures.length} Custom Features` : 'All Recommended'}</span>
            </div>
            ${customFeatures && customFeatures.length > 0 ? `
              <div class="flex flex-wrap gap-1 max-h-20 overflow-y-auto p-1.5 bg-[#090C12] rounded-lg border border-[#252C38]">
                ${customFeatures.map(f => `<span class="bg-[#151B26] text-[#4F67FF] border border-[#252C38] px-1.5 py-0.5 rounded text-[10px] font-mono">${f}</span>`).join("")}
              </div>
            ` : `
              <div class="text-[11px] text-[#8B95A7]">Using all features selected and recommended by statistical profiling.</div>
            `}
          </div>

          <!-- Compute Budget -->
          <div class="space-y-2">
            <label class="text-xs font-semibold text-slate-300 uppercase tracking-wider">Compute Budget</label>
            <div class="grid grid-cols-3 gap-3">
              <div class="budget-card p-2.5 rounded-lg border text-center cursor-pointer text-xs ${this.budget === 'quick' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-budget="quick">
                <div class="font-bold text-slate-200">Quick</div>
                <div class="text-[10px] text-slate-400">~10s • Baselines</div>
              </div>

              <div class="budget-card p-2.5 rounded-lg border text-center cursor-pointer text-xs ${this.budget === 'balanced' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-budget="balanced">
                <div class="font-bold text-slate-200">Balanced</div>
                <div class="text-[10px] text-slate-400">~1m • LightGBM + HPO</div>
              </div>

              <div class="budget-card p-2.5 rounded-lg border text-center cursor-pointer text-xs ${this.budget === 'thorough' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-budget="thorough">
                <div class="font-bold text-slate-200">Thorough</div>
                <div class="text-[10px] text-slate-400">~5m • Full Ensemble</div>
              </div>
            </div>
          </div>

          <!-- Validation Strategy -->
          <div class="space-y-2">
            <div class="flex items-center justify-between">
              <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">Validation Strategy</label>
              <span class="text-[10px] text-[#4F67FF] font-mono">CV Splitting</span>
            </div>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
              <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'stratified_kfold' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="stratified_kfold">
                <div class="font-bold text-[#F7F8FA] font-sans">Stratified K-Fold</div>
                <div class="text-[10px] text-[#8B95A7] font-sans">Balanced classes</div>
              </div>

              <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'kfold' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="kfold">
                <div class="font-bold text-[#F7F8FA] font-sans">Standard K-Fold</div>
                <div class="text-[10px] text-[#8B95A7] font-sans">Shuffled 5 folds</div>
              </div>

              <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'time_series' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="time_series">
                <div class="font-bold text-[#F7F8FA] font-sans">Time-Series</div>
                <div class="text-[10px] text-[#8B95A7] font-sans">Temporal splits</div>
              </div>

              <div class="val-strategy-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${this.validationStrategy === 'holdout' ? 'border-[#4F67FF] bg-[#151B26]' : 'border-[#252C38] bg-[#090C12]'}" data-strategy="holdout">
                <div class="font-bold text-[#F7F8FA] font-sans">Holdout (80/20)</div>
                <div class="text-[10px] text-[#8B95A7] font-sans">Fast single split</div>
              </div>
            </div>
          </div>

          <!-- Footer Actions -->
          <div class="pt-4 border-t border-[#27272e] flex items-center justify-end space-x-3">
            <button id="btnCancelModal" class="btn-ghost">
              CANCEL
            </button>
            <button id="btnStartAutoML" ${!hasTarget ? "disabled" : ""} class="btn-signal ${!hasTarget ? 'opacity-50 cursor-not-allowed' : ''}">
              <span class="inline-flex items-center gap-1.5">${icon("play", "icon-sm")} <span>START TRAINING</span></span>
            </button>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _bindEvents() {
    this.container.querySelector("#btnCloseModal")?.addEventListener("click", () => this.destroy());
    this.container.querySelector("#btnCancelModal")?.addEventListener("click", () => this.destroy());
    this.container.querySelector("#btnGoToDatasets")?.addEventListener("click", () => {
      this.destroy();
      store.setNav("datasets");
    });

    this.container.querySelectorAll(".mode-card").forEach(el => {
      el.addEventListener("click", () => {
        this.mode = el.getAttribute("data-mode");
        this.render();
      });
    });

    this.container.querySelectorAll(".budget-card").forEach(el => {
      el.addEventListener("click", () => {
        this.budget = el.getAttribute("data-budget");
        this.render();
      });
    });

    this.container.querySelectorAll(".val-strategy-card").forEach(el => {
      el.addEventListener("click", () => {
        this.validationStrategy = el.getAttribute("data-strategy");
        this.render();
      });
    });

    this.container.querySelector("#btnStartAutoML")?.addEventListener("click", async () => {
      const btn = this.container.querySelector("#btnStartAutoML");
      const modalBox = this.container.querySelector(".workbench-card");
      if (!btn || !modalBox) return;

      try {
        const state = store.getState();
        const activeRun = state.runs.find(r => isRunActive(r)) || state.runs[0] || null;
        if (!activeRun) {
          alert("No active AutoML run found. Please create a run or register a dataset first.");
          return;
        }

        let selectedModels = ["lightgbm"];
        if (this.mode === "guided" || this.mode === "manual") {
          const checked = Array.from(this.container.querySelectorAll(".model-check:checked")).map(c => c.value);
          if (checked.length > 0) {
            selectedModels = checked;
          }
        } else {
          selectedModels = ["lightgbm", "xgboost", "catboost"];
        }

        // Render Live Animated Training Overlay with Spinner and Progress Bar
        modalBox.innerHTML = `
          <div class="p-8 text-center space-y-6">
            <div class="relative w-20 h-20 mx-auto">
              <!-- Outer glowing rotating ring -->
              <svg class="animate-spin w-20 h-20 text-indigo-500" viewBox="0 0 24 24" fill="none">
                <circle class="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3"></circle>
                <path class="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <!-- Center pulse icon -->
              <div class="absolute inset-0 flex items-center justify-center text-[#4F67FF] animate-pulse">${icon("zap", "icon-lg", 24)}</div>
            </div>

            <div class="space-y-2">
              <h3 id="trainingStatusTitle" class="text-base font-bold text-slate-100">Training AutoML Pipeline...</h3>
              <p id="trainingStatusDesc" class="text-xs text-indigo-300 font-mono">Waiting for worker</p>
            </div>

            <!-- Animated Progress Bar -->
            <div class="space-y-1.5 max-w-md mx-auto">
              <div class="flex justify-between text-[11px] font-mono text-slate-400">
                <span id="trainingPercentLabel">0%</span>
                <span class="text-slate-500">Verified progress</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden p-0.5 border border-slate-700">
                <div id="trainingProgressBar" class="bg-indigo-500 h-2 rounded-full progress-striped transition-all duration-500 ease-out" style="width: 0%"></div>
              </div>
            </div>

            <div class="text-[11px] text-slate-400 font-mono bg-slate-950 p-2.5 rounded border border-slate-800/80">
              Models: <span class="text-slate-200 font-bold">${selectedModels.join(", ")}</span> • Budget: <span class="text-indigo-400 font-bold">${this.budget}</span> • Strategy: <span class="text-[#4F67FF] font-bold">${this.validationStrategy}</span>
            </div>
          </div>
        `;

        const progressBar = modalBox.querySelector("#trainingProgressBar");
        const percentLabel = modalBox.querySelector("#trainingPercentLabel");
        const statusDesc = modalBox.querySelector("#trainingStatusDesc");

        const res = await api.createAndRunExperiment({
          run_id: activeRun.id,
          mode: this.mode,
          budget: this.budget,
          models: selectedModels,
          feature_names: customFeatures && customFeatures.length > 0 ? customFeatures : undefined,
          validation_strategy: this.validationStrategy,
        }, job => {
          const percent = job.total ? Math.round(job.completed * 100 / job.total) : 0;
          if (progressBar) progressBar.style.width = `${percent}%`;
          if (percentLabel) percentLabel.textContent = `${percent}%`;
          if (statusDesc) statusDesc.textContent = `${job.status}: ${job.message}`;
        });

        if (progressBar) progressBar.style.width = "100%";
        if (percentLabel) percentLabel.textContent = "100%";
        if (statusDesc) {
          statusDesc.className = "text-xs text-emerald-400 font-mono font-bold";
          statusDesc.textContent = `Completed! ${res.primary_metric}: ${res.primary_score.toFixed(5)}`;
        }

        setTimeout(() => {
          this.destroy();
          store.setNav("studio");
        }, 800);

      } catch (err) {
        alert("Error launching experiment: " + err.message);
        this.destroy();
      }
    });
  }

  destroy() {
    if (this.container) {
      this.container.innerHTML = "";
    }
  }
}
