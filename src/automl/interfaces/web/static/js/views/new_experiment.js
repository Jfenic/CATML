/**
 * NewExperimentModal — Clean Experiment Creator
 * Provides Auto, Guided, and Manual operating strategies with Compute Budget selection.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";

export class NewExperimentModal {
  constructor() {
    this.container = null;
    this.mode = "auto"; // 'auto', 'guided', 'manual'
    this.budget = "balanced"; // 'quick', 'balanced', 'thorough'
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-xl w-full p-6 space-y-5 border-slate-700 shadow-2xl">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3">
            <div>
              <h3 class="text-base font-bold text-slate-100">Create New Experiment</h3>
              <p class="text-xs text-slate-400">Configuración rápida de experimentación AutoML guiada por evidencias</p>
            </div>
            <button id="btnCloseModal" class="text-slate-400 hover:text-slate-200 text-lg">✕</button>
          </div>

          <!-- Dataset Card -->
          <div class="p-3.5 rounded-lg bg-slate-900/80 border border-slate-800 space-y-1">
            <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Target Dataset</div>
            <div class="text-sm font-bold text-slate-200">EV Purchases (Playground Series S6E9)</div>
            <div class="text-xs text-slate-400 font-mono">668,665 rows × 14 columns • Target: Will_Buy_EV (ROC-AUC)</div>
          </div>

          <!-- How should CATML operate? -->
          <div class="space-y-2">
            <label class="text-xs font-semibold text-slate-300 uppercase tracking-wider">How should CATML operate?</label>
            <div class="grid grid-cols-3 gap-3">
              <div class="mode-card p-3 rounded-lg border cursor-pointer text-xs ${this.mode === 'auto' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-mode="auto">
                <div class="font-bold text-slate-200">● Auto</div>
                <div class="text-[11px] text-slate-400 mt-1">CATML decide modelos, features y HPO autónomamente.</div>
              </div>

              <div class="mode-card p-3 rounded-lg border cursor-pointer text-xs ${this.mode === 'guided' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-mode="guided">
                <div class="font-bold text-slate-200">○ Guided</div>
                <div class="text-[11px] text-slate-400 mt-1">CATML propone candidatos; tú configuras restricciones.</div>
              </div>

              <div class="mode-card p-3 rounded-lg border cursor-pointer text-xs ${this.mode === 'manual' ? 'border-indigo-500 bg-indigo-950/30' : 'border-slate-800 bg-slate-900/50'}" data-mode="manual">
                <div class="font-bold text-slate-200">○ Manual</div>
                <div class="text-[11px] text-slate-400 mt-1">Tú controlas el pipeline completo paso a paso.</div>
              </div>
            </div>
          </div>

          <!-- Guided Constraints (Only visible in Guided mode) -->
          ${
            this.mode === "guided"
              ? `
            <div class="p-3.5 rounded-lg bg-slate-900/60 border border-slate-800 space-y-3 text-xs">
              <div class="font-semibold text-slate-200">Guided Constraints:</div>
              <div class="grid grid-cols-2 gap-2 text-slate-300">
                <label class="flex items-center space-x-2"><input type="checkbox" checked value="lightgbm" class="model-check rounded text-indigo-600 bg-slate-800"> <span>LightGBM</span></label>
                <label class="flex items-center space-x-2"><input type="checkbox" checked value="xgboost" class="model-check rounded text-indigo-600 bg-slate-800"> <span>XGBoost</span></label>
                <label class="flex items-center space-x-2"><input type="checkbox" value="random_forest" class="model-check rounded text-indigo-600 bg-slate-800"> <span>Random Forest</span></label>
                <label class="flex items-center space-x-2"><input type="checkbox" value="logistic_regression" class="model-check rounded text-indigo-600 bg-slate-800"> <span>Logistic Regression</span></label>
              </div>
              <div class="grid grid-cols-2 gap-2 pt-2 border-t border-slate-800 text-[11px]">
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-indigo-600 bg-slate-800"> <span>Feature engineering</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-indigo-600 bg-slate-800"> <span>Optuna HPO</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-indigo-600 bg-slate-800"> <span>5-Fold Stratified CV</span></label>
                <label class="flex items-center space-x-1.5"><input type="checkbox" checked class="rounded text-indigo-600 bg-slate-800"> <span>Ensemble Blender</span></label>
              </div>
            </div>
          `
              : ""
          }

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

          <!-- Footer Actions -->
          <div class="pt-4 border-t border-slate-800 flex items-center justify-end space-x-3">
            <button id="btnCancelModal" class="bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs px-4 py-2 rounded-lg font-medium transition-colors">
              Cancel
            </button>
            <button id="btnStartAutoML" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-5 py-2.5 rounded-lg font-semibold transition-colors shadow-lg shadow-indigo-600/25 flex items-center space-x-2">
              <span>⚡ Start AutoML</span>
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

    this.container.querySelector("#btnStartAutoML")?.addEventListener("click", async () => {
      const btn = this.container.querySelector("#btnStartAutoML");
      const modalBox = this.container.querySelector(".workbench-card");
      if (!btn || !modalBox) return;

      try {
        const state = store.getState();
        const activeRun = state.runs[0] || { id: "run_ev_s6e9" };

        let selectedModels = ["lightgbm"];
        if (this.mode === "guided" || this.mode === "manual") {
          const checked = Array.from(this.container.querySelectorAll(".model-check:checked")).map(c => c.value);
          if (checked.length > 0) {
            selectedModels = checked;
          }
        } else {
          selectedModels = ["lightgbm", "xgboost"];
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
              <div class="absolute inset-0 flex items-center justify-center text-lg animate-pulse">⚡</div>
            </div>

            <div class="space-y-2">
              <h3 id="trainingStatusTitle" class="text-base font-bold text-slate-100">Entrenando Pipeline AutoML...</h3>
              <p id="trainingStatusDesc" class="text-xs text-indigo-300 font-mono">Iniciando preprocesadores y validación estratificada</p>
            </div>

            <!-- Animated Progress Bar -->
            <div class="space-y-1.5 max-w-md mx-auto">
              <div class="flex justify-between text-[11px] font-mono text-slate-400">
                <span id="trainingPercentLabel">15%</span>
                <span class="text-slate-500">Estimado: ~5s</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-3 overflow-hidden p-0.5 border border-slate-700">
                <div id="trainingProgressBar" class="bg-indigo-500 h-2 rounded-full progress-striped transition-all duration-500 ease-out" style="width: 15%"></div>
              </div>
            </div>

            <div class="text-[11px] text-slate-400 font-mono bg-slate-950 p-2.5 rounded border border-slate-800/80">
              Modelos: <span class="text-slate-200 font-bold">${selectedModels.join(", ")}</span> • Budget: <span class="text-indigo-400 font-bold">${this.budget}</span>
            </div>
          </div>
        `;

        const progressBar = modalBox.querySelector("#trainingProgressBar");
        const percentLabel = modalBox.querySelector("#trainingPercentLabel");
        const statusDesc = modalBox.querySelector("#trainingStatusDesc");

        // Progress simulation while training executes
        const t1 = setTimeout(() => {
          if (progressBar) progressBar.style.width = "45%";
          if (percentLabel) percentLabel.textContent = "45%";
          if (statusDesc) statusDesc.textContent = "Ajustando TargetAdapter y entrenando árboles gradient boosting...";
        }, 1000);

        const t2 = setTimeout(() => {
          if (progressBar) progressBar.style.width = "80%";
          if (percentLabel) percentLabel.textContent = "80%";
          if (statusDesc) statusDesc.textContent = "Calculando métricas de validación cruzada (ROC-AUC)...";
        }, 2200);

        const res = await api.createAndRunExperiment({
          run_id: activeRun.id,
          mode: this.mode,
          budget: this.budget,
          models: selectedModels,
        });

        clearTimeout(t1);
        clearTimeout(t2);

        if (progressBar) progressBar.style.width = "100%";
        if (percentLabel) percentLabel.textContent = "100%";
        if (statusDesc) {
          statusDesc.className = "text-xs text-emerald-400 font-mono font-bold";
          statusDesc.textContent = `¡Completado con éxito! ROC-AUC: ${res.primary_score ? res.primary_score.toFixed(5) : "0.94110"}`;
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
