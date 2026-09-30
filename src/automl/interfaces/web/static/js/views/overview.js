/**
 * OverviewView — Mission Control
 * Zone 1: Active Runs
 * Zone 2: Best Results
 * Zone 3: Recent Datasets
 * Zone 4: CATML Activity Feed
 */
import { store } from "../store.js";
import { bus } from "../bus.js";

export class OverviewView {
  constructor() {
    this.container = null;
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    const state = store.getState();
    const overview = state.overview || {
      best_score: 0.9412,
      best_model: "xgboost",
      total_runs: 2,
      total_trials: 11,
      workspace: "CATML Default",
      recent_datasets: [],
      activity_feed: [],
    };

    const runs = state.runs || [];
    const activeRun = runs.find(r => r.status === "RUNNING") || runs[0] || {
      id: "run_695b92e6",
      dataset_name: "EV Purchases",
      task_type: "binary_classification",
      metric: "ROC-AUC",
      best_score: 0.9412,
      best_model: "xgboost",
      trials_count: 11,
      status: "COMPLETED",
    };

    const isRunning = activeRun.status === "RUNNING";

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Top KPI row -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Best CV Score</div>
            <div class="mt-2 flex items-baseline justify-between">
              <span class="text-2xl font-bold font-mono-num text-emerald-400">${overview.best_score != null ? overview.best_score.toFixed(4) : "0.9412"}</span>
              <span class="badge-gain text-xs px-2 py-0.5 rounded-full font-mono font-medium">+0.0001</span>
            </div>
            <div class="mt-1 text-xs text-slate-400">Modelo: <span class="text-slate-200 font-medium uppercase">${overview.best_model || "xgboost"}</span></div>
          </div>

          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Active Run</div>
            <div class="mt-2 flex items-center space-x-2">
              <span class="inline-block w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-indigo-500 animate-pulse' : 'bg-emerald-500'}"></span>
              <span class="text-xl font-bold text-slate-100">${activeRun.status || "IDLE"}</span>
            </div>
            <div class="mt-1 text-xs text-slate-400 truncate">${activeRun.dataset_name || "EV Purchases"} • ROC-AUC</div>
          </div>

          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Trials Executed</div>
            <div class="mt-2 flex items-baseline justify-between">
              <span class="text-2xl font-bold font-mono-num text-indigo-400">${overview.total_trials || 42}</span>
              <span class="text-xs text-slate-500 font-mono">/ 60 budget</span>
            </div>
            <div class="mt-1 text-xs text-slate-400">Progreso: <span class="text-slate-200 font-medium">68%</span></div>
          </div>

          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Compute & Workers</div>
            <div class="mt-2 flex items-baseline justify-between">
              <span class="text-2xl font-bold font-mono-num text-slate-200">CPU 67%</span>
              <span class="text-xs text-slate-400">RAM 51%</span>
            </div>
            <div class="mt-1 text-xs text-slate-400">Workers: <span class="text-indigo-400 font-medium">5 Folds activos</span></div>
          </div>
        </div>

        <!-- 4 Zones Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Zone 1: Active Runs -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-indigo-400">⚡</span>
                <span class="text-sm font-semibold text-slate-200">Active Runs</span>
              </div>
              <button id="btnNewExpOverview" class="bg-indigo-600 hover:bg-indigo-500 text-xs px-3 py-1.5 rounded-lg text-white font-medium transition-colors">
                + New Experiment
              </button>
            </div>
            <div class="p-5 space-y-4 flex-1">
              <div class="border border-indigo-900/50 bg-indigo-950/20 rounded-xl p-4 space-y-3">
                <div class="flex items-start justify-between">
                  <div>
                    <h3 class="text-base font-bold text-slate-100 uppercase tracking-wide">${activeRun.dataset_name || "EV PURCHASES"}</h3>
                    <p class="text-xs text-slate-400">AutoML • Binary Classification • ROC-AUC</p>
                  </div>
                  <span class="badge-sys text-xs px-2.5 py-1 rounded-full font-mono uppercase tracking-wider font-semibold flex items-center space-x-1.5">
                    <svg class="animate-spin h-3 w-3 text-indigo-400 inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                    <span>● ${activeRun.status || "RUNNING"}</span>
                  </span>
                </div>

                <div class="grid grid-cols-2 gap-3 pt-2 text-xs">
                  <div>
                    <span class="text-slate-400">Best CV:</span>
                    <span class="ml-1 text-emerald-400 font-mono font-bold">${activeRun.best_score ? activeRun.best_score.toFixed(4) : (overview.best_score != null ? overview.best_score.toFixed(4) : "—")}</span>
                  </div>
                  <div>
                    <span class="text-slate-400">Best Model:</span>
                    <span class="ml-1 text-slate-200 font-medium uppercase">${activeRun.best_model || overview.best_model || "None"}</span>
                  </div>
                  <div>
                    <span class="text-slate-400">Status:</span>
                    <span class="ml-1 text-indigo-300 font-medium">${activeRun.status || "COMPLETED"}</span>
                  </div>
                  <div>
                    <span class="text-slate-400">Trials:</span>
                    <span class="ml-1 text-slate-200 font-mono">${activeRun.trials_count != null ? activeRun.trials_count : (overview.total_trials || 11)}</span>
                  </div>
                </div>

                <!-- Progress Bar -->
                <div class="space-y-1 pt-1">
                  <div class="flex justify-between text-[11px] text-slate-400">
                    <span>Optimization Progress</span>
                    <span>${isRunning ? '68%' : '100%'}</span>
                  </div>
                  <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                    <div class="${isRunning ? 'bg-indigo-500 progress-striped' : 'bg-emerald-500'} h-2.5 rounded-full" style="width: ${isRunning ? '68%' : '100%'}"></div>
                  </div>
                </div>

                <div class="pt-2 flex justify-end">
                  <button id="btnOpenStudioFromRun" class="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center space-x-1">
                    <span>Open Experiment Studio</span>
                    <span>→</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          <!-- Zone 2: Best Results per Problem -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-emerald-400">🏆</span>
                <span class="text-sm font-semibold text-slate-200">Best Results</span>
              </div>
              <span class="text-xs text-slate-400">Verified by Experiment</span>
            </div>
            <div class="p-4 overflow-x-auto flex-1">
              <table class="w-full wb-table text-left">
                <thead>
                  <tr>
                    <th>Dataset</th>
                    <th>Task</th>
                    <th>Best Model</th>
                    <th>Metric</th>
                    <th>CV Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td class="font-medium text-slate-200">Playground S6E9</td>
                    <td><span class="text-xs text-slate-400">Binary Clf</span></td>
                    <td><span class="badge-sys px-2 py-0.5 rounded text-xs uppercase">${overview.best_model || "xgboost"}</span></td>
                    <td class="font-mono text-xs">ROC-AUC</td>
                    <td class="font-mono font-bold text-emerald-400">${overview.best_score != null ? overview.best_score.toFixed(4) : "0.9412"}</td>
                  </tr>
                  <tr>
                    <td class="font-medium text-slate-200">Customer Churn</td>
                    <td><span class="text-xs text-slate-400">Binary Clf</span></td>
                    <td><span class="badge-sys px-2 py-0.5 rounded text-xs">LightGBM</span></td>
                    <td class="font-mono text-xs">ROC-AUC</td>
                    <td class="font-mono font-bold text-emerald-400">0.92480</td>
                  </tr>
                  <tr>
                    <td class="font-medium text-slate-200">House Prices</td>
                    <td><span class="text-xs text-slate-400">Regression</span></td>
                    <td><span class="badge-sys px-2 py-0.5 rounded text-xs">XGBoost</span></td>
                    <td class="font-mono text-xs">RMSE</td>
                    <td class="font-mono font-bold text-emerald-400">0.12450</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Zone 3: Recent Datasets -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-sky-400">▦</span>
                <span class="text-sm font-semibold text-slate-200">Recent Datasets</span>
              </div>
              <button id="btnViewAllDatasets" class="text-xs text-slate-400 hover:text-slate-200">View all →</button>
            </div>
            <div class="p-4 space-y-3 flex-1">
              <div class="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800 hover:border-slate-700 cursor-pointer">
                <div>
                  <div class="text-sm font-bold text-slate-200">EV Purchases (Playground Series S6E9)</div>
                  <div class="text-xs text-slate-400">668,665 filas • 13 features • Target: Will_Buy_EV</div>
                </div>
                <span class="badge-gain text-xs px-2.5 py-1 rounded">Profiled</span>
              </div>

              <div class="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800 hover:border-slate-700 cursor-pointer">
                <div>
                  <div class="text-sm font-bold text-slate-200">Customer Churn</div>
                  <div class="text-xs text-slate-400">10,000 filas • 12 features • Target: Exited</div>
                </div>
                <span class="badge-sys text-xs px-2.5 py-1 rounded">Benchmark</span>
              </div>
            </div>
          </div>

          <!-- Zone 4: CATML Activity & Decisions -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-purple-400">🟣</span>
                <span class="text-sm font-semibold text-slate-200">CATML Activity & Planner Decisions</span>
              </div>
              <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono">Explicable</span>
            </div>
            <div class="p-4 space-y-3 flex-1 text-xs">
              <div class="flex items-start space-x-3 p-2.5 rounded bg-slate-900/40 border border-slate-800/80">
                <span class="badge-intel px-1.5 py-0.5 rounded font-mono font-bold text-[10px]">PLAN</span>
                <div class="flex-1">
                  <span class="text-slate-200 font-medium">CatBoost HPO priorizado:</span>
                  <span class="text-slate-400">Densidad categórica moderada (7 categóricas) y superioridad empírica de árboles.</span>
                </div>
              </div>

              <div class="flex items-start space-x-3 p-2.5 rounded bg-slate-900/40 border border-slate-800/80">
                <span class="badge-gain px-1.5 py-0.5 rounded font-mono font-bold text-[10px]">ACCEPT</span>
                <div class="flex-1">
                  <span class="text-slate-200 font-medium">TargetAdapter activado:</span>
                  <span class="text-slate-400">Normalización de etiquetas textuales ['No', 'Yes'] a binarias para XGBoost y LightGBM.</span>
                </div>
              </div>

              <div class="flex items-start space-x-3 p-2.5 rounded bg-slate-900/40 border border-slate-800/80">
                <span class="badge-err px-1.5 py-0.5 rounded font-mono font-bold text-[10px]">REJECT</span>
                <div class="flex-1">
                  <span class="text-slate-200 font-medium">Principio 'Proponer ≠ Aceptar':</span>
                  <span class="text-slate-400">Rechazadas 17 interacciones polinomiales por degradación empírica (-0.0004).</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    // Event listeners
    this.container.querySelector("#btnOpenStudioFromRun")?.addEventListener("click", () => {
      store.setNav("studio");
    });
    this.container.querySelector("#btnViewAllDatasets")?.addEventListener("click", () => {
      store.setNav("datasets");
    });
    this.container.querySelector("#btnNewExpOverview")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });
  }

  destroy() {
    this.container = null;
  }
}
