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
      best_score: null,
      best_model: null,
      total_runs: 0,
      total_trials: 0,
      workspace: "CATML Default",
      recent_datasets: [],
      activity_feed: [],
    };

    const runs = state.runs || [];
    const activeRun = runs.find(r => r.status === "RUNNING") || runs[0] || null;
    const isRunning = activeRun && activeRun.status === "RUNNING";

    const bestScoreText = overview.best_score != null && overview.best_score > 0
      ? Number(overview.best_score).toFixed(4)
      : (activeRun && activeRun.best_score != null ? Number(activeRun.best_score).toFixed(4) : "—");

    const bestModelText = (overview.best_model && overview.best_model !== "-")
      ? overview.best_model
      : (activeRun && activeRun.best_model ? activeRun.best_model : "None");

    const trialsCount = overview.total_trials || (activeRun && activeRun.trials_count != null ? activeRun.trials_count : 0);

    // Zone 1 markup
    let zone1Html = "";
    if (activeRun) {
      zone1Html = `
        <div class="border border-indigo-900/50 bg-indigo-950/20 rounded-xl p-4 space-y-3">
          <div class="flex items-start justify-between">
            <div>
              <h3 class="text-base font-bold text-slate-100 uppercase tracking-wide">${activeRun.dataset_name || activeRun.id}</h3>
              <p class="text-xs text-slate-400">AutoML • ${(activeRun.task_type || "Classification").replace("_", " ")} • ${activeRun.metric || "CV"}</p>
            </div>
            <span class="${isRunning ? 'badge-sys' : 'badge-gain'} text-xs px-2.5 py-1 rounded-full font-mono uppercase tracking-wider font-semibold flex items-center space-x-1.5">
              ${isRunning ? '<svg class="animate-spin h-3 w-3 text-indigo-400 inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>' : ''}
              <span>● ${activeRun.status || "COMPLETED"}</span>
            </span>
          </div>

          <div class="grid grid-cols-2 gap-3 pt-2 text-xs">
            <div>
              <span class="text-slate-400">Best CV:</span>
              <span class="ml-1 text-emerald-400 font-mono font-bold">${activeRun.best_score != null ? Number(activeRun.best_score).toFixed(4) : "—"}</span>
            </div>
            <div>
              <span class="text-slate-400">Best Model:</span>
              <span class="ml-1 text-slate-200 font-medium uppercase">${activeRun.best_model || "None"}</span>
            </div>
            <div>
              <span class="text-slate-400">Status:</span>
              <span class="ml-1 text-indigo-300 font-medium">${activeRun.status || "READY"}</span>
            </div>
            <div>
              <span class="text-slate-400">Trials:</span>
              <span class="ml-1 text-slate-200 font-mono">${activeRun.trials_count != null ? activeRun.trials_count : 0}</span>
            </div>
          </div>

          <!-- Progress Bar -->
          <div class="space-y-1 pt-1">
            <div class="flex justify-between text-[11px] text-slate-400">
              <span>Optimization Progress</span>
              <span>${isRunning ? 'In Progress' : 'Completed'}</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
              <div class="${isRunning ? 'bg-indigo-500 progress-striped w-2/3' : 'bg-emerald-500 w-full'} h-2.5 rounded-full"></div>
            </div>
          </div>

          <div class="pt-2 flex justify-end">
            <button id="btnOpenStudioFromRun" class="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center space-x-1">
              <span>Open Experiment Studio</span>
              <span>→</span>
            </button>
          </div>
        </div>
      `;
    } else {
      zone1Html = `
        <div class="border border-slate-800 bg-slate-900/30 rounded-xl p-8 text-center space-y-3">
          <span class="text-3xl text-slate-600 block">⚡</span>
          <p class="text-sm font-medium text-slate-300">No active runs in current workspace.</p>
          <p class="text-xs text-slate-500 max-w-sm mx-auto">Create a new experiment or register a dataset to launch autonomous model training and HPO.</p>
          <button id="btnNewExpOverviewEmpty" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-4 py-2 rounded-lg font-medium transition-colors shadow-lg shadow-indigo-600/20">
            + New Experiment
          </button>
        </div>
      `;
    }

    // Zone 2 markup (Best Results Table)
    const runsRows = runs.length > 0
      ? runs.map(r => `
          <tr class="cursor-pointer hover:bg-slate-900/60 run-row-item" data-run-id="${r.id}">
            <td class="font-medium text-slate-200">${r.dataset_name || r.id}</td>
            <td><span class="text-xs text-slate-400 capitalize">${(r.task_type || "Classification").replace("_", " ")}</span></td>
            <td><span class="badge-sys px-2 py-0.5 rounded text-xs uppercase">${r.best_model || "—"}</span></td>
            <td class="font-mono text-xs">${r.metric || "CV"}</td>
            <td class="font-mono font-bold text-emerald-400">${r.best_score != null ? Number(r.best_score).toFixed(4) : "—"}</td>
          </tr>
        `).join("")
      : `
          <tr>
            <td colspan="5" class="text-center text-slate-500 py-6 text-xs">No runs recorded yet. Start an experiment to see best results.</td>
          </tr>
        `;

    // Zone 3 markup (Recent Datasets)
    const recentDatasets = overview.recent_datasets || [];
    const datasetsHtml = recentDatasets.length > 0
      ? recentDatasets.map(d => `
          <div class="flex items-center justify-between p-3 rounded-lg bg-slate-900/60 border border-slate-800 hover:border-slate-700 cursor-pointer dataset-item-card" data-dataset-id="${d.id}">
            <div>
              <div class="text-sm font-bold text-slate-200">${d.name}</div>
              <div class="text-xs text-slate-400">${d.rows ? Number(d.rows).toLocaleString() + ' filas • ' : ''}${d.features ? d.features + ' features • ' : ''}Target: ${d.target || d.target_column || "—"}</div>
            </div>
            <span class="badge-gain text-xs px-2.5 py-1 rounded">Profiled</span>
          </div>
        `).join("")
      : `
          <div class="text-center text-slate-500 py-8 text-xs space-y-2">
            <p>No datasets registered in workspace.</p>
            <button id="btnRegisterDSOverview" class="text-indigo-400 hover:text-indigo-300 font-semibold underline">Register a dataset →</button>
          </div>
        `;

    // Zone 4 markup (CATML Activity & Planner Decisions)
    const activityFeed = overview.activity_feed || [];
    const activityHtml = activityFeed.length > 0
      ? activityFeed.map(act => {
          const badgeClass = act.type === "ACCEPT" ? "badge-gain" : act.type === "REJECT" ? "badge-err" : act.type === "PLAN" ? "badge-intel" : "badge-sys";
          return `
            <div class="flex items-start space-x-3 p-2.5 rounded bg-slate-900/40 border border-slate-800/80">
              <span class="${badgeClass} px-1.5 py-0.5 rounded font-mono font-bold text-[10px]">${act.type || "INFO"}</span>
              <div class="flex-1">
                <span class="text-slate-200 font-medium">${act.title}:</span>
                <span class="text-slate-400 ml-1">${act.description}</span>
              </div>
            </div>
          `;
        }).join("")
      : `
          <div class="text-center text-slate-500 py-6 text-xs">No planner activity recorded.</div>
        `;

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Top KPI row -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Best CV Score</div>
            <div class="mt-2 flex items-baseline justify-between">
              <span class="text-2xl font-bold font-mono-num text-emerald-400">${bestScoreText}</span>
              <span class="badge-gain text-xs px-2 py-0.5 rounded-full font-mono font-medium">Optimal</span>
            </div>
            <div class="mt-1 text-xs text-slate-400">Modelo: <span class="text-slate-200 font-medium uppercase">${bestModelText}</span></div>
          </div>

          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Active Run</div>
            <div class="mt-2 flex items-center space-x-2">
              <span class="inline-block w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-indigo-500 animate-pulse' : (activeRun ? 'bg-emerald-500' : 'bg-slate-600')}"></span>
              <span class="text-xl font-bold text-slate-100">${activeRun ? activeRun.status : "IDLE"}</span>
            </div>
            <div class="mt-1 text-xs text-slate-400 truncate">${activeRun ? `${activeRun.dataset_name || activeRun.id} • ${activeRun.metric || 'CV'}` : "No run selected"}</div>
          </div>

          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Trials Executed</div>
            <div class="mt-2 flex items-baseline justify-between">
              <span class="text-2xl font-bold font-mono-num text-indigo-400">${trialsCount}</span>
              <span class="text-xs text-slate-500 font-mono">${runs.length} runs</span>
            </div>
            <div class="mt-1 text-xs text-slate-400">Progreso: <span class="text-slate-200 font-medium">${isRunning ? 'En curso' : 'Completado'}</span></div>
          </div>

          <div class="workbench-card p-4">
            <div class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Compute & Workers</div>
            <div class="mt-2 flex items-baseline justify-between">
              <span class="text-2xl font-bold font-mono-num text-slate-200">${isRunning ? 'Active' : 'Idle'}</span>
              <span class="text-xs text-slate-400">Stratified CV</span>
            </div>
            <div class="mt-1 text-xs text-slate-400">Workers: <span class="text-indigo-400 font-medium">${isRunning ? 'Folds activos' : 'Listo'}</span></div>
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
              ${zone1Html}
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
                  ${runsRows}
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
              ${datasetsHtml}
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
              ${activityHtml}
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
    this.container.querySelector("#btnRegisterDSOverview")?.addEventListener("click", () => {
      store.setNav("datasets");
    });
    this.container.querySelector("#btnNewExpOverview")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });
    this.container.querySelector("#btnNewExpOverviewEmpty")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    this.container.querySelectorAll(".run-row-item").forEach(el => {
      el.addEventListener("click", () => {
        const runId = el.getAttribute("data-run-id");
        if (runId) {
          bus.emit("run:select", runId);
        }
      });
    });

    this.container.querySelectorAll(".dataset-item-card").forEach(el => {
      el.addEventListener("click", () => {
        const datasetId = el.getAttribute("data-dataset-id");
        if (datasetId) {
          store.setState({ activeDatasetId: datasetId });
          store.setNav("datasets");
        }
      });
    });
  }

  destroy() {
    this.container = null;
  }
}
