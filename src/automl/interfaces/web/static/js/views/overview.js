/**
 * OverviewView — Neo-Industrial Mission Control & Visual ML Lab
 * Zone 1: Active Experiment Equipment Block
 * Zone 2: Best Results Leaderboard
 * Zone 3: Recent Datasets Inventory
 * Zone 4: Hypothesis & Activity Audit Log
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

    // Zone 1 markup (Active Experiment)
    let zone1Html = "";
    if (activeRun) {
      zone1Html = `
        <div class="border border-[#27272e] bg-[#16171c] p-4 space-y-4">
          <div class="flex items-start justify-between border-b border-[#27272e] pb-3">
            <div>
              <div class="text-[10px] font-mono text-[#D8D6CF]/60 uppercase tracking-widest">EXPERIMENT / ID: ${activeRun.id}</div>
              <h3 class="text-base font-bold font-mono text-[#F1EFE9] uppercase tracking-wide mt-0.5">${activeRun.dataset_name || activeRun.id}</h3>
              <p class="text-xs text-[#D8D6CF]/70 font-mono mt-0.5">TASK: ${(activeRun.task_type || "Classification").toUpperCase()} • METRIC: ${activeRun.metric || "CV"}</p>
            </div>
            <span class="${isRunning ? 'badge-warn' : 'badge-gain'} text-xs px-2.5 py-1 rounded-sm font-mono uppercase tracking-wider font-semibold flex items-center space-x-1.5">
              ${isRunning ? '<svg class="animate-spin h-3 w-3 text-[#E7C84B] inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>' : ''}
              <span>● ${activeRun.status || "READY"}</span>
            </span>
          </div>

          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 py-1 font-mono text-xs">
            <div class="border-l-2 border-[#63D49A] pl-2">
              <span class="text-[10px] text-[#D8D6CF]/60 uppercase block">BEST CV SCORE</span>
              <span class="text-[#63D49A] font-bold text-sm">${activeRun.best_score != null ? Number(activeRun.best_score).toFixed(4) : "—"}</span>
            </div>
            <div class="border-l-2 border-[#9C7CFF] pl-2">
              <span class="text-[10px] text-[#D8D6CF]/60 uppercase block">BEST MODEL</span>
              <span class="text-[#F1EFE9] font-medium uppercase text-sm">${activeRun.best_model || "None"}</span>
            </div>
            <div class="border-l-2 border-[#5C78FF] pl-2">
              <span class="text-[10px] text-[#D8D6CF]/60 uppercase block">STATUS</span>
              <span class="text-[#5C78FF] font-medium text-sm">${activeRun.status || "READY"}</span>
            </div>
            <div class="border-l-2 border-[#E7C84B] pl-2">
              <span class="text-[10px] text-[#D8D6CF]/60 uppercase block">TRIALS TESTED</span>
              <span class="text-[#F1EFE9] font-bold text-sm">${activeRun.trials_count != null ? activeRun.trials_count : 0}</span>
            </div>
          </div>

          <!-- Industrial Progress Bar -->
          <div class="space-y-1.5 pt-2 border-t border-[#27272e]">
            <div class="flex justify-between text-[11px] font-mono text-[#94A3B8]/70">
              <span>EXPLORATION STATE</span>
              <span>${isRunning ? 'SEARCHING SPACE (72%)' : 'EXECUTION COMPLETED (100%)'}</span>
            </div>
            <div class="w-full bg-[#111111] h-2 border border-[#27272e] overflow-hidden">
              <div class="${isRunning ? 'bg-[#4F67FF] progress-striped w-2/3' : 'bg-[#22C55E] w-full'} h-full"></div>
            </div>
          </div>

          <div class="pt-2 flex justify-between items-center">
            <a href="/api/models/export?run_id=${activeRun.id}" class="btn-signal text-xs flex items-center space-x-1.5" title="Descargar artefacto de inferencia autónomo (sin dependencias de workspace)">
              <span>⚡</span>
              <span>EXPORT WINNING MODEL (.PKL)</span>
            </a>
            <button id="btnOpenStudioFromRun" class="btn-technical text-xs flex items-center space-x-1.5">
              <span>OPEN EXPERIMENT STUDIO</span>
              <span>→</span>
            </button>
          </div>
        </div>
      `;
    } else {
      zone1Html = `
        <div class="border border-[#27272e] bg-[#16171c] p-8 text-center space-y-3">
          <span class="text-3xl text-[#94A3B8]/40 block font-mono">⚡</span>
          <p class="text-sm font-mono font-medium text-[#FFFFFF]">NO ACTIVE EXPERIMENT IN WORKSPACE</p>
          <p class="text-xs text-[#94A3B8]/60 max-w-sm mx-auto">Create a new experiment or select a dataset to launch autonomous model search, feature selection and HPO.</p>
          <button id="btnNewExpOverviewEmpty" class="btn-signal mt-2">
            + NEW EXPERIMENT
          </button>
        </div>
      `;
    }

    // Zone 2 markup (Best Results Leaderboard)
    const runsRows = runs.length > 0
      ? runs.map((r, idx) => `
          <tr class="cursor-pointer hover:bg-[#1c1d24] run-row-item ${idx === 0 ? 'champion-lead bg-[#1c1d24]/50' : ''}" data-run-id="${r.id}">
            <td class="font-mono text-xs font-bold text-[#FFFFFF]">
              ${idx === 0 ? '<span class="text-[#4F67FF] mr-1">▌01</span>' : `<span class="text-[#94A3B8]/40 mr-1">${String(idx + 1).padStart(2, '0')}</span>`}
              ${r.dataset_name || r.id}
            </td>
            <td><span class="text-xs text-[#94A3B8]/70 font-mono capitalize">${(r.task_type || "Classification").replace("_", " ")}</span></td>
            <td><span class="badge-sys px-2 py-0.5 rounded-sm text-xs font-mono uppercase">${r.best_model || "—"}</span></td>
            <td class="font-mono text-xs text-[#94A3B8]/80">${r.metric || "CV"}</td>
            <td class="font-mono font-bold text-[#22C55E] text-sm">${r.best_score != null ? Number(r.best_score).toFixed(4) : "—"}</td>
            <td class="text-right">
              <a href="/api/models/export?run_id=${r.id}" onclick="event.stopPropagation()" class="px-2 py-1 text-[11px] font-mono font-bold text-[#4F67FF] hover:text-[#FFFFFF] bg-[#4F67FF]/10 hover:bg-[#4F67FF] border border-[#4F67FF]/30 rounded transition-colors inline-flex items-center space-x-1" title="Descargar ModelArtifact autónomo (.pkl)">
                <span>⬇</span>
                <span>.PKL</span>
              </a>
            </td>
          </tr>
        `).join("")
      : `
          <tr>
            <td colspan="6" class="text-center text-[#94A3B8]/50 py-6 text-xs font-mono">NO EXPERIMENT RUNS RECORDED YET.</td>
          </tr>
        `;

    // Zone 3 markup (Recent Datasets)
    const recentDatasets = overview.recent_datasets || [];
    const datasetsHtml = recentDatasets.length > 0
      ? recentDatasets.map(d => `
          <div class="flex items-center justify-between p-3 bg-[#111111] border border-[#27272e] hover:border-[#D8D6CF]/40 cursor-pointer dataset-item-card transition-colors" data-dataset-id="${d.id}">
            <div>
              <div class="text-xs font-mono font-bold text-[#F1EFE9]">${d.name}</div>
              <div class="text-[11px] font-mono text-[#D8D6CF]/60 mt-0.5">${d.rows ? Number(d.rows).toLocaleString() + ' ROWS • ' : ''}${d.features ? d.features + ' FEATURES • ' : ''}TARGET: ${d.target || d.target_column || "—"}</div>
            </div>
            <span class="badge-sys text-[10px] px-2 py-0.5 rounded-sm font-mono font-bold">PROFILED</span>
          </div>
        `).join("")
      : `
          <div class="text-center text-[#D8D6CF]/50 py-8 text-xs font-mono space-y-2">
            <p>NO DATASETS REGISTERED IN WORKSPACE.</p>
            <button id="btnRegisterDSOverview" class="text-[#5C78FF] hover:underline font-semibold">REGISTER DATASET →</button>
          </div>
        `;

    // Zone 4 markup (CATML Activity & Planner Decisions)
    const activityFeed = overview.activity_feed || [];
    const activityHtml = activityFeed.length > 0
      ? activityFeed.map(act => {
          const badgeClass = act.type === "ACCEPT" ? "badge-gain" : act.type === "REJECT" ? "badge-err" : act.type === "PLAN" ? "badge-intel" : "badge-sys";
          return `
            <div class="flex items-start space-x-3 p-2.5 bg-[#111111] border border-[#27272e]">
              <span class="${badgeClass} px-1.5 py-0.5 rounded-sm font-mono font-bold text-[10px]">${act.type || "INFO"}</span>
              <div class="flex-1 font-mono text-xs">
                <span class="text-[#F1EFE9] font-semibold">${act.title}:</span>
                <span class="text-[#D8D6CF]/70 ml-1">${act.description}</span>
              </div>
            </div>
          `;
        }).join("")
      : `
          <div class="text-center text-[#D8D6CF]/50 py-6 text-xs font-mono">NO PLANNER AUDIT ENTRIES RECORDED.</div>
        `;

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Hero Industrial Header -->
        <div class="border-b border-[#27272e] pb-4 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <div class="text-[10px] font-mono text-[#4F67FF] uppercase tracking-widest font-bold">CATML WORKSPACE / AGENT-NATIVE AUTOML</div>
            <h1 class="text-2xl font-bold font-sans text-[#FFFFFF] tracking-tight mt-1">Train better models. Keep control.</h1>
            <p class="text-xs text-[#94A3B8]/70 font-mono mt-0.5">Local-first • Agent-native • Production-ready model artifacts</p>
          </div>
          <div>
            <button id="btnNewExpOverview" class="btn-signal">
              <span>+</span>
              <span>NEW EXPERIMENT</span>
            </button>
          </div>
        </div>

        <!-- Top KPI row: Large Numbers in Monospace -->
        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div class="metric-block">
            <div class="metric-lbl">BEST VALIDATED SCORE</div>
            <div class="metric-val text-[#63D49A] mt-1">${bestScoreText}</div>
            <div class="text-[11px] font-mono text-[#D8D6CF]/60 mt-1 uppercase">MODEL: <span class="text-[#F1EFE9] font-medium">${bestModelText}</span></div>
          </div>

          <div class="metric-block">
            <div class="metric-lbl">ACTIVE EXPERIMENT</div>
            <div class="metric-val text-[#F1EFE9] mt-1 flex items-center space-x-2">
              <span class="inline-block w-2.5 h-2.5 rounded-sm ${isRunning ? 'bg-[#E7C84B] animate-pulse' : (activeRun ? 'bg-[#63D49A]' : 'bg-[#27272e]')}"></span>
              <span class="text-xl">${activeRun ? activeRun.status : "IDLE"}</span>
            </div>
            <div class="text-[11px] font-mono text-[#D8D6CF]/60 mt-1 truncate uppercase">${activeRun ? `${activeRun.dataset_name || activeRun.id}` : "NO ACTIVE RUN"}</div>
          </div>

          <div class="metric-block">
            <div class="metric-lbl">MODELS TESTED</div>
            <div class="metric-val text-[#5C78FF] mt-1">${trialsCount}</div>
            <div class="text-[11px] font-mono text-[#D8D6CF]/60 mt-1 uppercase">${runs.length} RUNS RECORDED</div>
          </div>

          <div class="metric-block">
            <div class="metric-lbl">COMPUTE HARDWARE</div>
            <div class="metric-val text-[#F1EFE9] mt-1 text-xl">${isRunning ? 'RUNNING' : 'ONLINE'}</div>
            <div class="text-[11px] font-mono text-[#D8D6CF]/60 mt-1 uppercase">STRATIFIED CV • 4 WORKERS</div>
          </div>
        </div>

        <!-- 4 Equipment Modules Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Zone 1: Active Runs Equipment Module -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-[#4F67FF] font-mono font-bold text-xs">MOD / 01</span>
                <span class="text-xs font-mono font-bold text-[#FFFFFF] uppercase tracking-wider">ACTIVE EXPERIMENT</span>
              </div>
              <span class="badge-warn text-[10px] px-2 py-0.5 rounded-sm font-mono font-semibold">LIVE</span>
            </div>
            <div class="p-4 space-y-4 flex-1">
              ${zone1Html}
            </div>
          </div>

          <!-- Zone 2: Best Results Leaderboard -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-[#22C55E] font-mono font-bold text-xs">MOD / 02</span>
                <span class="text-xs font-mono font-bold text-[#FFFFFF] uppercase tracking-wider">MODEL LEADERBOARD</span>
              </div>
              <span class="text-[10px] font-mono text-[#94A3B8]/60 uppercase">VERIFIED BY CV</span>
            </div>
            <div class="p-0 overflow-x-auto flex-1">
              <table class="w-full wb-table text-left">
                <thead>
                  <tr>
                    <th>DATASET / ID</th>
                    <th>TASK</th>
                    <th>BEST MODEL</th>
                    <th>METRIC</th>
                    <th>SCORE</th>
                    <th class="text-right">EXPORT</th>
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
                <span class="text-[#5C78FF] font-mono font-bold text-xs">MOD / 03</span>
                <span class="text-xs font-mono font-bold text-[#F1EFE9] uppercase tracking-wider">DATASETS REPOSITORY</span>
              </div>
              <button id="btnViewAllDatasets" class="text-xs font-mono text-[#5C78FF] hover:underline uppercase">VIEW ALL →</button>
            </div>
            <div class="p-4 space-y-2 flex-1">
              ${datasetsHtml}
            </div>
          </div>

          <!-- Zone 4: CATML Activity & Decisions -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-[#9C7CFF] font-mono font-bold text-xs">MOD / 04</span>
                <span class="text-xs font-mono font-bold text-[#F1EFE9] uppercase tracking-wider">AUDIT & PLANNER DECISIONS</span>
              </div>
              <span class="badge-intel text-[10px] px-2 py-0.5 rounded-sm font-mono font-bold">HYPOTHESIS</span>
            </div>
            <div class="p-4 space-y-2 flex-1 text-xs">
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
