/**
 * OverviewView — Mission Control & Visual ML Lab
 * Zone 1: Active Experiment Equipment Block
 * Zone 2: Best Results Leaderboard
 * Zone 3: Recent Datasets Inventory
 * Zone 4: Hypothesis & Activity Audit Log
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { icon } from "../icons.js";
import { escapeHtml } from "../utils.js";
import { isRunActive, isMinimizingMetric, getMetricDirectionInfo } from "../utils.js";

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

    const activeDataset = store.getActiveDataset();
    const dsRuns = store.getRunsForActiveDataset();
    const runs = dsRuns;
    const activeRun = store.getActiveRun() || dsRuns.find(r => isRunActive(r)) || null;
    const isRunning = isRunActive(activeRun);

    const activeMetric = activeRun ? activeRun.metric : (overview.best_metric && overview.best_metric !== "-" ? overview.best_metric : null);
    const metricDirInfo = activeMetric ? getMetricDirectionInfo(activeMetric) : null;

    const bestScoreText = (activeRun && activeRun.best_score != null)
      ? Number(activeRun.best_score).toFixed(4)
      : (overview.best_score != null && !activeDataset ? Number(overview.best_score).toFixed(4) : "—");

    const bestModelText = (activeRun && activeRun.best_model)
      ? activeRun.best_model
      : (activeRun ? "Sin entrenar" : "Ninguno");

    const trialsCount = activeRun && activeRun.trials_count != null ? activeRun.trials_count : 0;

    // Zone 1 markup (Active Experiment)
    let zone1Html = "";
    if (activeRun) {
      zone1Html = `
        <div class="space-y-4">
          <div class="flex items-start justify-between border-b border-[#242A36] pb-3">
            <div>
              <div class="text-[10px] font-mono text-[#8B95A7] uppercase tracking-wider">Experimento / ID: ${escapeHtml(activeRun.id)}</div>
              <h3 class="text-base font-semibold font-sans text-[#F7F8FA] mt-0.5">${escapeHtml(activeRun.dataset_name || activeRun.id)}</h3>
              <p class="text-xs text-[#8B95A7] font-sans mt-0.5">Tarea: ${escapeHtml((activeRun.task_type || "Clasificación").replace("_", " "))} • Métrica: ${escapeHtml(activeRun.metric || "CV")}</p>
            </div>
            <div class="flex items-center space-x-2">
              <a href="/api/reports/executive?run_id=${activeRun.id}&format=html" target="_blank" class="btn-technical text-xs px-2.5 py-1 flex items-center space-x-1.5 hover:text-white transition-colors" title="Ver informe ejecutivo de resultados">
                ${icon("file-text", "icon-sm")}<span>Informe ejecutivo</span>
              </a>
              <span class="${isRunning ? 'badge-warn' : 'badge-gain'} text-xs px-2.5 py-0.5 rounded-md font-mono font-medium flex items-center space-x-1.5">
                ${isRunning ? '<svg class="animate-spin h-3 w-3 text-[#F59E0B] inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>' : '<span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span>'}
                <span>${escapeHtml(activeRun.status === 'READY' ? 'LISTO' : (activeRun.status || 'LISTO'))}</span>
              </span>
            </div>
          </div>

          <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 py-1 text-xs">
            <div class="p-3 rounded-xl bg-[#161B26] border border-[#242A36]">
              <span class="text-[10px] text-[#8B95A7] font-sans font-medium uppercase block">Mejor Puntuación CV</span>
              <span class="text-[#22C55E] font-mono font-bold text-sm mt-0.5 block">${activeRun.best_score != null ? Number(activeRun.best_score).toFixed(4) : "—"}</span>
            </div>
            <div class="p-3 rounded-xl bg-[#161B26] border border-[#242A36]">
              <span class="text-[10px] text-[#8B95A7] font-sans font-medium uppercase block">Mejor Modelo</span>
              <span class="text-[#F7F8FA] font-sans font-medium text-sm mt-0.5 block truncate">${escapeHtml(activeRun.best_model || "Ninguno")}</span>
            </div>
            <div class="p-3 rounded-xl bg-[#161B26] border border-[#242A36]">
              <span class="text-[10px] text-[#8B95A7] font-sans font-medium uppercase block">Estado</span>
              <span class="text-[#4F67FF] font-mono font-medium text-sm mt-0.5 block">${escapeHtml(activeRun.status === 'READY' ? 'LISTO' : (activeRun.status || 'LISTO'))}</span>
            </div>
            <div class="p-3 rounded-xl bg-[#161B26] border border-[#242A36]">
              <span class="text-[10px] text-[#8B95A7] font-sans font-medium uppercase block">Ensayos Evaluados</span>
              <span class="text-[#F7F8FA] font-mono font-bold text-sm mt-0.5 block">${activeRun.trials_count != null ? activeRun.trials_count : 0}</span>
            </div>
          </div>

          <!-- Progress Bar con datos reales -->
          <div class="space-y-1.5 pt-2 border-t border-[#242A36]">
            <div class="flex justify-between text-[11px] font-sans text-[#8B95A7]">
              <span>Estado de Exploración</span>
              <span class="font-mono">
                ${isRunning
                  ? 'Búsqueda en espacio de modelos (activa)'
                  : trialsCount > 0
                  ? `Ejecución completada (${trialsCount} ensayos)`
                  : 'Listo para entrenar (0 ensayos)'}
              </span>
            </div>
            <div class="w-full bg-[#080A0F] h-1.5 rounded-full overflow-hidden border border-[#242A36]">
              ${isRunning
                ? `<div class="bg-[#4F67FF] progress-striped w-1/2 h-full rounded-full"></div>`
                : trialsCount > 0
                ? `<div class="bg-[#22C55E] w-full h-full rounded-full"></div>`
                : `<div class="bg-transparent w-0 h-full"></div>`}
            </div>
          </div>

          <div class="pt-2 flex flex-wrap justify-between items-center gap-3">
            <a href="/api/models/export?run_id=${encodeURIComponent(activeRun.id)}" class="btn-signal text-xs flex items-center space-x-1.5" title="Descargar artefacto de inferencia autónomo (.pkl)">
              ${icon("download", "icon-sm")}
              <span>Exportar modelo ganador (.pkl)</span>
            </a>
            <button id="btnOpenStudioFromRun" class="btn-technical text-xs flex items-center space-x-1.5">
              <span>Abrir Studio de Experimentos</span>
              ${icon("arrow-right", "icon-sm text-[#8B95A7]")}
            </button>
          </div>
        </div>
      `;
    } else {
      zone1Html = `
        <div class="p-8 text-center space-y-3">
          <span class="text-[#8B95A7]/40 block">${icon("flask-conical", "icon-xl mx-auto")}</span>
          <p class="text-sm font-sans font-medium text-[#F7F8FA]">Sin experimento activo en el espacio de trabajo</p>
          <p class="text-xs text-[#8B95A7] max-w-sm mx-auto font-sans">Crea un nuevo experimento o selecciona un dataset para iniciar la búsqueda autónoma de modelos, ingeniería de variables y HPO.</p>
          <button id="btnNewExpOverviewEmpty" class="btn-signal mt-2 flex items-center space-x-1.5 mx-auto">
            ${icon("plus", "icon-sm")}
            <span>Nuevo Experimento</span>
          </button>
        </div>
      `;
    }

    // Zone 2 markup (Best Results Leaderboard)
    const runsRows = runs.length > 0
      ? runs.map((r, idx) => `
          <tr class="cursor-pointer hover:bg-[#161B26]/60 run-row-item ${idx === 0 ? 'champion-lead bg-[#161B26]/30' : ''}" data-run-id="${escapeHtml(r.id)}">
            <td class="font-mono text-xs font-semibold text-[#F7F8FA]">
              ${idx === 0 ? '<span class="text-[#4F67FF] mr-1">▌01</span>' : `<span class="text-[#8B95A7]/60 mr-1">${String(idx + 1).padStart(2, '0')}</span>`}
              ${escapeHtml(r.dataset_name || r.id)}
            </td>
            <td><span class="text-xs text-[#8B95A7] font-sans capitalize">${escapeHtml((r.task_type || "Clasificación").replace("_", " "))}</span></td>
            <td><span class="badge-sys px-2 py-0.5 rounded text-xs font-mono uppercase">${escapeHtml(r.best_model || "—")}</span></td>
            <td class="font-mono text-xs text-[#8B95A7]">${escapeHtml(r.metric || "CV")} <span class="text-[10px] text-[#8B95A7]/70">(${isMinimizingMetric(r.metric) ? 'menor es mejor' : 'mayor es mejor'})</span></td>
            <td class="font-mono font-bold text-[#22C55E] text-sm">${r.best_score != null ? Number(r.best_score).toFixed(4) : "—"}</td>
            <td class="text-right">
              <a href="/api/models/export?run_id=${encodeURIComponent(r.id)}" onclick="event.stopPropagation()" class="px-2.5 py-1 text-[11px] font-mono font-semibold text-[#4F67FF] hover:text-white bg-[#4F67FF]/10 hover:bg-[#4F67FF] border border-[#4F67FF]/25 rounded-md transition-all inline-flex items-center space-x-1" title="Descargar ModelArtifact (.pkl)">
                ${icon("download", "icon-sm")}
                <span>.pkl</span>
              </a>
            </td>
          </tr>
        `).join("")
      : `
          <tr>
            <td colspan="6" class="text-center text-[#8B95A7]/60 py-6 text-xs font-sans">
              No hay ejecuciones completadas en el espacio de trabajo.
            </td>
          </tr>
        `;

    // Zone 3 markup (Recent Datasets)
    const datasets = overview.recent_datasets || [];
    const datasetsHtml = datasets.length > 0
      ? datasets.slice(0, 4).map(ds => `
          <div class="flex items-center justify-between p-3 rounded-xl bg-[#161B26] border border-[#252C38] cursor-pointer hover:border-[#4F67FF]/40 transition-colors dataset-item-card" data-dataset-id="${escapeHtml(ds.id)}">
            <div>
              <div class="font-mono font-semibold text-xs text-[#F7F8FA]">${escapeHtml(ds.name)}</div>
              <div class="text-[11px] text-[#8B95A7] font-sans mt-0.5">${escapeHtml(ds.task_type || "Clasificación")} • ${ds.rows != null ? Number(ds.rows).toLocaleString() : "—"} filas</div>
            </div>
            <span class="badge-sys text-[10px] px-2 py-0.5 rounded font-mono">${ds.features != null ? `${ds.features} cols` : "Listo"}</span>
          </div>
        `).join("")
      : `
          <div class="text-center text-[#8B95A7]/60 py-6 text-xs font-sans">
            No hay datasets registrados todavía.
            <button id="btnRegisterDSOverview" class="text-[#4F67FF] hover:underline font-semibold ml-1">Registrar dataset →</button>
          </div>
        `;

    // Zone 4 markup (Activity & Planner Decisions)
    const activityFeed = overview.activity_feed || [];
    const activityHtml = activityFeed.length > 0
      ? activityFeed.map(act => {
          const badgeClass = act.type === "ACCEPT" ? "badge-gain" : act.type === "REJECT" ? "badge-err" : act.type === "PLAN" ? "badge-intel" : "badge-sys";
          return `
            <div class="flex items-start space-x-3 p-3 rounded-xl bg-[#161B26] border border-[#252C38]">
              <span class="${badgeClass} px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold">${escapeHtml(act.type || "INFO")}</span>
              <div class="flex-1 font-sans text-xs">
                <span class="text-[#F7F8FA] font-medium">${escapeHtml(act.title)}:</span>
                <span class="text-[#8B95A7] ml-1">${escapeHtml(act.description)}</span>
              </div>
            </div>
          `;
        }).join("")
      : `
          <div class="text-center text-[#8B95A7]/60 py-6 text-xs font-sans">No hay registros de auditoría del planificador.</div>
        `;

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Hero Header -->
        <div class="border-b border-[#242A36] pb-5 flex flex-col md:flex-row md:items-end justify-between gap-4">
          <div>
            <div class="text-[11px] font-mono text-[#4F67FF] uppercase tracking-wider font-semibold">Espacio de Trabajo • AutoML Nativo de Agentes</div>
            <h1 class="text-2xl font-bold font-sans text-[#F7F8FA] tracking-tight mt-1">Entrena mejores modelos. Mantén el control.</h1>
            <p class="text-xs text-[#8B95A7] font-sans mt-0.5">Local-first • Nativo de agentes • Artefactos listos para producción</p>
          </div>
          <div>
            <button id="btnNewExpOverview" class="btn-signal flex items-center space-x-1.5">
              ${icon("plus", "icon-sm")}
              <span>Nuevo Experimento</span>
            </button>
          </div>
        </div>

        <!-- Top KPI row -->
        <div class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div class="metric-block">
            <div class="metric-lbl">Mejor resultado observado${activeMetric ? ` (${escapeHtml(activeMetric.toUpperCase())})` : ""}</div>
            <div class="metric-val text-[#22C55E] mt-1">${bestScoreText}</div>
            <div class="text-[11px] font-sans text-[#8B95A7] mt-1 flex items-center justify-between">
              <span>Modelo: <span class="text-[#F7F8FA] font-medium font-mono">${escapeHtml(bestModelText)}</span></span>
              ${metricDirInfo ? `<span class="text-[10px] font-mono text-[#8B95A7]">(${metricDirInfo.label})</span>` : ""}
            </div>
          </div>

          <div class="metric-block">
            <div class="metric-lbl">Experimento Activo</div>
            <div class="metric-val text-[#F7F8FA] mt-1 flex items-center space-x-2">
              <span class="inline-block w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-[#F59E0B] animate-pulse' : (activeRun ? 'bg-[#22C55E]' : 'bg-[#242A36]')}"></span>
              <span class="text-xl">${activeRun ? escapeHtml(activeRun.status === 'READY' ? 'LISTO' : activeRun.status) : "EN REPOSO"}</span>
            </div>
            <div class="text-[11px] font-sans text-[#8B95A7] mt-1 truncate">${activeRun ? `${escapeHtml(activeRun.dataset_name || activeRun.id)}` : "Sin ejecución activa"}</div>
          </div>

          <div class="metric-block">
            <div class="metric-lbl">Modelos Evaluados</div>
            <div class="metric-val text-[#4F67FF] mt-1">${trialsCount}</div>
            <div class="text-[11px] font-sans text-[#8B95A7] mt-1">${runs.length} ejecuciones registradas</div>
          </div>

          <div class="metric-block">
            <div class="metric-lbl">Motor de Cómputo</div>
            <div class="metric-val text-[#F7F8FA] mt-1 text-xl">${isRunning ? 'EN EJECUCIÓN' : 'ACTIVO'}</div>
            <div class="text-[11px] font-sans text-[#8B95A7] mt-1">Cross-Validation Estratificada • Pliegues Locales</div>
          </div>
        </div>

        <!-- 4 Equipment Modules Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Zone 1: Active Runs Module -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                ${icon("zap", "icon-sm text-[#F59E0B]")}
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Experimento Activo</span>
              </div>
              <span class="badge-warn text-[10px] px-2 py-0.5 rounded font-mono font-medium">En vivo</span>
            </div>
            <div class="p-5 space-y-4 flex-1">
              ${zone1Html}
            </div>
          </div>

          <!-- Zone 2: Best Results Leaderboard -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                ${icon("trophy", "icon-sm text-[#22C55E]")}
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Clasificación de Modelos (Leaderboard)</span>
              </div>
              <span class="badge-sys text-[10px] px-2 py-0.5 rounded font-mono font-medium">Evaluado por Cross-Validation</span>
            </div>
            <div class="p-0 overflow-x-auto flex-1">
              <table class="w-full wb-table text-left">
                <thead>
                  <tr>
                    <th>Dataset / ID</th>
                    <th>Tarea</th>
                    <th>Mejor Modelo</th>
                    <th>Métrica</th>
                    <th>Puntuación</th>
                    <th class="text-right">Exportar</th>
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
                ${icon("database", "icon-sm text-[#4F67FF]")}
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Inventario de Datasets</span>
              </div>
              <button id="btnViewAllDatasets" class="text-xs font-sans text-[#4F67FF] hover:underline font-medium">Ver todos →</button>
            </div>
            <div class="p-5 space-y-2.5 flex-1">
              ${datasetsHtml}
            </div>
          </div>

          <!-- Zone 4: Activity & Decisions -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                ${icon("bot", "icon-sm text-[#6956E8]")}
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Auditoría y Decisiones del Planificador</span>
              </div>
              <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono font-medium">Hipótesis</span>
            </div>
            <div class="p-5 space-y-2.5 flex-1 text-xs">
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
