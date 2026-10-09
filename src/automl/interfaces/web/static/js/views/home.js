/**
 * HomeView — Visual ML Laboratory Home
 * Implements the unified laboratory model: 1 active workspace, 1 active dataset at a time.
 * - Hero Card: "Continuar con {dataset_name}" + Acciones secundarias
 * - Ejecuciones en curso (Active runs / Training progress)
 * - Trabajos recientes (Background jobs & activity)
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml, isRunActive } from "../utils.js";

export class HomeView {
  constructor() {
    this.container = null;
    this.datasets = [];
    this.runs = [];
    this.jobs = [];
    this.activeDataset = null;
    this.loading = true;
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
        <div class="text-sm font-sans font-medium">Cargando contexto del laboratorio local...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const [datasets, runs, jobs] = await Promise.all([
        api.getDatasets().catch(() => []),
        api.getRuns().catch(() => []),
        api.getJobs().catch(() => []),
      ]);

      this.datasets = datasets;
      this.runs = runs;
      this.jobs = jobs;

      store.setState({ datasets, runs });

      // Determine active dataset
      const state = store.getState();
      let activeId = state.activeDatasetId;

      if (!activeId || !datasets.some(d => d.id === activeId)) {
        if (datasets.length > 0) {
          activeId = datasets[0].id;
        } else if (runs.length > 0 && runs[0].dataset_id) {
          activeId = runs[0].dataset_id;
        }
      }

      if (activeId) {
        store.setActiveDataset(activeId);
        this.activeDataset = datasets.find(d => d.id === activeId) || null;
      }
    } catch (e) {
      console.warn("[HomeView] Error fetching laboratory data:", e);
    } finally {
      this.loading = false;
    }
  }

  render() {
    const ds = this.activeDataset || store.getActiveDataset();
    const runs = this.runs || [];
    const dsRuns = ds ? runs.filter(r => r.dataset_id === ds.id) : runs;
    const activeRun = dsRuns.find(r => isRunActive(r)) || dsRuns[0] || null;
    const isRunning = isRunActive(activeRun);

    // Filter recent jobs for this dataset or run
    const recentJobs = (this.jobs || []).slice(0, 6);

    this.container.innerHTML = `
      <div class="space-y-6 max-w-7xl mx-auto">
        <!-- 1. Laboratorio Header & Welcome -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#242A36] pb-4">
          <div>
            <div class="text-[11px] font-mono uppercase tracking-wider text-[#8B95A7] flex items-center space-x-2">
              <span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span>
              <span>Laboratorio Local • CATML Engine</span>
            </div>
            <h1 class="text-xl font-bold font-sans text-[#F7F8FA] mt-1">Centro de Control</h1>
          </div>

          <div class="flex items-center space-x-2.5">
            <button id="btnHomeNewExpDirect" class="btn-signal text-xs flex items-center space-x-1.5">
              ${icon("play", "icon-sm")}
              <span>Nuevo experimento</span>
            </button>
          </div>
        </div>

        <!-- 2. Hero Card: Continuar con {dataset activo} -->
        ${ds ? this._renderActiveDatasetHero(ds, activeRun) : this._renderEmptyDatasetHero()}

        <!-- 3. Grid de dos columnas: Ejecuciones en curso + Trabajos recientes -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Columna Izquierda: Ejecuciones en curso -->
          <div class="workbench-card p-5 space-y-4">
            <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
              <div class="flex items-center space-x-2">
                <span class="text-[#4F67FF]">${icon("activity", "icon-sm")}</span>
                <h3 class="text-sm font-semibold font-sans text-[#F7F8FA]">Ejecuciones en curso</h3>
              </div>
              ${activeRun ? `
                <span class="${isRunning ? 'badge-warn' : 'badge-gain'} text-[11px] px-2.5 py-0.5 rounded-md font-mono font-medium flex items-center space-x-1">
                  ${isRunning
                    ? '<svg class="animate-spin h-3 w-3 text-[#F59E0B] inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span>ENTRENANDO</span>'
                    : '<span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span><span>COMPLETADO</span>'
                  }
                </span>
              ` : `
                <span class="text-xs text-[#8B95A7] font-mono">Sin ejecuciones</span>
              `}
            </div>

            ${activeRun ? `
              <div class="space-y-4">
                <div class="p-4 rounded-xl bg-[#11151E] border border-[#242A36] space-y-3">
                  <div class="flex items-center justify-between">
                    <div>
                      <div class="text-[10px] font-mono text-[#8B95A7] uppercase tracking-wider">Run ID: ${escapeHtml(activeRun.id)}</div>
                      <div class="text-sm font-semibold font-sans text-[#F7F8FA] mt-0.5">${escapeHtml(activeRun.dataset_name || activeRun.id)}</div>
                    </div>
                    <div class="text-right">
                      <div class="text-[10px] text-[#8B95A7] font-sans">Métrica</div>
                      <div class="text-xs font-mono font-bold text-[#4F67FF] uppercase">${escapeHtml(activeRun.metric || "ROC-AUC")}</div>
                    </div>
                  </div>

                  <div class="grid grid-cols-3 gap-2 text-center pt-1 border-t border-[#242A36]/60">
                    <div class="p-2 rounded-lg bg-[#161B26]">
                      <div class="text-[10px] text-[#8B95A7] font-sans">Mejor Modelo</div>
                      <div class="text-xs font-semibold text-[#F7F8FA] font-sans truncate mt-0.5">${escapeHtml(activeRun.best_model || "Pendiente")}</div>
                    </div>
                    <div class="p-2 rounded-lg bg-[#161B26]">
                      <div class="text-[10px] text-[#8B95A7] font-sans">Puntaje CV</div>
                      <div class="text-xs font-bold text-[#22C55E] font-mono mt-0.5">${activeRun.best_score != null ? Number(activeRun.best_score).toFixed(4) : "—"}</div>
                    </div>
                    <div class="p-2 rounded-lg bg-[#161B26]">
                      <div class="text-[10px] text-[#8B95A7] font-sans">Ensayos</div>
                      <div class="text-xs font-bold text-[#F7F8FA] font-mono mt-0.5">${activeRun.trials_count != null ? activeRun.trials_count : 0}</div>
                    </div>
                  </div>

                  <!-- Barra de progreso -->
                  <div class="space-y-1 pt-1">
                    <div class="flex justify-between text-[10px] font-sans text-[#8B95A7]">
                      <span>Estado del entrenamiento</span>
                      <span class="font-mono">${isRunning ? 'Optimizando modelos (en curso)' : 'Entrenamiento finalizado'}</span>
                    </div>
                    <div class="w-full bg-[#080A0F] h-1.5 rounded-full overflow-hidden border border-[#242A36]">
                      <div class="${isRunning ? 'bg-[#4F67FF] progress-striped w-3/4' : 'bg-[#22C55E] w-full'} h-full rounded-full"></div>
                    </div>
                  </div>
                </div>

                <div class="flex justify-end">
                  <button id="btnHomeGoToExperiments" class="text-xs text-[#4F67FF] hover:text-[#53C8FF] font-sans font-medium flex items-center space-x-1.5 transition-colors">
                    <span>Ver experimentos en detalle</span>
                    ${icon("arrow-right", "icon-sm")}
                  </button>
                </div>
              </div>
            ` : `
              <div class="p-8 text-center text-[#8B95A7] font-sans text-xs bg-[#11151E] rounded-xl border border-[#242A36] space-y-3">
                <p>No hay ejecuciones registradas para el dataset activo.</p>
                <button id="btnHomeStartFirstExp" class="btn-signal text-xs">
                  <span>Lanzar primer experimento</span>
                </button>
              </div>
            `}
          </div>

          <!-- Columna Derecha: Trabajos recientes (Background Jobs) -->
          <div class="workbench-card p-5 space-y-4">
            <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
              <div class="flex items-center space-x-2">
                <span class="text-[#4F67FF]">${icon("cpu", "icon-sm")}</span>
                <h3 class="text-sm font-semibold font-sans text-[#F7F8FA]">Trabajos recientes</h3>
              </div>
              <span class="text-xs font-mono text-[#8B95A7]">${recentJobs.length} registrados</span>
            </div>

            ${recentJobs.length > 0 ? `
              <div class="space-y-2 max-h-80 overflow-y-auto pr-1">
                ${recentJobs.map(job => {
                  const isSuccess = job.status === "completed";
                  const isFailed = job.status === "failed";
                  const badgeClass = isSuccess ? "badge-gain" : isFailed ? "badge-err" : "badge-warn";
                  const timeSec = job.result && job.result.training_time_seconds ? `${job.result.training_time_seconds.toFixed(1)}s` : "";
                  const modelTitle = (job.payload && job.payload.model_ids && job.payload.model_ids.join(", ")) || job.operation || "Tarea";

                  return `
                    <div class="p-3 rounded-lg bg-[#11151E] border border-[#242A36] hover:border-[#384355] transition-colors flex items-center justify-between text-xs font-sans">
                      <div class="space-y-0.5 min-w-0 pr-3">
                        <div class="flex items-center space-x-2">
                          <span class="font-medium text-[#F7F8FA] truncate capitalize">${escapeHtml(modelTitle)}</span>
                          <span class="${badgeClass} text-[9px] px-1.5 py-0.2 rounded font-mono uppercase">${escapeHtml(job.status)}</span>
                        </div>
                        <div class="text-[10px] text-[#8B95A7] font-mono truncate">
                          ID: ${escapeHtml(job.id.substring(0, 16))}... ${timeSec ? `• Duración: ${timeSec}` : ''}
                        </div>
                      </div>
                      <div class="text-right text-[11px] font-mono text-[#8B95A7] whitespace-nowrap">
                        ${job.result && job.result.primary_score != null ? `<span class="text-[#22C55E] font-bold">${Number(job.result.primary_score).toFixed(4)}</span>` : '—'}
                      </div>
                    </div>
                  `;
                }).join("")}
              </div>
            ` : `
              <div class="p-8 text-center text-[#8B95A7] font-sans text-xs bg-[#11151E] rounded-xl border border-[#242A36]">
                Sin actividad de cola registrada en este momento.
              </div>
            `}
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _renderActiveDatasetHero(ds, activeRun) {
    const rowCountText = ds.row_count != null ? Number(ds.row_count).toLocaleString() + " filas" : "Desconocido";
    const colCountText = ds.column_count != null ? `${ds.column_count} columnas` : "";
    const taskTypeClean = (ds.task_type || "binary_classification").replace("_", " ");
    const targetClean = ds.target_column || "Sin target definido";

    return `
      <div class="workbench-card p-6 border-[#4F67FF]/30 bg-gradient-to-br from-[#10151E] via-[#11151E] to-[#141B28] space-y-5">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="space-y-1.5">
            <div class="flex items-center space-x-2">
              <span class="px-2 py-0.5 rounded-md bg-[#4F67FF]/15 text-[#4F67FF] text-[10px] font-mono font-semibold uppercase tracking-wider border border-[#4F67FF]/30">
                Dataset Activo
              </span>
              <span class="text-xs font-mono text-[#8B95A7]">${escapeHtml(ds.id)}</span>
            </div>
            <h2 class="text-2xl font-bold font-sans text-[#F7F8FA] tracking-tight">
              ${escapeHtml(ds.name || ds.id)}
            </h2>
            <p class="text-xs text-[#8B95A7] font-sans max-w-2xl">
              Ruta: <span class="font-mono text-[#F7F8FA]/90">${escapeHtml(ds.path || "Cargado en memoria")}</span>
            </p>
          </div>

          <!-- Acciones principales -->
          <div class="flex flex-wrap items-center gap-2.5">
            <button id="btnHomeContinue" class="btn-signal text-sm px-5 py-2.5 flex items-center space-x-2 shadow-lg shadow-[#4F67FF]/20">
              <span>Continuar con ${escapeHtml(ds.name || 'dataset')}</span>
              ${icon("arrow-right", "icon-sm")}
            </button>
            <button id="btnHomeAddDataset" class="bg-[#161B26] hover:bg-[#1E2536] text-[#F7F8FA] border border-[#242A36] px-4 py-2.5 rounded-lg text-xs font-sans font-medium flex items-center space-x-1.5 transition-colors">
              ${icon("plus", "icon-sm")}
              <span>Añadir dataset</span>
            </button>
            <button id="btnHomeOpenDataset" class="bg-[#161B26] hover:bg-[#1E2536] text-[#F7F8FA] border border-[#242A36] px-4 py-2.5 rounded-lg text-xs font-sans font-medium flex items-center space-x-1.5 transition-colors">
              ${icon("folder", "icon-sm")}
              <span>Abrir dataset guardado</span>
            </button>
          </div>
        </div>

        <!-- Metadatos del Dataset Activo -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-[#242A36]">
          <div class="p-3 rounded-xl bg-[#0D1017]/80 border border-[#242A36]/80">
            <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">Dimensiones</div>
            <div class="text-sm font-bold font-mono text-[#F7F8FA] mt-0.5">${rowCountText} • ${colCountText}</div>
          </div>
          <div class="p-3 rounded-xl bg-[#0D1017]/80 border border-[#242A36]/80">
            <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">Variable Objetivo</div>
            <div class="text-sm font-bold font-mono text-[#4F67FF] mt-0.5 truncate">${escapeHtml(targetClean)}</div>
          </div>
          <div class="p-3 rounded-xl bg-[#0D1017]/80 border border-[#242A36]/80">
            <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">Tipo de Problema</div>
            <div class="text-sm font-semibold font-sans text-[#F7F8FA] mt-0.5 capitalize">${escapeHtml(taskTypeClean)}</div>
          </div>
          <div class="p-3 rounded-xl bg-[#0D1017]/80 border border-[#242A36]/80">
            <div class="text-[10px] font-sans font-medium text-[#8B95A7] uppercase">Mejor Resultado</div>
            <div class="text-sm font-bold font-mono text-[#22C55E] mt-0.5">
              ${activeRun && activeRun.best_score != null ? `${activeRun.best_model || 'Modelo'} (${Number(activeRun.best_score).toFixed(4)})` : 'Sin evaluar'}
            </div>
          </div>
        </div>
      </div>
    `;
  }

  _renderEmptyDatasetHero() {
    return `
      <div class="workbench-card p-8 border-dashed border-[#242A36] text-center space-y-4">
        <div class="inline-block p-3 rounded-2xl bg-[#161B26] border border-[#242A36] text-[#4F67FF]">
          ${icon("database", "icon-lg")}
        </div>
        <div class="max-w-md mx-auto space-y-1">
          <h2 class="text-lg font-bold font-sans text-[#F7F8FA]">Sin dataset activo</h2>
          <p class="text-xs text-[#8B95A7] font-sans">
            Para comenzar a explorar, generar hipótesis y entrenar modelos en el laboratorio, selecciona un dataset o añade uno nuevo.
          </p>
        </div>
        <div class="flex items-center justify-center space-x-3 pt-2">
          <button id="btnHomeAddDatasetEmpty" class="btn-signal text-xs flex items-center space-x-2">
            ${icon("plus", "icon-sm")}
            <span>Añadir dataset</span>
          </button>
          <button id="btnHomeOpenDatasetEmpty" class="bg-[#161B26] hover:bg-[#1E2536] text-[#F7F8FA] border border-[#242A36] px-4 py-2 rounded-lg text-xs font-sans font-medium transition-colors">
            <span>Abrir dataset guardado</span>
          </button>
        </div>
      </div>
    `;
  }

  _bindEvents() {
    this.container.querySelector("#btnHomeContinue")?.addEventListener("click", () => {
      store.setNav("dataset");
    });

    this.container.querySelector("#btnHomeAddDataset")?.addEventListener("click", () => {
      store.setNav("dataset");
      bus.emit("dataset:show-register");
    });

    this.container.querySelector("#btnHomeAddDatasetEmpty")?.addEventListener("click", () => {
      store.setNav("dataset");
      bus.emit("dataset:show-register");
    });

    this.container.querySelector("#btnHomeOpenDataset")?.addEventListener("click", () => {
      bus.emit("modal:switch-dataset");
    });

    this.container.querySelector("#btnHomeOpenDatasetEmpty")?.addEventListener("click", () => {
      bus.emit("modal:switch-dataset");
    });

    this.container.querySelector("#btnHomeGoToExperiments")?.addEventListener("click", () => {
      store.setNav("experiments");
    });

    this.container.querySelector("#btnHomeNewExpDirect")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnHomeStartFirstExp")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });
  }

  destroy() {
    // cleanup
  }
}
