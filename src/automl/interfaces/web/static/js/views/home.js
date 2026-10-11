/**
 * HomeView — Visual ML Laboratory Home (Fase P1)
 * Implements the guided 4-stage ML lifecycle and honest execution state fidelity:
 * 1. Flujo de Trabajo Guiado: Datos → Exploración → Experimentos → Resultados
 * 2. Hero Card: Estado actual con detalles técnicos colapsables y acción principal según estado.
 * 3. Ejecuciones en curso: Muestra EXCLUSIVAMENTE operaciones activas reales; el motor en reposo
 *    muestra el último modelo completado de forma separada sin fingir que está en curso.
 * 4. Trabajos recientes: Historial verídico con distinción estricta de todos los estados.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import {
  escapeHtml,
  isRunActive,
  isJobActive,
  getJobStatusLabel,
} from "../utils.js";

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

      const { activeDataset } = store.resolveContext({ datasets, runs });
      this.activeDataset = activeDataset;
    } catch (e) {
      console.warn("[HomeView] Error fetching laboratory data:", e);
    } finally {
      this.loading = false;
    }
  }

  render() {
    const ds = this.activeDataset || store.getActiveDataset();
    const runs = this.runs || [];
    const dsRuns = ds ? runs.filter(r => r.dataset_id === ds.id) : [];
    const activeRun = store.getActiveRun() || dsRuns.find(r => isRunActive(r)) || dsRuns[0] || null;

    // Strict separation: verify if there is an actual active operation in progress
    const activeJobs = (this.jobs || []).filter(j => isJobActive(j));
    const activeJob = activeJobs[0] || null;
    const isRunRunning = Boolean(activeRun && isRunActive(activeRun));
    const isTraining = Boolean(activeJob || isRunRunning);

    const trialsCount = activeRun && activeRun.trials_count != null ? activeRun.trials_count : 0;
    const hasResults = trialsCount > 0;

    // Filter recent completed/failed/historical jobs
    const recentJobs = (this.jobs || []).slice(0, 8);

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
            <button id="btnHomeNewExpDirect" class="btn-signal text-xs flex items-center space-x-1.5 shadow-md shadow-[#4F67FF]/20">
              ${icon("play", "icon-sm")}
              <span>Nuevo experimento</span>
            </button>
          </div>
        </div>

        <!-- 2. Flujo de Trabajo Guiado (4 Etapas del Ciclo de ML) -->
        ${this._renderGuidedWorkflowBanner(ds, activeRun, isTraining, hasResults, trialsCount)}

        <!-- 3. Hero Card: Estado actual y dataset activo -->
        ${ds ? this._renderActiveDatasetHero(ds, activeRun, isTraining, hasResults) : this._renderEmptyDatasetHero()}

        <!-- 4. Grid de dos columnas: Ejecuciones en curso + Trabajos recientes -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Columna Izquierda: Ejecuciones en curso -->
          <div class="workbench-card p-5 space-y-4">
            <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
              <div class="flex items-center space-x-2">
                <span class="text-[#4F67FF]">${icon("activity", "icon-sm")}</span>
                <h3 class="text-sm font-semibold font-sans text-[#F7F8FA]">Ejecuciones en curso</h3>
              </div>

              <!-- Badge con estado estrictamente verídico -->
              ${isTraining ? `
                <span class="badge-warn text-[11px] px-2.5 py-0.5 rounded-md font-mono font-medium flex items-center space-x-1.5">
                  <svg class="animate-spin h-3 w-3 text-[#F59E0B] inline" viewBox="0 0 24 24" fill="none">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>ENTRENANDO</span>
                </span>
              ` : (hasResults ? `
                <span class="badge-gain text-[11px] px-2.5 py-0.5 rounded-md font-mono font-medium flex items-center space-x-1.5">
                  <span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span>
                  <span>MOTOR EN REPOSO</span>
                </span>
              ` : `
                <span class="text-[#8B95A7] bg-[#161B26] border border-[#242A36] text-[11px] px-2.5 py-0.5 rounded-md font-mono font-medium flex items-center space-x-1.5">
                  <span class="w-1.5 h-1.5 rounded-full bg-[#8B95A7]"></span>
                  <span>LISTO PARA ENTRENAR</span>
                </span>
              `)}
            </div>

            <!-- Contenido según estado real -->
            ${isTraining ? `
              <!-- Caso 1: Entrenamiento o trabajo activo en curso -->
              <div class="space-y-4">
                <div class="p-4 rounded-xl bg-[#11151E] border border-[#4F67FF]/40 space-y-3">
                  <div class="flex items-center justify-between">
                    <div>
                      <div class="text-[10px] font-mono text-[#8B95A7] uppercase tracking-wider">
                        ${activeJob ? `Trabajo ID: ${escapeHtml(activeJob.id.substring(0, 16))}...` : `Run ID: ${escapeHtml(activeRun.id)}`}
                      </div>
                      <div class="text-sm font-semibold font-sans text-[#F7F8FA] mt-0.5">
                        ${activeJob ? escapeHtml(activeJob.operation || 'Optimización de modelos') : escapeHtml(activeRun.dataset_name || activeRun.id)}
                      </div>
                    </div>
                    <div class="text-right">
                      <div class="text-[10px] text-[#8B95A7] font-sans">Estado</div>
                      <div class="text-xs font-mono font-bold text-[#4F67FF] uppercase">
                        ${activeJob ? escapeHtml(getJobStatusLabel(activeJob.status)) : 'EN PROCESO'}
                      </div>
                    </div>
                  </div>

                  <!-- Barra de progreso con datos reales sin porcentajes ficticios -->
                  <div class="space-y-1.5 pt-1">
                    <div class="flex justify-between text-[10px] font-sans text-[#8B95A7]">
                      <span>Progreso de la tarea</span>
                      <span class="font-mono">
                        ${activeJob && activeJob.total > 0
                          ? `${activeJob.completed || 0}/${activeJob.total} pasos (${Math.round(((activeJob.completed || 0) * 100) / activeJob.total)}%)`
                          : 'Optimizando modelos (en curso)'}
                      </span>
                    </div>
                    <div class="w-full bg-[#080A0F] h-1.5 rounded-full overflow-hidden border border-[#242A36]">
                      ${activeJob && activeJob.total > 0 ? `
                        <div class="bg-[#4F67FF] progress-striped h-full rounded-full transition-all duration-300" style="width: ${Math.max(8, Math.round(((activeJob.completed || 0) * 100) / activeJob.total))}%"></div>
                      ` : `
                        <div class="bg-[#4F67FF] h-full w-full rounded-full animate-pulse"></div>
                      `}
                    </div>
                  </div>
                </div>

                <div class="flex items-center justify-between pt-1">
                  <span class="text-[11px] text-[#8B95A7] font-sans">Operación ejecutándose en segundo plano.</span>
                  <button id="btnHomeGoToExperiments" class="btn-signal text-xs flex items-center space-x-1.5">
                    <span>Ver progreso en detalle</span>
                    ${icon("arrow-right", "icon-sm")}
                  </button>
                </div>
              </div>
            ` : (hasResults ? `
              <!-- Caso 2: Sin ejecuciones activas, pero con historial previo disponible -->
              <div class="space-y-4">
                <div class="p-4 rounded-xl bg-[#11151E] border border-[#242A36] space-y-3">
                  <div class="flex items-center justify-between">
                    <div>
                      <div class="text-[10px] font-mono text-[#8B95A7] uppercase tracking-wider">Última Ejecución · Run ID: ${escapeHtml(activeRun.id)}</div>
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
                      <div class="text-xs font-semibold text-[#F7F8FA] font-sans truncate mt-0.5">${activeRun.best_model ? escapeHtml(activeRun.best_model) : '<span class="text-[#8B95A7] font-normal">—</span>'}</div>
                    </div>
                    <div class="p-2 rounded-lg bg-[#161B26]">
                      <div class="text-[10px] text-[#8B95A7] font-sans">Puntaje CV</div>
                      <div class="text-xs font-bold text-[#22C55E] font-mono mt-0.5">${activeRun.best_score != null ? Number(activeRun.best_score).toFixed(4) : '<span class="text-[#8B95A7] font-mono font-normal">—</span>'}</div>
                    </div>
                    <div class="p-2 rounded-lg bg-[#161B26]">
                      <div class="text-[10px] text-[#8B95A7] font-sans">Ensayos</div>
                      <div class="text-xs font-bold text-[#F7F8FA] font-mono mt-0.5">${trialsCount}</div>
                    </div>
                  </div>

                  <div class="text-[11px] text-[#8B95A7] pt-1">
                    No hay ejecuciones activas en este momento. Los modelos de esta sesión están evaluados y listos para consultar.
                  </div>
                </div>

                <div class="flex items-center justify-between pt-1">
                  <button id="btnHomeStartFirstExpInner" class="btn-technical text-xs flex items-center space-x-1.5">
                    ${icon("play", "icon-sm")}
                    <span>Lanzar nuevo experimento</span>
                  </button>
                  <button id="btnHomeGoToExperiments" class="text-xs text-[#4F67FF] hover:text-[#53C8FF] font-sans font-medium flex items-center space-x-1.5 transition-colors">
                    <span>Ver experimentos en detalle</span>
                    ${icon("arrow-right", "icon-sm")}
                  </button>
                </div>
              </div>
            ` : (activeRun ? `
              <!-- Caso 3: Run creado pero sin ensayos ejecutados todavía -->
              <div class="p-8 text-center text-[#8B95A7] font-sans text-xs bg-[#11151E] rounded-xl border border-[#242A36] space-y-3">
                <p>Dataset preparado. Sin modelos entrenados todavía.</p>
                <button id="btnHomeStartFirstExpInner" class="btn-signal text-xs flex items-center space-x-1.5 mx-auto">
                  ${icon("play", "icon-sm")}
                  <span>Lanzar primer experimento</span>
                </button>
              </div>
            ` : `
              <!-- Caso 4: Sin ejecuciones registradas para este dataset -->
              <div class="p-8 text-center text-[#8B95A7] font-sans text-xs bg-[#11151E] rounded-xl border border-[#242A36] space-y-3">
                <p>No hay ejecuciones registradas para el dataset activo.</p>
                <button id="btnHomeStartFirstExp" class="btn-signal text-xs flex items-center space-x-1.5 mx-auto">
                  ${icon("play", "icon-sm")}
                  <span>Lanzar primer experimento</span>
                </button>
              </div>
            `))}
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
                  const isRunning = isJobActive(job);
                  const isCancelled = job.status === "cancelled" || job.status === "interrupted";
                  const badgeClass = isSuccess ? "badge-gain" : isFailed ? "badge-err" : isRunning ? "badge-warn" : isCancelled ? "text-[#8B95A7] bg-[#161B26]" : "badge-sys";
                  const timeSec = job.result && job.result.training_time_seconds ? `${job.result.training_time_seconds.toFixed(1)}s` : "";
                  const modelTitle = (job.payload && job.payload.model_ids && job.payload.model_ids.join(", ")) || job.operation || "Tarea";

                  return `
                    <div class="p-3 rounded-lg bg-[#11151E] border border-[#242A36] hover:border-[#384355] transition-colors flex items-center justify-between text-xs font-sans">
                      <div class="space-y-0.5 min-w-0 pr-3">
                        <div class="flex items-center space-x-2">
                          <span class="font-medium text-[#F7F8FA] truncate capitalize">${escapeHtml(modelTitle)}</span>
                          <span class="${badgeClass} text-[9px] px-1.5 py-0.2 rounded font-mono uppercase">${escapeHtml(getJobStatusLabel(job.status))}</span>
                        </div>
                        <div class="text-[10px] text-[#8B95A7] font-mono truncate">
                          ID: ${escapeHtml(job.id.substring(0, 16))}... ${timeSec ? `• Duración: ${timeSec}` : ''}
                        </div>
                        ${job.error ? `<div class="text-[10px] text-[#EF4444] font-mono truncate">${escapeHtml(job.error)}</div>` : ''}
                      </div>
                      <div class="text-right text-[11px] font-mono text-[#8B95A7] whitespace-nowrap">
                        ${job.result && job.result.primary_score != null ? `<span class="text-[#22C55E] font-bold">${escapeHtml(job.result.primary_metric || 'Score')}: ${Number(job.result.primary_score).toFixed(4)}</span>` : '—'}
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

  /**
   * Guided 4-Stage Workflow Stepper: Datos → Exploración → Experimentos → Resultados
   */
  _renderGuidedWorkflowBanner(ds, activeRun, isTraining, hasResults, trialsCount) {
    const hasData = Boolean(ds);
    const step1Label = hasData ? (ds.row_count ? `${Number(ds.row_count).toLocaleString()} filas` : "Preparado") : "Pendiente";
    const step2Label = hasData ? "Estudios e hipótesis" : "Requiere datos";
    const step3Label = isTraining ? "Entrenando en vivo" : (hasResults ? `${trialsCount} ensayos evaluados` : "Por configurar");
    const step4Label = hasResults ? (activeRun && activeRun.best_score != null ? `Score: ${Number(activeRun.best_score).toFixed(4)}` : "Disponible") : "Pendiente";

    return `
      <div class="workbench-card p-4 space-y-3 bg-[#0E121B] border-[#242A36]">
        <div class="flex items-center justify-between text-xs font-sans pb-1">
          <div class="flex items-center space-x-2">
            <span class="text-[#4F67FF]">${icon("workflow", "icon-sm")}</span>
            <span class="font-bold text-[#F7F8FA] uppercase tracking-wider text-[11px] font-mono">Ciclo de Trabajo AutoML</span>
          </div>
          <span class="text-[#8B95A7] text-[11px] font-sans">
            Guía de 4 etapas orientada a reproducibilidad y evidencia científica
          </span>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          <!-- Etapa 1: Datos -->
          <div id="cardFlowData" class="p-3 rounded-xl bg-[#11151E] border ${hasData ? 'border-[#22C55E]/40 hover:border-[#22C55E]' : 'border-[#242A36] hover:border-[#384355]'} transition-all cursor-pointer group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-[10px] font-mono uppercase tracking-wider text-[#8B95A7]">01 DATOS</span>
              <span class="${hasData ? 'badge-gain' : 'badge-sys'} text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">${hasData ? 'COMPLETO' : 'PENDIENTE'}</span>
            </div>
            <div class="text-xs font-bold text-[#F7F8FA] group-hover:text-[#4F67FF] transition-colors">Ingesta & Calidad</div>
            <div class="text-[10px] text-[#8B95A7] mt-0.5 font-mono truncate">${escapeHtml(step1Label)}</div>
          </div>

          <!-- Etapa 2: Exploración -->
          <div id="cardFlowExplore" class="p-3 rounded-xl bg-[#11151E] border ${hasData ? 'border-[#53C8FF]/30 hover:border-[#53C8FF]' : 'border-[#242A36] opacity-60'} transition-all cursor-pointer group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-[10px] font-mono uppercase tracking-wider text-[#8B95A7]">02 EXPLORACIÓN</span>
              <span class="${hasData ? 'badge-gain' : 'badge-sys'} text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">${hasData ? 'DISPONIBLE' : 'BLOQUEADO'}</span>
            </div>
            <div class="text-xs font-bold text-[#F7F8FA] group-hover:text-[#53C8FF] transition-colors">Estudios & Hipótesis</div>
            <div class="text-[10px] text-[#8B95A7] mt-0.5 font-mono truncate">${escapeHtml(step2Label)}</div>
          </div>

          <!-- Etapa 3: Experimentos -->
          <div id="cardFlowExp" class="p-3 rounded-xl bg-[#11151E] border ${isTraining ? 'border-[#F59E0B]/60 shadow-md shadow-[#F59E0B]/10' : hasResults ? 'border-[#22C55E]/40 hover:border-[#22C55E]' : 'border-[#242A36] hover:border-[#384355]'} transition-all cursor-pointer group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-[10px] font-mono uppercase tracking-wider text-[#8B95A7]">03 EXPERIMENTOS</span>
              <span class="${isTraining ? 'badge-warn' : hasResults ? 'badge-gain' : 'badge-sys'} text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">${isTraining ? 'ACTIVO' : hasResults ? 'EVALUADO' : 'POR INICIAR'}</span>
            </div>
            <div class="text-xs font-bold text-[#F7F8FA] group-hover:text-[#4F67FF] transition-colors">Modelos & HPO</div>
            <div class="text-[10px] text-[#8B95A7] mt-0.5 font-mono truncate">${escapeHtml(step3Label)}</div>
          </div>

          <!-- Etapa 4: Resultados -->
          <div id="cardFlowResults" class="p-3 rounded-xl bg-[#11151E] border ${hasResults ? 'border-[#22C55E]/40 hover:border-[#22C55E]' : 'border-[#242A36] opacity-60'} transition-all cursor-pointer group">
            <div class="flex items-center justify-between mb-1.5">
              <span class="text-[10px] font-mono uppercase tracking-wider text-[#8B95A7]">04 RESULTADOS</span>
              <span class="${hasResults ? 'badge-gain' : 'badge-sys'} text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">${hasResults ? 'LISTO' : 'PENDIENTE'}</span>
            </div>
            <div class="text-xs font-bold text-[#F7F8FA] group-hover:text-[#22C55E] transition-colors">Leaderboard & Artefactos</div>
            <div class="text-[10px] text-[#8B95A7] mt-0.5 font-mono truncate">${escapeHtml(step4Label)}</div>
          </div>
        </div>
      </div>
    `;
  }

  _renderActiveDatasetHero(ds, activeRun, isTraining, hasResults) {
    const rowCountText = ds.row_count != null ? Number(ds.row_count).toLocaleString() + " filas" : "Desconocido";
    const colCountText = ds.column_count != null ? `${ds.column_count} columnas` : "";
    const taskTypeClean = (ds.task_type || "binary_classification").replace("_", " ");
    const targetClean = ds.target_column || "Sin target definido";

    // Primary CTA selection strictly based on state (Section 5.3)
    let primaryCtaHtml = "";
    if (isTraining) {
      primaryCtaHtml = `
        <button id="btnHomeContinue" class="btn-signal text-sm px-5 py-2.5 flex items-center space-x-2 shadow-lg shadow-[#4F67FF]/20">
          <span>Ver progreso de entrenamiento</span>
          ${icon("arrow-right", "icon-sm")}
        </button>
      `;
    } else if (hasResults) {
      primaryCtaHtml = `
        <button id="btnHomeContinue" class="btn-signal text-sm px-5 py-2.5 flex items-center space-x-2 shadow-lg shadow-[#4F67FF]/20">
          <span>Ver resultados de ${escapeHtml(ds.name || 'dataset')}</span>
          ${icon("arrow-right", "icon-sm")}
        </button>
      `;
    } else {
      primaryCtaHtml = `
        <button id="btnHomeContinue" class="btn-signal text-sm px-5 py-2.5 flex items-center space-x-2 shadow-lg shadow-[#4F67FF]/20">
          <span>Continuar con ${escapeHtml(ds.name || 'dataset')}</span>
          ${icon("arrow-right", "icon-sm")}
        </button>
      `;
    }

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
          </div>

          <!-- Acciones principales -->
          <div class="flex flex-wrap items-center gap-2.5">
            ${primaryCtaHtml}
            <button id="btnHomeAddDataset" class="bg-[#161B26] hover:bg-[#1E2536] text-[#F7F8FA] border border-[#242A36] px-4 py-2.5 rounded-lg text-xs font-sans font-medium flex items-center space-x-1.5 transition-colors">
              ${icon("plus", "icon-sm")}
              <span>Añadir dataset</span>
            </button>
            <button id="btnHomeOpenDataset" class="bg-[#161B26] hover:bg-[#1E2536] text-[#F7F8FA] border border-[#242A36] px-4 py-2.5 rounded-lg text-xs font-sans font-medium flex items-center space-x-1.5 transition-colors">
              ${icon("layers", "icon-sm")}
              <span>Abrir dataset guardado</span>
            </button>
          </div>
        </div>

        <!-- Metadatos de alto nivel del Dataset Activo -->
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

        <!-- Detalles técnicos colapsables para mantener la vista limpia (Section 5.4) -->
        <details class="text-xs text-[#8B95A7] pt-2 border-t border-[#242A36]/60 cursor-pointer">
          <summary class="hover:text-[#F7F8FA] flex items-center space-x-1 select-none font-mono text-[11px]">
            <span>Detalles técnicos del entorno de datos</span>
          </summary>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2 pt-2 bg-[#090C12] p-3 rounded-lg border border-[#242A36] text-[11px] font-mono">
            <div><span class="text-[#8B95A7]/70">Ruta de origen:</span> <span class="text-[#F7F8FA] break-all">${escapeHtml(ds.path || "En memoria")}</span></div>
            <div><span class="text-[#8B95A7]/70">ID único:</span> <span class="text-[#F7F8FA]">${escapeHtml(ds.id)}</span></div>
          </div>
        </details>
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
            Para comenzar a explorar, formular hipótesis y entrenar modelos en el laboratorio, selecciona un dataset existente o añade uno nuevo.
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
      const activeRun = store.getActiveRun();
      const trialsCount = activeRun && activeRun.trials_count != null ? activeRun.trials_count : 0;
      if (trialsCount > 0) {
        store.setNav("experiments");
      } else {
        store.setNav("dataset");
      }
    });

    this.container.querySelector("#btnHomeAddDataset")?.addEventListener("click", () => {
      bus.emit("modal:register-dataset");
    });

    this.container.querySelector("#btnHomeAddDatasetEmpty")?.addEventListener("click", () => {
      bus.emit("modal:register-dataset");
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

    this.container.querySelector("#btnHomeStartFirstExpInner")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    // Guided workflow banner cards navigation
    this.container.querySelector("#cardFlowData")?.addEventListener("click", () => {
      store.setNav("dataset");
    });

    this.container.querySelector("#cardFlowExplore")?.addEventListener("click", () => {
      store.setNav("explore");
    });

    this.container.querySelector("#cardFlowExp")?.addEventListener("click", () => {
      store.setNav("experiments");
    });

    this.container.querySelector("#cardFlowResults")?.addEventListener("click", () => {
      store.setNav("evidence");
    });
  }

  destroy() {
    // cleanup
  }
}
