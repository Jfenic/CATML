/**
 * AgentDrawer — Autonomous Agent Lateral Co-pilot
 * Embodies the core CATML principle: "Proponer ≠ Aceptar"
 * Proposes explicit hypothesis objects requiring empirical critic verification before promotion.
 * Adheres strictly to centralized context: activeDataset and activeRun.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml } from "../utils.js";
import { isRunActive } from "../utils.js";

export class AgentDrawer {
  constructor() {
    this.container = null;
    this.hypotheses = [];
    this.currentContextKey = null;
  }

  async mount(container) {
    this.container = container;
    await this.refresh();
  }

  async refresh() {
    if (!this.container) return;
    const activeDataset = store.getActiveDataset();
    const activeRun = store.getActiveRun();
    const contextKey = `${activeDataset ? activeDataset.id : 'none'}_${activeRun ? activeRun.id : 'none'}`;

    // If context changed, immediately clear stale hypotheses from memory before fetching
    if (this.currentContextKey !== contextKey) {
      this.hypotheses = [];
      this.currentContextKey = contextKey;
    }

    if (!activeRun) {
      this.hypotheses = [];
      this.render();
      return;
    }

    try {
      const resp = await api.getAgentHypotheses(activeRun.id);
      // Verify context has not changed while awaiting response
      if (this.currentContextKey === contextKey) {
        this.hypotheses = resp.hypotheses || [];
      }
    } catch {
      this.hypotheses = [];
    }
    this.render();
  }

  render() {
    if (!this.container) return;
    const activeDataset = store.getActiveDataset();
    const activeRun = store.getActiveRun();
    const dsName = activeDataset ? (activeDataset.name || activeDataset.id) : "No active dataset";

    // 1. Build Context Header & Action Cards
    let contextCardHtml = "";
    let nextActionCardHtml = "";

    if (!activeDataset) {
      contextCardHtml = `
        <div class="bg-[#11151E] border border-[#242A36] rounded-xl p-4 space-y-2.5">
          <div class="text-[10px] uppercase font-mono tracking-wider text-[#8B95A7] font-semibold">Contexto del Laboratorio</div>
          <div class="p-3 bg-[#161B26] border border-[#242A36] rounded-lg text-center space-y-2">
            <div class="text-xs font-semibold text-[#F7F8FA]">Sin dataset seleccionado</div>
            <p class="text-[11px] text-[#8B95A7]">Selecciona un dataset para comenzar.</p>
            <button id="btnAgentSelectDsPrompt" class="btn-signal text-xs px-3 py-1.5 rounded-lg inline-flex items-center gap-1.5 mt-1">
              ${icon("database", "icon-sm")}
              <span>Seleccionar dataset</span>
            </button>
          </div>
        </div>
      `;

      nextActionCardHtml = `
        <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-2">
          <div class="flex items-center justify-between">
            <span class="text-[10px] uppercase font-mono tracking-wider text-[#4F67FF] font-semibold">Siguiente Acción</span>
            <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-[#4F67FF]/10 text-[#4F67FF] border border-[#4F67FF]/20 font-medium">Requerido</span>
          </div>
          <p class="text-xs text-[#8B95A7] font-sans leading-relaxed">
            Registra un dataset en el workspace para iniciar la exploración autónoma, formulación de hipótesis y entrenamiento de modelos.
          </p>
        </div>
      `;
    } else if (!activeRun) {
      const taskType = activeDataset.task_type ? activeDataset.task_type.replace("_", " ") : "No configurada";
      contextCardHtml = `
        <div class="bg-[#11151E] border border-[#242A36] rounded-xl p-4 space-y-2.5">
          <div class="text-[10px] uppercase font-mono tracking-wider text-[#8B95A7] font-semibold">Contexto Activo</div>
          <div class="space-y-1.5 font-mono text-xs">
            <div class="flex justify-between"><span class="text-[#8B95A7]">Dataset:</span> <span class="text-[#F7F8FA] font-semibold">${escapeHtml(dsName)}</span></div>
            <div class="flex justify-between"><span class="text-[#8B95A7]">Ejecución:</span> <span class="text-[#8B95A7]">Sin ejecución activa</span></div>
            <div class="flex justify-between"><span class="text-[#8B95A7]">Tarea:</span> <span class="text-[#F7F8FA] capitalize">${escapeHtml(taskType)}</span></div>
            <div class="flex justify-between"><span class="text-[#8B95A7]">Estado:</span> <span class="text-[#F59E0B] font-semibold flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#F59E0B]"></span>Listo para entrenar</span></div>
          </div>
        </div>
      `;

      nextActionCardHtml = `
        <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-[10px] uppercase font-mono tracking-wider text-[#4F67FF] font-semibold">Siguiente Paso</span>
            <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-[#4F67FF]/10 text-[#4F67FF] border border-[#4F67FF]/20 font-medium">Entrenamiento</span>
          </div>
          <p class="text-xs text-[#F7F8FA] font-sans leading-relaxed">
            Dataset preparado. Configura y lanza una primera ejecución de experimentos sobre <code class="font-mono text-[#4F67FF] bg-[#11151E] px-1 py-0.5 rounded">${escapeHtml(activeDataset.name)}</code> para obtener hipótesis y comparativas.
          </p>
          <div class="pt-2 border-t border-[#242A36]">
            <button id="btnAgentLaunchFirstExp" class="w-full btn-signal text-xs py-2 flex items-center justify-center gap-1.5">
              ${icon("play", "icon-sm")}
              <span>Lanzar primer experimento</span>
            </button>
          </div>
        </div>
      `;
    } else {
      const isRunning = isRunActive(activeRun);
      const statusText = isRunning ? "Entrenando" : "Listo";
      const statusColor = isRunning ? "#F59E0B" : "#22C55E";
      const taskType = activeRun.task_type || activeDataset.task_type || "Auto";

      contextCardHtml = `
        <div class="bg-[#11151E] border border-[#242A36] rounded-xl p-4 space-y-2.5">
          <div class="text-[10px] uppercase font-mono tracking-wider text-[#8B95A7] font-semibold">Contexto Activo</div>
          <div class="space-y-1.5 font-mono text-xs">
            <div class="flex justify-between"><span class="text-[#8B95A7]">Dataset:</span> <span class="text-[#F7F8FA] font-semibold">${escapeHtml(dsName)}</span></div>
            <div class="flex justify-between"><span class="text-[#8B95A7]">Ejecución:</span> <span class="text-[#4F67FF] font-semibold">${escapeHtml(activeRun.id)}</span></div>
            <div class="flex justify-between"><span class="text-[#8B95A7]">Tarea:</span> <span class="text-[#F7F8FA] capitalize">${escapeHtml(taskType)}</span></div>
            <div class="flex justify-between"><span class="text-[#8B95A7]">Estado:</span> <span class="font-semibold flex items-center gap-1.5" style="color: ${statusColor}"><span class="w-1.5 h-1.5 rounded-full" style="background-color: ${statusColor}"></span>${escapeHtml(statusText)}</span></div>
          </div>
        </div>
      `;

      nextActionCardHtml = `
        <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-3">
          <div class="flex items-center justify-between">
            <span class="text-[10px] uppercase font-mono tracking-wider text-[#4F67FF] font-semibold">Siguiente Acción Recomendada</span>
            <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-[#4F67FF]/10 text-[#4F67FF] border border-[#4F67FF]/20 font-medium">Plan Empírico</span>
          </div>
          <p class="text-xs text-[#F7F8FA] font-sans leading-relaxed">
            Evaluar modelos candidatos y validación cruzada en el dataset <code class="font-mono text-[#4F67FF] bg-[#11151E] px-1 py-0.5 rounded">${escapeHtml(activeDataset.name)}</code> (Run: ${escapeHtml(activeRun.id)}).
          </p>
          <div class="flex items-center space-x-2 pt-2 border-t border-[#242A36]">
            <button id="btnAgentReviewPlan" class="flex-1 bg-[#11151E] hover:bg-[#1A202C] text-[#F7F8FA] border border-[#242A36] py-1.5 rounded-lg text-xs font-medium transition-colors">
              Revisar plan
            </button>
            <button id="btnAgentRunPlan" class="flex-1 bg-[#4F67FF] hover:bg-[#3D56FF] text-white py-1.5 rounded-lg text-xs font-semibold transition-all shadow-sm flex items-center justify-center gap-1">
              ${icon("play", "icon-sm")}
              <span>Ejecutar</span>
            </button>
          </div>
        </div>
      `;
    }

    // 2. Build Hypotheses HTML
    let hypothesesHtml = "";
    if (!activeDataset || !activeRun) {
      hypothesesHtml = `
        <div class="p-6 text-center text-[#8B95A7]/70 text-xs font-sans rounded-xl bg-[#11151E] border border-[#242A36]">
          Sin hipótesis de ejecución activa.
          <div class="text-[10px] text-[#8B95A7]/50 mt-1">Las propuestas autónomas se generan y verifican dentro de cada ejecución.</div>
        </div>
      `;
    } else if (!this.hypotheses || this.hypotheses.length === 0) {
      hypothesesHtml = `
        <div class="p-6 text-center text-[#8B95A7]/70 text-xs font-sans rounded-xl bg-[#11151E] border border-[#242A36]">
          No hay hipótesis registradas para la ejecución <span class="font-mono text-[#F7F8FA]">${escapeHtml(activeRun.id)}</span>.
          <div class="text-[10px] text-[#8B95A7]/50 mt-1">Las propuestas del agente siguen el principio: Proponer ≠ Aceptar.</div>
        </div>
      `;
    } else {
      hypothesesHtml = this.hypotheses.map(hyp => {
        const isPromoted = hyp.critic_decision === "PROMOTE" || hyp.status === "ACCEPTED";
        const isRejected = hyp.critic_decision === "REJECT" || hyp.status === "REJECTED";
        const badgeClass = isPromoted ? "badge-gain" : isRejected ? "badge-err" : "badge-warn";
        const badgeIcon = isPromoted ? icon("check", "icon-sm") : isRejected ? icon("x", "icon-sm") : `<span class="w-1.5 h-1.5 rounded-full bg-[#F59E0B]"></span>`;
        const badgeText = isPromoted ? "Promovida" : isRejected ? "Rechazada" : "Propuesta (Pendiente)";

        const empiricalResults = (hyp.before_score !== null && hyp.before_score !== undefined)
          ? `
            <div class="bg-[#0D1017] p-2.5 rounded-lg border border-[#242A36] font-mono text-[11px] space-y-1">
              <div class="flex justify-between"><span class="text-[#8B95A7]">Benchmark:</span> <span class="text-[#F7F8FA]">${escapeHtml(hyp.before_score)}</span></div>
              ${hyp.after_score !== null && hyp.after_score !== undefined ? `<div class="flex justify-between"><span class="text-[#8B95A7]">Experimento:</span> <span class="${isPromoted ? 'text-[#22C55E]' : isRejected ? 'text-[#EF4444]' : 'text-[#F7F8FA]'} font-semibold">${escapeHtml(hyp.after_score)}</span></div>` : ''}
              ${hyp.delta ? `<div class="flex justify-between"><span class="text-[#8B95A7]">Diferencia observada:</span> <span class="${isPromoted ? 'text-[#22C55E]' : 'text-[#EF4444]'} font-bold">${escapeHtml(hyp.delta)}</span></div>` : ''}
            </div>
          `
          : '';

        const actionButtons = (!isPromoted && !isRejected)
          ? `
            <div class="flex items-center space-x-2 pt-2 border-t border-[#242A36]">
              <button class="btn-hyp-action flex-1 bg-[#11151E] hover:bg-[#1E2536] text-[#8B95A7] hover:text-[#F7F8FA] border border-[#242A36] py-1.5 rounded-lg font-medium text-xs transition-colors flex items-center justify-center gap-1" data-hyp="${escapeHtml(hyp.id)}" data-action="reject">
                ${icon("x", "icon-sm")}
                <span>Rechazar</span>
              </button>
              <button class="btn-hyp-action flex-1 bg-[#4F67FF] hover:bg-[#3D56FF] text-white py-1.5 rounded-lg font-semibold text-xs transition-colors shadow-sm flex items-center justify-center gap-1" data-hyp="${escapeHtml(hyp.id)}" data-action="approve">
                ${icon("check", "icon-sm")}
                <span>Aprobar prueba</span>
              </button>
            </div>
          `
          : '';

        return `
          <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <div>
                <span class="font-mono font-semibold text-[#F7F8FA]">Hipótesis #${hyp.id ? escapeHtml(String(hyp.id).slice(0, 8)) : ''}</span>
                <div class="text-[10px] font-mono text-[#8B95A7]">Run: ${escapeHtml(activeRun.id)} • Dataset: ${escapeHtml(activeDataset.name)}</div>
              </div>
              <span class="${badgeClass} text-[10px] px-2 py-0.5 rounded font-mono font-medium flex items-center gap-1">
                ${badgeIcon} <span>${badgeText}</span>
              </span>
            </div>
            <div class="space-y-1">
              <div class="text-[#8B95A7] text-[10px] uppercase font-mono">Enunciado</div>
              <p class="text-[#F7F8FA] font-sans leading-relaxed">${escapeHtml(hyp.statement || '')}</p>
            </div>
            ${hyp.action ? `
              <div class="space-y-1">
                <div class="text-[#8B95A7] text-[10px] uppercase font-mono">Acción Propuesta</div>
                <p class="font-mono text-[#4F67FF] text-xs">${escapeHtml(hyp.action)}</p>
              </div>
            ` : ''}
            ${empiricalResults}
            ${hyp.critic_reason ? `
              <div class="p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] space-y-1 text-[11px]">
                <div class="font-medium text-[#F7F8FA]">Crítico Evaluador: ${escapeHtml(hyp.critic_decision || 'Evaluado')}</div>
                <div class="text-[#8B95A7] text-[10px] leading-relaxed">${escapeHtml(hyp.critic_reason)}</div>
              </div>
            ` : ''}
            ${actionButtons}
          </div>
        `;
      }).join("");
    }

    this.container.innerHTML = `
      <div class="h-full flex flex-col bg-[#0D1017] border-l border-[#242A36] shadow-2xl">
        <!-- Top Drawer Header -->
        <div class="p-4 border-b border-[#242A36] bg-[#080A0F] flex items-center justify-between">
          <div class="flex items-center space-x-2.5">
            <span class="text-[#6956E8]">${icon("bot", "icon-md", 18)}</span>
            <div>
              <h3 class="text-sm font-semibold text-[#F7F8FA] font-sans">CATML Copiloto Autónomo</h3>
              <p class="text-[11px] text-[#8B95A7] font-sans">Asistente de experimentación</p>
            </div>
          </div>
          <button id="btnCloseAgentDrawer" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1.5 rounded-lg hover:bg-[#161B26] transition-colors" title="Cerrar panel">${icon("x", "icon-sm")}</button>
        </div>

        <!-- Hypotheses & Interactive Stream Container -->
        <div class="p-4 space-y-4 overflow-y-auto flex-1 text-xs">
          <!-- Active Context Card -->
          ${contextCardHtml}

          <!-- Suggested Next Action Card -->
          ${nextActionCardHtml}

          <!-- Section Label -->
          <div class="flex items-center justify-between pt-1">
            <span class="text-[11px] font-mono uppercase tracking-wider text-[#8B95A7]">Registro de Hipótesis</span>
            <span class="text-[10px] font-mono text-[#8B95A7]/70">Proponer ≠ Aceptar</span>
          </div>

          ${hypothesesHtml}
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _bindEvents() {
    this.container.querySelector("#btnCloseAgentDrawer")?.addEventListener("click", () => {
      store.toggleAgentDrawer(false);
    });

    this.container.querySelector("#btnAgentSelectDsPrompt")?.addEventListener("click", () => {
      store.toggleAgentDrawer(false);
      bus.emit("modal:switch-dataset");
    });

    this.container.querySelector("#btnAgentLaunchFirstExp")?.addEventListener("click", () => {
      store.toggleAgentDrawer(false);
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnAgentReviewPlan")?.addEventListener("click", () => {
      store.toggleAgentDrawer(false);
      store.setNav("studio");
    });

    this.container.querySelector("#btnAgentRunPlan")?.addEventListener("click", async () => {
      try {
        await api.sendAgentAction("plan_execute", "approve");
        alert("Plan de experimentación programado para ejecución.");
        store.toggleAgentDrawer(false);
        store.setNav("studio");
      } catch (err) {
        alert("Agent action failed: " + (err.message || "Failed"));
        store.toggleAgentDrawer(false);
      }
    });

    this.container.querySelectorAll(".btn-hyp-action").forEach(btn => {
      btn.addEventListener("click", async () => {
        const hypId = btn.getAttribute("data-hyp");
        const action = btn.getAttribute("data-action");
        try {
          await api.sendAgentAction(hypId, action);
          alert(`Hipótesis ${hypId} ${action === "approve" ? "aprobada para ejecución!" : "rechazada."}`);
          await this.refresh();
        } catch (err) {
          alert("Error: " + err.message);
        }
      });
    });
  }

  destroy() {
    this.container = null;
  }
}
