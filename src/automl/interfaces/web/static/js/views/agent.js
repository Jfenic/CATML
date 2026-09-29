/**
 * AgentDrawer — V0.9 & V1.0 Autonomous Agent Lateral Drawer
 * Embodies the core CATML principle: "Proponer ≠ Aceptar"
 * Proposes explicit hypothesis objects requiring empirical critic verification before promotion.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";

export class AgentDrawer {
  constructor() {
    this.container = null;
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="h-full flex flex-col bg-slate-900 border-l border-slate-800 shadow-2xl">
        <!-- Top Drawer Header -->
        <div class="p-4 border-b border-slate-800 bg-slate-950 flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <span class="text-purple-400 text-lg">✦</span>
            <div>
              <h3 class="text-sm font-bold text-slate-100 uppercase tracking-wider">CATML Agent</h3>
              <p class="text-[10px] text-purple-300 font-mono">Principio: Proponer ≠ Aceptar</p>
            </div>
          </div>
          <button id="btnCloseAgentDrawer" class="text-slate-400 hover:text-slate-200 text-lg px-2">✕</button>
        </div>

        <!-- Hypotheses Stream Container -->
        <div class="p-4 space-y-4 overflow-y-auto flex-1 text-xs">
          <div class="text-[11px] text-slate-400 bg-purple-950/20 border border-purple-900/40 p-3 rounded-lg">
            El agente formula hipótesis causales a partir de la distribución de errores y datos. Cada propuesta requiere validación empírica en holdout antes de integrarse al pipeline.
          </div>

          <!-- Hypothesis #12 (PROMOTED) -->
          <div class="workbench-card p-4 space-y-3 border-emerald-800/60 bg-emerald-950/10">
            <div class="flex items-center justify-between">
              <span class="font-mono font-bold text-emerald-400">HYPOTHESIS #12</span>
              <span class="badge-gain text-[10px] px-2 py-0.5 rounded font-mono font-bold">✓ PROMOTED</span>
            </div>

            <div class="space-y-1">
              <div class="text-slate-400 text-[10px] uppercase font-semibold">Statement</div>
              <p class="text-slate-200">Income / Age puede capturar la capacidad de compra del vehículo eléctrico ajustada por ciclo vital.</p>
            </div>

            <div class="space-y-1">
              <div class="text-slate-400 text-[10px] uppercase font-semibold">Proposed Action</div>
              <p class="font-mono text-indigo-300">Feature interaction: Income_div_Age</p>
            </div>

            <!-- Empirical Results -->
            <div class="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-[11px] space-y-1">
              <div class="flex justify-between"><span>Before (Benchmark):</span> <span>0.94621 ROC-AUC</span></div>
              <div class="flex justify-between"><span>After Experiment:</span>  <span class="text-emerald-400 font-bold">0.94648 ROC-AUC</span></div>
              <div class="flex justify-between"><span>Delta:</span>             <span class="text-emerald-400 font-bold">+0.00027</span></div>
            </div>

            <div class="p-2 rounded bg-purple-950/30 border border-purple-900/50 space-y-1 text-[11px]">
              <div class="font-semibold text-purple-300">Critic Decision: PROMOTE</div>
              <div class="text-slate-400 text-[10px]">Mejora reproducible en 4/5 folds con reducción de varianza residual. Aceptado en pipeline.</div>
            </div>
          </div>

          <!-- Hypothesis #13 (REJECTED by Rule 4) -->
          <div class="workbench-card p-4 space-y-3 border-rose-800/60 bg-rose-950/10">
            <div class="flex items-center justify-between">
              <span class="font-mono font-bold text-rose-400">HYPOTHESIS #13</span>
              <span class="badge-err text-[10px] px-2 py-0.5 rounded font-mono font-bold">✕ REJECTED</span>
            </div>

            <div class="space-y-1">
              <div class="text-slate-400 text-[10px] uppercase font-semibold">Statement</div>
              <p class="text-slate-200">Matriz combinatoria de 22 interacciones polinomiales automáticas.</p>
            </div>

            <div class="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-[11px] space-y-1">
              <div class="flex justify-between"><span>Before:</span> <span>0.94110 ROC-AUC</span></div>
              <div class="flex justify-between"><span>After:</span>  <span class="text-rose-400">0.94093 ROC-AUC</span></div>
              <div class="flex justify-between"><span>Delta:</span>  <span class="text-rose-400 font-bold">-0.00017</span></div>
            </div>

            <div class="p-2 rounded bg-rose-950/30 border border-rose-900/50 space-y-1 text-[11px]">
              <div class="font-semibold text-rose-300">Critic Decision: REJECT</div>
              <div class="text-slate-400 text-[10px]">Degradación de rendimiento por colinealidad. Rechazado estrictamente según principio 'Proponer ≠ Aceptar'.</div>
            </div>
          </div>

          <!-- Hypothesis #14 (PENDING HUMAN APPROVAL) -->
          <div class="workbench-card p-4 space-y-3 border-indigo-700/60 bg-indigo-950/20">
            <div class="flex items-center justify-between">
              <span class="font-mono font-bold text-indigo-300">HYPOTHESIS #14</span>
              <span class="badge-warn text-[10px] px-2 py-0.5 rounded font-mono font-bold">PROPOSED</span>
            </div>

            <div class="space-y-1">
              <div class="text-slate-400 text-[10px] uppercase font-semibold">Statement</div>
              <p class="text-slate-200">Target encoding suavizado m-estimate (m=10) para Region y Vehicle_Type reduce cardinalidad de ruido.</p>
            </div>

            <div class="space-y-1 text-slate-400 font-mono text-[11px]">
              <div>Coste estimado: 1 corrida 5-fold CV (~6s)</div>
              <div>Current Benchmark: 0.94621 ROC-AUC</div>
            </div>

            <!-- Action buttons -->
            <div class="flex items-center space-x-2 pt-2 border-t border-slate-800">
              <button class="btn-hyp-action flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 rounded font-semibold text-xs transition-colors" data-hyp="14" data-action="reject">
                Reject
              </button>
              <button class="btn-hyp-action flex-1 bg-indigo-600 hover:bg-indigo-500 text-white py-1.5 rounded font-semibold text-xs transition-colors shadow-lg shadow-indigo-600/20" data-hyp="14" data-action="approve">
                Approve Experiment
              </button>
            </div>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _bindEvents() {
    this.container.querySelector("#btnCloseAgentDrawer")?.addEventListener("click", () => {
      store.toggleAgentDrawer(false);
    });

    this.container.querySelectorAll(".btn-hyp-action").forEach(btn => {
      btn.addEventListener("click", async () => {
        const hypId = btn.getAttribute("data-hyp");
        const action = btn.getAttribute("data-action");
        try {
          await api.sendAgentAction(`hyp_${hypId}`, action);
          alert(`Hypothesis #${hypId} ${action === "approve" ? "approved for execution!" : "rejected."}`);
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
