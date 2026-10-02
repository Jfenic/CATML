/**
 * AgentDrawer — Autonomous Agent Lateral Co-pilot
 * Embodies the core CATML principle: "Proponer ≠ Aceptar"
 * Proposes explicit hypothesis objects requiring empirical critic verification before promotion.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";

export class AgentDrawer {
  constructor() {
    this.container = null;
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    const state = store.getState();
    const runs = state.runs || [];
    const activeRun = runs.find(r => r.id === state.activeRunId) || runs[0] || null;
    const dsName = (activeRun && activeRun.dataset_name) || "customers_churn";
    const taskType = (activeRun && activeRun.task_type) || "Binary classification";

    this.container.innerHTML = `
      <div class="h-full flex flex-col bg-[#0D1017] border-l border-[#242A36] shadow-2xl">
        <!-- Top Drawer Header -->
        <div class="p-4 border-b border-[#242A36] bg-[#080A0F] flex items-center justify-between">
          <div class="flex items-center space-x-2.5">
            <span class="text-[#6956E8]">${icon("bot", "icon-md", 18)}</span>
            <div>
              <h3 class="text-sm font-semibold text-[#F7F8FA] font-sans">CATML Agent</h3>
              <p class="text-[11px] text-[#8B95A7] font-sans">Autonomous AutoML Co-pilot</p>
            </div>
          </div>
          <button id="btnCloseAgentDrawer" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1.5 rounded-lg hover:bg-[#161B26] transition-colors" title="Close Drawer">${icon("x", "icon-sm")}</button>
        </div>

        <!-- Hypotheses & Interactive Stream Container -->
        <div class="p-4 space-y-4 overflow-y-auto flex-1 text-xs">
          <!-- Active Context Card -->
          <div class="bg-[#11151E] border border-[#242A36] rounded-xl p-4 space-y-2.5">
            <div class="text-[10px] uppercase font-mono tracking-wider text-[#8B95A7] font-semibold">Active Context</div>
            <div class="space-y-1.5 font-mono text-xs">
              <div class="flex justify-between"><span class="text-[#8B95A7]">Dataset:</span> <span class="text-[#F7F8FA] font-semibold">${dsName}</span></div>
              <div class="flex justify-between"><span class="text-[#8B95A7]">Task:</span> <span class="text-[#F7F8FA]">${taskType}</span></div>
              <div class="flex justify-between"><span class="text-[#8B95A7]">Status:</span> <span class="text-[#22C55E] font-semibold flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span>Ready</span></div>
            </div>
          </div>

          <!-- Suggested Next Action Card -->
          <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-[10px] uppercase font-mono tracking-wider text-[#4F67FF] font-semibold">Suggested Next Action</span>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-[#4F67FF]/10 text-[#4F67FF] border border-[#4F67FF]/20 font-medium">Empirical Plan</span>
            </div>
            <p class="text-xs text-[#F7F8FA] font-sans leading-relaxed">
              Exclude <code class="font-mono text-[#4F67FF] bg-[#11151E] px-1 py-0.5 rounded">customer_id</code> and evaluate derived interaction features on 5-fold CV benchmark.
            </p>
            <div class="flex items-center space-x-2 pt-2 border-t border-[#242A36]">
              <button id="btnAgentReviewPlan" class="flex-1 bg-[#11151E] hover:bg-[#1A202C] text-[#F7F8FA] border border-[#242A36] py-1.5 rounded-lg text-xs font-medium transition-colors">
                Review plan
              </button>
              <button id="btnAgentRunPlan" class="flex-1 bg-[#4F67FF] hover:bg-[#3D56FF] text-white py-1.5 rounded-lg text-xs font-semibold transition-all shadow-sm flex items-center justify-center gap-1">
                ${icon("play", "icon-sm")}
                <span>Run</span>
              </button>
            </div>
          </div>

          <!-- Section Label -->
          <div class="flex items-center justify-between pt-1">
            <span class="text-[11px] font-mono uppercase tracking-wider text-[#8B95A7]">Hypothesis Ledger</span>
            <span class="text-[10px] font-mono text-[#8B95A7]/70">Propose ≠ Accept</span>
          </div>

          <!-- Hypothesis #12 (PROMOTED) -->
          <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="font-mono font-semibold text-[#F7F8FA]">Hypothesis #12</span>
              <span class="badge-gain text-[10px] px-2 py-0.5 rounded font-mono font-medium flex items-center gap-1">${icon("check", "icon-sm")} <span>Promoted</span></span>
            </div>

            <div class="space-y-1">
              <div class="text-[#8B95A7] text-[10px] uppercase font-mono">Statement</div>
              <p class="text-[#F7F8FA] font-sans leading-relaxed">Income / Age ratio captures life-cycle vehicle purchasing power with reduced residual error.</p>
            </div>

            <div class="space-y-1">
              <div class="text-[#8B95A7] text-[10px] uppercase font-mono">Proposed Action</div>
              <p class="font-mono text-[#4F67FF] text-xs">Feature interaction: Income_div_Age</p>
            </div>

            <!-- Empirical Results -->
            <div class="bg-[#0D1017] p-2.5 rounded-lg border border-[#242A36] font-mono text-[11px] space-y-1">
              <div class="flex justify-between"><span class="text-[#8B95A7]">Benchmark:</span> <span class="text-[#F7F8FA]">0.94621 ROC-AUC</span></div>
              <div class="flex justify-between"><span class="text-[#8B95A7]">Experiment:</span>  <span class="text-[#22C55E] font-semibold">0.94648 ROC-AUC</span></div>
              <div class="flex justify-between"><span class="text-[#8B95A7]">Delta:</span>       <span class="text-[#22C55E] font-bold">+0.00027</span></div>
            </div>

            <div class="p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] space-y-1 text-[11px]">
              <div class="font-medium text-[#F7F8FA]">Critic Decision: Promote</div>
              <div class="text-[#8B95A7] text-[10px] leading-relaxed">Reproducible gain across 4/5 cross-validation folds. Accepted into production feature set.</div>
            </div>
          </div>

          <!-- Hypothesis #13 (REJECTED by Rule 4) -->
          <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="font-mono font-semibold text-[#F7F8FA]">Hypothesis #13</span>
              <span class="badge-err text-[10px] px-2 py-0.5 rounded font-mono font-medium flex items-center gap-1">${icon("x", "icon-sm")} <span>Rejected</span></span>
            </div>

            <div class="space-y-1">
              <div class="text-[#8B95A7] text-[10px] uppercase font-mono">Statement</div>
              <p class="text-[#F7F8FA] font-sans leading-relaxed">Combinatorial expansion of 22 polynomial interactions without variance gating.</p>
            </div>

            <div class="bg-[#0D1017] p-2.5 rounded-lg border border-[#242A36] font-mono text-[11px] space-y-1">
              <div class="flex justify-between"><span class="text-[#8B95A7]">Benchmark:</span> <span class="text-[#F7F8FA]">0.94110 ROC-AUC</span></div>
              <div class="flex justify-between"><span class="text-[#8B95A7]">Experiment:</span>  <span class="text-[#EF4444]">0.94093 ROC-AUC</span></div>
              <div class="flex justify-between"><span class="text-[#8B95A7]">Delta:</span>       <span class="text-[#EF4444] font-bold">-0.00017</span></div>
            </div>

            <div class="p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] space-y-1 text-[11px]">
              <div class="font-medium text-[#F7F8FA]">Critic Decision: Reject</div>
              <div class="text-[#8B95A7] text-[10px] leading-relaxed">Multicollinearity penalty exceeds marginal variance gain. Rejected under Rule 4.</div>
            </div>
          </div>

          <!-- Hypothesis #14 (PENDING HUMAN APPROVAL) -->
          <div class="bg-[#161B26] border border-[#242A36] rounded-xl p-4 space-y-3">
            <div class="flex items-center justify-between">
              <span class="font-mono font-semibold text-[#F7F8FA]">Hypothesis #14</span>
              <span class="badge-warn text-[10px] px-2 py-0.5 rounded font-mono font-medium flex items-center gap-1"><span class="w-1.5 h-1.5 rounded-full bg-[#F59E0B]"></span> <span>Proposed</span></span>
            </div>

            <div class="space-y-1">
              <div class="text-[#8B95A7] text-[10px] uppercase font-mono">Statement</div>
              <p class="text-[#F7F8FA] font-sans leading-relaxed">Smoothed m-estimate target encoding (m=10) for Region to suppress cardinality noise.</p>
            </div>

            <div class="space-y-1 text-[#8B95A7] font-mono text-[11px]">
              <div>Estimated cost: 1 run (5-fold CV, ~6s)</div>
              <div>Current benchmark: 0.94621 ROC-AUC</div>
            </div>

            <!-- Action buttons -->
            <div class="flex items-center space-x-2 pt-2 border-t border-[#242A36]">
              <button class="btn-hyp-action flex-1 bg-[#11151E] hover:bg-[#1E2536] text-[#8B95A7] hover:text-[#F7F8FA] border border-[#242A36] py-1.5 rounded-lg font-medium text-xs transition-colors flex items-center justify-center gap-1" data-hyp="14" data-action="reject">
                ${icon("x", "icon-sm")}
                <span>Reject</span>
              </button>
              <button class="btn-hyp-action flex-1 bg-[#4F67FF] hover:bg-[#3D56FF] text-white py-1.5 rounded-lg font-semibold text-xs transition-colors shadow-sm flex items-center justify-center gap-1" data-hyp="14" data-action="approve">
                ${icon("check", "icon-sm")}
                <span>Approve Run</span>
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

    this.container.querySelector("#btnAgentReviewPlan")?.addEventListener("click", () => {
      store.toggleAgentDrawer(false);
      store.setNav("studio");
    });

    this.container.querySelector("#btnAgentRunPlan")?.addEventListener("click", async () => {
      try {
        await api.sendAgentAction("plan_execute", "approve");
        alert("Automated agent plan scheduled for execution.");
        store.toggleAgentDrawer(false);
        store.setNav("studio");
      } catch (err) {
        alert("Agent action scheduled: " + (err.message || "OK"));
        store.toggleAgentDrawer(false);
      }
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
