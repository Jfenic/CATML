/**
 * CompareView — Multi-Experiment Comparator & Diff Inspector
 * Side-by-side comparison, progression tracking & explicit difference breakdown.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";

export class CompareView {
  constructor() {
    this.container = null;
    this.chart = null;
    this.experiments = [];
    this.selected = [];
    this.activeRun = null;
  }

  async mount(container) {
    this.container = container;
    this.renderLoading();
    await this.fetchData();
    this.render();
  }

  renderLoading() {
    this.container.innerHTML = `
      <div class="workbench-card p-12 text-center text-slate-400 space-y-3">
        <div class="animate-spin text-2xl text-indigo-400">⚡</div>
        <div class="text-sm font-medium">Cargando experimentos para comparación...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const state = store.getState();
      const runs = state.runs || [];
      this.activeRun = runs.find(r => r.id === state.activeRunId)
        || runs.find(r => r.status === "RUNNING")
        || runs[0]
        || null;

      if (this.activeRun) {
        this.experiments = await api.getExperiments(this.activeRun.id).catch(() => []);
        if (this.experiments.length > 0) {
          this.selected = this.experiments.slice(0, 3).map(e => e.id);
        }
      }
    } catch (e) {
      console.warn("CompareView fetchData error:", e);
    }
  }

  render() {
    if (!this.activeRun || this.experiments.length === 0) {
      this.container.innerHTML = `
        <div class="space-y-6">
          <div class="workbench-card p-12 text-center space-y-4">
            <span class="text-4xl text-slate-600 block">⇄</span>
            <h3 class="text-base font-bold text-slate-200">No Experiments Found to Compare</h3>
            <p class="text-xs text-slate-400 max-w-sm mx-auto">You need at least 2 completed or active experiments in the workspace to perform side-by-side comparison.</p>
            <button id="btnNewExpCompareEmpty" class="btn-signal">
              + CREATE FIRST EXPERIMENT
            </button>
          </div>
        </div>
      `;
      this.container.querySelector("#btnNewExpCompareEmpty")?.addEventListener("click", () => {
        bus.emit("modal:new-experiment");
      });
      return;
    }

    const selectedExps = this.experiments.filter(e => this.selected.includes(e.id));
    const metricName = this.activeRun.metric || "CV Score";

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Top Selector Row -->
        <div class="workbench-card p-4 bg-[#16171c] border-[#27272e] flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex items-center space-x-3">
            <span class="text-[#E5512D] font-mono font-bold text-lg">⇄</span>
            <div>
              <h2 class="text-base font-bold font-mono text-[#F1EFE9]">EXPERIMENT COMPARISON & LINEAGE</h2>
              <p class="text-xs text-[#D8D6CF]/70 font-mono">Selecciona dos o más experimentos para inspeccionar diferencias algorítmicas y ganancias métricas</p>
            </div>
          </div>

          <!-- Experiment Checkboxes -->
          <div class="flex flex-wrap items-center gap-2 text-xs font-mono">
            ${this.experiments.map(exp => `
              <label class="flex items-center space-x-1.5 cursor-pointer px-3 py-1.5 rounded-sm border ${this.selected.includes(exp.id) ? 'border-[#E5512D] text-[#F1EFE9] bg-[#1c1d24]' : 'border-[#27272e] text-[#D8D6CF]/60 bg-[#111111]'}">
                <input type="checkbox" ${this.selected.includes(exp.id) ? "checked" : ""} class="exp-compare-chk rounded-sm text-[#E5512D] focus:ring-0 bg-[#111111]" data-exp-id="${exp.id}">
                <span>${exp.name || exp.id}</span>
              </label>
            `).join("")}
          </div>
        </div>

        <!-- Comparative Table -->
        <div class="workbench-card overflow-hidden">
          <div class="workbench-panel-header flex items-center justify-between">
            <span class="text-sm font-semibold text-slate-200">Side-by-Side Metrics & Architectures</span>
            <span class="text-xs text-slate-400">Metric: ${metricName}</span>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full wb-table text-left">
              <thead>
                <tr>
                  <th class="w-48">Dimension</th>
                  ${selectedExps.map(e => `
                    <th class="font-mono text-slate-200">${e.name || e.id}</th>
                  `).join("")}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="font-medium text-slate-300">Validation Score</td>
                  ${selectedExps.map(e => `
                    <td class="font-mono font-bold text-emerald-400">
                      ${e.best_score != null ? Number(e.best_score).toFixed(5) : "—"}
                    </td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Models / Family</td>
                  ${selectedExps.map(e => `
                    <td><span class="badge-sys px-2 py-0.5 rounded text-xs uppercase">${(e.model_ids || []).join(", ") || "N/A"}</span></td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Features Count</td>
                  ${selectedExps.map(e => `
                    <td class="font-mono text-slate-300">${e.features_count || (e.feature_names ? e.feature_names.length : 0)} features</td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Optimization Trials</td>
                  ${selectedExps.map(e => `
                    <td class="font-mono text-slate-300">${e.trials_count != null ? e.trials_count : (e.trials ? e.trials.length : 0)} trials</td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Execution Status</td>
                  ${selectedExps.map(e => `
                    <td><span class="${e.status === 'COMPLETED' ? 'badge-gain' : 'badge-sys'} text-[10px] px-2 py-0.5 rounded font-mono">${e.status}</span></td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Hypothesis / Rationale</td>
                  ${selectedExps.map(e => `
                    <td class="text-xs text-slate-400 max-w-xs truncate">${e.hypothesis || "Baseline model training"}</td>
                  `).join("")}
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Progression Chart & Differences Inspector -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Progression Chart -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <span class="text-sm font-semibold text-slate-200">Evolution Progression</span>
              <span class="text-xs font-mono text-emerald-400">${metricName}</span>
            </div>
            <div class="p-4 flex-1 flex flex-col justify-between">
              <div class="h-64 w-full relative">
                <canvas id="compareEvolutionChart"></canvas>
              </div>
              <div class="mt-3 flex justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
                ${selectedExps.map((e, idx) => `
                  <span>#${idx + 1}: ${e.name || e.id}</span>
                `).join("")}
              </div>
            </div>
          </div>

          <!-- Show Differences Inspector -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-amber-400">🔍</span>
                <span class="text-sm font-semibold text-slate-200">Differences Inspector</span>
              </div>
              ${selectedExps.length >= 2 && selectedExps[0].best_score != null && selectedExps[1].best_score != null ? `
                <span class="badge-gain text-xs px-2 py-0.5 rounded font-mono font-bold">
                  Δ: ${(selectedExps[1].best_score - selectedExps[0].best_score >= 0 ? '+' : '') + (selectedExps[1].best_score - selectedExps[0].best_score).toFixed(5)}
                </span>
              ` : ''}
            </div>
            <div class="p-5 space-y-4 flex-1 text-xs">
              ${selectedExps.length >= 2 ? `
                <div class="space-y-1.5">
                  <div class="text-slate-400 uppercase font-semibold tracking-wider text-[10px]">Model Architecture Differences</div>
                  <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono space-y-1 text-slate-300">
                    <div>Exp 1: <span class="text-indigo-400 font-bold">${selectedExps[0].name}</span> (${(selectedExps[0].model_ids || []).join(", ")})</div>
                    <div>Exp 2: <span class="text-purple-400 font-bold">${selectedExps[1].name}</span> (${(selectedExps[1].model_ids || []).join(", ")})</div>
                  </div>
                </div>

                <div class="space-y-1.5">
                  <div class="text-slate-400 uppercase font-semibold tracking-wider text-[10px]">Hypothesis Evaluation</div>
                  <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono space-y-1 text-slate-300">
                    <div>${selectedExps[1].hypothesis || "Optuna Bayesian hyperparameter search exploration"}</div>
                  </div>
                </div>
              ` : `
                <div class="text-center text-slate-500 py-12">Select at least 2 experiments above to inspect differences.</div>
              `}
            </div>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
    this._initChart(selectedExps);
  }

  _bindEvents() {
    this.container.querySelectorAll(".exp-compare-chk").forEach(chk => {
      chk.addEventListener("change", e => {
        const id = chk.getAttribute("data-exp-id");
        if (e.target.checked) {
          if (!this.selected.includes(id)) this.selected.push(id);
        } else {
          this.selected = this.selected.filter(x => x !== id);
        }
        this.render();
      });
    });
  }

  _initChart(selectedExps) {
    const canvas = this.container.querySelector("#compareEvolutionChart");
    if (!canvas || !window.Chart) return;

    if (this.chart) {
      this.chart.destroy();
    }

    const labels = selectedExps.map(e => e.name || e.id);
    const data = selectedExps.map(e => e.best_score || 0.0);

    const minScore = Math.min(...data.filter(d => d > 0));
    const maxScore = Math.max(...data);
    const pad = (maxScore - minScore) * 0.1 || 0.01;

    this.chart = new window.Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: `${this.activeRun ? this.activeRun.metric : 'Score'} Progression`,
            data: data,
            borderColor: "#6366f1",
            backgroundColor: "rgba(99, 102, 241, 0.15)",
            pointBackgroundColor: ["#94a3b8", "#6366f1", "#a855f7", "#10b981"],
            pointRadius: 6,
            pointHoverRadius: 8,
            fill: true,
            tension: 0.2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `Score: ${ctx.raw.toFixed(5)}`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(51, 65, 85, 0.2)" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
          },
          y: {
            min: Math.max(0, minScore - pad),
            max: Math.min(1.0, maxScore + pad),
            grid: { color: "rgba(51, 65, 85, 0.2)" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
          },
        },
      },
    });
  }

  destroy() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
    this.container = null;
  }
}
