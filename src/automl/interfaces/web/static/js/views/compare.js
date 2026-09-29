/**
 * CompareView — Multi-Experiment Comparator & Diff Inspector
 * Side-by-side comparison, progression tracking & explicit difference breakdown.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";

export class CompareView {
  constructor() {
    this.container = null;
    this.chart = null;
    this.selected = ["exp_34", "exp_38", "exp_42"];
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Top Selector Row -->
        <div class="workbench-card p-4 bg-slate-900/80 border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex items-center space-x-3">
            <span class="text-indigo-400 font-bold text-lg">⇄</span>
            <div>
              <h2 class="text-base font-bold text-slate-100">Experiment Comparison & Lineage</h2>
              <p class="text-xs text-slate-400">Selecciona dos o más experimentos para inspeccionar diferencias algorítmicas y ganancias métricas</p>
            </div>
          </div>

          <!-- Experiment Checkboxes -->
          <div class="flex items-center space-x-3 text-xs">
            <label class="flex items-center space-x-1.5 cursor-pointer bg-slate-950 px-3 py-1.5 rounded border border-slate-800">
              <input type="checkbox" checked class="rounded text-indigo-600 focus:ring-0 bg-slate-800" data-exp="exp_34">
              <span class="font-mono text-slate-300">#34 (LGBM Base)</span>
            </label>
            <label class="flex items-center space-x-1.5 cursor-pointer bg-slate-950 px-3 py-1.5 rounded border border-slate-800">
              <input type="checkbox" checked class="rounded text-indigo-600 focus:ring-0 bg-slate-800" data-exp="exp_38">
              <span class="font-mono text-slate-300">#38 (CatBoost HPO)</span>
            </label>
            <label class="flex items-center space-x-1.5 cursor-pointer bg-slate-950 px-3 py-1.5 rounded border border-indigo-600/40">
              <input type="checkbox" checked class="rounded text-indigo-600 focus:ring-0 bg-slate-800" data-exp="exp_42">
              <span class="font-mono text-emerald-400 font-bold">#42 (Ensemble)</span>
            </label>
          </div>
        </div>

        <!-- Comparative Table -->
        <div class="workbench-card overflow-hidden">
          <div class="workbench-panel-header flex items-center justify-between">
            <span class="text-sm font-semibold text-slate-200">Side-by-Side Metrics & Architectures</span>
            <span class="text-xs text-slate-400">Metric: 5-Fold Stratified ROC-AUC</span>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full wb-table text-left">
              <thead>
                <tr>
                  <th class="w-48">Dimension</th>
                  <th class="font-mono">#34 Baseline</th>
                  <th class="font-mono">#38 CatBoost HPO</th>
                  <th class="font-mono text-emerald-400">#42 Ensemble Best</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="font-medium text-slate-300">CV ROC-AUC</td>
                  <td class="font-mono text-slate-300">0.94210</td>
                  <td class="font-mono text-slate-300">0.94470</td>
                  <td class="font-mono font-bold text-emerald-400">0.94621 <span class="badge-gain text-[10px] px-1 py-0.5 rounded font-normal ml-1">+0.00151</span></td>
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">CV std</td>
                  <td class="font-mono text-slate-400">±0.0019</td>
                  <td class="font-mono text-slate-400">±0.0014</td>
                  <td class="font-mono font-medium text-indigo-300">±0.0012 (Menor varianza)</td>
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Algorithm / Family</td>
                  <td><span class="badge-sys px-2 py-0.5 rounded text-xs">LightGBM</span></td>
                  <td><span class="badge-sys px-2 py-0.5 rounded text-xs">CatBoost</span></td>
                  <td><span class="badge-intel px-2 py-0.5 rounded text-xs font-bold">Weighted Blender Ensemble</span></td>
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Active Features</td>
                  <td class="font-mono text-slate-300">13 features</td>
                  <td class="font-mono text-slate-300">18 features</td>
                  <td class="font-mono text-slate-300">18 features (+2 interactions)</td>
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Optimization Trials</td>
                  <td class="font-mono text-slate-300">1 trial</td>
                  <td class="font-mono text-slate-300">60 trials</td>
                  <td class="font-mono text-slate-300">88 trials total</td>
                </tr>
                <tr>
                  <td class="font-medium text-slate-300">Total Runtime</td>
                  <td class="font-mono text-slate-300">5.03s</td>
                  <td class="font-mono text-slate-300">21m 14s</td>
                  <td class="font-mono text-slate-300">34m 02s</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Progression Chart & Show Differences Inspector -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Progression Chart -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <span class="text-sm font-semibold text-slate-200">Performance Over Evolution Phases</span>
              <span class="text-xs font-mono text-emerald-400">Step Progression</span>
            </div>
            <div class="p-4 flex-1 flex flex-col justify-between">
              <div class="h-64 w-full relative">
                <canvas id="compareEvolutionChart"></canvas>
              </div>
              <div class="mt-3 flex justify-between text-xs text-slate-400 pt-2 border-t border-slate-800">
                <span>Phase 1: Baseline</span>
                <span>Phase 2: HPO</span>
                <span>Phase 3: Features</span>
                <span>Phase 4: Ensemble</span>
              </div>
            </div>
          </div>

          <!-- Show Differences Inspector -->
          <div class="workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-amber-400">🔍</span>
                <span class="text-sm font-semibold text-slate-200">Show Differences: #38 ➔ #42</span>
              </div>
              <span class="badge-gain text-xs px-2 py-0.5 rounded font-mono font-bold">+0.00153 ROC-AUC</span>
            </div>
            <div class="p-5 space-y-4 flex-1 text-xs">
              <!-- Added features -->
              <div class="space-y-1.5">
                <div class="text-slate-400 uppercase font-semibold tracking-wider text-[10px]">Feature Engineering (Added)</div>
                <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono space-y-1 text-emerald-400">
                  <div>+ Income × Age interaction (captura etapa de vida)</div>
                  <div>+ Home_Ownership × Region (geografía y patrimonio)</div>
                </div>
              </div>

              <!-- Changed Hyperparameters -->
              <div class="space-y-1.5">
                <div class="text-slate-400 uppercase font-semibold tracking-wider text-[10px]">Hyperparameters Tuned (Changed)</div>
                <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono space-y-1 text-slate-300">
                  <div>CatBoost depth: <span class="text-slate-400 line-through">7</span> ➔ <span class="text-indigo-400 font-bold">8</span></div>
                  <div>learning_rate:  <span class="text-slate-400 line-through">0.040</span> ➔ <span class="text-indigo-400 font-bold">0.031</span></div>
                  <div>l2_leaf_reg:    <span class="text-slate-400 line-through">3.0</span> ➔ <span class="text-indigo-400 font-bold">4.2</span></div>
                </div>
              </div>

              <!-- Added to ensemble -->
              <div class="space-y-1.5">
                <div class="text-slate-400 uppercase font-semibold tracking-wider text-[10px]">Ensemble Blend Composition</div>
                <div class="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono space-y-1 text-purple-300">
                  <div>• LightGBM #34 (Weight: 0.35)</div>
                  <div>• CatBoost #38 (Weight: 0.65)</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    this._initChart();
  }

  _initChart() {
    const canvas = this.container.querySelector("#compareEvolutionChart");
    if (!canvas || !window.Chart) return;

    if (this.chart) {
      this.chart.destroy();
    }

    this.chart = new window.Chart(canvas, {
      type: "line",
      data: {
        labels: ["Baseline #34", "CatBoost HPO #38", "Feature Eng", "Ensemble #42"],
        datasets: [
          {
            label: "ROC-AUC Progression",
            data: [0.94210, 0.94470, 0.94520, 0.94621],
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
              label: (ctx) => `Score: ${ctx.raw.toFixed(5)} ROC-AUC`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(51, 65, 85, 0.2)" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
          },
          y: {
            min: 0.94,
            max: 0.948,
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
