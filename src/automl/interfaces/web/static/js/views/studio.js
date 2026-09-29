/**
 * StudioView — Experiment Studio
 * Real-time experiment workbench: Planner explicability, HPO, Controls & Evaluation.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";

export class StudioView {
  constructor() {
    this.container = null;
    this.activeTab = "hpo";
    this.activeDecisionModal = null;
    this.chart = null;
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    const state = store.getState();
    const runs = state.runs || [];
    const activeRun = runs.find(r => r.status === "RUNNING") || runs[0] || {
      id: "run_ev_s6e9",
      dataset_name: "EV Purchases",
      status: "RUNNING",
      best_score: 0.94621,
      best_model: "Ensemble #7",
    };

    const isRunning = activeRun.status === "RUNNING";
    const isPaused = activeRun.status === "PAUSED";

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Top Sticky Control Header -->
        <div class="workbench-card p-4 bg-slate-900/90 border-slate-800">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-3">
                <h2 class="text-lg font-bold text-slate-100">Experiment #42 / ${activeRun.dataset_name || "EV Purchases"}</h2>
                <span id="runStatusBadge" class="${isRunning ? 'badge-sys' : isPaused ? 'badge-warn' : 'badge-gain'} text-xs px-2.5 py-0.5 rounded-full font-mono font-semibold">
                  ● ${activeRun.status || "RUNNING"}
                </span>
              </div>
              <p class="text-xs text-slate-400 mt-0.5">AutoML Pipeline • Metric: ROC-AUC • 5-Fold Stratified Cross-Validation</p>
            </div>

            <!-- Key metrics row -->
            <div class="flex items-center space-x-6">
              <div>
                <div class="text-[11px] uppercase tracking-wider text-slate-400">Best CV</div>
                <div class="text-lg font-bold font-mono-num text-emerald-400">0.94621 <span class="text-xs font-normal text-emerald-500 font-mono">+0.0031</span></div>
              </div>
              <div>
                <div class="text-[11px] uppercase tracking-wider text-slate-400">Best Model</div>
                <div class="text-base font-semibold text-slate-200">Ensemble #7</div>
              </div>
              <div>
                <div class="text-[11px] uppercase tracking-wider text-slate-400">Trials</div>
                <div class="text-base font-mono font-semibold text-indigo-400">37 / 60</div>
              </div>
              <div>
                <div class="text-[11px] uppercase tracking-wider text-slate-400">Resources</div>
                <div class="text-xs font-mono text-slate-300">CPU 67% • RAM 51%</div>
              </div>

              <!-- Action Controls -->
              <div class="flex items-center space-x-2 pl-2 border-l border-slate-800">
                ${
                  isRunning
                    ? `<button id="btnPauseRun" class="bg-amber-600/20 hover:bg-amber-600/30 text-amber-300 border border-amber-600/40 text-xs px-3 py-1.5 rounded-lg font-medium transition-colors">⏸ Pause</button>`
                    : `<button id="btnResumeRun" class="bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-600/40 text-xs px-3 py-1.5 rounded-lg font-medium transition-colors">▶ Resume</button>`
                }
                <button id="btnStopRun" class="bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-600/40 text-xs px-3 py-1.5 rounded-lg font-medium transition-colors">■ Stop</button>
                <button id="btnCloneRun" class="bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs px-3 py-1.5 rounded-lg font-medium transition-colors">Clone</button>
                <button id="btnExportRun" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3 py-1.5 rounded-lg font-medium transition-colors">Export</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Middle Section: Planner on Left, Performance on Right -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Left: AUTOML PLAN (Explicable) -->
          <div class="lg:col-span-5 workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-purple-400 font-bold">🧠</span>
                <span class="text-sm font-semibold text-slate-200">AUTOML PLAN & DECISIONS</span>
              </div>
              <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono">Explicable</span>
            </div>

            <div class="p-4 space-y-3.5 flex-1">
              <!-- Step 1 -->
              <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800 flex items-start justify-between">
                <div class="space-y-1">
                  <div class="flex items-center space-x-2">
                    <span class="text-emerald-400 font-bold">1 ✓</span>
                    <span class="text-xs font-bold text-slate-200">Baseline Logistic Regression</span>
                  </div>
                  <div class="text-[11px] text-slate-400">ROC-AUC: <span class="font-mono text-slate-300">0.93120</span></div>
                </div>
                <button data-why="step1" class="btn-why badge-intel hover:bg-purple-900/40 text-[10px] px-2 py-0.5 rounded transition-colors">
                  Why this?
                </button>
              </div>

              <!-- Step 2 -->
              <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800 flex items-start justify-between">
                <div class="space-y-1">
                  <div class="flex items-center space-x-2">
                    <span class="text-emerald-400 font-bold">2 ✓</span>
                    <span class="text-xs font-bold text-slate-200">LightGBM Gradient Boosting</span>
                  </div>
                  <div class="text-[11px] text-slate-400">
                    ROC-AUC: <span class="font-mono text-emerald-400 font-semibold">0.94370</span>
                    <span class="text-emerald-500 font-mono text-[10px] ml-1">↑ +0.0125</span>
                  </div>
                </div>
                <button data-why="step2" class="btn-why badge-intel hover:bg-purple-900/40 text-[10px] px-2 py-0.5 rounded transition-colors">
                  Why this?
                </button>
              </div>

              <!-- Step 3 (Active) -->
              <div class="p-3.5 rounded-lg bg-indigo-950/30 border border-indigo-700/60 space-y-2">
                <div class="flex items-start justify-between">
                  <div class="space-y-0.5">
                    <div class="flex items-center space-x-2">
                      <span class="inline-block w-2 h-2 rounded-full bg-indigo-400 animate-pulse"></span>
                      <span class="text-xs font-bold text-indigo-200">3 ● CatBoost HPO</span>
                    </div>
                    <span class="badge-sys text-[10px] px-1.5 py-0.2 rounded font-mono font-medium">Priority: HIGH</span>
                  </div>
                  <button data-why="step3" class="btn-why badge-intel hover:bg-purple-900/40 text-[10px] px-2.5 py-1 rounded font-semibold transition-colors">
                    Why this?
                  </button>
                </div>

                <div class="text-[11px] text-slate-300 pl-4 border-l-2 border-indigo-500/40 space-y-0.5">
                  <p class="font-semibold text-slate-200">Why selected?</p>
                  <p>• 7 características categóricas en el dataset</p>
                  <p>• Cardinalidad moderada sin dispersión extrema</p>
                  <p>• Modelos de árboles superan baseline (+0.0125)</p>
                </div>
              </div>

              <!-- Step 4 -->
              <div class="p-3 rounded-lg bg-slate-900/30 border border-slate-800/80 flex items-start justify-between opacity-80">
                <div class="space-y-1">
                  <div class="flex items-center space-x-2">
                    <span class="text-slate-500 font-bold">4 ○</span>
                    <span class="text-xs font-medium text-slate-300">XGBoost HPO</span>
                  </div>
                  <div class="text-[11px] text-slate-500">Priority: MEDIUM</div>
                </div>
                <button data-why="step4" class="btn-why badge-intel hover:bg-purple-900/40 text-[10px] px-2 py-0.5 rounded transition-colors">
                  Why this?
                </button>
              </div>

              <!-- Step 5 -->
              <div class="p-3 rounded-lg bg-slate-900/30 border border-slate-800/80 flex items-start justify-between opacity-60">
                <div class="space-y-1">
                  <div class="flex items-center space-x-2">
                    <span class="text-slate-500 font-bold">5 ○</span>
                    <span class="text-xs font-medium text-slate-300">Feature Interactions</span>
                  </div>
                  <div class="text-[11px] text-slate-500">Waiting (Propose ≠ Accept)</div>
                </div>
                <button data-why="step5" class="btn-why badge-intel hover:bg-purple-900/40 text-[10px] px-2 py-0.5 rounded transition-colors">
                  Why this?
                </button>
              </div>

              <!-- Step 6 -->
              <div class="p-3 rounded-lg bg-slate-900/30 border border-slate-800/80 flex items-start justify-between opacity-60">
                <div class="space-y-1">
                  <div class="flex items-center space-x-2">
                    <span class="text-slate-500 font-bold">6 ○</span>
                    <span class="text-xs font-medium text-slate-300">Ensemble Blender</span>
                  </div>
                  <div class="text-[11px] text-slate-500">Waiting</div>
                </div>
                <button data-why="step6" class="btn-why badge-intel hover:bg-purple-900/40 text-[10px] px-2 py-0.5 rounded transition-colors">
                  Why this?
                </button>
              </div>
            </div>
          </div>

          <!-- Right: Performance & Optimization Progression -->
          <div class="lg:col-span-7 workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-emerald-400">📈</span>
                <span class="text-sm font-semibold text-slate-200">Optimization Progress (ROC-AUC over Trials)</span>
              </div>
              <span class="text-xs font-mono text-slate-400">Best: 0.94621</span>
            </div>

            <div class="p-4 flex-1 flex flex-col justify-between">
              <div class="h-64 w-full relative">
                <canvas id="performanceChart"></canvas>
              </div>

              <div class="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
                <div>Modelos explorados: <span class="text-slate-200 font-mono">Logistic, LightGBM, CatBoost</span></div>
                <div class="flex items-center space-x-4">
                  <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-full bg-slate-500"></span><span>Baseline</span></span>
                  <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-full bg-indigo-500"></span><span>LightGBM</span></span>
                  <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span><span>CatBoost HPO</span></span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Bottom Tabs: Models | HPO | Features | Resources | Logs -->
        <div class="workbench-card">
          <div class="border-b border-slate-800 px-4 flex items-center space-x-6 text-xs font-medium">
            <button class="tab-btn py-3 border-b-2 ${this.activeTab === 'hpo' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="hpo">
              🎯 HPO & Hyperparameters
            </button>
            <button class="tab-btn py-3 border-b-2 ${this.activeTab === 'models' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="models">
              📊 Models & Leaderboard
            </button>
            <button class="tab-btn py-3 border-b-2 ${this.activeTab === 'resources' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="resources">
              ⚙ Resources & Workers
            </button>
            <button class="tab-btn py-3 border-b-2 ${this.activeTab === 'logs' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="logs">
              📜 Logs & Events
            </button>
          </div>

          <div class="p-5" id="tabContent">
            ${this._renderTabContent()}
          </div>
        </div>
      </div>

      <!-- Why This? Planner Explanation Modal -->
      <div id="whyModal" class="hidden fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-lg w-full p-6 space-y-4 border-purple-800/60 shadow-2xl shadow-purple-900/20">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <span class="text-purple-400 text-base">🟣</span>
              <h3 class="text-base font-bold text-slate-100">Planner Decision Explanation</h3>
            </div>
            <button id="btnCloseWhyModal" class="text-slate-400 hover:text-slate-200 text-lg">✕</button>
          </div>
          <div id="whyModalBody" class="text-xs space-y-3">
            <!-- Injected dynamically -->
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
    this._initChart();
  }

  _renderTabContent() {
    if (this.activeTab === "hpo") {
      return `
        <div class="space-y-6">
          <div class="flex items-center justify-between">
            <div>
              <h4 class="text-sm font-bold text-slate-200">HYPERPARAMETER SEARCH (Optuna TPE)</h4>
              <p class="text-xs text-slate-400">Best Trial: <span class="font-mono text-emerald-400 font-bold">#37</span> • ROC-AUC: <span class="font-mono text-emerald-400 font-bold">0.94582</span></p>
            </div>
            <button id="btnStopHPOAndPromote" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-4 py-2 rounded-lg font-semibold transition-colors shadow-lg shadow-indigo-600/20">
              ⚡ Stop HPO and promote best
            </button>
          </div>

          <!-- Parameter Importance Section -->
          <div class="space-y-3">
            <div class="text-xs font-semibold text-slate-300 uppercase tracking-wider">Parameter Importance</div>
            <div class="space-y-2">
              <div>
                <div class="flex justify-between text-xs font-mono text-slate-300 mb-1">
                  <span>depth</span>
                  <span class="text-indigo-400">34%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div class="bg-indigo-500 h-2 rounded-full" style="width: 34%"></div>
                </div>
              </div>

              <div>
                <div class="flex justify-between text-xs font-mono text-slate-300 mb-1">
                  <span>learning_rate</span>
                  <span class="text-indigo-400">23%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div class="bg-indigo-500 h-2 rounded-full" style="width: 23%"></div>
                </div>
              </div>

              <div>
                <div class="flex justify-between text-xs font-mono text-slate-300 mb-1">
                  <span>l2_leaf_reg</span>
                  <span class="text-indigo-400">17%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div class="bg-indigo-500 h-2 rounded-full" style="width: 17%"></div>
                </div>
              </div>

              <div>
                <div class="flex justify-between text-xs font-mono text-slate-300 mb-1">
                  <span>iterations</span>
                  <span class="text-indigo-400">12%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div class="bg-indigo-500 h-2 rounded-full" style="width: 12%"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      `;
    } else if (this.activeTab === "models") {
      return `
        <div class="overflow-x-auto">
          <table class="w-full wb-table text-left">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Validation Metric</th>
                <th>Score</th>
                <th>Duration</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="font-mono font-bold text-amber-400">#1</td>
                <td class="font-medium text-slate-200">CatBoost (Trial 37)</td>
                <td class="font-mono text-xs">ROC-AUC</td>
                <td class="font-mono font-bold text-emerald-400">0.94621</td>
                <td class="font-mono text-xs text-slate-400">14.2s</td>
                <td><span class="badge-gain text-[10px] px-2 py-0.5 rounded">Optimal</span></td>
              </tr>
              <tr>
                <td class="font-mono font-bold text-slate-400">#2</td>
                <td class="font-medium text-slate-200">LightGBM (Baseline)</td>
                <td class="font-mono text-xs">ROC-AUC</td>
                <td class="font-mono font-bold text-slate-200">0.94110</td>
                <td class="font-mono text-xs text-slate-400">5.03s</td>
                <td><span class="badge-sys text-[10px] px-2 py-0.5 rounded">Verified</span></td>
              </tr>
              <tr>
                <td class="font-mono font-bold text-slate-400">#3</td>
                <td class="font-medium text-slate-200">Logistic Regression</td>
                <td class="font-mono text-xs">ROC-AUC</td>
                <td class="font-mono font-bold text-slate-400">0.93120</td>
                <td class="font-mono text-xs text-slate-400">0.42s</td>
                <td><span class="badge-sys text-[10px] px-2 py-0.5 rounded">Baseline</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      `;
    } else if (this.activeTab === "resources") {
      return `
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
          <div class="space-y-3">
            <div class="font-semibold text-slate-200 uppercase tracking-wider">System Hardware Load</div>
            <div class="space-y-2">
              <div>
                <div class="flex justify-between font-mono text-slate-300 mb-1">
                  <span>CPU (AMD Ryzen 9 / 16 threads)</span>
                  <span class="text-indigo-400">67%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2"><div class="bg-indigo-500 h-2 rounded-full" style="width: 67%"></div></div>
              </div>
              <div>
                <div class="flex justify-between font-mono text-slate-300 mb-1">
                  <span>RAM (16.2 / 32.0 GB)</span>
                  <span class="text-indigo-400">51%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2"><div class="bg-indigo-500 h-2 rounded-full" style="width: 51%"></div></div>
              </div>
            </div>
          </div>

          <div class="space-y-3">
            <div class="font-semibold text-slate-200 uppercase tracking-wider">Parallel Cross-Validation Workers</div>
            <div class="space-y-1.5 font-mono">
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>CatBoost Fold 1</span>
                <span class="text-emerald-400">● Completed (0.9464)</span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>CatBoost Fold 2</span>
                <span class="text-emerald-400">● Completed (0.9459)</span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>CatBoost Fold 3</span>
                <span class="text-indigo-400 animate-pulse">● Running...</span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between opacity-60">
                <span>CatBoost Fold 4</span>
                <span class="text-slate-400">◌ Queued</span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between opacity-60">
                <span>CatBoost Fold 5</span>
                <span class="text-slate-400">◌ Queued</span>
              </div>
            </div>
          </div>
        </div>
      `;
    } else {
      return `
        <div class="font-mono text-xs space-y-1 text-slate-400 bg-slate-950 p-4 rounded-lg border border-slate-800 max-h-56 overflow-y-auto">
          <div><span class="text-indigo-400">[18:24:10]</span> [CATML Engine] Registered dataset 'EV Purchases' (668665 rows)</div>
          <div><span class="text-indigo-400">[18:24:12]</span> [Profiler] Column 'id' detected as identifier (cardinality 1.0) -> Excluded</div>
          <div><span class="text-indigo-400">[18:24:14]</span> [Planner] TargetAdapter activated: encoded ['No', 'Yes'] -> [0, 1]</div>
          <div><span class="text-indigo-400">[18:24:15]</span> [SklearnTrainer] LightGBM fitted in 5.03s -> CV ROC-AUC: 0.94110</div>
          <div><span class="text-indigo-400">[18:24:25]</span> [OptunaOptimizer] Trial 37 reached best score: 0.94621</div>
          <div><span class="text-emerald-400">[18:24:30]</span> [Checkpoint] Model state saved to .automl/s6e9_automl/checkpoint.pkl</div>
        </div>
      `;
    }
  }

  _bindEvents() {
    // Tabs click
    this.container.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeTab = btn.getAttribute("data-tab");
        this.container.querySelector("#tabContent").innerHTML = this._renderTabContent();
        this.container.querySelectorAll(".tab-btn").forEach(b => {
          b.className = `tab-btn py-3 border-b-2 ${
            b.getAttribute("data-tab") === this.activeTab
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`;
        });
      });
    });

    // Run actions (Pause / Resume / Stop / Clone)
    this.container.querySelector("#btnPauseRun")?.addEventListener("click", async () => {
      try {
        const state = store.getState();
        const activeRun = state.runs[0];
        if (activeRun) {
          await api.pauseRun(activeRun.id);
          activeRun.status = "PAUSED";
          this.render();
        }
      } catch (e) {
        alert("Error pausing run: " + e.message);
      }
    });

    this.container.querySelector("#btnResumeRun")?.addEventListener("click", async () => {
      try {
        const state = store.getState();
        const activeRun = state.runs[0];
        if (activeRun) {
          await api.resumeRun(activeRun.id);
          activeRun.status = "RUNNING";
          this.render();
        }
      } catch (e) {
        alert("Error resuming run: " + e.message);
      }
    });

    this.container.querySelector("#btnStopRun")?.addEventListener("click", async () => {
      if (confirm("Are you sure you want to stop this experiment run?")) {
        const state = store.getState();
        const activeRun = state.runs[0];
        if (activeRun) {
          await api.cancelRun(activeRun.id);
          activeRun.status = "CANCELLED";
          this.render();
        }
      }
    });

    this.container.querySelector("#btnCloneRun")?.addEventListener("click", async () => {
      const state = store.getState();
      const activeRun = state.runs[0];
      if (activeRun) {
        await api.cloneRun(activeRun.id, `${activeRun.dataset_name}_cloned`);
        alert("Run cloned successfully.");
      }
    });

    this.container.querySelector("#btnStopHPOAndPromote")?.addEventListener("click", () => {
      alert("HPO stopped. Best parameters promoted to active Model Pipeline.");
    });

    // "Why this?" buttons
    const whyModal = this.container.querySelector("#whyModal");
    const whyModalBody = this.container.querySelector("#whyModalBody");
    const btnCloseWhyModal = this.container.querySelector("#btnCloseWhyModal");

    btnCloseWhyModal?.addEventListener("click", () => {
      whyModal.classList.add("hidden");
    });

    this.container.querySelectorAll(".btn-why").forEach(btn => {
      btn.addEventListener("click", () => {
        const step = btn.getAttribute("data-why");
        this._showWhyModal(step, whyModal, whyModalBody);
      });
    });
  }

  _showWhyModal(step, modal, body) {
    let content = "";
    if (step === "step3") {
      content = `
        <div class="space-y-3 font-sans">
          <div class="border-b border-purple-800/40 pb-2">
            <span class="text-xs uppercase text-slate-400 font-semibold tracking-wider">Candidate</span>
            <div class="text-base font-bold text-purple-300 font-mono">CATBOOST_HPO</div>
          </div>
          <div class="grid grid-cols-2 gap-2 text-xs">
            <div><span class="text-slate-400">Priority Score:</span> <span class="font-mono text-emerald-400 font-bold">0.87</span></div>
            <div><span class="text-slate-400">Planner Rule:</span> <span class="font-mono text-slate-200">high_categorical_density</span></div>
            <div><span class="text-slate-400">Engine Source:</span> <span class="font-mono text-slate-200">RuleBasedPlanner</span></div>
          </div>
          <div class="space-y-1.5 pt-2">
            <div class="font-semibold text-slate-300">Empirical Evidence:</div>
            <div class="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-[11px] space-y-1">
              <div>Categorical Ratio: <span class="text-indigo-400">0.46 (6 / 13 cols)</span></div>
              <div>Missing Ratio:     <span class="text-indigo-400">0.021</span></div>
              <div>LightGBM Baseline Gain: <span class="text-emerald-400">+0.0125 ROC-AUC</span></div>
            </div>
          </div>
          <div class="pt-2 text-slate-300 text-xs">
            CatBoost utiliza codificación adaptativa de combinaciones categóricas en tiempo de construcción del árbol, reduciendo el sobreajuste que ocurre con one-hot encoding tradicional.
          </div>
        </div>
      `;
    } else {
      content = `
        <div class="space-y-2 text-xs">
          <div class="font-bold text-purple-300">Decision rule for ${step.toUpperCase()}</div>
          <p class="text-slate-300">Decisión inferida automáticamente por el motor de priorización basada en las características estadísticas del DatasetProfile y la estrategia de validación.</p>
        </div>
      `;
    }
    body.innerHTML = content;
    modal.classList.remove("hidden");
  }

  _initChart() {
    const canvas = this.container.querySelector("#performanceChart");
    if (!canvas || !window.Chart) return;

    if (this.chart) {
      this.chart.destroy();
    }

    const labels = Array.from({ length: 37 }, (_, i) => i + 1);
    const scores = [
      0.9312, 0.934, 0.9365, 0.938, 0.9411, 0.9412, 0.9415, 0.942, 0.9425, 0.9431,
      0.9433, 0.9437, 0.944, 0.9442, 0.9445, 0.9448, 0.945, 0.9451, 0.9452, 0.9453,
      0.9453, 0.9455, 0.9455, 0.9456, 0.9457, 0.9457, 0.9458, 0.9458, 0.9459, 0.946,
      0.946, 0.9461, 0.9461, 0.9461, 0.9462, 0.9462, 0.94621
    ];

    this.chart = new window.Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "ROC-AUC Progression",
            data: scores,
            borderColor: "#10b981",
            backgroundColor: "rgba(16, 185, 129, 0.1)",
            fill: true,
            tension: 0.25,
            pointRadius: 2,
            pointHoverRadius: 5,
            pointBackgroundColor: "#10b981",
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
              label: (ctx) => `Trial ${ctx.label}: ${ctx.raw.toFixed(5)} ROC-AUC`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(51, 65, 85, 0.2)" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
          },
          y: {
            min: 0.93,
            max: 0.95,
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
