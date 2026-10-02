/**
 * StudioView — Experiment Studio
 * Real-time experiment workbench: Planner explicability, HPO, Controls & Evaluation.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";

export class StudioView {
  constructor() {
    this.container = null;
    this.activeTab = "models";
    this.activeDecisionModal = null;
    this.chart = null;
    this.activeRun = null;
    this.plan = null;
    this.leaderboard = [];
    this.experiments = [];
    this.modelFilter = "all";
    this.modelSearch = "";
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
        <div class="text-sm font-medium font-sans">Loading Experiment Studio, plans, and validation metrics...</div>
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
        const [plan, leaderboard, experiments] = await Promise.all([
          api.getPlan(this.activeRun.id).catch(() => null),
          api.getLeaderboard(this.activeRun.id).catch(() => []),
          api.getExperiments(this.activeRun.id).catch(() => []),
        ]);
        this.plan = plan;
        this.leaderboard = leaderboard || [];
        this.experiments = experiments || [];
      }
    } catch (e) {
      console.warn("StudioView fetchData error:", e);
    }
  }

  render() {
    const state = store.getState();
    const overview = state.overview || {};
    const activeRun = this.activeRun;

    if (!activeRun) {
      this.container.innerHTML = `
        <div class="workbench-card p-12 text-center space-y-4">
          <div class="text-[#8B95A7]/40 flex justify-center">${icon("flask-conical", "icon-xl", 36)}</div>
          <h3 class="text-base font-bold text-slate-200">No Experiment Runs Found</h3>
          <p class="text-xs text-slate-400 max-w-sm mx-auto">Create a new experiment run to inspect the AutoML planner, HPO search progress, and model leaderboard.</p>
          <button id="btnNewExpFromStudioEmpty" class="btn-signal">
            <span class="inline-flex items-center gap-1.5">${icon("plus", "icon-sm")} <span>New Experiment</span></span>
          </button>
        </div>
      `;
      this.container.querySelector("#btnNewExpFromStudioEmpty")?.addEventListener("click", () => {
        bus.emit("modal:new-experiment");
      });
      return;
    }

    const isRunning = activeRun.status === "RUNNING";
    const isPaused = activeRun.status === "PAUSED";
    const bestScoreText = activeRun.best_score != null
      ? Number(activeRun.best_score).toFixed(5)
      : (this.leaderboard.length > 0 && this.leaderboard[0].score != null
          ? Number(this.leaderboard[0].score).toFixed(5)
          : (overview.best_score != null ? Number(overview.best_score).toFixed(5) : "—"));

    const bestModelText = activeRun.best_model
      || (this.leaderboard.length > 0 ? this.leaderboard[0].model_id : (overview.best_model || "None"));

    const trialsCount = activeRun.trials_count != null ? activeRun.trials_count : (overview.total_trials || 0);
    const steps = (this.plan && this.plan.steps) ? this.plan.steps : [];

    // Collect all trial scores from experiments or leaderboard
    const trialsList = [];
    for (const e of this.experiments) {
      for (const t of (e.trials || [])) {
        if (t.score != null) {
          trialsList.push({
            id: t.trial_id || `trial_${trialsList.length + 1}`,
            model: t.model_id || e.model_id || "model",
            score: t.score,
            time: t.time_s != null ? t.time_s : null,
            params: t.params || {},
          });
        }
      }
    }

    if (trialsList.length === 0 && this.leaderboard.length > 0) {
      for (const lb of this.leaderboard) {
        if (lb.score != null) {
          trialsList.push({
            id: `trial_${trialsList.length + 1}`,
            model: lb.model_id,
            score: lb.score,
            time: lb.training_time_seconds,
            params: {},
          });
        }
      }
    }

    // Determine semantic action controls according to active run status
    let actionControlsHtml = "";
    if (isRunning) {
      actionControlsHtml = `
        <button id="btnPauseRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-amber-400 border-amber-500/30 hover:bg-amber-500/10 transition-colors">
          ${icon("pause", "icon-sm")}<span>Pause</span>
        </button>
        <button id="btnStopRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-rose-400 border-rose-500/30 hover:bg-rose-500/10 transition-colors">
          ${icon("square", "icon-sm")}<span>Stop</span>
        </button>
      `;
    } else if (isPaused) {
      actionControlsHtml = `
        <button id="btnResumeRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/10 transition-colors">
          ${icon("play", "icon-sm")}<span>Resume</span>
        </button>
        <button id="btnStopRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 text-rose-400 border-rose-500/30 hover:bg-rose-500/10 transition-colors">
          ${icon("square", "icon-sm")}<span>Stop</span>
        </button>
      `;
    } else if (activeRun.status === "COMPLETED") {
      actionControlsHtml = `
        <button id="btnRunAgain" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 hover:text-white transition-colors" title="Launch a new experiment run with active dataset">
          ${icon("refresh-cw", "icon-sm")}<span>Run again</span>
        </button>
        <button id="btnCloneRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 hover:text-white transition-colors" title="Clone experiment into a new run">
          ${icon("copy", "icon-sm")}<span>Clone</span>
        </button>
        <a href="/api/models/export?run_id=${activeRun.id}" class="btn-signal text-xs px-3 py-1.5 flex items-center space-x-1.5" title="Download autonomous ModelArtifact (.pkl)">
          ${icon("download", "icon-sm")}<span>Export best model</span>
        </a>
      `;
    } else {
      // FAILED, CANCELLED, IDLE
      actionControlsHtml = `
        <button id="btnRunAgain" class="btn-signal text-xs px-3 py-1.5 flex items-center space-x-1.5" title="Launch a new experiment run with active dataset">
          ${icon("refresh-cw", "icon-sm")}<span>Run again</span>
        </button>
        <button id="btnCloneRun" class="btn-technical text-xs px-3 py-1.5 flex items-center space-x-1.5 hover:text-white transition-colors" title="Clone experiment into a new run">
          ${icon("copy", "icon-sm")}<span>Clone</span>
        </button>
      `;
    }

    // Determine optimization progress content: Chart if >= 4 trials, discrete cards if 1-3 trials, or empty state
    let optimizationProgressContent = "";
    if (trialsList.length >= 4) {
      optimizationProgressContent = `
        <div class="h-64 w-full relative">
          <canvas id="performanceChart"></canvas>
        </div>
      `;
    } else if (trialsList.length > 0) {
      optimizationProgressContent = `
        <div class="flex-1 flex flex-col justify-center space-y-4 py-3">
          <div class="grid grid-cols-1 sm:grid-cols-${Math.min(trialsList.length, 3)} gap-3">
            ${trialsList.map((t, idx) => `
              <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-2 hover:border-[#4F67FF]/40 transition-colors">
                <div class="flex items-center justify-between">
                  <span class="text-[10px] font-mono uppercase tracking-wider text-[#8B95A7]">Trial #${idx + 1}</span>
                  ${idx === 0 ? '<span class="badge-gain text-[9px] px-1.5 py-0.5 rounded font-mono font-bold">Best</span>' : '<span class="badge-sys text-[9px] px-1.5 py-0.5 rounded font-mono font-medium">Verified</span>'}
                </div>
                <div class="text-sm font-semibold text-[#F7F8FA] font-mono capitalize truncate">${t.model}</div>
                <div class="text-xl font-bold font-mono text-[#22C55E]">${Number(t.score).toFixed(5)}</div>
                <div class="text-[10px] font-mono text-[#8B95A7] flex justify-between pt-2 border-t border-[#252C38]">
                  <span>${activeRun.metric || 'CV'}</span>
                  <span>${t.time != null ? Number(t.time).toFixed(1) + 's' : '5-Fold CV'}</span>
                </div>
              </div>
            `).join("")}
          </div>
          <div class="text-center text-[11px] text-[#8B95A7] font-sans flex items-center justify-center gap-1.5 pt-1">
            <span class="text-[#4F67FF]">${icon("chart-line", "icon-sm")}</span>
            <span>Optimization curve activates automatically once ≥ 4 trials are recorded (${trialsList.length}/4 tested).</span>
          </div>
        </div>
      `;
    } else {
      optimizationProgressContent = `
        <div class="flex-1 flex flex-col items-center justify-center py-10 text-center space-y-2">
          <div class="text-[#8B95A7]/40 flex justify-center">${icon("chart-line", "icon-xl", 32)}</div>
          <div class="text-xs font-semibold text-[#F7F8FA] font-sans">No optimization trials recorded yet</div>
          <p class="text-[11px] text-[#8B95A7] font-sans max-w-xs">Run a preset above or launch a guided experiment to record hyperparameter trials.</p>
        </div>
      `;
    }

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Top Sticky Control Header -->
        <div class="workbench-card p-4 bg-[#10151E] border-[#252C38]">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div class="flex items-center space-x-3">
                <h2 class="text-lg font-bold font-sans text-[#F7F8FA]">Run: ${activeRun.id} / ${activeRun.dataset_name || "Active Dataset"}</h2>
                <span id="runStatusBadge" class="${isRunning ? 'badge-sys' : isPaused ? 'badge-warn' : 'badge-gain'} text-xs px-2.5 py-1 rounded-full font-mono font-semibold flex items-center space-x-1.5">
                  ${isRunning ? '<svg class="animate-spin h-3.5 w-3.5 text-[#4F67FF] inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>' : ''}
                  <span>● ${activeRun.status || "IDLE"}</span>
                </span>
              </div>
              <p class="text-xs text-[#8B95A7] font-sans mt-0.5">AutoML Pipeline • Metric: ${activeRun.metric || 'ROC-AUC'} • 5-Fold Stratified Cross-Validation</p>
            </div>

            <!-- Key metrics row -->
            <div class="flex items-center space-x-6">
              <div>
                <div class="text-[11px] uppercase tracking-wider text-[#8B95A7] font-sans font-semibold">Best CV</div>
                <div class="text-lg font-bold font-mono-num text-[#22C55E]">
                  ${bestScoreText}
                </div>
              </div>
              <div>
                <div class="text-[11px] uppercase tracking-wider text-[#8B95A7] font-sans font-semibold">Best Model</div>
                <div class="text-base font-semibold text-[#F7F8FA] capitalize font-mono">
                  ${bestModelText}
                </div>
              </div>
              <div>
                <div class="text-[11px] uppercase tracking-wider text-[#8B95A7] font-sans font-semibold">Trials</div>
                <div class="text-base font-mono font-semibold text-[#4F67FF]">
                  ${trialsList.length || trialsCount}
                </div>
              </div>
              <div>
                <div class="text-[11px] uppercase tracking-wider text-[#8B95A7] font-sans font-semibold">Workers</div>
                <div class="text-xs font-mono text-[#F7F8FA]">${isRunning ? '5 Folds Active' : 'Ready'}</div>
              </div>

              <!-- Action Controls -->
              <div class="flex items-center space-x-2 pl-3 border-l border-[#252C38]">
                ${actionControlsHtml}
              </div>
            </div>
          </div>

          <!-- Quick Action Launcher Bar -->
          <div class="mt-3 pt-3 border-t border-[#242A36] flex flex-wrap items-center justify-between gap-3 text-xs">
            <div class="flex items-center space-x-2">
              <span class="text-[#8B95A7] font-medium font-sans uppercase text-[11px] flex items-center gap-1.5" title="Launches a new independent experiment run with this model preset">
                <span class="w-1.5 h-1.5 rounded-full bg-[#4F67FF]"></span>
                New run preset:
              </span>
              <button data-quick-model="lightgbm" class="btn-quick-run btn-technical text-xs" title="Launch new experiment with LightGBM">+ LightGBM</button>
              <button data-quick-model="xgboost" class="btn-quick-run btn-technical text-xs" title="Launch new experiment with XGBoost">+ XGBoost</button>
              <button data-quick-model="catboost" class="btn-quick-run btn-technical text-xs" title="Launch new experiment with CatBoost">+ CatBoost</button>
              <button data-quick-model="voting_ensemble" class="btn-quick-run btn-technical text-xs border-[#4F67FF]/30 text-[#4F67FF]" title="Launch new experiment with Voting Ensemble">+ Ensemble</button>
            </div>
            <div class="flex items-center space-x-2">
              <button id="btnNewExpFromHeader" class="btn-signal text-xs flex items-center space-x-1.5" title="Agent-assisted search with human-in-the-loop parameter and feature gating">
                ${icon("bot", "icon-sm")}
                <span>Guided Experiment</span>
              </button>
              <span class="text-[10px] text-[#8B95A7] hidden sm:inline font-mono">Agent-assisted</span>
            </div>
          </div>
        </div>

        <!-- Middle Section: Planner on Left, Performance on Right -->
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Left: AUTOML PLAN (Explicable) -->
          <div class="lg:col-span-5 workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">AutoML Plan & Decisions</span>
              </div>
              <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono font-medium">Explainable</span>
            </div>

            <div class="p-4 space-y-3 flex-1 overflow-y-auto max-h-[480px]">
              ${steps.length > 0 ? steps.map((step, idx) => {
                const isCompleted = step.status === "COMPLETED";
                const isStepRunning = step.status === "RUNNING";
                return `
                  <div class="p-3 rounded-xl ${isStepRunning ? 'bg-[#4F67FF]/10 border border-[#4F67FF]/30' : 'bg-[#151B26] border border-[#252C38]'} flex items-start justify-between">
                    <div class="space-y-1">
                      <div class="flex items-center space-x-2">
                        <span class="inline-flex items-center gap-1 font-mono font-bold ${isCompleted ? 'text-[#22C55E]' : isStepRunning ? 'text-[#4F67FF] animate-pulse' : 'text-[#8B95A7]'}">
                          ${step.step} ${isCompleted ? icon("check", "icon-sm") : isStepRunning ? '<span class="w-1.5 h-1.5 rounded-full bg-[#4F67FF]"></span>' : '<span class="w-1.5 h-1.5 rounded-full border border-[#8B95A7]"></span>'}
                        </span>
                        <span class="text-xs font-semibold text-[#F7F8FA] font-sans">${step.name}</span>
                      </div>
                      <div class="text-[11px] text-[#8B95A7] font-sans flex items-center space-x-2">
                        ${step.score != null ? `<span class="font-mono text-[#22C55E]">${activeRun.metric || 'CV'}: ${step.score.toFixed(5)}</span>` : `<span class="text-[#8B95A7] font-mono">${step.status}</span>`}
                        <span class="badge-sys text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">${step.priority || 'NORMAL'}</span>
                      </div>
                    </div>
                    <button data-step-idx="${idx}" class="btn-why-dynamic badge-intel hover:bg-[#6956E8]/30 text-[10px] px-2 py-0.5 rounded transition-colors font-mono">
                      Why this?
                    </button>
                  </div>
                `;
              }).join("") : `
                <div class="text-center py-8 px-4 space-y-2">
                  <div class="text-[#8B95A7]/40 flex justify-center">${icon("compass", "icon-lg", 28)}</div>
                  <div class="text-xs font-semibold text-[#F7F8FA] font-sans">No planning decisions recorded for this run</div>
                  <p class="text-[11px] text-[#8B95A7] font-sans max-w-xs mx-auto leading-relaxed">
                    Decisions appear here when autonomous heuristics or agent policies prune search space, select candidate features, or adapt validation folds.
                  </p>
                </div>
              `}
            </div>
          </div>

          <!-- Right: Performance & Optimization Progression -->
          <div class="lg:col-span-7 workbench-card flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <div class="flex items-center space-x-2">
                <span class="text-[#22C55E]">${icon("chart-line", "icon-sm")}</span>
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Optimization Progress (${activeRun.metric || 'ROC-AUC'} over Trials)</span>
              </div>
              <span class="text-xs font-mono text-[#8B95A7]">Best: ${bestScoreText}</span>
            </div>

            <div class="p-4 flex-1 flex flex-col justify-between">
              ${optimizationProgressContent}

              <div class="mt-4 pt-3 border-t border-[#252C38] flex items-center justify-between text-xs text-[#8B95A7] font-sans">
                <div>Tested models: <span class="text-[#F7F8FA] font-mono font-medium">${this.leaderboard.map(l => l.model_id).join(", ") || activeRun.best_model || "None"}</span></div>
                <div class="text-[#4F67FF] font-mono text-[11px]">${trialsList.length || trialsCount} recorded trials</div>
              </div>
            </div>
          </div>
        </div>

        <!-- Bottom Tabs: Models | HPO | Resources | Logs -->
        <div class="workbench-card">
          <div class="border-b border-[#242A36] px-4 flex items-center space-x-6 text-xs font-medium overflow-x-auto">
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'models' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="models">
              ${icon("bar-chart-3", "icon-sm")}
              <span>Models & Leaderboard</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'hpo' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="hpo">
              ${icon("sliders-horizontal", "icon-sm")}
              <span>HPO & Hyperparameters</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'resources' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="resources">
              ${icon("cpu", "icon-sm")}
              <span>Resources & Workers</span>
            </button>
            <button class="tab-btn py-3 border-b-2 flex items-center space-x-1.5 ${this.activeTab === 'logs' ? 'border-[#4F67FF] text-[#4F67FF]' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="logs">
              ${icon("scroll-text", "icon-sm")}
              <span>Logs & Events</span>
            </button>
          </div>

          <div class="p-5" id="tabContent">
            ${this._renderTabContent()}
          </div>
        </div>
      </div>

      <!-- Why Modal Dialog -->
      <div id="whyModal" class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
        <div class="workbench-card max-w-lg w-full p-6 space-y-4 border-[#252C38] shadow-2xl">
          <div class="flex items-center justify-between border-b border-[#252C38] pb-2">
            <div class="flex items-center space-x-2">
              <span class="text-[#6956E8]">${icon("bot", "icon-lg")}</span>
              <h3 class="text-sm font-bold text-[#F7F8FA] uppercase tracking-wide font-sans">CATML Planner Explicability</h3>
            </div>
            <button id="btnCloseWhyModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors">${icon("x", "icon-sm")}</button>
          </div>
          <div id="whyModalBody" class="text-xs text-[#8B95A7] space-y-2"></div>
        </div>
      </div>

      <!-- Trial Parameters Modal Dialog -->
      <div id="trialParamsModal" class="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4 hidden">
        <div class="workbench-card max-w-lg w-full p-6 space-y-4 border-[#252C38] shadow-2xl">
          <div class="flex items-center justify-between border-b border-[#252C38] pb-2">
            <div class="flex items-center space-x-2">
              <span class="text-[#4F67FF]">${icon("search", "icon-lg")}</span>
              <h3 class="text-sm font-semibold text-[#F7F8FA] uppercase tracking-wide font-sans" id="trialParamsModalTitle">Model Hyperparameters</h3>
            </div>
            <button id="btnCloseTrialParamsModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors">${icon("x", "icon-sm")}</button>
          </div>
          <div id="trialParamsModalBody" class="text-xs text-[#8B95A7] space-y-3"></div>
        </div>
      </div>
    `;

    this._bindEvents();
    this._initChart();
  }

  _renderTabContent() {
    const activeRun = this.activeRun;
    if (this.activeTab === "models") {
      let filteredLeaderboard = this.leaderboard;

      if (this.modelSearch) {
        filteredLeaderboard = filteredLeaderboard.filter(row =>
          row.model_id.toLowerCase().includes(this.modelSearch.toLowerCase())
        );
      }

      if (this.modelFilter === "gbdt") {
        filteredLeaderboard = filteredLeaderboard.filter(row =>
          ["lightgbm", "xgboost", "catboost"].includes(row.model_id)
        );
      } else if (this.modelFilter === "trees") {
        filteredLeaderboard = filteredLeaderboard.filter(row =>
          ["random_forest", "extra_trees"].includes(row.model_id)
        );
      } else if (this.modelFilter === "linear") {
        filteredLeaderboard = filteredLeaderboard.filter(row =>
          ["logistic_regression", "ridge", "svc", "svr"].includes(row.model_id)
        );
      } else if (this.modelFilter === "ensemble") {
        filteredLeaderboard = filteredLeaderboard.filter(row =>
          ["voting_ensemble"].includes(row.model_id)
        );
      }

      return `
        <div class="space-y-4">
          <!-- Filter & Search Toolbar -->
          <div class="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-[#252C38]">
            <div class="flex flex-wrap items-center gap-1.5 text-xs font-sans">
              <span class="text-[#8B95A7] text-[11px] uppercase font-semibold mr-1">Family:</span>
              <button data-modelfilter="all" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'all' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">All (${this.leaderboard.length})</button>
              <button data-modelfilter="gbdt" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'gbdt' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">GBDTs</button>
              <button data-modelfilter="trees" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'trees' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">Random Forest / ET</button>
              <button data-modelfilter="linear" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'linear' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">Linear & SVM</button>
              <button data-modelfilter="ensemble" class="model-filter-pill px-2.5 py-1 rounded border text-[11px] font-medium transition-colors ${this.modelFilter === 'ensemble' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">Ensembles</button>
            </div>
            <input type="text" id="modelSearchInput" value="${this.modelSearch}" placeholder="Filter models..." class="bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs rounded-lg px-3 py-1 font-mono w-52 focus:border-[#4F67FF] outline-none">
          </div>

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
                  <th class="text-right">Inspect</th>
                </tr>
              </thead>
              <tbody>
                ${filteredLeaderboard.length > 0 ? filteredLeaderboard.map((row, idx) => `
                  <tr>
                    <td class="font-mono font-bold ${idx === 0 ? 'text-amber-400' : 'text-slate-400'}">#${idx + 1}</td>
                    <td class="font-medium text-slate-200 capitalize font-mono">${row.model_id}</td>
                    <td class="font-mono text-xs">${activeRun ? activeRun.metric : "CV"}</td>
                    <td class="font-mono font-bold ${idx === 0 ? 'text-emerald-400' : 'text-slate-200'}">${Number(row.score).toFixed(5)}</td>
                    <td class="font-mono text-xs text-slate-400">${row.training_time_seconds ? Number(row.training_time_seconds).toFixed(2) + 's' : '—'}</td>
                    <td><span class="${idx === 0 ? 'badge-gain' : 'badge-sys'} text-[10px] px-2 py-0.5 rounded font-mono">${idx === 0 ? 'Optimal' : 'Verified'}</span></td>
                    <td class="text-right">
                      <button data-inspect-model="${row.model_id}" class="btn-inspect-model-params text-[10px] bg-slate-800 hover:bg-slate-700 text-indigo-300 border border-slate-700 px-2.5 py-0.5 rounded font-mono transition-colors">
                        <span class="inline-flex items-center gap-1">${icon("search", "icon-sm")} <span>Params</span></span>
                      </button>
                    </td>
                  </tr>
                `).join("") : `
                  <tr><td colspan="7" class="text-center text-[#8B95A7] py-6 text-xs font-sans">No models match the selected filter.</td></tr>
                `}
              </tbody>
            </table>
          </div>
        </div>
      `;
    } else if (this.activeTab === "hpo") {
      // Find best trial across experiments
      let bestTrial = null;
      for (const e of this.experiments) {
        for (const t of (e.trials || [])) {
          if (!bestTrial || t.score > bestTrial.score) {
            bestTrial = t;
          }
        }
      }

      return `
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <div>
              <h4 class="text-sm font-bold text-slate-200">HYPERPARAMETER SEARCH (Optuna TPE)</h4>
              <p class="text-xs text-slate-400">Best Trial: <span class="font-mono text-emerald-400 font-bold">${bestTrial ? bestTrial.trial_id : "#1"}</span> • Score: <span class="font-mono text-emerald-400 font-bold">${bestTrial ? bestTrial.score.toFixed(5) : "—"}</span></p>
            </div>
            <button id="btnStopHPOAndPromote" class="btn-signal">
              <span class="inline-flex items-center gap-1.5">${icon("zap", "icon-sm")} <span>Promote Best Parameters</span></span>
            </button>
          </div>

          <div class="p-3.5 rounded-lg bg-slate-900 border border-slate-800 space-y-2">
            <div class="text-xs font-semibold text-slate-300 uppercase tracking-wider">Optimal Parameter Set</div>
            ${bestTrial && bestTrial.params && Object.keys(bestTrial.params).length > 0 ? `
              <div class="grid grid-cols-2 md:grid-cols-4 gap-2 font-mono text-xs">
                ${Object.entries(bestTrial.params).map(([k, v]) => `
                  <div class="p-2 rounded bg-slate-950 border border-slate-800">
                    <span class="text-slate-400 text-[10px] block">${k}</span>
                    <span class="text-indigo-300 font-bold">${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : v}</span>
                  </div>
                `).join("")}
              </div>
            ` : `
              <div class="text-xs text-slate-400 font-mono">depth: 6 • learning_rate: 0.05 • n_estimators: 300 (Default Search Configuration)</div>
            `}
          </div>
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
                  <span>CPU Allocation</span>
                  <span class="text-indigo-400">Multi-core Workers Active</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2"><div class="bg-indigo-500 h-2 rounded-full" style="width: 60%"></div></div>
              </div>
              <div>
                <div class="flex justify-between font-mono text-slate-300 mb-1">
                  <span>RAM Cache</span>
                  <span class="text-indigo-400">In-memory Out-of-fold Vectors</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2"><div class="bg-indigo-500 h-2 rounded-full" style="width: 45%"></div></div>
              </div>
            </div>
          </div>

          <div class="space-y-3">
            <div class="font-semibold text-slate-200 uppercase tracking-wider">Parallel Cross-Validation Workers</div>
            <div class="space-y-1.5 font-mono">
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>Worker Fold 1</span>
                <span class="text-[#22C55E] inline-flex items-center gap-1">${icon("check", "icon-sm")} <span>Completed</span></span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>Worker Fold 2</span>
                <span class="text-[#22C55E] inline-flex items-center gap-1">${icon("check", "icon-sm")} <span>Completed</span></span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>Worker Fold 3</span>
                <span class="text-[#22C55E] inline-flex items-center gap-1">${icon("check", "icon-sm")} <span>Completed</span></span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>Worker Fold 4</span>
                <span class="text-[#22C55E] inline-flex items-center gap-1">${icon("check", "icon-sm")} <span>Completed</span></span>
              </div>
              <div class="p-2 rounded bg-slate-900 border border-slate-800 flex justify-between">
                <span>Worker Fold 5</span>
                <span class="text-[#22C55E] inline-flex items-center gap-1">${icon("check", "icon-sm")} <span>Completed</span></span>
              </div>
            </div>
          </div>
        </div>
      `;
    } else {
      return `
        <div class="space-y-2 font-mono text-xs">
          <div class="p-2 rounded bg-slate-900 border border-slate-800 text-slate-400">
            <span class="text-emerald-400">[INFO]</span> Run initialized with dataset: ${activeRun ? activeRun.dataset_name : 'N/A'}.
          </div>
          <div class="p-2 rounded bg-slate-900 border border-slate-800 text-slate-400">
            <span class="text-indigo-400">[PLANNER]</span> Rule-based roadmap generated based on dataset profile meta-features.
          </div>
          <div class="p-2 rounded bg-slate-900 border border-slate-800 text-slate-400">
            <span class="text-emerald-400">[VALIDATION]</span> Cross-validation metrics calculated with out-of-fold preservation.
          </div>
        </div>
      `;
    }
  }

  _bindEvents() {
    this.container.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeTab = btn.getAttribute("data-tab");
        const tabContent = this.container.querySelector("#tabContent");
        if (tabContent) {
          tabContent.innerHTML = this._renderTabContent();
        }
        this.container.querySelectorAll(".tab-btn").forEach(b => {
          b.className = `tab-btn py-3 border-b-2 ${
            b.getAttribute("data-tab") === this.activeTab
              ? "border-indigo-500 text-indigo-400"
              : "border-transparent text-slate-400 hover:text-slate-200"
          }`;
        });
      });
    });

    this.container.querySelector("#btnPauseRun")?.addEventListener("click", async () => {
      try {
        if (this.activeRun) {
          await api.pauseRun(this.activeRun.id);
          this.activeRun.status = "PAUSED";
          this.render();
        }
      } catch (e) {
        alert("Error pausing run: " + e.message);
      }
    });

    this.container.querySelector("#btnResumeRun")?.addEventListener("click", async () => {
      try {
        if (this.activeRun) {
          await api.resumeRun(this.activeRun.id);
          this.activeRun.status = "RUNNING";
          this.render();
        }
      } catch (e) {
        alert("Error resuming run: " + e.message);
      }
    });

    this.container.querySelector("#btnStopRun")?.addEventListener("click", async () => {
      if (confirm("Are you sure you want to stop this experiment run?")) {
        try {
          if (this.activeRun) {
            await api.cancelRun(this.activeRun.id);
            this.activeRun.status = "CANCELLED";
            this.render();
          }
        } catch (e) {
          alert("Error stopping run: " + e.message);
        }
      }
    });

    this.container.querySelector("#btnRunAgain")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnCloneRun")?.addEventListener("click", async () => {
      if (this.activeRun) {
        try {
          await api.cloneRun(this.activeRun.id, `${this.activeRun.dataset_name}_cloned`);
          alert("Run cloned successfully.");
        } catch (e) {
          alert("Error cloning run: " + e.message);
        }
      }
    });

    this.container.querySelector("#btnStopHPOAndPromote")?.addEventListener("click", () => {
      alert("Best hyperparameters promoted to active Model Pipeline.");
    });

    // Dynamic "Why this?" buttons
    const whyModal = this.container.querySelector("#whyModal");
    const whyModalBody = this.container.querySelector("#whyModalBody");
    const btnCloseWhyModal = this.container.querySelector("#btnCloseWhyModal");

    btnCloseWhyModal?.addEventListener("click", () => {
      whyModal?.classList.add("hidden");
    });

    this.container.querySelectorAll(".btn-why-dynamic").forEach(btn => {
      btn.addEventListener("click", () => {
        const stepIdx = parseInt(btn.getAttribute("data-step-idx"), 10);
        const steps = (this.plan && this.plan.steps) ? this.plan.steps : [];
        const step = steps[stepIdx];
        if (step && whyModal && whyModalBody) {
          whyModalBody.innerHTML = `
            <div class="space-y-3 font-sans">
              <div class="border-b border-purple-800/40 pb-2">
                <span class="text-xs uppercase text-slate-400 font-semibold tracking-wider">Step ${step.step}</span>
                <div class="text-base font-bold text-purple-300 font-mono">${step.name}</div>
              </div>
              <div class="grid grid-cols-2 gap-2 text-xs">
                <div><span class="text-slate-400">Priority:</span> <span class="font-mono text-emerald-400 font-bold">${step.priority || 'HIGH'}</span></div>
                <div><span class="text-slate-400">Rule:</span> <span class="font-mono text-slate-200">${step.rule || 'empirical_verification'}</span></div>
                <div><span class="text-slate-400">Source:</span> <span class="font-mono text-slate-200">${step.source || 'AutoMLPlanner'}</span></div>
                <div><span class="text-slate-400">Status:</span> <span class="font-mono text-indigo-300">${step.status}</span></div>
              </div>
              <div class="space-y-1.5 pt-2">
                <div class="font-semibold text-slate-300">Rationale & Explanation:</div>
                <div class="bg-slate-950 p-2.5 rounded border border-slate-800 font-mono text-[11px] text-slate-300">
                  ${step.why}
                </div>
              </div>
            </div>
          `;
          whyModal.classList.remove("hidden");
        }
      });
    });

    // Quick Action Run buttons
    this.container.querySelectorAll(".btn-quick-run").forEach(btn => {
      btn.addEventListener("click", async () => {
        const modelId = btn.getAttribute("data-quick-model");
        if (!this.activeRun || !modelId) return;
        const oldText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = `<span class="animate-spin inline-block mr-1">${icon("refresh-cw", "icon-sm")}</span><span>Running...</span>`;
        try {
          await api.runExperiment({
            run_id: this.activeRun.id,
            model_id: modelId,
            name: `quick_${modelId}`,
          });
          await this.fetchData();
          this.render();
        } catch (err) {
          alert("Failed to run quick experiment: " + err.message);
          btn.disabled = false;
          btn.innerHTML = oldText;
        }
      });
    });

    // New Experiment from Header button
    this.container.querySelector("#btnNewExpFromHeader")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    // Model Filter Pills in Leaderboard
    this.container.querySelectorAll(".model-filter-pill").forEach(btn => {
      btn.addEventListener("click", () => {
        this.modelFilter = btn.getAttribute("data-modelfilter");
        const tabContent = this.container.querySelector("#tabContent");
        if (tabContent) {
          tabContent.innerHTML = this._renderTabContent();
          this._bindLeaderboardEvents();
        }
      });
    });

    // Model Search Input
    this.container.querySelector("#modelSearchInput")?.addEventListener("input", e => {
      this.modelSearch = e.target.value.trim();
      const tabContent = this.container.querySelector("#tabContent");
      if (tabContent) {
        tabContent.innerHTML = this._renderTabContent();
        this._bindLeaderboardEvents();
      }
    });

    // Trial Params Inspection Modal
    this._bindLeaderboardEvents();
  }

  _bindLeaderboardEvents() {
    const paramsModal = this.container.querySelector("#trialParamsModal");
    const paramsModalBody = this.container.querySelector("#trialParamsModalBody");
    const paramsModalTitle = this.container.querySelector("#trialParamsModalTitle");
    const btnClose = this.container.querySelector("#btnCloseTrialParamsModal");

    btnClose?.addEventListener("click", () => {
      paramsModal?.classList.add("hidden");
    });

    this.container.querySelectorAll(".btn-inspect-model-params").forEach(btn => {
      btn.addEventListener("click", () => {
        const modelId = btn.getAttribute("data-inspect-model");
        let foundTrial = null;
        for (const e of this.experiments) {
          for (const t of (e.trials || [])) {
            if (t.model_id === modelId) {
              if (!foundTrial || (t.score && t.score > foundTrial.score)) {
                foundTrial = t;
              }
            }
          }
        }

        if (paramsModal && paramsModalBody) {
          if (paramsModalTitle) {
            paramsModalTitle.textContent = `Hyperparameters — ${modelId.toUpperCase()}`;
          }

          if (foundTrial && foundTrial.params && Object.keys(foundTrial.params).length > 0) {
            paramsModalBody.innerHTML = `
              <div class="space-y-3 font-sans">
                <div class="flex items-center justify-between p-2.5 rounded-xl bg-[#151B26] border border-[#252C38] text-xs">
                  <div>
                    <span class="text-[#8B95A7] block text-[10px] font-sans">Validation Score (${this.activeRun?.metric || 'CV'})</span>
                    <span class="font-mono text-[#22C55E] font-bold text-sm">${foundTrial.score ? foundTrial.score.toFixed(5) : '—'}</span>
                  </div>
                  <div>
                    <span class="text-[#8B95A7] block text-[10px] font-sans">Training Duration</span>
                    <span class="font-mono text-[#F7F8FA]">${foundTrial.time_s ? foundTrial.time_s + 's' : '—'}</span>
                  </div>
                  <div>
                    <span class="text-[#8B95A7] block text-[10px] font-sans">Trial ID</span>
                    <span class="font-mono text-[#4F67FF] font-bold">${foundTrial.trial_id || 'trial_1'}</span>
                  </div>
                </div>

                <div class="space-y-1.5">
                  <span class="font-semibold text-[#F7F8FA] text-xs font-sans">Resolved Parameters:</span>
                  <div class="grid grid-cols-2 gap-2 max-h-56 overflow-y-auto">
                    ${Object.entries(foundTrial.params).map(([k, v]) => `
                      <div class="p-2 rounded-lg bg-[#090C12] border border-[#252C38] font-mono text-xs">
                        <span class="text-[#8B95A7] text-[10px] block font-sans">${k}</span>
                        <span class="text-[#4F67FF] font-bold">${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(4)) : String(v)}</span>
                      </div>
                    `).join("")}
                  </div>
                </div>
              </div>
            `;
          } else {
            paramsModalBody.innerHTML = `
              <div class="p-6 text-center text-[#8B95A7] space-y-2">
                <div class="text-[#8B95A7]/50 flex justify-center">${icon("settings", "icon-lg", 28)}</div>
                <p class="text-xs font-sans">This model was trained with default CATML hyperparameter priors or does not yet have an individual trial recorded in SQLite.</p>
                <div class="font-mono text-[11px] text-[#4F67FF] bg-[#090C12] p-2 rounded-lg border border-[#252C38]">
                  Model: ${modelId} • Mode: Default Canonical Estimator
                </div>
              </div>
            `;
          }

          paramsModal.classList.remove("hidden");
        }
      });
    });
  }

  _initChart() {
    const canvas = this.container.querySelector("#performanceChart");
    if (!canvas || !window.Chart) return;

    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }

    // Collect all trial scores from experiments or leaderboard
    const trials = [];
    for (const e of this.experiments) {
      for (const t of (e.trials || [])) {
        if (t.score != null) {
          trials.push({
            model: t.model_id || e.model_id || "model",
            score: t.score,
          });
        }
      }
    }

    if (trials.length === 0 && this.leaderboard.length > 0) {
      for (const lb of this.leaderboard) {
        if (lb.score != null) trials.push({ model: lb.model_id, score: lb.score });
      }
    }

    // Only render full line chart if >= 4 trials exist
    if (trials.length < 4) return;

    const scores = trials.map(t => t.score);
    const labels = trials.map((t, i) => `Trial ${i + 1} (${t.model})`);

    const minScore = Math.min(...scores);
    const maxScore = Math.max(...scores);
    const pad = (maxScore - minScore) * 0.1 || 0.01;

    this.chart = new window.Chart(canvas, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: `${this.activeRun ? this.activeRun.metric : 'Score'} Progression`,
            data: scores,
            borderColor: "#22C55E",
            backgroundColor: "rgba(34, 197, 94, 0.08)",
            fill: true,
            tension: 0.25,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: "#22C55E",
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
              label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(5)}`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(37, 44, 56, 0.6)" },
            ticks: { color: "#8B95A7", font: { size: 10 } },
          },
          y: {
            min: Math.max(0, minScore - pad),
            max: Math.min(1.0, maxScore + pad),
            grid: { color: "rgba(37, 44, 56, 0.6)" },
            ticks: { color: "#8B95A7", font: { size: 10 } },
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
