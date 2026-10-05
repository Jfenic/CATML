/**
 * CompareView — Multi-Experiment Comparator & Diff Inspector
 * Side-by-side comparison, progression tracking & explicit difference breakdown.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";

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
      <div class="workbench-card p-12 text-center text-[#8B95A7] space-y-3">
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-medium font-sans">Loading experiments for comparison...</div>
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
            <div class="text-[#8B95A7]/40 flex justify-center">${icon("arrow-left-right", "icon-xl", 32)}</div>
            <h3 class="text-base font-semibold text-[#F7F8FA] font-sans">No experiments found to compare</h3>
            <p class="text-xs text-[#8B95A7] max-w-sm mx-auto font-sans">You need at least 2 completed or active experiments in the workspace to perform side-by-side comparison.</p>
            <button id="btnNewExpCompareEmpty" class="btn-signal">
              <span class="inline-flex items-center gap-1.5">${icon("plus", "icon-sm")} <span>Create first experiment</span></span>
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
        <div class="workbench-card p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex items-center space-x-3">
            <span class="text-[#4F67FF]">${icon("arrow-left-right", "icon-lg")}</span>
            <div>
              <h2 class="text-base font-semibold font-sans text-[#F7F8FA]">Experiment Comparison & Lineage</h2>
              <p class="text-xs text-[#8B95A7] font-sans">Select two or more experiments to inspect algorithmic differences and metric gains.</p>
            </div>
          </div>

          <!-- Experiment Checkboxes -->
          <div class="flex flex-wrap items-center gap-2 text-xs font-mono">
            ${this.experiments.map(exp => `
              <label class="flex items-center space-x-1.5 cursor-pointer px-3 py-1.5 rounded-lg border ${this.selected.includes(exp.id) ? 'border-[#4F67FF] text-[#F7F8FA] bg-[#161B26]' : 'border-[#242A36] text-[#8B95A7] bg-[#11151E]'}">
                <input type="checkbox" ${this.selected.includes(exp.id) ? "checked" : ""} class="exp-compare-chk rounded text-[#4F67FF] focus:ring-0 bg-[#080A0F] border border-[#242A36]" data-exp-id="${exp.id}">
                <span>${exp.name || exp.id}</span>
              </label>
            `).join("")}
          </div>
        </div>

        <!-- Comparative Table -->
        <div class="workbench-card overflow-hidden">
          <div class="workbench-panel-header flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="text-[#8B95A7]">${icon("boxes", "icon-sm")}</span>
              <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Side-by-Side Metrics & Architectures</span>
            </div>
            <div class="flex items-center gap-3">
              <span class="text-xs font-mono text-[#8B95A7]">Metric: ${metricName}</span>
              <button id="btnOpenEnsembleModal" ${selectedExps.length < 2 ? "disabled" : ""} class="btn-signal text-xs py-1 px-3 inline-flex items-center gap-1.5 ${selectedExps.length < 2 ? 'opacity-40 cursor-not-allowed' : ''}">
                ${icon("layers", "icon-sm")}
                <span>Build Ensemble (${selectedExps.length})</span>
              </button>
            </div>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full wb-table text-left">
              <thead>
                <tr>
                  <th class="w-48">Dimension</th>
                  ${selectedExps.map(e => `
                    <th class="font-mono text-[#F7F8FA]">${e.name || e.id}</th>
                  `).join("")}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="font-medium text-[#8B95A7]">Validation Score</td>
                  ${selectedExps.map(e => `
                    <td class="font-mono font-bold text-[#22C55E]">
                      ${e.best_score != null ? Number(e.best_score).toFixed(5) : "—"}
                    </td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-[#8B95A7]">Models / Family</td>
                  ${selectedExps.map(e => `
                    <td><span class="badge-sys px-2 py-0.5 rounded text-xs font-mono uppercase">${(e.model_ids || []).join(", ") || "N/A"}</span></td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-[#8B95A7]">Features Count</td>
                  ${selectedExps.map(e => `
                    <td class="font-mono text-[#F7F8FA]">${e.features_count || (e.feature_names ? e.feature_names.length : 0)} features</td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-[#8B95A7]">Optimization Trials</td>
                  ${selectedExps.map(e => `
                    <td class="font-mono text-[#F7F8FA]">${e.trials_count != null ? e.trials_count : (e.trials ? e.trials.length : 0)} trials</td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-[#8B95A7]">Execution Status</td>
                  ${selectedExps.map(e => `
                    <td><span class="${e.status === 'COMPLETED' ? 'badge-gain' : 'badge-sys'} text-[10px] px-2 py-0.5 rounded font-mono font-medium">${e.status}</span></td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-[#8B95A7]">Hypothesis / Rationale</td>
                  ${selectedExps.map(e => `
                    <td class="text-xs text-[#8B95A7] max-w-xs truncate font-sans">${e.hypothesis || "Baseline model training"}</td>
                  `).join("")}
                </tr>
                <tr>
                  <td class="font-medium text-[#8B95A7]">Artifact Export</td>
                  ${selectedExps.map(e => `
                    <td>
                      <a href="/api/models/export?experiment_id=${e.id}" class="btn-signal text-[11px] py-1 px-2.5 inline-flex items-center space-x-1" title="Download autonomous artifact (.pkl)">
                        ${icon("download", "icon-sm")}
                        <span>Download .pkl</span>
                      </a>
                    </td>
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
              <div class="flex items-center gap-2">
                <span class="text-[#8B95A7]">${icon("chart-line", "icon-sm")}</span>
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Evolution Progression</span>
              </div>
              <span class="text-xs font-mono text-[#22C55E]">${metricName}</span>
            </div>
            <div class="p-5 flex-1 flex flex-col justify-between">
              <div class="h-64 w-full relative">
                <canvas id="compareEvolutionChart"></canvas>
              </div>
              <div class="mt-3 flex justify-between text-xs text-[#8B95A7] pt-2 border-t border-[#242A36]">
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
                <span class="text-[#8B95A7]">${icon("scan-search", "icon-sm")}</span>
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Differences Inspector</span>
              </div>
              ${selectedExps.length >= 2 && selectedExps[0].best_score != null && selectedExps[1].best_score != null ? `
                <span class="badge-gain text-xs px-2 py-0.5 rounded font-mono font-medium">
                  Δ: ${(selectedExps[1].best_score - selectedExps[0].best_score >= 0 ? '+' : '') + (selectedExps[1].best_score - selectedExps[0].best_score).toFixed(5)}
                </span>
              ` : ''}
            </div>
            <div class="p-5 space-y-4 flex-1 text-xs">
              ${selectedExps.length >= 2 ? `
                <div class="space-y-1.5">
                  <div class="text-[#8B95A7] uppercase font-sans font-medium tracking-wider text-[10px]">Model Architecture Differences</div>
                  <div class="bg-[#161B26] p-3 rounded-xl border border-[#242A36] font-mono space-y-1 text-[#F7F8FA]">
                    <div>Exp 1: <span class="text-[#4F67FF] font-semibold">${selectedExps[0].name}</span> (${(selectedExps[0].model_ids || []).join(", ")})</div>
                    <div>Exp 2: <span class="text-[#6956E8] font-semibold">${selectedExps[1].name}</span> (${(selectedExps[1].model_ids || []).join(", ")})</div>
                  </div>
                </div>

                <div class="space-y-1.5">
                  <div class="text-[#8B95A7] uppercase font-sans font-medium tracking-wider text-[10px]">Hypothesis Evaluation</div>
                  <div class="bg-[#161B26] p-3 rounded-xl border border-[#242A36] font-sans space-y-1 text-[#8B95A7]">
                    <div>${selectedExps[1].hypothesis || "Optuna Bayesian hyperparameter search exploration"}</div>
                  </div>
                </div>
              ` : `
                <div class="text-center text-[#8B95A7]/60 py-12 font-sans">Select at least 2 experiments above to inspect differences.</div>
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

    this.container.querySelector("#btnOpenEnsembleModal")?.addEventListener("click", () => {
      const selectedExps = this.experiments.filter(e => this.selected.includes(e.id));
      this._openEnsembleModal(selectedExps);
    });
  }

  _openEnsembleModal(selectedExps) {
    const candidateModels = [];
    selectedExps.forEach(e => {
      (e.model_ids || []).forEach(m => {
        if (!candidateModels.includes(m) && m !== "voting_ensemble" && m !== "oof_blend") candidateModels.push(m);
      });
    });
    if (candidateModels.length < 2) {
      ["lightgbm", "xgboost", "catboost"].forEach(m => {
        if (!candidateModels.includes(m)) candidateModels.push(m);
      });
    }

    let selectedMethod = "average";
    let selectedMeta = "ridge";
    let selectedFolds = 5;
    let selectedModels = [...candidateModels];

    const modalId = "ensembleBuilderModal";
    let modalEl = document.getElementById(modalId);
    if (!modalEl) {
      modalEl = document.createElement("div");
      modalEl.id = modalId;
      document.body.appendChild(modalEl);
    }

    const renderModal = () => {
      modalEl.innerHTML = `
        <div class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div class="workbench-card max-w-xl w-full p-6 space-y-5 border-slate-700 shadow-2xl">
            <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
              <div>
                <h3 class="text-base font-bold text-[#F7F8FA] font-sans">Visual Ensemble Builder</h3>
                <p class="text-xs text-[#8B95A7] font-sans">Multi-model blending and Level-2 meta-learning</p>
              </div>
              <button id="btnCloseEnsembleModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1 rounded hover:bg-[#161B26] transition-colors">${icon("x", "icon-sm")}</button>
            </div>

            <!-- Base Models Selection -->
            <div class="space-y-2">
              <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">Include Base Models</label>
              <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-mono">
                ${candidateModels.map(m => `
                  <label class="flex items-center space-x-2 p-2 rounded-lg border cursor-pointer ${selectedModels.includes(m) ? 'border-[#4F67FF] bg-[#161B26] text-[#F7F8FA]' : 'border-[#242A36] bg-[#0F131C] text-[#8B95A7]'}">
                    <input type="checkbox" ${selectedModels.includes(m) ? "checked" : ""} class="ens-model-chk rounded text-[#4F67FF] bg-[#090C12] border-[#242A36]" data-model="${m}">
                    <span>${m}</span>
                  </label>
                `).join("")}
              </div>
            </div>

            <!-- Ensemble Method Selection -->
            <div class="space-y-2">
              <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">Ensemble Strategy</label>
              <div class="grid grid-cols-2 sm:grid-cols-4 gap-2">
                <div class="ens-strat-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${selectedMethod === 'average' ? 'border-[#4F67FF] bg-[#161B26]' : 'border-[#242A36] bg-[#090C12]'}" data-method="average">
                  <div class="font-bold text-[#F7F8FA]">Average</div>
                  <div class="text-[10px] text-[#8B95A7]">Soft voting</div>
                </div>

                <div class="ens-strat-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${selectedMethod === 'rank' ? 'border-[#4F67FF] bg-[#161B26]' : 'border-[#242A36] bg-[#090C12]'}" data-method="rank">
                  <div class="font-bold text-[#F7F8FA]">Rank Avg</div>
                  <div class="text-[10px] text-[#8B95A7]">ROC-AUC boost</div>
                </div>

                <div class="ens-strat-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${selectedMethod === 'simplex' ? 'border-[#4F67FF] bg-[#161B26]' : 'border-[#242A36] bg-[#090C12]'}" data-method="simplex">
                  <div class="font-bold text-[#F7F8FA]">Simplex</div>
                  <div class="text-[10px] text-[#8B95A7]">Nelder-Mead HPO</div>
                </div>

                <div class="ens-strat-card p-2.5 rounded-xl border text-center cursor-pointer text-xs ${selectedMethod === 'stacked' ? 'border-[#4F67FF] bg-[#161B26]' : 'border-[#242A36] bg-[#090C12]'}" data-method="stacked">
                  <div class="font-bold text-[#F7F8FA]">Stacked L2</div>
                  <div class="text-[10px] text-[#8B95A7]">Meta-learner</div>
                </div>
              </div>
            </div>

            <!-- Meta-Learner (only visible if stacked) -->
            ${selectedMethod === "stacked" ? `
              <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-2 text-xs font-sans">
                <div class="font-semibold text-[#F7F8FA]">Level-2 Meta-Learner Architecture:</div>
                <div class="grid grid-cols-3 gap-2">
                  <label class="p-2 rounded-lg border cursor-pointer text-center ${selectedMeta === 'ridge' ? 'border-[#4F67FF] bg-[#1D2433] text-[#F7F8FA]' : 'border-[#252C38] text-[#8B95A7]'}">
                    <input type="radio" name="metaModel" value="ridge" ${selectedMeta === 'ridge' ? 'checked' : ''} class="hidden">
                    <div class="font-bold">Ridge</div>
                    <div class="text-[10px] text-[#8B95A7]">L2 Regularized</div>
                  </label>
                  <label class="p-2 rounded-lg border cursor-pointer text-center ${selectedMeta === 'logistic_regression' ? 'border-[#4F67FF] bg-[#1D2433] text-[#F7F8FA]' : 'border-[#252C38] text-[#8B95A7]'}">
                    <input type="radio" name="metaModel" value="logistic_regression" ${selectedMeta === 'logistic_regression' ? 'checked' : ''} class="hidden">
                    <div class="font-bold">Logistic Reg</div>
                    <div class="text-[10px] text-[#8B95A7]">Sigmoid Calibrated</div>
                  </label>
                  <label class="p-2 rounded-lg border cursor-pointer text-center ${selectedMeta === 'lasso' ? 'border-[#4F67FF] bg-[#1D2433] text-[#F7F8FA]' : 'border-[#252C38] text-[#8B95A7]'}">
                    <input type="radio" name="metaModel" value="lasso" ${selectedMeta === 'lasso' ? 'checked' : ''} class="hidden">
                    <div class="font-bold">Lasso</div>
                    <div class="text-[10px] text-[#8B95A7]">Sparse Selector</div>
                  </label>
                </div>
              </div>
            ` : ""}

            <!-- Validation Folds -->
            <div class="space-y-2">
              <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">Validation Folds</label>
              <div class="grid grid-cols-3 gap-2">
                ${[3, 5, 10].map(k => `
                  <div class="ens-fold-card p-2 rounded-lg border text-center cursor-pointer text-xs ${selectedFolds === k ? 'border-[#4F67FF] bg-[#151B26] text-[#F7F8FA]' : 'border-[#252C38] bg-[#090C12] text-[#8B95A7]'}" data-folds="${k}">
                    <span class="font-bold">${k} Folds</span>
                  </div>
                `).join("")}
              </div>
            </div>

            <!-- Custom Name -->
            <div class="space-y-1">
              <label class="text-xs font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">Ensemble Experiment Name</label>
              <input type="text" id="ensCustomName" value="Ensemble (${selectedMethod.toUpperCase()} - ${selectedModels.length} models)" class="w-full bg-[#090C12] border border-[#252C38] rounded-lg p-2 text-xs font-mono text-[#F7F8FA] focus:border-[#4F67FF]">
            </div>

            <!-- Footer Actions -->
            <div class="pt-4 border-t border-[#242A36] flex items-center justify-end space-x-3">
              <button id="btnCancelEnsemble" class="btn-ghost">CANCEL</button>
              <button id="btnSubmitEnsemble" ${selectedModels.length < 2 ? 'disabled' : ''} class="btn-signal ${selectedModels.length < 2 ? 'opacity-50 cursor-not-allowed' : ''}">
                <span class="inline-flex items-center gap-1.5">${icon("sparkles", "icon-sm")} <span>BUILD & EVALUATE ENSEMBLE</span></span>
              </button>
            </div>
          </div>
        </div>
      `;

      // Event bindings
      modalEl.querySelector("#btnCloseEnsembleModal")?.addEventListener("click", () => modalEl.remove());
      modalEl.querySelector("#btnCancelEnsemble")?.addEventListener("click", () => modalEl.remove());

      modalEl.querySelectorAll(".ens-model-chk").forEach(chk => {
        chk.addEventListener("change", e => {
          const m = chk.getAttribute("data-model");
          if (e.target.checked) {
            if (!selectedModels.includes(m)) selectedModels.push(m);
          } else {
            selectedModels = selectedModels.filter(x => x !== m);
          }
          renderModal();
        });
      });

      modalEl.querySelectorAll(".ens-strat-card").forEach(c => {
        c.addEventListener("click", () => {
          selectedMethod = c.getAttribute("data-method");
          renderModal();
        });
      });

      modalEl.querySelectorAll("input[name='metaModel']").forEach(r => {
        r.addEventListener("change", () => {
          selectedMeta = r.value;
        });
      });

      modalEl.querySelectorAll(".ens-fold-card").forEach(f => {
        f.addEventListener("click", () => {
          selectedFolds = parseInt(f.getAttribute("data-folds"), 10);
          renderModal();
        });
      });

      modalEl.querySelector("#btnSubmitEnsemble")?.addEventListener("click", async () => {
        const btn = modalEl.querySelector("#btnSubmitEnsemble");
        const customName = modalEl.querySelector("#ensCustomName")?.value.trim();
        if (!btn || selectedModels.length < 2) return;

        btn.disabled = true;
        btn.innerHTML = `
          <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" viewBox="0 0 24 24" fill="none">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span>Training Ensemble...</span>
        `;

        try {
          const res = await api.buildEnsemble({
            run_id: this.activeRun.id,
            models: selectedModels,
            method: selectedMethod,
            meta_model: selectedMeta,
            folds: selectedFolds,
            name: customName || undefined,
          });

          btn.innerHTML = `<span class="inline-flex items-center gap-1.5">${icon("check", "icon-sm")} <span>Score: ${Number(res.score).toFixed(5)}</span></span>`;
          bus.emit("run:updated");

          setTimeout(async () => {
            modalEl.remove();
            await this.fetchData();
            if (res.experiment_id && !this.selected.includes(res.experiment_id)) {
              this.selected.push(res.experiment_id);
            }
            this.render();
          }, 800);
        } catch (err) {
          alert("Error building ensemble: " + err.message);
          btn.disabled = false;
          btn.innerHTML = `<span class="inline-flex items-center gap-1.5">${icon("sparkles", "icon-sm")} <span>BUILD & EVALUATE ENSEMBLE</span></span>`;
        }
      });
    };

    renderModal();
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
            borderColor: "#4F67FF",
            backgroundColor: "rgba(79, 103, 255, 0.12)",
            pointBackgroundColor: ["#8B95A7", "#4F67FF", "#6956E8", "#22C55E"],
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
            grid: { color: "rgba(36, 42, 54, 0.4)" },
            ticks: { color: "#8B95A7", font: { size: 10 } },
          },
          y: {
            min: Math.max(0, minScore - pad),
            max: Math.min(1.0, maxScore + pad),
            grid: { color: "rgba(36, 42, 54, 0.4)" },
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
