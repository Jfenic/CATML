/**
 * DatasetsView — Dataset Inspector: "¿Qué entendió CATML?"
 * Raw Dataset -> CATML Understanding -> Experiment Plan
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";

export class DatasetsView {
  constructor() {
    this.container = null;
    this.profile = null;
    this.datasets = [];
    this.activeDatasetId = null;
    this.showRegisterForm = false;
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
        <div class="text-sm font-medium">Analizando perfil del dataset e inferencias de CATML...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const state = store.getState();
      this.datasets = await api.getDatasets().catch(() => []);

      this.activeDatasetId = state.activeDatasetId
        || (this.datasets.length ? this.datasets[0].id : null)
        || (state.runs && state.runs[0] ? state.runs[0].dataset_id : null)
        || (state.overview && state.overview.recent_datasets && state.overview.recent_datasets[0] ? state.overview.recent_datasets[0].id : null);

      if (this.activeDatasetId) {
        this.profile = await api.getDatasetProfile(this.activeDatasetId).catch(() => null);
      }
    } catch (e) {
      console.warn("Could not load dataset profile:", e);
    }
  }

  render() {
    if (!this.profile && this.datasets.length === 0) {
      this._renderEmptyOrRegisterState();
      return;
    }

    const p = this.profile;
    if (!p) {
      this._renderEmptyOrRegisterState();
      return;
    }

    const columns = p.columns || [];
    const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float"))).length;
    const catCols = columns.filter(c => c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string")).length;
    const excludedCount = columns.filter(c => c.is_identifier || c.catml_action === "Exclude").length;
    const currentDs = this.datasets.find(d => d.id === this.activeDatasetId);
    const datasetName = currentDs ? currentDs.name : (p.name || p.dataset_id);

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Architecture Concept Banner & Dataset Switcher -->
        <div class="workbench-card p-4 bg-indigo-950/20 border-indigo-900/60">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div class="flex items-center space-x-3 text-xs">
              <span class="text-slate-400 font-semibold uppercase">Flujo Conceptual:</span>
              <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 font-mono">1. Raw Dataset</span>
              <span class="text-indigo-400 font-bold">➔</span>
              <span class="px-2 py-1 rounded bg-purple-950/50 border border-purple-800 text-purple-300 font-mono font-bold">2. CATML Understanding</span>
              <span class="text-indigo-400 font-bold">➔</span>
              <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 font-mono">3. Experiment Plan</span>
            </div>

            <div class="flex items-center space-x-3">
              ${this.datasets.length > 1 ? `
                <select id="datasetSelect" class="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:border-indigo-500 font-mono">
                  ${this.datasets.map(d => `
                    <option value="${d.id}" ${d.id === this.activeDatasetId ? "selected" : ""}>${d.name}</option>
                  `).join("")}
                </select>
              ` : `
                <span class="font-mono text-xs text-indigo-300 font-semibold bg-slate-900 px-3 py-1 rounded-lg border border-slate-800">${datasetName}</span>
              `}

              <button id="btnToggleRegisterForm" class="bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs px-3 py-1.5 rounded-lg border border-slate-700 transition-colors">
                + Register New
              </button>

              <button id="btnNewExperimentFromDS" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3.5 py-1.5 rounded-lg font-semibold transition-colors shadow-lg shadow-indigo-600/20">
                + Create Experiment
              </button>
            </div>
          </div>
        </div>

        ${this.showRegisterForm ? this._getRegisterFormHtml() : ""}

        <!-- 1. CATML Understanding & Detection Header -->
        <div class="workbench-card p-6 space-y-4">
          <div class="flex items-center justify-between border-b border-slate-800 pb-4">
            <div class="flex items-center space-x-3">
              <span class="text-purple-400 text-xl font-bold">🟣</span>
              <div>
                <h3 class="text-base font-bold text-slate-100 uppercase tracking-wide">CATML Interpretation — ${datasetName}</h3>
                <p class="text-xs text-slate-400">Diagnóstico estadístico y semántico autónomo del problema tabular</p>
              </div>
            </div>
            <span class="badge-gain text-xs px-3 py-1 rounded-full font-mono font-bold">Profiled</span>
          </div>

          <!-- Diagnostic Metrics Grid -->
          <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 text-center">
            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Task</div>
              <div class="text-xs font-bold text-slate-100 mt-1 truncate capitalize">${(p.task_type || "Classification").replace("_", " ")}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Target</div>
              <div class="text-xs font-bold text-indigo-400 font-mono mt-1 truncate">${p.target_column || "—"}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Metric</div>
              <div class="text-xs font-bold text-emerald-400 font-mono mt-1">${(p.task_type || "").includes("regression") ? "RMSE" : "ROC-AUC"}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Rows</div>
              <div class="text-xs font-bold text-slate-100 font-mono-num mt-1">${p.row_count != null ? Number(p.row_count).toLocaleString() : "—"}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Features</div>
              <div class="text-xs font-bold text-slate-100 font-mono-num mt-1">${columns.length > 0 ? (columns.filter(c => c.name !== p.target_column).length) : 0}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Numerical</div>
              <div class="text-xs font-bold text-slate-200 font-mono-num mt-1">${numCols}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Categorical</div>
              <div class="text-xs font-bold text-slate-200 font-mono-num mt-1">${catCols}</div>
            </div>

            <div class="p-3 rounded-lg bg-rose-950/20 border border-rose-800/40">
              <div class="text-[10px] uppercase text-rose-300 font-semibold tracking-wider">Excluded</div>
              <div class="text-xs font-bold text-rose-400 font-mono-num mt-1">${excludedCount}</div>
            </div>
          </div>
        </div>

        <!-- 2. Schema Table with "Acción CATML" -->
        <div class="workbench-card overflow-hidden">
          <div class="workbench-panel-header flex items-center justify-between">
            <span class="text-sm font-semibold text-slate-200">Schema Diagnostics & Action Registry</span>
            <span class="text-xs text-slate-400">Total ${columns.length} columns inspected</span>
          </div>

          <div class="overflow-x-auto">
            <table class="w-full wb-table text-left">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Detectado como</th>
                  <th>Missing</th>
                  <th>Cardinalidad</th>
                  <th>Acción CATML</th>
                  <th>Justificación del Motor</th>
                </tr>
              </thead>
              <tbody>
                ${columns.map(c => `
                  <tr>
                    <td class="font-mono font-medium text-slate-100">${c.name}</td>
                    <td>
                      <span class="text-xs ${c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'text-indigo-300' : 'text-purple-300'} font-mono">
                        ${c.is_identifier ? 'Identifier' : (c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'Numerical' : 'Categorical')}
                      </span>
                    </td>
                    <td class="font-mono text-xs ${c.missing_ratio > 0 ? 'text-amber-400 font-bold' : 'text-slate-400'}">
                      ${c.missing_ratio != null ? (c.missing_ratio * 100).toFixed(1) + '%' : '0.0%'}
                    </td>
                    <td class="font-mono text-xs text-slate-300">
                      ${c.unique_count != null ? Number(c.unique_count).toLocaleString() : '—'}
                    </td>
                    <td>
                      ${
                        c.name === p.target_column || c.catml_action === "Target"
                          ? `<span class="badge-sys text-[11px] px-2 py-0.5 rounded font-mono font-bold">Target</span>`
                          : c.catml_action === "Exclude" || c.is_identifier
                          ? `<span class="badge-err text-[11px] px-2 py-0.5 rounded font-mono font-bold">Exclude</span>`
                          : c.catml_action === "Encode"
                          ? `<span class="badge-intel text-[11px] px-2 py-0.5 rounded font-mono font-bold">Encode</span>`
                          : `<span class="badge-gain text-[11px] px-2 py-0.5 rounded font-mono font-bold">Keep</span>`
                      }
                    </td>
                    <td class="text-xs text-slate-400">${c.action_reason || (c.name === p.target_column ? "Target variable for optimization" : "Predictive feature")}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _renderEmptyOrRegisterState() {
    this.container.innerHTML = `
      <div class="space-y-6">
        <div class="workbench-card p-8 text-center space-y-4">
          <span class="text-4xl text-slate-600 block">▦</span>
          <div>
            <h3 class="text-base font-bold text-slate-200">No Profiled Datasets Found</h3>
            <p class="text-xs text-slate-400 mt-1">Register a tabular CSV or Parquet dataset to generate autonomous statistical profiles and candidate pipelines.</p>
          </div>
        </div>
        ${this._getRegisterFormHtml()}
      </div>
    `;
    this._bindRegisterFormEvents();
  }

  _getRegisterFormHtml() {
    return `
      <div class="workbench-card p-6 space-y-4 border-indigo-900/60 bg-indigo-950/20" id="registerFormCard">
        <div class="flex items-center justify-between border-b border-slate-800 pb-3">
          <div class="flex items-center space-x-2">
            <span class="text-indigo-400 font-bold">＋</span>
            <span class="text-sm font-semibold text-slate-200">Register New Dataset</span>
          </div>
          ${this.profile ? `<button id="btnCloseRegisterForm" class="text-slate-400 hover:text-slate-200 text-xs">✕ Close</button>` : ""}
        </div>

        <form id="formRegisterDataset" class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div>
            <label class="block text-slate-300 font-medium mb-1">Dataset Name *</label>
            <input type="text" id="regName" required placeholder="e.g. My Tabular Dataset" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500">
          </div>

          <div>
            <label class="block text-slate-300 font-medium mb-1">File Path (CSV or Parquet) *</label>
            <input type="text" id="regPath" required placeholder="e.g. data/train.csv" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500 font-mono">
          </div>

          <div>
            <label class="block text-slate-300 font-medium mb-1">Target Column *</label>
            <input type="text" id="regTarget" required placeholder="e.g. target" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500 font-mono">
          </div>

          <div>
            <label class="block text-slate-300 font-medium mb-1">Task Type</label>
            <select id="regTaskType" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-2 text-slate-100 focus:border-indigo-500">
              <option value="binary_classification">Binary Classification</option>
              <option value="multiclass_classification">Multiclass Classification</option>
              <option value="regression">Regression</option>
            </select>
          </div>

          <div class="md:col-span-2 flex justify-end pt-2">
            <button type="submit" id="btnSubmitRegister" class="bg-indigo-600 hover:bg-indigo-500 text-white font-semibold px-4 py-2 rounded-lg transition-colors shadow-lg shadow-indigo-600/20">
              Register & Profile Dataset
            </button>
          </div>
        </form>
      </div>
    `;
  }

  _bindEvents() {
    this.container.querySelector("#btnNewExperimentFromDS")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnToggleRegisterForm")?.addEventListener("click", () => {
      this.showRegisterForm = !this.showRegisterForm;
      this.render();
    });

    this.container.querySelector("#datasetSelect")?.addEventListener("change", async e => {
      const newId = e.target.value;
      if (newId && newId !== this.activeDatasetId) {
        this.activeDatasetId = newId;
        store.setState({ activeDatasetId: newId });
        this.renderLoading();
        this.profile = await api.getDatasetProfile(newId).catch(() => null);
        this.render();
      }
    });

    this._bindRegisterFormEvents();
  }

  _bindRegisterFormEvents() {
    this.container.querySelector("#btnCloseRegisterForm")?.addEventListener("click", () => {
      this.showRegisterForm = false;
      this.render();
    });

    this.container.querySelector("#formRegisterDataset")?.addEventListener("submit", async e => {
      e.preventDefault();
      const btn = this.container.querySelector("#btnSubmitRegister");
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Profiling...";
      }

      try {
        const payload = {
          name: this.container.querySelector("#regName").value.trim(),
          path: this.container.querySelector("#regPath").value.trim(),
          target_column: this.container.querySelector("#regTarget").value.trim(),
          task_type: this.container.querySelector("#regTaskType").value,
        };

        const res = await api.registerDataset(payload);
        this.showRegisterForm = false;
        store.setState({ activeDatasetId: res.dataset_id });
        this.renderLoading();
        await this.fetchData();
        this.render();
      } catch (err) {
        alert("Failed to register dataset: " + err.message);
        if (btn) {
          btn.disabled = false;
          btn.textContent = "Register & Profile Dataset";
        }
      }
    });
  }

  destroy() {
    this.container = null;
  }
}
