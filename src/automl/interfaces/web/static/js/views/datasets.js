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
      const datasetId = state.activeDatasetId || "ds_s6e9";
      this.profile = await api.getDatasetProfile(datasetId).catch(() => null);
    } catch (_) {}
  }

  render() {
    const p = this.profile || {
      dataset_id: "ds_ev_purchases",
      task_type: "binary_classification",
      target_column: "Will_Buy_EV",
      row_count: 668665,
      column_count: 14,
      columns: [
        { name: "id", dtype: "int64", missing_ratio: 0.0, unique_count: 668665, is_identifier: true, catml_action: "Exclude", action_reason: "ID candidate (99.99% unique)" },
        { name: "Age", dtype: "float64", missing_ratio: 0.0, unique_count: 72, is_identifier: false, catml_action: "Keep", action_reason: "Numerical predictive feature" },
        { name: "Income", dtype: "float64", missing_ratio: 0.003, unique_count: 512, is_identifier: false, catml_action: "Impute & Keep", action_reason: "Missing 0.3%; median imputation" },
        { name: "Region", dtype: "object", missing_ratio: 0.0, unique_count: 12, is_identifier: false, catml_action: "Encode", action_reason: "Categorical (Target / Ordinal encoding)" },
        { name: "Vehicle_Type", dtype: "object", missing_ratio: 0.0, unique_count: 7, is_identifier: false, catml_action: "Encode", action_reason: "Categorical (Target / Ordinal encoding)" },
        { name: "Household_Size", dtype: "int64", missing_ratio: 0.0, unique_count: 8, is_identifier: false, catml_action: "Keep", action_reason: "Numerical predictive feature" },
        { name: "Commute_Distance", dtype: "float64", missing_ratio: 0.0, unique_count: 450, is_identifier: false, catml_action: "Keep", action_reason: "Numerical predictive feature" },
        { name: "Home_Ownership", dtype: "object", missing_ratio: 0.0, unique_count: 3, is_identifier: false, catml_action: "Encode", action_reason: "Categorical encoding" },
        { name: "Current_Vehicles", dtype: "int64", missing_ratio: 0.0, unique_count: 5, is_identifier: false, catml_action: "Keep", action_reason: "Numerical predictive feature" },
        { name: "Charging_Access", dtype: "object", missing_ratio: 0.0, unique_count: 2, is_identifier: false, catml_action: "Encode", action_reason: "Binary flag encoding" },
        { name: "Tax_Incentive_Eligible", dtype: "object", missing_ratio: 0.0, unique_count: 2, is_identifier: false, catml_action: "Encode", action_reason: "Binary flag encoding" },
        { name: "Fuel_Cost_Monthly", dtype: "float64", missing_ratio: 0.0, unique_count: 320, is_identifier: false, catml_action: "Keep", action_reason: "Numerical predictive feature" },
        { name: "Public_Transit_Access", dtype: "object", missing_ratio: 0.0, unique_count: 3, is_identifier: false, catml_action: "Encode", action_reason: "Categorical encoding" },
        { name: "Will_Buy_EV", dtype: "object", missing_ratio: 0.0, unique_count: 2, is_identifier: false, catml_action: "Target", action_reason: "Target label; TargetAdapter ['No','Yes']->[0,1]" },
      ],
    };

    const numCols = p.columns.filter(c => c.dtype.includes("int") || c.dtype.includes("float")).length;
    const catCols = p.columns.filter(c => c.dtype === "object" || c.dtype === "category").length;
    const excludedCount = p.columns.filter(c => c.is_identifier).length;

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Architecture Concept Banner -->
        <div class="workbench-card p-4 bg-indigo-950/20 border-indigo-900/60">
          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-3 text-xs">
              <span class="text-slate-400 font-semibold uppercase">Flujo Conceptual:</span>
              <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 font-mono">1. Raw Dataset</span>
              <span class="text-indigo-400 font-bold">➔</span>
              <span class="px-2 py-1 rounded bg-purple-950/50 border border-purple-800 text-purple-300 font-mono font-bold">2. CATML Understanding</span>
              <span class="text-indigo-400 font-bold">➔</span>
              <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 font-mono">3. Experiment Plan</span>
            </div>
            <button id="btnNewExperimentFromDS" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3.5 py-1.5 rounded-lg font-semibold transition-colors shadow-lg shadow-indigo-600/20">
              + Create Experiment
            </button>
          </div>
        </div>

        <!-- 1. CATML Understanding & Detection Header -->
        <div class="workbench-card p-6 space-y-4">
          <div class="flex items-center justify-between border-b border-slate-800 pb-4">
            <div class="flex items-center space-x-3">
              <span class="text-purple-400 text-xl font-bold">🟣</span>
              <div>
                <h3 class="text-base font-bold text-slate-100 uppercase tracking-wide">CATML Interpretation</h3>
                <p class="text-xs text-slate-400">Diagnóstico estadístico y semántico autónomo del problema</p>
              </div>
            </div>
            <span class="badge-gain text-xs px-3 py-1 rounded-full font-mono font-bold">Validated</span>
          </div>

          <!-- Diagnostic Metrics Grid -->
          <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 text-center">
            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Task</div>
              <div class="text-xs font-bold text-slate-100 mt-1 truncate">Binary Clf</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Target</div>
              <div class="text-xs font-bold text-indigo-400 font-mono mt-1 truncate">${p.target_column}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Metric</div>
              <div class="text-xs font-bold text-emerald-400 font-mono mt-1">ROC-AUC</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Rows</div>
              <div class="text-xs font-bold text-slate-100 font-mono-num mt-1">${p.row_count.toLocaleString()}</div>
            </div>

            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Features</div>
              <div class="text-xs font-bold text-slate-100 font-mono-num mt-1">${p.columns.length - 1}</div>
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
              <div class="text-xs font-bold text-rose-400 font-mono-num mt-1">${excludedCount} (id)</div>
            </div>
          </div>
        </div>

        <!-- 2. Schema Table with "Acción CATML" -->
        <div class="workbench-card overflow-hidden">
          <div class="workbench-panel-header flex items-center justify-between">
            <span class="text-sm font-semibold text-slate-200">Schema Diagnostics & Action Registry</span>
            <span class="text-xs text-slate-400">Total ${p.columns.length} columns inspected</span>
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
                ${p.columns.map(c => `
                  <tr>
                    <td class="font-mono font-medium text-slate-100">${c.name}</td>
                    <td>
                      <span class="text-xs ${c.dtype.includes('float') || c.dtype.includes('int') ? 'text-indigo-300' : 'text-purple-300'} font-mono">
                        ${c.is_identifier ? 'Identifier' : c.dtype.includes('float') || c.dtype.includes('int') ? 'Numerical' : 'Categorical'}
                      </span>
                    </td>
                    <td class="font-mono text-xs ${c.missing_ratio > 0 ? 'text-amber-400 font-bold' : 'text-slate-400'}">
                      ${(c.missing_ratio * 100).toFixed(1)}%
                    </td>
                    <td class="font-mono text-xs text-slate-300">
                      ${c.unique_count ? c.unique_count.toLocaleString() : '—'}
                    </td>
                    <td>
                      ${
                        c.catml_action === "Exclude"
                          ? `<span class="badge-err text-[11px] px-2 py-0.5 rounded font-mono font-bold">Exclude</span>`
                          : c.catml_action === "Encode"
                          ? `<span class="badge-intel text-[11px] px-2 py-0.5 rounded font-mono font-bold">Encode</span>`
                          : c.catml_action === "Target"
                          ? `<span class="badge-sys text-[11px] px-2 py-0.5 rounded font-mono font-bold">Target</span>`
                          : `<span class="badge-gain text-[11px] px-2 py-0.5 rounded font-mono font-bold">Keep</span>`
                      }
                    </td>
                    <td class="text-xs text-slate-400">${c.action_reason || "Predictive feature"}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    this.container.querySelector("#btnNewExperimentFromDS")?.addEventListener("click", () => {
      bus.emit("modal:new-experiment");
    });
  }

  destroy() {
    this.container = null;
  }
}
