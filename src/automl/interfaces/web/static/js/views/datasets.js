/**
 * DatasetsView — Dataset Inspector & Interactive Feature Selection
 * Diagnóstico estadístico sólido, exploración de variables, recomendaciones inteligentes,
 * y selección de columnas tanto manual como automática para experimentos.
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
    this.selectedFeatures = new Set();
    this.activeTab = "schema"; // 'schema' | 'stats' | 'preview' | 'categories'
    this.filterType = "all"; // 'all' | 'numeric' | 'categorical' | 'selected' | 'excluded'
    this.searchQuery = "";
  }

  async mount(container) {
    this.container = container;
    this.renderLoading();
    await this.fetchData();
    this._initSelectedFeatures();
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

  _initSelectedFeatures() {
    if (!this.profile) return;
    const target = this.profile.target_column;
    const columns = this.profile.columns || [];

    // Initialize with recommended features (non-identifier, non-target)
    this.selectedFeatures.clear();
    columns.forEach(col => {
      if (col.name !== target && !col.is_identifier && col.catml_action !== "Exclude") {
        this.selectedFeatures.add(col.name);
      }
    });

    // Save to store so NewExperimentModal immediately knows active selection
    store.setState({ customFeatures: Array.from(this.selectedFeatures) });
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
    const featureCols = columns.filter(c => c.name !== p.target_column);
    const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float")) && c.name !== p.target_column).length;
    const catCols = columns.filter(c => c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string") && c.name !== p.target_column).length;
    const excludedCount = columns.filter(c => c.is_identifier || c.catml_action === "Exclude").length;
    const currentDs = this.datasets.find(d => d.id === this.activeDatasetId);
    const datasetName = currentDs ? currentDs.name : (p.name || p.dataset_id);

    const recommendations = p.recommendations || [];

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
              <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300 font-mono">3. Feature Selection & Plan</span>
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

              <button id="btnNewExperimentFromDS" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-3.5 py-1.5 rounded-lg font-semibold transition-colors shadow-lg shadow-indigo-600/20 flex items-center space-x-1.5">
                <span>⚡ Create Experiment</span>
              </button>
            </div>
          </div>
        </div>

        ${this.showRegisterForm ? this._getRegisterFormHtml() : ""}

        <!-- Diagnostic Metrics Grid -->
        <div class="workbench-card p-6 space-y-4">
          <div class="flex items-center justify-between border-b border-slate-800 pb-4">
            <div class="flex items-center space-x-3">
              <span class="text-purple-400 text-xl font-bold">🟣</span>
              <div>
                <h3 class="text-base font-bold text-slate-100 uppercase tracking-wide">CATML Interpretation — ${datasetName}</h3>
                <p class="text-xs text-slate-400">Diagnóstico estadístico, roles de columnas y selección guiada por evidencias</p>
              </div>
            </div>
            <span class="badge-gain text-xs px-3 py-1 rounded-full font-mono font-bold">Profiled</span>
          </div>

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
              <div class="text-xs font-bold text-slate-100 font-mono-num mt-1">${featureCols.length}</div>
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

        <!-- 2. CATML Smart Recommendations Panel -->
        ${recommendations.length > 0 ? `
          <div class="workbench-card p-5 space-y-3 border-indigo-900/40 bg-gradient-to-r from-slate-900/80 to-indigo-950/30">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
              <div class="flex items-center space-x-2">
                <span class="text-amber-400">💡</span>
                <span class="text-sm font-bold text-slate-200 uppercase tracking-wide">Recomendaciones del Motor Estadístico</span>
                <span class="text-xs font-mono text-indigo-300 bg-slate-800 px-2 py-0.5 rounded">${recommendations.length} detectadas</span>
              </div>
              <button id="btnApplyRecommendations" class="bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/50 text-xs px-3.5 py-1.5 rounded-lg font-semibold transition-all flex items-center space-x-1.5">
                <span>⚡ Aplicar Recomendaciones Automáticas</span>
              </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 pt-1">
              ${recommendations.slice(0, 6).map(r => `
                <div class="p-3 rounded-lg bg-slate-900/90 border ${r.severity === 'danger' ? 'border-rose-800/60 text-rose-300' : r.severity === 'warning' ? 'border-amber-800/60 text-amber-300' : r.severity === 'success' ? 'border-emerald-800/60 text-emerald-300' : 'border-indigo-800/60 text-indigo-300'} space-y-1 text-xs">
                  <div class="flex items-center justify-between">
                    <span class="font-bold text-slate-200">${r.title}</span>
                    <span class="text-[10px] px-1.5 py-0.5 rounded font-mono uppercase font-bold ${r.severity === 'danger' ? 'bg-rose-950 text-rose-400' : r.severity === 'warning' ? 'bg-amber-950 text-amber-400' : r.severity === 'success' ? 'bg-emerald-950 text-emerald-400' : 'bg-indigo-950 text-indigo-400'}">${r.badge}</span>
                  </div>
                  <p class="text-[11px] text-slate-400">${r.description}</p>
                </div>
              `).join("")}
            </div>
          </div>
        ` : ""}

        <!-- 3. Interactive Selection & Exploration Toolbar -->
        <div class="workbench-card p-4 space-y-3">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <!-- Search & Type Filter Pills -->
            <div class="flex flex-wrap items-center gap-2">
              <input type="text" id="featureSearchInput" value="${this.searchQuery}" placeholder="Filtrar características por nombre..." class="bg-slate-900 border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 focus:border-indigo-500 w-52 font-mono">

              <div class="flex items-center space-x-1 text-xs">
                <button data-filter="all" class="filter-pill px-2.5 py-1 rounded-md border text-[11px] font-medium transition-colors ${this.filterType === 'all' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'}">Todas (${featureCols.length})</button>
                <button data-filter="numeric" class="filter-pill px-2.5 py-1 rounded-md border text-[11px] font-medium transition-colors ${this.filterType === 'numeric' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'}">Numéricas (${numCols})</button>
                <button data-filter="categorical" class="filter-pill px-2.5 py-1 rounded-md border text-[11px] font-medium transition-colors ${this.filterType === 'categorical' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'}">Categóricas (${catCols})</button>
                <button data-filter="selected" class="filter-pill px-2.5 py-1 rounded-md border text-[11px] font-medium transition-colors ${this.filterType === 'selected' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'}">Seleccionadas (<span id="pillSelectedCount">${this.selectedFeatures.size}</span>)</button>
              </div>
            </div>

            <!-- Batch Selection Controls -->
            <div class="flex flex-wrap items-center gap-2">
              <span class="text-[11px] text-slate-400 uppercase font-semibold">Selección:</span>
              <button id="btnSelectAll" class="bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs px-2.5 py-1 rounded transition-colors">✓ Todas</button>
              <button id="btnDeselectAll" class="bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs px-2.5 py-1 rounded transition-colors">✕ Ninguna</button>
              <button id="btnSelectTop5" class="bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs px-2.5 py-1 rounded transition-colors">🎯 Top 5 Señal</button>
              <button id="btnSelectTop10" class="bg-slate-900 hover:bg-slate-800 border border-slate-700 text-slate-300 text-xs px-2.5 py-1 rounded transition-colors">🎯 Top 10 Señal</button>
            </div>
          </div>

          <!-- Bottom Status Counter & Launch CTA -->
          <div class="pt-3 border-t border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div class="flex items-center space-x-2 text-xs">
              <span class="text-slate-400">Estado de selección:</span>
              <span id="selectionLiveCount" class="font-mono font-bold text-indigo-300 bg-indigo-950/60 border border-indigo-800/80 px-2.5 py-1 rounded">
                ${this.selectedFeatures.size} de ${featureCols.length} características seleccionadas (${featureCols.length - this.selectedFeatures.size} excluidas)
              </span>
            </div>

            <button id="btnLaunchWithSelection" class="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs px-4 py-2 rounded-lg transition-colors shadow-lg shadow-emerald-600/20 flex items-center space-x-2">
              <span>🚀 Lanzar Experimento con esta Selección (<span id="ctaSelectedCount">${this.selectedFeatures.size}</span>)</span>
            </button>
          </div>
        </div>

        <!-- 4. Multi-Tab Exploration Card -->
        <div class="workbench-card overflow-hidden">
          <div class="border-b border-slate-800 px-4 flex items-center space-x-6 text-xs font-medium bg-slate-950/40">
            <button class="view-tab-btn py-3 border-b-2 ${this.activeTab === 'schema' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="schema">
              ▦ Schema & Selección
            </button>
            <button class="view-tab-btn py-3 border-b-2 ${this.activeTab === 'stats' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="stats">
              📊 Estadísticas Descriptivas
            </button>
            <button class="view-tab-btn py-3 border-b-2 ${this.activeTab === 'preview' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="preview">
              🔍 Muestra de Datos Reales (Raw)
            </button>
            <button class="view-tab-btn py-3 border-b-2 ${this.activeTab === 'categories' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-400 hover:text-slate-200'}" data-tab="categories">
              🏷️ Distribución Categórica
            </button>
          </div>

          <div class="p-0" id="tabContentContainer">
            ${this._renderActiveTabContent(p)}
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
  }

  _renderActiveTabContent(p) {
    if (this.activeTab === "schema") {
      return this._renderSchemaTab(p);
    } else if (this.activeTab === "stats") {
      return this._renderStatsTab(p);
    } else if (this.activeTab === "preview") {
      return this._renderPreviewTab(p);
    } else if (this.activeTab === "categories") {
      return this._renderCategoriesTab(p);
    }
    return "";
  }

  _getFilteredColumns(columns) {
    return columns.filter(c => {
      // Search filter
      if (this.searchQuery && !c.name.toLowerCase().includes(this.searchQuery.toLowerCase())) {
        return false;
      }
      // Type / Status filter
      if (this.filterType === "numeric") {
        return c.dtype && (c.dtype.includes("int") || c.dtype.includes("float"));
      } else if (this.filterType === "categorical") {
        return c.dtype && (c.dtype === "object" || c.dtype === "category" || c.dtype === "string");
      } else if (this.filterType === "selected") {
        return this.selectedFeatures.has(c.name);
      } else if (this.filterType === "excluded") {
        return !this.selectedFeatures.has(c.name) && c.name !== this.profile?.target_column;
      }
      return true;
    });
  }

  _renderSchemaTab(p) {
    const columns = this._getFilteredColumns(p.columns || []);

    return `
      <div class="overflow-x-auto">
        <table class="w-full wb-table text-left">
          <thead>
            <tr>
              <th class="w-10 text-center">
                <input type="checkbox" id="checkSelectAllRows" class="rounded text-indigo-600 bg-slate-800" ${this.selectedFeatures.size > 0 ? "checked" : ""}>
              </th>
              <th>Feature</th>
              <th>Tipo</th>
              <th>Missing %</th>
              <th>Valores Únicos</th>
              <th>Correlación (r)</th>
              <th>Acción CATML</th>
              <th>Justificación del Motor</th>
            </tr>
          </thead>
          <tbody>
            ${columns.map(c => {
              const isTarget = c.name === p.target_column;
              const isChecked = this.selectedFeatures.has(c.name);
              const corr = c.target_correlation;

              return `
                <tr class="${isChecked ? 'bg-indigo-950/10' : 'opacity-75'}">
                  <td class="text-center">
                    ${isTarget ? `<span class="text-slate-500 font-mono text-xs">🎯</span>` : `
                      <input type="checkbox" data-col="${c.name}" class="col-toggle-checkbox rounded text-indigo-600 bg-slate-800 cursor-pointer" ${isChecked ? "checked" : ""}>
                    `}
                  </td>
                  <td class="font-mono font-medium text-slate-100 flex items-center space-x-2">
                    <span>${c.name}</span>
                    ${isTarget ? `<span class="badge-sys text-[9px] px-1.5 py-0.2 rounded font-bold">TARGET</span>` : ""}
                  </td>
                  <td>
                    <span class="text-xs ${c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'text-indigo-300' : 'text-purple-300'} font-mono">
                      ${c.is_identifier ? 'Identifier' : (c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'Numerical' : 'Categorical')}
                    </span>
                  </td>
                  <td class="font-mono text-xs ${c.null_count > 0 ? 'text-amber-400 font-bold' : 'text-slate-400'}">
                    ${p.row_count ? ((c.null_count / p.row_count) * 100).toFixed(1) + '%' : '0.0%'}
                  </td>
                  <td class="font-mono text-xs text-slate-300">
                    ${c.unique_count != null ? Number(c.unique_count).toLocaleString() : '—'}
                  </td>
                  <td class="font-mono text-xs">
                    ${corr != null ? `
                      <span class="${Math.abs(corr) >= 0.25 ? 'text-emerald-400 font-bold' : 'text-slate-400'}">
                        ${corr > 0 ? '+' : ''}${corr.toFixed(3)}
                      </span>
                    ` : '<span class="text-slate-600">—</span>'}
                  </td>
                  <td>
                    ${
                      isTarget
                        ? `<span class="badge-sys text-[11px] px-2 py-0.5 rounded font-mono font-bold">Target</span>`
                        : c.catml_action === "Exclude" || c.is_identifier
                        ? `<span class="badge-err text-[11px] px-2 py-0.5 rounded font-mono font-bold">Exclude</span>`
                        : c.catml_action === "Encode"
                        ? `<span class="badge-intel text-[11px] px-2 py-0.5 rounded font-mono font-bold">Encode</span>`
                        : `<span class="badge-gain text-[11px] px-2 py-0.5 rounded font-mono font-bold">Keep</span>`
                    }
                  </td>
                  <td class="text-xs text-slate-400">${c.action_reason || (isTarget ? "Variable objetivo a predecir" : "Característica predictiva")}</td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  _renderStatsTab(p) {
    const columns = this._getFilteredColumns(p.columns || []);

    return `
      <div class="overflow-x-auto">
        <table class="w-full wb-table text-left">
          <thead>
            <tr>
              <th>Feature</th>
              <th>Tipo</th>
              <th>Count</th>
              <th>Nulos %</th>
              <th>Media (Mean)</th>
              <th>Std</th>
              <th>Mín</th>
              <th>25%</th>
              <th>Mediana (50%)</th>
              <th>75%</th>
              <th>Máx</th>
              <th>Skewness</th>
              <th>Target r</th>
            </tr>
          </thead>
          <tbody>
            ${columns.map(c => {
              const isTarget = c.name === p.target_column;
              const hasStats = c.mean != null;
              return `
                <tr>
                  <td class="font-mono font-medium text-slate-100">${c.name} ${isTarget ? '🎯' : ''}</td>
                  <td class="font-mono text-xs text-indigo-300">${c.dtype}</td>
                  <td class="font-mono text-xs text-slate-300">${p.row_count ? (p.row_count - c.null_count).toLocaleString() : '—'}</td>
                  <td class="font-mono text-xs ${c.null_count > 0 ? 'text-amber-400' : 'text-slate-400'}">${p.row_count ? ((c.null_count / p.row_count) * 100).toFixed(1) + '%' : '0%'}</td>
                  <td class="font-mono text-xs text-slate-200">${hasStats ? c.mean.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-slate-400">${hasStats ? c.std.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-slate-300">${hasStats ? c.min.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-slate-400">${hasStats ? c.q25.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-indigo-300 font-bold">${hasStats ? c.median.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-slate-400">${hasStats ? c.q75.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-slate-300">${hasStats ? c.max.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs ${c.skew != null && Math.abs(c.skew) > 1.5 ? 'text-amber-400' : 'text-slate-400'}">${c.skew != null ? c.skew.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs ${c.target_correlation != null && Math.abs(c.target_correlation) >= 0.25 ? 'text-emerald-400 font-bold' : 'text-slate-500'}">
                    ${c.target_correlation != null ? (c.target_correlation > 0 ? '+' : '') + c.target_correlation.toFixed(3) : '—'}
                  </td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  _renderPreviewTab(p) {
    const rows = p.preview_rows || [];
    const columns = p.columns || [];

    if (rows.length === 0) {
      return `
        <div class="p-8 text-center text-slate-500 text-xs">
          No hay filas de previsualización disponibles para este dataset.
        </div>
      `;
    }

    return `
      <div class="p-4 space-y-2">
        <div class="text-xs text-slate-400">Muestra de las primeras 8 filas reales del dataset:</div>
        <div class="overflow-x-auto border border-slate-800 rounded-lg">
          <table class="w-full wb-table text-left">
            <thead>
              <tr class="bg-slate-900/80">
                <th class="w-12 text-slate-500 text-center font-mono">#</th>
                ${columns.map(c => `
                  <th class="font-mono text-xs ${c.name === p.target_column ? 'text-indigo-400' : 'text-slate-300'}">${c.name}</th>
                `).join("")}
              </tr>
            </thead>
            <tbody>
              ${rows.map((row, idx) => `
                <tr>
                  <td class="text-center font-mono text-xs text-slate-500">${idx + 1}</td>
                  ${columns.map(c => {
                    const val = row[c.name];
                    const isTarget = c.name === p.target_column;
                    return `
                      <td class="font-mono text-xs ${isTarget ? 'text-indigo-300 font-bold bg-indigo-950/20' : val === null ? 'text-slate-600 italic' : 'text-slate-200'}">
                        ${val !== null && val !== undefined ? String(val) : 'null'}
                      </td>
                    `;
                  }).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  _renderCategoriesTab(p) {
    const catCols = (p.columns || []).filter(c => c.top_categories && c.top_categories.length > 0);

    if (catCols.length === 0) {
      return `
        <div class="p-8 text-center text-slate-500 text-xs">
          No se detectaron características categóricas con distribución de frecuencias.
        </div>
      `;
    }

    return `
      <div class="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        ${catCols.map(c => `
          <div class="workbench-card p-4 space-y-3 bg-slate-900/60 border-slate-800">
            <div class="flex items-center justify-between border-b border-slate-800 pb-2">
              <span class="font-mono font-bold text-slate-200 text-xs">${c.name}</span>
              <span class="text-[10px] font-mono text-purple-400 bg-purple-950/40 px-2 py-0.5 rounded">${c.unique_count} categorías</span>
            </div>

            <div class="space-y-2">
              ${c.top_categories.map(cat => `
                <div class="space-y-1">
                  <div class="flex justify-between text-[11px] font-mono">
                    <span class="text-slate-300 truncate max-w-[150px]">${cat.value}</span>
                    <span class="text-slate-400">${cat.pct}% (${cat.count})</span>
                  </div>
                  <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div class="bg-indigo-500 h-1.5 rounded-full" style="width: ${Math.min(cat.pct, 100)}%"></div>
                  </div>
                </div>
              `).join("")}
            </div>
          </div>
        `).join("")}
      </div>
    `;
  }

  _updateSelectionUi() {
    const p = this.profile;
    if (!p) return;
    const featureCols = (p.columns || []).filter(c => c.name !== p.target_column);

    // Save to store so NewExperimentModal immediately knows active selection
    store.setState({ customFeatures: Array.from(this.selectedFeatures) });

    const liveCount = this.container.querySelector("#selectionLiveCount");
    if (liveCount) {
      liveCount.textContent = `${this.selectedFeatures.size} de ${featureCols.length} características seleccionadas (${featureCols.length - this.selectedFeatures.size} excluidas)`;
    }

    const pillCount = this.container.querySelector("#pillSelectedCount");
    if (pillCount) {
      pillCount.textContent = this.selectedFeatures.size;
    }

    const ctaCount = this.container.querySelector("#ctaSelectedCount");
    if (ctaCount) {
      ctaCount.textContent = this.selectedFeatures.size;
    }

    // Update checkboxes in table
    this.container.querySelectorAll(".col-toggle-checkbox").forEach(cb => {
      const colName = cb.getAttribute("data-col");
      cb.checked = this.selectedFeatures.has(colName);
      const row = cb.closest("tr");
      if (row) {
        if (cb.checked) {
          row.classList.add("bg-indigo-950/10");
          row.classList.remove("opacity-75");
        } else {
          row.classList.remove("bg-indigo-950/10");
          row.classList.add("opacity-75");
        }
      }
    });
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
      store.setState({ customFeatures: Array.from(this.selectedFeatures) });
      bus.emit("modal:new-experiment");
    });

    this.container.querySelector("#btnLaunchWithSelection")?.addEventListener("click", () => {
      store.setState({ customFeatures: Array.from(this.selectedFeatures) });
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
        this._initSelectedFeatures();
        this.render();
      }
    });

    // Recommendations Application
    this.container.querySelector("#btnApplyRecommendations")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      const excludedCols = new Set(
        (this.profile.recommendations || [])
          .filter(r => r.action === "exclude")
          .map(r => r.column)
      );

      this.selectedFeatures.clear();
      (this.profile.columns || []).forEach(c => {
        if (c.name !== target && !c.is_identifier && !excludedCols.has(c.name)) {
          this.selectedFeatures.add(c.name);
        }
      });
      this._updateSelectionUi();
    });

    // Batch Selection Buttons
    this.container.querySelector("#btnSelectAll")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      (this.profile.columns || []).forEach(c => {
        if (c.name !== target) this.selectedFeatures.add(c.name);
      });
      this._updateSelectionUi();
    });

    this.container.querySelector("#btnDeselectAll")?.addEventListener("click", () => {
      this.selectedFeatures.clear();
      this._updateSelectionUi();
    });

    this.container.querySelector("#btnSelectTop5")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      const sorted = (this.profile.columns || [])
        .filter(c => c.name !== target && c.target_correlation != null)
        .sort((a, b) => Math.abs(b.target_correlation) - Math.abs(a.target_correlation));

      this.selectedFeatures.clear();
      sorted.slice(0, 5).forEach(c => this.selectedFeatures.add(c.name));
      this._updateSelectionUi();
    });

    this.container.querySelector("#btnSelectTop10")?.addEventListener("click", () => {
      if (!this.profile) return;
      const target = this.profile.target_column;
      const sorted = (this.profile.columns || [])
        .filter(c => c.name !== target && c.target_correlation != null)
        .sort((a, b) => Math.abs(b.target_correlation) - Math.abs(a.target_correlation));

      this.selectedFeatures.clear();
      sorted.slice(0, 10).forEach(c => this.selectedFeatures.add(c.name));
      this._updateSelectionUi();
    });

    // Checkbox toggles in table
    this.container.querySelectorAll(".col-toggle-checkbox").forEach(cb => {
      cb.addEventListener("change", e => {
        const col = e.target.getAttribute("data-col");
        if (e.target.checked) {
          this.selectedFeatures.add(col);
        } else {
          this.selectedFeatures.delete(col);
        }
        this._updateSelectionUi();
      });
    });

    // Master row checkbox
    this.container.querySelector("#checkSelectAllRows")?.addEventListener("change", e => {
      const checked = e.target.checked;
      const filtered = this._getFilteredColumns(this.profile?.columns || []);
      filtered.forEach(c => {
        if (c.name !== this.profile?.target_column) {
          if (checked) {
            this.selectedFeatures.add(c.name);
          } else {
            this.selectedFeatures.delete(c.name);
          }
        }
      });
      this._updateSelectionUi();
    });

    // Search input
    this.container.querySelector("#featureSearchInput")?.addEventListener("input", e => {
      this.searchQuery = e.target.value.trim();
      const tabContent = this.container.querySelector("#tabContentContainer");
      if (tabContent && this.profile) {
        tabContent.innerHTML = this._renderActiveTabContent(this.profile);
        this._bindTabSpecificEvents();
      }
    });

    // Filter pills
    this.container.querySelectorAll(".filter-pill").forEach(btn => {
      btn.addEventListener("click", () => {
        this.filterType = btn.getAttribute("data-filter");
        this.render();
      });
    });

    // View tabs
    this.container.querySelectorAll(".view-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeTab = btn.getAttribute("data-tab");
        this.render();
      });
    });

    this._bindRegisterFormEvents();
    this._bindTabSpecificEvents();
  }

  _bindTabSpecificEvents() {
    this.container.querySelectorAll(".col-toggle-checkbox").forEach(cb => {
      cb.addEventListener("change", e => {
        const col = e.target.getAttribute("data-col");
        if (e.target.checked) {
          this.selectedFeatures.add(col);
        } else {
          this.selectedFeatures.delete(col);
        }
        this._updateSelectionUi();
      });
    });
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
        this._initSelectedFeatures();
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
