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
    this.activeTab = "schema"; // 'schema' | 'stats' | 'preview' | 'categories' | 'correlation'
    this.filterType = "all"; // 'all' | 'numeric' | 'categorical' | 'selected' | 'excluded'
    this.searchQuery = "";
    this.activeModalVar = null;
    this.activeModalTab = "boxplot"; // 'boxplot' | 'histogram' | 'pattern'
    this.calcMode = "formula"; // 'formula' | 'python_code'
    this.calcFeatureName = "";
    this.calcExpression = "";
    this.calcEvaluationResult = null;
    this.calcSuggestions = [];
    this.calcEvaluating = false;
    this.calcApplying = false;
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
              <button id="btnOpenCalculator" class="bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/50 text-xs px-2.5 py-1 rounded transition-colors font-semibold flex items-center space-x-1"><span>⚡ Calculadora de Features</span></button>
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

            <button id="btnLaunchWithSelection" class="btn-signal">
              <span>▶ LANZAR EXPERIMENTO (<span id="ctaSelectedCount">${this.selectedFeatures.size}</span>)</span>
            </button>
          </div>
        </div>

        <!-- 4. Multi-Tab Exploration Equipment Module -->
        <div class="workbench-card overflow-hidden">
          <div class="border-b border-[#27272e] px-4 flex items-center space-x-6 text-xs font-mono font-medium bg-[#111111] overflow-x-auto">
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap ${this.activeTab === 'schema' ? 'border-[#4F67FF] text-[#FFFFFF] font-bold' : 'border-transparent text-[#94A3B8]/70 hover:text-[#FFFFFF]'}" data-tab="schema">
              ▦ SCHEMA & SELECCIÓN
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap ${this.activeTab === 'stats' ? 'border-[#4F67FF] text-[#FFFFFF] font-bold' : 'border-transparent text-[#94A3B8]/70 hover:text-[#FFFFFF]'}" data-tab="stats">
              📊 ESTADÍSTICAS DESCRIPTIVAS
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap ${this.activeTab === 'preview' ? 'border-[#4F67FF] text-[#FFFFFF] font-bold' : 'border-transparent text-[#94A3B8]/70 hover:text-[#FFFFFF]'}" data-tab="preview">
              🔍 MUESTRA RAW
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap ${this.activeTab === 'categories' ? 'border-[#4F67FF] text-[#FFFFFF] font-bold' : 'border-transparent text-[#94A3B8]/70 hover:text-[#FFFFFF]'}" data-tab="categories">
              🏷️ DISTRIBUCIÓN CATEGÓRICA
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap ${this.activeTab === 'correlation' ? 'border-[#4F67FF] text-[#FFFFFF] font-bold' : 'border-transparent text-[#94A3B8]/70 hover:text-[#FFFFFF]'}" data-tab="correlation">
              🔗 MATRIZ DE CORRELACIÓN
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap ${this.activeTab === 'calculator' ? 'border-[#4F67FF] text-[#FFFFFF] font-bold' : 'border-transparent text-[#94A3B8]/70 hover:text-[#FFFFFF]'}" data-tab="calculator">
              ⚡ CALCULADORA DE FEATURES
            </button>
          </div>

          <div class="p-0" id="tabContentContainer">
            ${this._renderActiveTabContent(p)}
          </div>
        </div>

        <!-- Variable Visual Analytics Modal Container -->
        <div id="variableModalContainer"></div>
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
    } else if (this.activeTab === "correlation") {
      return this._renderCorrelationTab(p);
    } else if (this.activeTab === "calculator") {
      return this._renderCalculatorTab(p);
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
              <th class="text-center">Gráficos</th>
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
                  <td class="text-center">
                    <button class="btn-visualize-var bg-indigo-950/80 hover:bg-indigo-900 border border-indigo-700/60 text-indigo-300 hover:text-white px-2 py-1 rounded text-[11px] font-mono transition-colors" data-var="${c.name}">
                      📊 Ver
                    </button>
                  </td>
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
              <th class="text-center">Gráficos</th>
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
                  <td class="text-center">
                    <button class="btn-visualize-var bg-indigo-950/80 hover:bg-indigo-900 border border-indigo-700/60 text-indigo-300 hover:text-white px-2 py-1 rounded text-[11px] font-mono transition-colors" data-var="${c.name}">
                      📊 Ver
                    </button>
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

  _renderCorrelationTab(p) {
    const cm = p.correlation_matrix;
    if (!cm || !cm.columns || !cm.columns.length) {
      return `
        <div class="p-8 text-center text-slate-400 space-y-2">
          <p class="text-sm font-semibold">No se computó matriz de correlación numérica.</p>
          <p class="text-xs text-slate-500">Se requieren al menos 2 características numéricas en el dataset.</p>
        </div>
      `;
    }

    const cols = cm.columns;
    const matrix = cm.matrix || [];

    return `
      <div class="p-4 space-y-4">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
          <div>
            <span class="font-bold text-slate-200">Matriz de Correlación de Pearson ($r \\in [-1, 1]$)</span>
            <p class="text-[11px] text-slate-400">Analiza patrones bivariantes, covarianza con la variable objetivo y detecta pares multicolineales.</p>
          </div>
          <div class="flex flex-wrap items-center gap-3 text-[11px] font-mono">
            <span class="flex items-center space-x-1.5"><span class="w-3 h-3 rounded bg-emerald-600 inline-block"></span><span>Positiva fuerte (&gt; 0.5)</span></span>
            <span class="flex items-center space-x-1.5"><span class="w-3 h-3 rounded bg-slate-800 border border-slate-700 inline-block"></span><span>Neutra (-0.2 a 0.2)</span></span>
            <span class="flex items-center space-x-1.5"><span class="w-3 h-3 rounded bg-rose-600 inline-block"></span><span>Negativa (&lt; -0.2)</span></span>
            <span class="flex items-center space-x-1.5"><span class="w-3 h-3 rounded ring-1 ring-amber-400 bg-amber-950 inline-block"></span><span>Colinealidad (&gt; 0.88)</span></span>
          </div>
        </div>

        <div class="overflow-x-auto max-h-[580px] border border-slate-800 rounded-lg">
          <table class="w-full text-center text-xs border-collapse">
            <thead class="sticky top-0 bg-slate-950 z-10 border-b border-slate-800">
              <tr>
                <th class="p-2.5 text-left font-mono text-[11px] text-slate-400 bg-slate-950 sticky left-0 z-20 border-r border-slate-800 min-w-[130px]">Variable</th>
                ${cols.map(c => `
                  <th class="p-2 font-mono text-[11px] text-slate-300 min-w-[70px] max-w-[110px] truncate" title="${c}">
                    ${c === p.target_column ? '🎯 ' + c : c}
                  </th>
                `).join("")}
                <th class="p-2 font-mono text-[11px] text-slate-400">Análisis</th>
              </tr>
            </thead>
            <tbody>
              ${cols.map((rowName, rIdx) => `
                <tr class="border-b border-slate-900 hover:bg-slate-900/50">
                  <td class="p-2 text-left font-mono font-medium text-slate-200 bg-slate-950/95 sticky left-0 z-10 border-r border-slate-800 truncate max-w-[150px]" title="${rowName}">
                    ${rowName === p.target_column ? '🎯 ' + rowName : rowName}
                  </td>
                  ${cols.map((colName, cIdx) => {
                    const val = (matrix[rIdx] && matrix[rIdx][cIdx] != null) ? matrix[rIdx][cIdx] : 0.0;
                    const isDiag = (rIdx === cIdx);
                    let cellBg = "bg-slate-950 text-slate-400";
                    let ringStyle = "";
                    if (!isDiag) {
                      if (val >= 0.70) cellBg = "bg-emerald-800 text-white font-bold";
                      else if (val >= 0.40) cellBg = "bg-emerald-950 text-emerald-300 font-semibold";
                      else if (val >= 0.15) cellBg = "bg-emerald-950/40 text-emerald-400";
                      else if (val <= -0.50) cellBg = "bg-rose-900 text-white font-bold";
                      else if (val <= -0.20) cellBg = "bg-rose-950 text-rose-300 font-semibold";
                      else if (val <= -0.10) cellBg = "bg-amber-950/30 text-amber-400";
                      if (Math.abs(val) >= 0.88) ringStyle = "ring-1 ring-amber-400 ring-inset";
                    } else {
                      cellBg = "bg-slate-900 text-slate-500 font-bold";
                    }

                    return `
                      <td class="p-2 font-mono text-[11px] ${cellBg} ${ringStyle} transition-colors" title="${rowName} ↔ ${colName}: r = ${val}">
                        ${isDiag ? '1.00' : (val > 0 ? '+' : '') + val.toFixed(2)}
                      </td>
                    `;
                  }).join("")}
                  <td class="p-1.5 bg-slate-950/60 text-center">
                    <button class="btn-visualize-var bg-indigo-950 hover:bg-indigo-900 border border-indigo-700/60 text-indigo-300 px-2 py-0.5 rounded text-[10px] font-mono" data-var="${rowName}">
                      📊 Ver
                    </button>
                  </td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  }

  _renderVariableModal() {
    if (!this.activeModalVar || !this.profile) return "";
    const col = this.profile.columns.find(c => c.name === this.activeModalVar);
    if (!col) return "";

    const isNumeric = col.dtype && (col.dtype.includes("int") || col.dtype.includes("float"));
    const isTarget = col.name === this.profile.target_column;

    return `
      <div id="variableVisualModal" class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-3xl w-full p-6 space-y-4 border-slate-700 shadow-2xl max-h-[90vh] overflow-y-auto">
          <!-- Header -->
          <div class="flex items-start justify-between border-b border-slate-800 pb-3">
            <div>
              <div class="flex items-center space-x-2">
                <span class="text-xs font-mono px-2 py-0.5 rounded font-bold ${isNumeric ? 'bg-indigo-950 text-indigo-300 border border-indigo-800' : 'bg-purple-950 text-purple-300 border border-purple-800'}">
                  ${isNumeric ? 'NUMÉRICA' : 'CATEGÓRICA'}
                </span>
                <h3 class="text-lg font-bold text-slate-100 font-mono">${col.name}</h3>
                ${isTarget ? `<span class="badge-sys text-[10px] px-2 py-0.5 rounded font-bold">TARGET</span>` : ""}
              </div>
              <p class="text-xs text-slate-400 mt-1">
                Tipo: <span class="font-mono text-slate-300">${col.dtype}</span> •
                Nulos: <span class="font-mono ${col.null_count > 0 ? 'text-amber-400' : 'text-slate-300'}">${col.null_count} (${((col.null_count / (this.profile.row_count || 1)) * 100).toFixed(1)}%)</span> •
                Únicos: <span class="font-mono text-slate-300">${col.unique_count != null ? col.unique_count.toLocaleString() : '—'}</span>
                ${col.target_correlation != null ? ` • Correlación con Target: <span class="font-mono font-bold ${Math.abs(col.target_correlation) >= 0.25 ? 'text-emerald-400' : 'text-slate-300'}">${col.target_correlation > 0 ? '+' : ''}${col.target_correlation.toFixed(3)}</span>` : ""}
              </p>
            </div>
            <button id="btnCloseVarModal" class="text-slate-400 hover:text-white p-1 text-lg leading-none">✕</button>
          </div>

          <!-- Modal Tabs -->
          <div class="flex items-center space-x-2 border-b border-slate-800 pb-2 text-xs">
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${this.activeModalTab === 'boxplot' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'}" data-modal-tab="boxplot">
              📦 Diagrama de Caja y Bigotes (Box Plot)
            </button>
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${this.activeModalTab === 'histogram' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'}" data-modal-tab="histogram">
              📊 Histograma & Distribución
            </button>
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${this.activeModalTab === 'pattern' ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'}" data-modal-tab="pattern">
              🎯 Relación con el Target
            </button>
          </div>

          <!-- Modal Tab Content -->
          <div class="pt-2">
            ${this._renderVarModalContent(col)}
          </div>
        </div>
      </div>
    `;
  }

  _renderVarModalContent(col) {
    if (this.activeModalTab === "boxplot") {
      return this._renderBoxPlotContent(col);
    } else if (this.activeModalTab === "histogram") {
      return this._renderHistogramContent(col);
    } else if (this.activeModalTab === "pattern") {
      return this._renderTargetPatternContent(col);
    }
    return "";
  }

  _renderBoxPlotContent(col) {
    const bp = col.box_plot;
    if (!bp || bp.min == null || bp.max == null) {
      return `
        <div class="p-8 text-center text-slate-400 space-y-2">
          <span class="text-2xl text-slate-500 block">📦</span>
          <p class="text-sm">El diagrama de caja y bigotes está diseñado para variables numéricas cuantitativas.</p>
          <p class="text-xs text-slate-500">Para variables categóricas, consulta las pestañas de Histograma o Relación con Target.</p>
        </div>
      `;
    }

    const byTarget = bp.by_target || [];
    const hasTargetSplit = byTarget.length >= 2;

    const minVal = bp.min;
    const maxVal = bp.max;
    const range = (maxVal - minVal) || 1;

    const svgWidth = 640;
    const svgHeight = hasTargetSplit ? (60 + byTarget.length * 60) : 120;
    const padL = 90;
    const padR = 40;
    const plotW = svgWidth - padL - padR;

    const scaleX = val => padL + Math.max(0, Math.min(plotW, ((val - minVal) / range) * plotW));

    let svgRows = "";
    if (hasTargetSplit) {
      byTarget.forEach((t, idx) => {
        const yCenter = 45 + idx * 55;
        const color = idx === 0 ? "#6366f1" : (idx === 1 ? "#10b981" : "#f59e0b");
        const xMin = scaleX(t.min);
        const xQ1 = scaleX(t.q25);
        const xMed = scaleX(t.median);
        const xQ3 = scaleX(t.q75);
        const xMax = scaleX(t.max);
        const xMean = t.mean != null ? scaleX(t.mean) : null;

        svgRows += `
          <text x="${padL - 10}" y="${yCenter + 4}" fill="${color}" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">
            Clase ${t.class_label}
          </text>
          <line x1="${xMin}" y1="${yCenter}" x2="${xMax}" y2="${yCenter}" stroke="${color}" stroke-width="2" stroke-dasharray="3,2" />
          <line x1="${xMin}" y1="${yCenter - 9}" x2="${xMin}" y2="${yCenter + 9}" stroke="${color}" stroke-width="2" />
          <line x1="${xMax}" y1="${yCenter - 9}" x2="${xMax}" y2="${yCenter + 9}" stroke="${color}" stroke-width="2" />
          <rect x="${xQ1}" y="${yCenter - 13}" width="${Math.max(2, xQ3 - xQ1)}" height="26" fill="${color}" fill-opacity="0.25" stroke="${color}" stroke-width="2" rx="3" />
          <line x1="${xMed}" y1="${yCenter - 13}" x2="${xMed}" y2="${yCenter + 13}" stroke="#ffffff" stroke-width="3" />
          ${xMean != null ? `
            <polygon points="${xMean},${yCenter - 6} ${xMean + 5},${yCenter} ${xMean},${yCenter + 6} ${xMean - 5},${yCenter}" fill="#fbbf24" stroke="#ffffff" stroke-width="1" />
          ` : ""}
        `;
      });
    } else {
      const yCenter = 50;
      const color = "#6366f1";
      const xMin = scaleX(bp.min);
      const xQ1 = scaleX(bp.q25);
      const xMed = scaleX(bp.median);
      const xQ3 = scaleX(bp.q75);
      const xMax = scaleX(bp.max);
      const xMean = bp.mean != null ? scaleX(bp.mean) : null;

      svgRows = `
        <text x="${padL - 10}" y="${yCenter + 4}" fill="${color}" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">
          Distribución
        </text>
        <line x1="${xMin}" y1="${yCenter}" x2="${xMax}" y2="${yCenter}" stroke="${color}" stroke-width="2" stroke-dasharray="3,2" />
        <line x1="${xMin}" y1="${yCenter - 12}" x2="${xMin}" y2="${yCenter + 12}" stroke="${color}" stroke-width="2" />
        <line x1="${xMax}" y1="${yCenter - 12}" x2="${xMax}" y2="${yCenter + 12}" stroke="${color}" stroke-width="2" />
        <rect x="${xQ1}" y="${yCenter - 16}" width="${Math.max(2, xQ3 - xQ1)}" height="32" fill="${color}" fill-opacity="0.25" stroke="${color}" stroke-width="2" rx="3" />
        <line x1="${xMed}" y1="${yCenter - 16}" x2="${xMed}" y2="${yCenter + 16}" stroke="#ffffff" stroke-width="3" />
        ${xMean != null ? `
          <polygon points="${xMean},${yCenter - 6} ${xMean + 5},${yCenter} ${xMean},${yCenter + 6} ${xMean - 5},${yCenter}" fill="#fbbf24" stroke="#ffffff" stroke-width="1" />
        ` : ""}
      `;
    }

    const yAxis = svgHeight - 20;
    const ticks = [0, 0.25, 0.5, 0.75, 1.0].map(frac => {
      const val = minVal + frac * range;
      const x = padL + frac * plotW;
      return `
        <line x1="${x}" y1="${yAxis - 4}" x2="${x}" y2="${yAxis}" stroke="#64748b" stroke-width="1" />
        <text x="${x}" y="${yAxis + 12}" fill="#94a3b8" font-size="9" font-family="monospace" text-anchor="middle">${val.toFixed(1)}</text>
      `;
    }).join("");

    return `
      <div class="space-y-4">
        <div class="p-3 bg-slate-950 rounded-lg border border-slate-800 flex flex-col items-center">
          <div class="w-full flex items-center justify-between text-[11px] text-slate-400 font-mono mb-2">
            <span>Visualización Matemática de Dispersión (Tukey Boxplot)</span>
            <div class="flex items-center space-x-3">
              <span class="flex items-center space-x-1"><span class="w-2.5 h-0.5 bg-white inline-block"></span><span>Mediana</span></span>
              <span class="flex items-center space-x-1"><span class="w-2 h-2 rotate-45 bg-amber-400 inline-block"></span><span>Media</span></span>
              <span class="flex items-center space-x-1"><span class="w-3 h-2 bg-indigo-500/40 border border-indigo-400 inline-block"></span><span>Rango IQR</span></span>
            </div>
          </div>

          <svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="w-full max-w-2xl overflow-visible">
            <line x1="${padL}" y1="${yAxis}" x2="${padL + plotW}" y2="${yAxis}" stroke="#475569" stroke-width="1" />
            ${ticks}
            ${svgRows}
          </svg>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full wb-table text-left text-xs">
            <thead>
              <tr>
                <th>Segmento / Clase</th>
                <th>Mínimo</th>
                <th>Q1 (25%)</th>
                <th>Mediana (50%)</th>
                <th>Q3 (75%)</th>
                <th>Máximo</th>
                <th>IQR</th>
                <th>Media</th>
              </tr>
            </thead>
            <tbody>
              ${hasTargetSplit ? byTarget.map(t => `
                <tr>
                  <td class="font-mono font-bold text-slate-200">${this.profile.target_column} = ${t.class_label}</td>
                  <td class="font-mono text-slate-400">${t.min != null ? t.min.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${t.q25 != null ? t.q25.toFixed(2) : '—'}</td>
                  <td class="font-mono font-bold text-indigo-300">${t.median != null ? t.median.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${t.q75 != null ? t.q75.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${t.max != null ? t.max.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${t.iqr != null ? t.iqr.toFixed(2) : '—'}</td>
                  <td class="font-mono text-amber-300">${t.mean != null ? t.mean.toFixed(2) : '—'}</td>
                </tr>
              `).join("") : `
                <tr>
                  <td class="font-mono font-bold text-slate-200">Global (${col.name})</td>
                  <td class="font-mono text-slate-400">${bp.min != null ? bp.min.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${bp.q25 != null ? bp.q25.toFixed(2) : '—'}</td>
                  <td class="font-mono font-bold text-indigo-300">${bp.median != null ? bp.median.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${bp.q75 != null ? bp.q75.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${bp.max != null ? bp.max.toFixed(2) : '—'}</td>
                  <td class="font-mono text-slate-400">${bp.iqr != null ? bp.iqr.toFixed(2) : '—'}</td>
                  <td class="font-mono text-amber-300">${bp.mean != null ? bp.mean.toFixed(2) : '—'}</td>
                </tr>
              `}
            </tbody>
          </table>
        </div>

        ${hasTargetSplit && byTarget.length === 2 ? `
          <div class="p-3 bg-indigo-950/30 border border-indigo-900/60 rounded-lg text-xs space-y-1">
            <span class="font-bold text-indigo-300">💡 Interpretación Bivariante frente al Target:</span>
            <p class="text-slate-300">
              La diferencia de medianas entre clases es de <span class="font-mono font-bold text-emerald-400">${Math.abs(byTarget[1].median - byTarget[0].median).toFixed(2)}</span>
              ${Math.abs(byTarget[1].median - byTarget[0].median) > (bp.iqr * 0.25) ?
                '(fuerte desplazamiento de distribución, excelente poder separador para LightGBM/XGBoost).' :
                '(distribuciones con solapamiento moderado; combinar con variables de interacción).'
              }
            </p>
          </div>
        ` : ""}
      </div>
    `;
  }

  _renderHistogramContent(col) {
    const isNumeric = col.dtype && (col.dtype.includes("int") || col.dtype.includes("float"));
    if (isNumeric && col.histogram && col.histogram.bins && col.histogram.bins.length > 0) {
      const h = col.histogram;
      const maxCount = Math.max(...h.counts, 1);

      return `
        <div class="space-y-4">
          <div class="flex items-center justify-between text-xs">
            <span class="text-slate-300 font-medium">Histograma de Frecuencias (10 Bins Equidistantes)</span>
            ${col.skew != null ? `
              <span class="font-mono text-xs px-2 py-0.5 rounded font-bold ${Math.abs(col.skew) > 1.0 ? 'bg-amber-950 text-amber-300 border border-amber-800' : 'bg-slate-900 text-slate-300 border border-slate-800'}">
                Asimetría (Skewness): ${col.skew.toFixed(2)}
              </span>
            ` : ""}
          </div>

          <div class="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
            <div class="h-44 flex items-end justify-between gap-1.5 pt-6 pb-2 border-b border-slate-800">
              ${h.bins.map((binLabel, i) => {
                const count = h.counts[i];
                const pct = h.percentages ? h.percentages[i] : ((count / (col.null_count + count)) * 100).toFixed(1);
                const barHeight = Math.max(4, Math.round((count / maxCount) * 100));

                return `
                  <div class="flex-1 flex flex-col items-center h-full justify-end group relative">
                    <span class="opacity-0 group-hover:opacity-100 transition-opacity absolute -top-5 text-[10px] font-mono text-indigo-300 bg-slate-900 px-1 rounded border border-slate-800 whitespace-nowrap z-10">
                      ${count.toLocaleString()} (${pct}%)
                    </span>
                    <div style="height: ${barHeight}%;" class="w-full bg-indigo-600/70 hover:bg-indigo-500 rounded-t transition-all"></div>
                  </div>
                `;
              }).join("")}
            </div>

            <div class="flex items-center justify-between text-[9px] text-slate-400 font-mono">
              <span class="truncate max-w-[80px]">${h.bins[0]}</span>
              <span class="truncate max-w-[80px]">${h.bins[Math.floor(h.bins.length / 2)]}</span>
              <span class="truncate max-w-[80px]">${h.bins[h.bins.length - 1]}</span>
            </div>
          </div>

          <div class="p-3 bg-slate-900 border border-slate-800 rounded-lg text-xs space-y-1">
            <span class="font-bold text-slate-200">📐 Diagnóstico de Distribución y Asimetría:</span>
            <p class="text-slate-400">
              ${col.skew == null ? 'Distribución cuantitativa sin coeficiente de asimetría computado.' :
                Math.abs(col.skew) <= 0.5 ? 'Distribución aproximadamente simétrica y acampanada. Óptima para modelos lineales y SVM sin transformación previa.' :
                col.skew > 0.5 ? `Asimetría positiva acentuada (+${col.skew.toFixed(2)}). Concentración de valores a la izquierda con cola larga a la derecha. CATML sugiere transformación logarítmica log1p(x) para estabilizar la varianza.` :
                `Asimetría negativa (${col.skew.toFixed(2)}). Cola larga a la izquierda. Se sugiere transformación de potencia o escalado robusto.`
              }
            </p>
          </div>
        </div>
      `;
    } else if (col.top_categories && col.top_categories.length > 0) {
      return `
        <div class="space-y-4">
          <div class="text-xs text-slate-300 font-medium">Distribución de Frecuencia Categórica</div>
          <div class="space-y-2.5 p-4 bg-slate-950 rounded-lg border border-slate-800">
            ${col.top_categories.map(c => `
              <div class="space-y-1">
                <div class="flex items-center justify-between text-xs">
                  <span class="font-mono font-medium text-slate-200">${c.value}</span>
                  <span class="font-mono text-slate-400">${c.count.toLocaleString()} (${c.pct}%)</span>
                </div>
                <div class="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                  <div class="h-full bg-purple-500 rounded-full" style="width: ${c.pct}%;"></div>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      `;
    }

    return `
      <div class="p-6 text-center text-slate-400 text-xs">
        No hay datos de distribución o frecuencias disponibles para esta característica.
      </div>
    `;
  }

  _renderTargetPatternContent(col) {
    const isTarget = col.name === this.profile.target_column;
    if (isTarget) {
      return `
        <div class="p-6 text-center text-slate-400 text-xs">
          Esta columna es la variable objetivo principal (Target).
        </div>
      `;
    }

    const isNumeric = col.dtype && (col.dtype.includes("int") || col.dtype.includes("float"));

    if (!isNumeric && col.top_categories && col.top_categories.length > 0) {
      const hasTargetRate = col.top_categories.some(c => c.target_rate != null);
      if (hasTargetRate) {
        return `
          <div class="space-y-4">
            <div class="flex items-center justify-between text-xs">
              <span class="text-slate-300 font-medium">Tasa de Incidencia del Target (%) por Categoría</span>
              <span class="text-[11px] font-mono text-indigo-400">Patrón de propensión</span>
            </div>

            <div class="space-y-3 p-4 bg-slate-950 rounded-lg border border-slate-800">
              ${col.top_categories.map(c => {
                const rate = c.target_rate != null ? c.target_rate : 0.0;
                let colorClass = "bg-indigo-600";
                if (rate >= 60) colorClass = "bg-emerald-500";
                else if (rate <= 40) colorClass = "bg-rose-500";

                return `
                  <div class="space-y-1">
                    <div class="flex items-center justify-between text-xs">
                      <span class="font-mono font-medium text-slate-200">${c.value}</span>
                      <span class="font-mono font-bold ${rate >= 50 ? 'text-emerald-400' : 'text-slate-300'}">
                        ${rate.toFixed(1)}% tasa positiva (${c.count.toLocaleString()} muestras)
                      </span>
                    </div>
                    <div class="h-2.5 w-full bg-slate-800 rounded-full overflow-hidden flex">
                      <div class="h-full ${colorClass} rounded-full transition-all" style="width: ${rate}%;"></div>
                    </div>
                  </div>
                `;
              }).join("")}
            </div>

            <div class="p-3 bg-slate-900 border border-slate-800 rounded-lg text-xs space-y-1">
              <span class="font-bold text-slate-200">🎯 Utilidad en Modelado:</span>
              <p class="text-slate-400">
                La variación en la tasa positiva entre categorías demuestra que esta variable aporta poder discriminante. CATML aplicará Target Encoding out-of-fold para capturar estas diferencias de propensión sin causar fuga de datos.
              </p>
            </div>
          </div>
        `;
      }
    } else if (isNumeric) {
      const corr = col.target_correlation;
      const bp = col.box_plot;
      const byTarget = bp?.by_target || [];

      return `
        <div class="space-y-4">
          <div class="grid grid-cols-2 gap-3 text-xs">
            <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
              <span class="text-slate-400 text-[10px] block">Correlación de Pearson con Target</span>
              <span class="font-mono text-base font-bold ${corr != null && Math.abs(corr) >= 0.25 ? 'text-emerald-400' : 'text-slate-300'}">
                ${corr != null ? (corr > 0 ? '+' : '') + corr.toFixed(4) : 'No calculada'}
              </span>
              <p class="text-[10px] text-slate-500">
                ${corr != null && Math.abs(corr) >= 0.4 ? 'Correlación fuerte: candidato prioritario.' :
                  corr != null && Math.abs(corr) >= 0.2 ? 'Correlación moderada: predictora útil.' :
                  'Correlación lineal baja: el modelo explotará patrones no lineales mediante árboles.'
                }
              </p>
            </div>

            <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
              <span class="text-slate-400 text-[10px] block">Separación Intercuartílica entre Clases</span>
              <span class="font-mono text-base font-bold text-indigo-300">
                ${byTarget.length === 2 ? Math.abs(byTarget[1].median - byTarget[0].median).toFixed(2) : (bp?.iqr != null ? bp.iqr.toFixed(2) : '—')}
              </span>
              <p class="text-[10px] text-slate-500">
                ${byTarget.length === 2 ? 'Delta absoluto entre medianas de clase 0 y clase 1.' : 'Rango intercuartílico (IQR = Q75 - Q25).'}
              </p>
            </div>
          </div>

          ${byTarget.length >= 2 ? `
            <div class="p-3 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
              <span class="text-xs font-semibold text-slate-300">Comparativa de Medias por Clase del Target</span>
              <div class="space-y-2">
                ${byTarget.map(t => `
                  <div class="flex items-center justify-between text-xs font-mono">
                    <span class="text-slate-400">${this.profile.target_column} = ${t.class_label}:</span>
                    <span class="font-bold text-slate-200">Media = ${t.mean.toFixed(2)} • Mediana = ${t.median.toFixed(2)} (${t.count.toLocaleString()} registros)</span>
                  </div>
                `).join("")}
              </div>
            </div>
          ` : ""}
        </div>
      `;
    }

    return `
      <div class="p-6 text-center text-slate-400 text-xs">
        No hay datos bivariantes suficientes para correlacionar con la variable objetivo.
      </div>
    `;
  }

  _renderCalculatorTab(p) {
    const columns = p.columns || [];
    const numCols = columns.filter(c => c.dtype && (c.dtype.includes("int") || c.dtype.includes("float")));
    const catCols = columns.filter(c => !numCols.includes(c));
    const evalRes = this.calcEvaluationResult;

    return `
      <div class="p-6 space-y-6">
        <!-- Header Banner -->
        <div class="p-4 rounded border border-[#27272e] bg-[#1a1a20] flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="space-y-1">
            <div class="flex items-center space-x-2">
              <span class="text-[#4F67FF] text-base">⚡</span>
              <h3 class="text-sm font-bold text-[#FFFFFF] font-mono uppercase tracking-wider">Calculadora de Características & Motor de Columnas Derivadas</h3>
            </div>
            <p class="text-xs text-[#94A3B8]/70 font-mono">
              Genera nuevas features complejas mediante fórmulas o código Python sandboxed con aislamiento de división por cero y comprobación de tipos.
            </p>
          </div>

          <div class="flex items-center space-x-2">
            <button id="btnCalcModeFormula" class="px-3 py-1.5 text-xs font-mono rounded border transition-colors ${this.calcMode === 'formula' ? 'bg-[#4F67FF] text-white border-[#4F67FF] font-bold' : 'bg-[#111111] text-[#94A3B8]/70 border-[#27272e] hover:text-[#FFFFFF]'}">
              🧮 MODO FÓRMULA
            </button>
            <button id="btnCalcModePython" class="px-3 py-1.5 text-xs font-mono rounded border transition-colors ${this.calcMode === 'python_code' ? 'bg-[#4F67FF] text-white border-[#4F67FF] font-bold' : 'bg-[#111111] text-[#94A3B8]/70 border-[#27272e] hover:text-[#FFFFFF]'}">
              🐍 CÓDIGO PYTHON
            </button>
            <button id="btnCalcSuggest" class="px-3 py-1.5 text-xs font-mono rounded border border-indigo-500/60 bg-indigo-950/40 text-indigo-300 hover:bg-indigo-900/50 transition-colors font-semibold flex items-center space-x-1">
              <span>🤖 Sugerir con IA</span>
            </button>
          </div>
        </div>

        <!-- Inputs Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Left Column: Name & Expression Editor -->
          <div class="lg:col-span-2 space-y-4">
            <div class="space-y-1.5">
              <label class="text-xs font-mono text-[#94A3B8] uppercase font-bold tracking-wider">Nombre de la Nueva Característica:</label>
              <input type="text" id="calcFeatureName" value="${this.calcFeatureName || ''}" placeholder="ej: debt_to_income_ratio" class="w-full bg-[#111111] border border-[#27272e] text-[#FFFFFF] text-xs font-mono px-3 py-2 rounded focus:border-[#4F67FF] outline-none">
            </div>

            <div class="space-y-1.5">
              <div class="flex items-center justify-between">
                <label class="text-xs font-mono text-[#94A3B8] uppercase font-bold tracking-wider">
                  ${this.calcMode === 'formula' ? 'Expresión Matemática / Fórmula:' : 'Función Python Sandboxed (df):'}
                </label>
                <span class="text-[10px] text-[#94A3B8]/60 font-mono">
                  ${this.calcMode === 'formula' ? 'Auto-protección contra división por 0 activa' : 'Restringido a numpy y pandas'}
                </span>
              </div>
              <textarea id="calcExpression" rows="${this.calcMode === 'python_code' ? 7 : 4}" placeholder="${this.calcMode === 'python_code' ? 'def compute_feature(df):\n    ratio = df[\'col_a\'] / (df[\'col_b\'] + 1e-5)\n    return np.log1p(ratio)' : 'ej: debt / (income + 1e-5)'}" class="w-full bg-[#111111] border border-[#27272e] text-[#FFFFFF] text-xs font-mono p-3 rounded focus:border-[#4F67FF] outline-none">${this.calcExpression || ''}</textarea>
            </div>

            <!-- Quick Operators Toolbar (Formula Mode) -->
            ${this.calcMode === 'formula' ? `
              <div class="space-y-2">
                <span class="text-[11px] font-mono text-[#94A3B8]/70 font-semibold uppercase">Operadores y Funciones Rápidas:</span>
                <div class="flex flex-wrap gap-1.5 text-xs font-mono">
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op=" + ">+</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op=" - ">-</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op=" * ">*</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op=" / ">/</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op=" ** 2">**2</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op="log1p()">log1p()</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op="sqrt()">sqrt()</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op="clip(col, 0, 100)">clip()</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op="zscore()">zscore()</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op="if_else(cond, x, y)">if_else()</button>
                  <button class="calc-op-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#94A3B8] hover:text-[#FFFFFF] hover:border-[#4F67FF]" data-op="fillna(col, 0)">fillna()</button>
                </div>
              </div>
            ` : ""}

            <!-- Action CTAs -->
            <div class="flex items-center space-x-3 pt-2">
              <button id="btnCalcValidate" class="px-4 py-2 bg-[#1e1e24] hover:bg-[#27272e] text-[#F1EFE9] border border-[#27272e] text-xs font-mono font-bold rounded flex items-center space-x-2 transition-all">
                <span>⚡ PROBAR Y PREVISUALIZAR</span>
              </button>
              <button id="btnCalcApply" class="btn-signal ${evalRes && evalRes.is_valid ? '' : 'opacity-50 cursor-not-allowed'}" ${evalRes && evalRes.is_valid ? '' : 'disabled'}>
                <span>+ APLICAR AL DATASET Y GUARDAR</span>
              </button>
            </div>
          </div>

          <!-- Right Column: Available Column Chips -->
          <div class="space-y-3 p-4 rounded border border-[#27272e] bg-[#141418]">
            <div class="flex items-center justify-between border-b border-[#27272e] pb-2">
              <span class="text-xs font-mono text-[#F1EFE9] font-bold uppercase tracking-wider">Columnas Disponibles</span>
              <span class="text-[10px] font-mono text-[#D8D6CF]/60">Clic para insertar</span>
            </div>
            <div class="space-y-2 max-h-64 overflow-y-auto pr-1">
              <div class="text-[10px] font-mono text-[#D8D6CF]/70 font-semibold uppercase">Numéricas:</div>
              <div class="flex flex-wrap gap-1.5">
                ${numCols.map(c => `
                  <button class="calc-col-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-indigo-300 font-mono text-xs hover:border-[#4F67FF] hover:text-[#FFFFFF] transition-colors" data-col="${c.name}">
                    ${c.name}
                  </button>
                `).join("")}
              </div>

              ${catCols.length > 0 ? `
                <div class="text-[10px] font-mono text-[#94A3B8]/70 font-semibold uppercase pt-2">Otras / Categóricas:</div>
                <div class="flex flex-wrap gap-1.5">
                  ${catCols.map(c => `
                    <button class="calc-col-chip px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-purple-300 font-mono text-xs hover:border-[#4F67FF] hover:text-[#FFFFFF] transition-colors" data-col="${c.name}">
                      ${c.name}
                    </button>
                  `).join("")}
                </div>
              ` : ""}
            </div>
          </div>
        </div>

        <!-- Evaluation Results & Diagnostics Card -->
        ${evalRes ? `
          <div class="p-5 rounded border ${evalRes.is_valid ? 'border-emerald-800/80 bg-emerald-950/20' : 'border-rose-800/80 bg-rose-950/20'} space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#27272e] pb-3">
              <div class="flex items-center space-x-3">
                <span class="text-xs px-2.5 py-1 rounded font-mono font-bold uppercase ${evalRes.is_valid ? 'bg-emerald-900/60 text-emerald-300 border border-emerald-700' : 'bg-rose-900/60 text-rose-300 border border-rose-700'}">
                  ${evalRes.is_valid ? '✓ FÓRMULA VÁLIDA' : '✕ ERROR DE COMPILACIÓN / VALIDACIÓN'}
                </span>
                <span class="text-sm font-bold font-mono text-[#F1EFE9]">${evalRes.feature_name}</span>
                <span class="text-xs font-mono text-[#D8D6CF]/70">(${evalRes.dtype || 'unknown'})</span>
              </div>

              <div class="flex items-center space-x-2 text-xs font-mono">
                <span class="px-2 py-0.5 rounded bg-[#111111] border border-[#27272e] text-[#D8D6CF]">
                  Filas: <strong>${evalRes.row_count}</strong>
                </span>
                <span class="px-2 py-0.5 rounded bg-[#111111] border border-[#27272e] ${evalRes.null_percentage > 0.2 ? 'text-amber-400' : 'text-[#D8D6CF]'}">
                  Nulos: <strong>${evalRes.null_count} (${(evalRes.null_percentage * 100).toFixed(1)}%)</strong>
                </span>
                <span class="px-2 py-0.5 rounded bg-[#111111] border border-[#27272e] ${evalRes.zero_division_occurred ? 'text-emerald-400 font-bold' : 'text-[#D8D6CF]'}">
                  Div/0: ${evalRes.zero_division_occurred ? 'Aislada (0.0/eps)' : 'Ninguna'}
                </span>
              </div>
            </div>

            ${evalRes.error_message ? `
              <div class="p-3 rounded bg-rose-950/40 border border-rose-800 text-rose-300 font-mono text-xs space-y-1">
                <div class="font-bold">Detalle del Fallo de Validación:</div>
                <div>${evalRes.error_message}</div>
              </div>
            ` : ""}

            ${evalRes.warnings && evalRes.warnings.length > 0 ? `
              <div class="space-y-1">
                ${evalRes.warnings.map(w => `
                  <div class="text-xs font-mono text-amber-300 flex items-center space-x-1.5">
                    <span>⚠️</span>
                    <span>${w}</span>
                  </div>
                `).join("")}
              </div>
            ` : ""}

            ${evalRes.is_valid ? `
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono pt-1">
                <!-- Summary Stats -->
                <div class="p-3 rounded bg-[#111111] border border-[#27272e] space-y-2">
                  <div class="text-[10px] text-[#D8D6CF]/70 uppercase font-bold tracking-wider">Estadísticas Descriptivas:</div>
                  <div class="grid grid-cols-4 gap-2 text-center">
                    <div class="p-1.5 rounded bg-[#1e1e24]">
                      <div class="text-[10px] text-[#D8D6CF]/60">Mín</div>
                      <div class="font-bold text-[#F1EFE9]">${evalRes.summary_stats?.min != null ? evalRes.summary_stats.min.toFixed(3) : '-'}</div>
                    </div>
                    <div class="p-1.5 rounded bg-[#1e1e24]">
                      <div class="text-[10px] text-[#D8D6CF]/60">Media</div>
                      <div class="font-bold text-[#F1EFE9]">${evalRes.summary_stats?.mean != null ? evalRes.summary_stats.mean.toFixed(3) : '-'}</div>
                    </div>
                    <div class="p-1.5 rounded bg-[#1e1e24]">
                      <div class="text-[10px] text-[#D8D6CF]/60">Std</div>
                      <div class="font-bold text-[#F1EFE9]">${evalRes.summary_stats?.std != null ? evalRes.summary_stats.std.toFixed(3) : '-'}</div>
                    </div>
                    <div class="p-1.5 rounded bg-[#1e1e24]">
                      <div class="text-[10px] text-[#D8D6CF]/60">Máx</div>
                      <div class="font-bold text-[#F1EFE9]">${evalRes.summary_stats?.max != null ? evalRes.summary_stats.max.toFixed(3) : '-'}</div>
                    </div>
                  </div>
                </div>

                <!-- Sample Preview -->
                <div class="p-3 rounded bg-[#111111] border border-[#27272e] space-y-2">
                  <div class="text-[10px] text-[#D8D6CF]/70 uppercase font-bold tracking-wider">Muestra de Valores Calculados (Primeras 5 Filas):</div>
                  <div class="flex items-center space-x-2">
                    ${(evalRes.sample_values || []).map((val, idx) => `
                      <span class="px-2 py-1 rounded bg-[#1e1e24] border border-[#27272e] text-[#F1EFE9] font-bold">
                        #${idx + 1}: ${val != null ? val : 'NaN'}
                      </span>
                    `).join("")}
                  </div>
                </div>
              </div>
            ` : ""}
          </div>
        ` : ""}

        <!-- AI Feature Suggestions Card -->
        ${this.calcSuggestions && this.calcSuggestions.length > 0 ? `
          <div class="p-4 rounded border border-indigo-900/60 bg-indigo-950/20 space-y-3">
            <div class="flex items-center justify-between border-b border-indigo-900/40 pb-2">
              <span class="text-xs font-mono font-bold text-indigo-300 uppercase tracking-wider">💡 Hipótesis de Features Propuestas por la IA:</span>
              <span class="text-[10px] font-mono text-[#D8D6CF]/70">${this.calcSuggestions.length} sugerencias validadas</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              ${this.calcSuggestions.map(sugg => `
                <div class="p-3 rounded bg-[#111111] border border-[#27272e] space-y-2">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-mono font-bold text-indigo-300">${sugg.definition.name}</span>
                    <button class="btn-load-suggestion px-2 py-0.5 rounded bg-indigo-600/30 hover:bg-indigo-600/60 text-indigo-200 border border-indigo-500/50 text-[10px] font-mono font-semibold" data-name="${sugg.definition.name}" data-expr="${encodeURIComponent(sugg.definition.expression)}" data-type="${sugg.definition.expression_type}">
                      Usar en Calculadora ➔
                    </button>
                  </div>
                  <div class="text-[11px] font-mono text-[#F1EFE9] bg-[#1a1a20] px-2 py-1 rounded border border-[#27272e] overflow-x-auto">
                    ${sugg.definition.expression}
                  </div>
                  <p class="text-[10px] text-[#D8D6CF]/70 font-mono">${sugg.definition.description || ''}</p>
                </div>
              `).join("")}
            </div>
          </div>
        ` : ""}
      </div>
    `;
  }

  _bindCalculatorEvents() {
    // Mode toggles
    this.container.querySelector("#btnCalcModeFormula")?.addEventListener("click", () => {
      this.calcMode = "formula";
      this.calcEvaluationResult = null;
      this.render();
    });

    this.container.querySelector("#btnCalcModePython")?.addEventListener("click", () => {
      this.calcMode = "python_code";
      this.calcEvaluationResult = null;
      this.render();
    });

    // Column chips insertion
    this.container.querySelectorAll(".calc-col-chip").forEach(btn => {
      btn.addEventListener("click", () => {
        const col = btn.getAttribute("data-col");
        const exprArea = this.container.querySelector("#calcExpression");
        if (exprArea) {
          exprArea.value += (exprArea.value.length && !exprArea.value.endsWith(" ") ? " " : "") + col;
          this.calcExpression = exprArea.value;
        }
      });
    });

    // Operator chips insertion
    this.container.querySelectorAll(".calc-op-chip").forEach(btn => {
      btn.addEventListener("click", () => {
        const op = btn.getAttribute("data-op");
        const exprArea = this.container.querySelector("#calcExpression");
        if (exprArea) {
          exprArea.value += op;
          this.calcExpression = exprArea.value;
        }
      });
    });

    // Validate & Preview
    this.container.querySelector("#btnCalcValidate")?.addEventListener("click", async () => {
      const name = this.container.querySelector("#calcFeatureName")?.value.trim();
      const expr = this.container.querySelector("#calcExpression")?.value.trim();
      if (!name || !expr) {
        alert("Por favor introduce el nombre y la expresión de la característica.");
        return;
      }
      this.calcFeatureName = name;
      this.calcExpression = expr;
      try {
        const res = await api.calculateDerivedFeature({
          dataset_id: this.activeDatasetId,
          name: name,
          expression: expr,
          expression_type: this.calcMode,
        });
        this.calcEvaluationResult = res.result;
        this.render();
      } catch (err) {
        alert(`Error al validar: ${err.message}`);
      }
    });

    // Apply & Save to Dataset
    this.container.querySelector("#btnCalcApply")?.addEventListener("click", async () => {
      const name = this.calcFeatureName || this.container.querySelector("#calcFeatureName")?.value.trim();
      const expr = this.calcExpression || this.container.querySelector("#calcExpression")?.value.trim();
      if (!name || !expr) return;

      const btn = this.container.querySelector("#btnCalcApply");
      if (btn) btn.textContent = "⏳ APLICANDO AL DATASET...";

      try {
        await api.applyDerivedFeature({
          dataset_id: this.activeDatasetId,
          name: name,
          expression: expr,
          expression_type: this.calcMode,
        });

        // Refetch profile so the new feature is included
        await this.fetchData();
        this.selectedFeatures.add(name);
        this.activeTab = "schema";
        this.calcEvaluationResult = null;
        this.render();
      } catch (err) {
        alert(`Error al aplicar la característica: ${err.message}`);
        this.render();
      }
    });

    // AI Suggestions
    this.container.querySelector("#btnCalcSuggest")?.addEventListener("click", async () => {
      const btn = this.container.querySelector("#btnCalcSuggest");
      if (btn) btn.textContent = "⏳ Analizando...";
      try {
        const res = await api.suggestDerivedFeatures(this.activeDatasetId);
        this.calcSuggestions = res.suggestions || [];
        this.render();
      } catch (err) {
        alert(`Error al generar sugerencias: ${err.message}`);
        this.render();
      }
    });

    // Load suggestion into calculator
    this.container.querySelectorAll(".btn-load-suggestion").forEach(btn => {
      btn.addEventListener("click", () => {
        this.calcFeatureName = btn.getAttribute("data-name");
        this.calcExpression = decodeURIComponent(btn.getAttribute("data-expr"));
        this.calcMode = btn.getAttribute("data-type") || "formula";
        this.calcEvaluationResult = null;
        this.render();
      });
    });

    // Quick toolbar button jump to calculator
    this.container.querySelector("#btnOpenCalculator")?.addEventListener("click", () => {
      this.activeTab = "calculator";
      this.render();
    });
  }

  _renderAndMountModal() {
    const modalContainer = this.container.querySelector("#variableModalContainer");
    if (!modalContainer) return;
    modalContainer.innerHTML = this._renderVariableModal();

    modalContainer.querySelector("#btnCloseVarModal")?.addEventListener("click", () => {
      this.activeModalVar = null;
      modalContainer.innerHTML = "";
    });

    modalContainer.querySelectorAll(".var-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        this.activeModalTab = btn.getAttribute("data-modal-tab");
        this._renderAndMountModal();
      });
    });
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
    this._bindCalculatorEvents();
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

    this.container.querySelectorAll(".btn-visualize-var").forEach(btn => {
      btn.addEventListener("click", () => {
        const varName = btn.getAttribute("data-var");
        this.activeModalVar = varName;
        this.activeModalTab = "boxplot";
        this._renderAndMountModal();
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
