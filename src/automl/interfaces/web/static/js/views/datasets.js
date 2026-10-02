/**
 * DatasetsView — Dataset Understanding & Interactive Feature Selection
 * Statistical profiling, variable exploration, evidence-based recommendations,
 * and column selection for experiment planning.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";

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
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-medium">Analyzing dataset profile and CATML inferences...</div>
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
        <div class="workbench-card p-4">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div class="flex flex-wrap items-center space-x-2 text-xs">
              <span class="text-[#8B95A7] font-medium font-sans uppercase text-[11px] tracking-wide">Flow:</span>
              <span class="px-2.5 py-1 rounded-md bg-[#161B26] border border-[#242A36] text-[#8B95A7] font-mono text-[11px]">1. Raw Dataset</span>
              <span class="text-[#4F67FF] font-bold">→</span>
              <span class="px-2.5 py-1 rounded-md bg-[#4F67FF]/10 border border-[#4F67FF]/25 text-[#4F67FF] font-mono text-[11px] font-semibold">2. Understanding</span>
              <span class="text-[#4F67FF] font-bold">→</span>
              <span class="px-2.5 py-1 rounded-md bg-[#161B26] border border-[#242A36] text-[#8B95A7] font-mono text-[11px]">3. Feature Plan</span>
            </div>

            <div class="flex items-center space-x-3">
              ${this.datasets.length > 1 ? `
                <select id="datasetSelect" class="bg-[#161B26] border border-[#242A36] text-[#F7F8FA] text-xs rounded-lg px-3 py-1.5 focus:border-[#4F67FF] font-mono outline-none">
                  ${this.datasets.map(d => `
                    <option value="${d.id}" ${d.id === this.activeDatasetId ? "selected" : ""}>${d.name}</option>
                  `).join("")}
                </select>
              ` : `
                <span class="font-mono text-xs text-[#F7F8FA] font-medium bg-[#161B26] px-3 py-1 rounded-lg border border-[#242A36]">${datasetName}</span>
              `}

              <button id="btnToggleRegisterForm" class="btn-technical text-xs">
                + Register New
              </button>

              <button id="btnNewExperimentFromDS" class="btn-signal text-xs">
                <span>Create Experiment</span>
              </button>
            </div>
          </div>
        </div>

        ${this.showRegisterForm ? this._getRegisterFormHtml() : ""}

        <!-- Diagnostic Metrics Grid -->
        <div class="workbench-card p-6 space-y-5">
          <div class="flex items-center justify-between border-b border-[#242A36] pb-4">
            <div>
              <div class="flex items-center space-x-2.5">
                <h3 class="text-base font-semibold text-[#F7F8FA] font-sans">Dataset understanding</h3>
                <span class="font-mono text-xs text-[#8B95A7] bg-[#151B26] px-2 py-0.5 rounded-md border border-[#252C38]">${datasetName}</span>
              </div>
              <p class="text-xs text-[#8B95A7] mt-0.5 font-sans">Statistical profile, feature roles, and evidence-based recommendations.</p>
            </div>
            <span class="badge-gain text-xs px-2.5 py-0.5 rounded-md font-mono font-medium">Profiled</span>
          </div>

          <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 text-left">
            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Task</div>
              <div class="text-sm font-semibold text-[#F7F8FA] mt-1 truncate capitalize">${(p.task_type || "Classification").replace("_", " ")}</div>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Target</div>
              <div class="text-sm font-semibold text-[#4F67FF] font-mono mt-1 truncate">${p.target_column || "—"}</div>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Metric</div>
              <div class="text-sm font-semibold text-[#22C55E] font-mono mt-1">${(p.task_type || "").includes("regression") ? "RMSE" : "ROC-AUC"}</div>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Rows</div>
              <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${p.row_count != null ? Number(p.row_count).toLocaleString() : "—"}</div>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Features</div>
              <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${featureCols.length}</div>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Numerical</div>
              <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${numCols}</div>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Categorical</div>
              <div class="text-sm font-semibold text-[#F7F8FA] font-mono mt-1">${catCols}</div>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26]">
              <div class="text-[11px] uppercase text-[#8B95A7] font-medium tracking-normal font-sans">Excluded</div>
              <div class="text-sm font-semibold text-[#EF4444] font-mono mt-1">${excludedCount}</div>
            </div>
          </div>
        </div>

        <!-- 2. Smart Recommendations Panel -->
        ${recommendations.length > 0 ? `
          <div class="workbench-card p-5 space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#252C38] pb-3">
              <div class="flex items-center space-x-2.5">
                <span class="text-[#4F67FF]">${icon("lightbulb", "icon-sm")}</span>
                <span class="text-sm font-semibold text-[#F7F8FA] font-sans">Statistical recommendations</span>
                <span class="text-xs font-mono text-[#8B95A7] bg-[#151B26] px-2 py-0.5 rounded-md">${recommendations.length} detected</span>
              </div>
              <button id="btnApplyRecommendations" class="btn-technical text-xs font-sans flex items-center space-x-1.5">
                ${icon("check", "icon-sm")}
                <span>Apply recommendations</span>
              </button>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5 pt-1">
              ${recommendations.slice(0, 6).map(r => {
                const isDanger = r.severity === 'danger' || r.type === 'exclude';
                const isSuccess = r.severity === 'success' || r.type === 'recommend';
                const isWarning = r.severity === 'warning' || r.type === 'warn';
                const badgeClass = isDanger ? 'badge-err' : isSuccess ? 'badge-gain' : isWarning ? 'badge-warn' : 'badge-intel';
                let badgeText = isDanger ? 'IDENTIFIER' : isSuccess ? 'HIGH RELEVANCE' : isWarning ? 'ATTENTION' : 'ENCODING';
                if (r.badge) {
                  const bLower = r.badge.toLowerCase();
                  if (bLower.includes('strong signal') || bLower.includes('high relevance') || bLower.includes('señal')) {
                    badgeText = 'HIGH RELEVANCE';
                  } else if (bLower.includes('zero variance')) {
                    badgeText = 'ZERO VARIANCE';
                  } else if (bLower.includes('null')) {
                    badgeText = 'HIGH NULLS';
                  } else if (bLower.includes('collinear')) {
                    badgeText = 'COLLINEARITY';
                  } else if (bLower.includes('cardinal')) {
                    badgeText = 'HIGH CARDINALITY';
                  } else if (bLower.includes('identifier') || bLower.includes('identificador')) {
                    badgeText = 'IDENTIFIER';
                  } else {
                    badgeText = r.badge.toUpperCase();
                  }
                }
                return `
                  <div class="p-4 rounded-xl bg-[#151B26] hover:bg-[#1A2230] transition-colors space-y-2">
                    <div class="flex items-center justify-between">
                      <span class="${badgeClass} text-[10px] px-2 py-0.5 rounded font-mono font-medium">${badgeText}</span>
                      ${r.column ? `<code class="text-[11px] font-mono text-[#8B95A7]">${r.column}</code>` : ""}
                    </div>
                    <h4 class="font-sans font-semibold text-xs text-[#F7F8FA]">${r.title}</h4>
                    <p class="text-xs text-[#8B95A7] font-sans leading-relaxed">${r.description}</p>
                  </div>
                `;
              }).join("")}
            </div>
          </div>
        ` : ""}

        <!-- 3. Interactive Selection & Exploration Toolbar -->
        <div class="workbench-card p-4 space-y-3">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <!-- Search & Segmented Filter Pills -->
            <div class="flex flex-wrap items-center gap-3">
              <input type="text" id="featureSearchInput" value="${this.searchQuery}" placeholder="Filter features..." class="bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs rounded-lg px-3 py-1.5 focus:border-[#4F67FF] w-48 font-mono outline-none">

              <div class="segmented-track">
                <button data-filter="all" class="segmented-pill ${this.filterType === 'all' ? 'active-signal' : ''}">All (${featureCols.length})</button>
                <button data-filter="numeric" class="segmented-pill ${this.filterType === 'numeric' ? 'active-signal' : ''}">Numeric (${numCols})</button>
                <button data-filter="categorical" class="segmented-pill ${this.filterType === 'categorical' ? 'active-signal' : ''}">Categorical (${catCols})</button>
                <button data-filter="selected" class="segmented-pill ${this.filterType === 'selected' ? 'active-signal' : ''}">Selected (<span id="pillSelectedCount">${this.selectedFeatures.size}</span>)</button>
              </div>
            </div>

            <!-- Batch Selection Controls -->
            <div class="flex items-center gap-1 text-xs">
              <span class="text-[11px] text-[#8B95A7] uppercase font-sans font-medium mr-1.5">Select:</span>
              <button id="btnSelectAll" class="btn-ghost text-xs py-1 px-2">All</button>
              <button id="btnDeselectAll" class="btn-ghost text-xs py-1 px-2">None</button>
              <span class="text-[#252C38] px-1">·</span>
              <button id="btnSelectTop5" class="btn-ghost text-xs py-1 px-2 text-[#4F67FF] hover:text-white">Top 5 Signal</button>
              <button id="btnSelectTop10" class="btn-ghost text-xs py-1 px-2 text-[#4F67FF] hover:text-white">Top 10 Signal</button>
              <span class="text-[#252C38] px-1">·</span>
              <button id="btnOpenCalculator" class="btn-ghost text-xs py-1 px-2 text-[#6956E8] hover:text-white">Feature Calculator</button>
            </div>
          </div>

          <!-- Bottom Status Counter & Dominant Launch CTA -->
          <div class="pt-3 border-t border-[#252C38]/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div class="flex items-center space-x-2 text-xs font-sans text-[#8B95A7]">
              <span>Selection status:</span>
              <span id="selectionLiveCount" class="font-mono text-xs text-[#F7F8FA]">
                <strong class="text-[#4F67FF]">${this.selectedFeatures.size}</strong> of ${featureCols.length} features selected (${featureCols.length - this.selectedFeatures.size} excluded)
              </span>
            </div>

            <button id="btnLaunchWithSelection" class="btn-signal font-sans font-semibold text-xs px-5 py-2 shadow-md shadow-[#4F67FF]/20 flex items-center space-x-1.5">
              ${icon("rocket", "icon-sm")}
              <span>Launch Experiment (<span id="ctaSelectedCount">${this.selectedFeatures.size}</span>)</span>
            </button>
          </div>
        </div>

        <!-- 4. Multi-Tab Exploration Equipment Module -->
        <div class="workbench-card overflow-hidden">
          <div class="border-b border-[#242A36] px-4 flex items-center space-x-6 text-xs font-sans font-medium bg-[#0D1017] overflow-x-auto">
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${this.activeTab === 'schema' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="schema">
              ${icon("database", "icon-sm")}
              <span>Schema & Selection</span>
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${this.activeTab === 'stats' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="stats">
              ${icon("bar-chart-3", "icon-sm")}
              <span>Descriptive Statistics</span>
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${this.activeTab === 'preview' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="preview">
              ${icon("scroll-text", "icon-sm")}
              <span>Raw Sample</span>
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${this.activeTab === 'categories' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="categories">
              ${icon("tags", "icon-sm")}
              <span>Categorical Distributions</span>
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${this.activeTab === 'correlation' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="correlation">
              ${icon("chart-line", "icon-sm")}
              <span>Correlation Matrix</span>
            </button>
            <button class="view-tab-btn py-3 border-b-2 whitespace-nowrap transition-colors flex items-center space-x-1.5 ${this.activeTab === 'calculator' ? 'border-[#4F67FF] text-[#F7F8FA] font-semibold' : 'border-transparent text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="calculator">
              ${icon("sparkles", "icon-sm")}
              <span>Feature Calculator</span>
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
                <input type="checkbox" id="checkSelectAllRows" class="rounded text-[#4F67FF] bg-[#161B26] border border-[#242A36]" ${this.selectedFeatures.size > 0 ? "checked" : ""}>
              </th>
              <th>Feature</th>
              <th>Type</th>
              <th>Missing %</th>
              <th>Unique Values</th>
              <th>Target Correlation (r)</th>
              <th>Engine Action</th>
              <th>Rationale</th>
              <th class="text-center">Visuals</th>
            </tr>
          </thead>
          <tbody>
            ${columns.map(c => {
              const isTarget = c.name === p.target_column;
              const isChecked = this.selectedFeatures.has(c.name);
              const corr = c.target_correlation;

              return `
                <tr class="${isChecked ? 'bg-[#161B26]/40' : 'opacity-70'}">
                  <td class="text-center">
                    ${isTarget ? `<span class="text-[#4F67FF] flex justify-center" title="Target Column">${icon("target", "icon-sm")}</span>` : `
                      <input type="checkbox" data-col="${c.name}" class="col-toggle-checkbox rounded text-[#4F67FF] bg-[#161B26] border border-[#242A36] cursor-pointer" ${isChecked ? "checked" : ""}>
                    `}
                  </td>
                  <td class="font-mono font-medium text-[#F7F8FA] flex items-center space-x-2">
                    <span>${c.name}</span>
                    ${isTarget ? `<span class="badge-sys text-[9px] px-1.5 py-0.2 rounded font-mono font-medium">TARGET</span>` : ""}
                  </td>
                  <td>
                    <span class="text-xs ${c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'text-[#4F67FF]' : 'text-[#6956E8]'} font-mono">
                      ${c.is_identifier ? 'Identifier' : (c.dtype && (c.dtype.includes('float') || c.dtype.includes('int')) ? 'Numerical' : 'Categorical')}
                    </span>
                  </td>
                  <td class="font-mono text-xs ${c.null_count > 0 ? 'text-[#F59E0B] font-semibold' : 'text-[#8B95A7]'}">
                    ${p.row_count ? ((c.null_count / p.row_count) * 100).toFixed(1) + '%' : '0.0%'}
                  </td>
                  <td class="font-mono text-xs text-[#F7F8FA]">
                    ${c.unique_count != null ? Number(c.unique_count).toLocaleString() : '—'}
                  </td>
                  <td class="font-mono text-xs">
                    ${corr != null ? `
                      <span class="${Math.abs(corr) >= 0.25 ? 'text-[#22C55E] font-semibold' : 'text-[#8B95A7]'}">
                        ${corr > 0 ? '+' : ''}${corr.toFixed(3)}
                      </span>
                    ` : '<span class="text-[#8B95A7]/50">—</span>'}
                  </td>
                  <td>
                    ${
                      isTarget
                        ? `<span class="badge-sys text-[11px] px-2 py-0.5 rounded font-mono font-medium">Target</span>`
                        : c.catml_action === "Exclude" || c.is_identifier
                        ? `<span class="badge-err text-[11px] px-2 py-0.5 rounded font-mono font-medium">Exclude</span>`
                        : c.catml_action === "Encode"
                        ? `<span class="badge-intel text-[11px] px-2 py-0.5 rounded font-mono font-medium">Encode</span>`
                        : `<span class="badge-gain text-[11px] px-2 py-0.5 rounded font-mono font-medium">Keep</span>`
                    }
                  </td>
                  <td class="text-xs text-[#8B95A7] font-sans">${c.action_reason || (isTarget ? "Target prediction objective" : "Predictive feature")}</td>
                  <td class="text-center">
                    <button class="btn-visualize-var btn-technical text-[11px] py-0.5 px-2" data-var="${c.name}">
                      View
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
              <th>Type</th>
              <th>Count</th>
              <th>Missing %</th>
              <th>Mean</th>
              <th>Std</th>
              <th>Min</th>
              <th>25%</th>
              <th>Median (50%)</th>
              <th>75%</th>
              <th>Max</th>
              <th>Skewness</th>
              <th>Target r</th>
              <th class="text-center">Visuals</th>
            </tr>
          </thead>
          <tbody>
            ${columns.map(c => {
              const isTarget = c.name === p.target_column;
              const hasStats = c.mean != null;
              return `
                <tr>
                  <td class="font-mono font-medium text-[#F7F8FA] flex items-center space-x-1.5">
                    <span>${c.name}</span>
                    ${isTarget ? `<span class="text-[#4F67FF]" title="Target Column">${icon("target", "icon-sm")}</span>` : ''}
                  </td>
                  <td class="font-mono text-xs text-[#4F67FF]">${c.dtype}</td>
                  <td class="font-mono text-xs text-[#F7F8FA]">${p.row_count ? (p.row_count - c.null_count).toLocaleString() : '—'}</td>
                  <td class="font-mono text-xs ${c.null_count > 0 ? 'text-[#F59E0B]' : 'text-[#8B95A7]'}">${p.row_count ? ((c.null_count / p.row_count) * 100).toFixed(1) + '%' : '0%'}</td>
                  <td class="font-mono text-xs text-[#F7F8FA]">${hasStats ? c.mean.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-[#8B95A7]">${hasStats ? c.std.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-[#F7F8FA]">${hasStats ? c.min.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-[#8B95A7]">${hasStats ? c.q25.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-[#4F67FF] font-semibold">${hasStats ? c.median.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-[#8B95A7]">${hasStats ? c.q75.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs text-[#F7F8FA]">${hasStats ? c.max.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs ${c.skew != null && Math.abs(c.skew) > 1.5 ? 'text-[#F59E0B]' : 'text-[#8B95A7]'}">${c.skew != null ? c.skew.toFixed(2) : '—'}</td>
                  <td class="font-mono text-xs ${c.target_correlation != null && Math.abs(c.target_correlation) >= 0.25 ? 'text-[#22C55E] font-semibold' : 'text-[#8B95A7]'}">
                    ${c.target_correlation != null ? (c.target_correlation > 0 ? '+' : '') + c.target_correlation.toFixed(3) : '—'}
                  </td>
                  <td class="text-center">
                    <button class="btn-visualize-var btn-technical text-[11px] py-0.5 px-2" data-var="${c.name}">
                      View
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
        <div class="p-8 text-center text-[#8B95A7] text-xs">
          No raw preview rows available for this dataset.
        </div>
      `;
    }

    return `
      <div class="p-4 space-y-2">
        <div class="text-xs text-[#8B95A7]">First 8 sampled rows from source dataset:</div>
        <div class="overflow-x-auto border border-[#242A36] rounded-xl">
          <table class="w-full wb-table text-left">
            <thead>
              <tr class="bg-[#0D1017]">
                <th class="w-12 text-[#8B95A7] text-center font-mono">#</th>
                ${columns.map(c => `
                  <th class="font-mono text-xs ${c.name === p.target_column ? 'text-[#4F67FF]' : 'text-[#8B95A7]'}">${c.name}</th>
                `).join("")}
              </tr>
            </thead>
            <tbody>
              ${rows.map((row, idx) => `
                <tr>
                  <td class="text-center font-mono text-xs text-[#8B95A7]">${idx + 1}</td>
                  ${columns.map(c => {
                    const val = row[c.name];
                    const isTarget = c.name === p.target_column;
                    return `
                      <td class="font-mono text-xs ${isTarget ? 'text-[#4F67FF] font-semibold bg-[#4F67FF]/10' : val === null ? 'text-[#8B95A7]/40 italic' : 'text-[#F7F8FA]'}">
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
        <div class="p-8 text-center text-[#8B95A7] text-xs">
          No categorical features detected with discrete frequency distributions.
        </div>
      `;
    }

    return `
      <div class="p-6 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        ${catCols.map(c => `
          <div class="workbench-card p-4 space-y-3 bg-[#161B26] border-[#242A36]">
            <div class="flex items-center justify-between border-b border-[#242A36] pb-2">
              <span class="font-mono font-medium text-[#F7F8FA] text-xs">${c.name}</span>
              <span class="text-[10px] font-mono text-[#6956E8] bg-[#6956E8]/10 border border-[#6956E8]/20 px-2 py-0.5 rounded">${c.unique_count} categories</span>
            </div>

            <div class="space-y-2">
              ${c.top_categories.map(cat => `
                <div class="space-y-1">
                  <div class="flex justify-between text-[11px] font-mono">
                    <span class="text-[#F7F8FA] truncate max-w-[150px]">${cat.value}</span>
                    <span class="text-[#8B95A7]">${cat.pct}% (${cat.count})</span>
                  </div>
                  <div class="w-full bg-[#11151E] rounded-full h-1.5 overflow-hidden">
                    <div class="bg-[#4F67FF] h-1.5 rounded-full" style="width: ${Math.min(cat.pct, 100)}%"></div>
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
        <div class="p-8 text-center text-[#8B95A7] space-y-2">
          <p class="text-sm font-semibold">Numerical correlation matrix not computed.</p>
          <p class="text-xs text-[#8B95A7]/70">Requires at least 2 numerical features in dataset.</p>
        </div>
      `;
    }

    const cols = cm.columns;
    const matrix = cm.matrix || [];

    return `
      <div class="p-4 space-y-4">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
          <div>
            <span class="font-semibold text-[#F7F8FA] font-sans">Pearson Correlation Matrix ($r \\in [-1, 1]$)</span>
            <p class="text-[11px] text-[#8B95A7]">Bivariate linear dependency, target covariance, and collinear feature detection.</p>
          </div>
          <div class="flex flex-wrap items-center gap-3 text-[11px] font-mono">
            <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-[#22C55E] inline-block"></span><span class="text-[#8B95A7]">Strong Positive (&gt; 0.5)</span></span>
            <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-[#161B26] border border-[#242A36] inline-block"></span><span class="text-[#8B95A7]">Neutral (-0.2 to 0.2)</span></span>
            <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm bg-[#EF4444] inline-block"></span><span class="text-[#8B95A7]">Negative (&lt; -0.2)</span></span>
            <span class="flex items-center space-x-1.5"><span class="w-2.5 h-2.5 rounded-sm ring-1 ring-[#F59E0B] bg-[#F59E0B]/20 inline-block"></span><span class="text-[#8B95A7]">Collinear (&gt; 0.88)</span></span>
          </div>
        </div>

        <div class="overflow-x-auto max-h-[580px] border border-[#242A36] rounded-xl">
          <table class="w-full text-center text-xs border-collapse">
            <thead class="sticky top-0 bg-[#0D1017] z-10 border-b border-[#242A36]">
              <tr>
                <th class="p-2.5 text-left font-mono text-[11px] text-[#8B95A7] bg-[#0D1017] sticky left-0 z-20 border-r border-[#242A36] min-w-[130px]">Feature</th>
                ${cols.map(c => `
                  <th class="p-2 font-mono text-[11px] text-[#8B95A7] min-w-[70px] max-w-[110px] truncate" title="${c}">
                    ${c === p.target_column ? `<span class="text-[#4F67FF] inline-flex items-center space-x-1">${icon("target", "icon-sm")}<span>${c}</span></span>` : c}
                  </th>
                `).join("")}
                <th class="p-2 font-mono text-[11px] text-[#8B95A7]">Visuals</th>
              </tr>
            </thead>
            <tbody>
              ${cols.map((rowName, rIdx) => `
                <tr class="border-b border-[#242A36]/60 hover:bg-[#161B26]/50">
                  <td class="p-2 text-left font-mono font-medium text-[#F7F8FA] bg-[#0D1017]/95 sticky left-0 z-10 border-r border-[#242A36] truncate max-w-[150px]" title="${rowName}">
                    ${rowName === p.target_column ? `<span class="text-[#4F67FF] inline-flex items-center space-x-1">${icon("target", "icon-sm")}<span>${rowName}</span></span>` : rowName}
                  </td>
                  ${cols.map((colName, cIdx) => {
                    const val = (matrix[rIdx] && matrix[rIdx][cIdx] != null) ? matrix[rIdx][cIdx] : 0.0;
                    const isDiag = (rIdx === cIdx);
                    let cellBg = "bg-[#080A0F] text-[#8B95A7]";
                    let ringStyle = "";
                    if (!isDiag) {
                      if (val >= 0.70) cellBg = "bg-emerald-950 text-emerald-300 font-bold";
                      else if (val >= 0.40) cellBg = "bg-emerald-950/60 text-emerald-400 font-semibold";
                      else if (val >= 0.15) cellBg = "bg-emerald-950/20 text-emerald-400";
                      else if (val <= -0.50) cellBg = "bg-rose-950 text-rose-300 font-bold";
                      else if (val <= -0.20) cellBg = "bg-rose-950/60 text-rose-400 font-semibold";
                      else if (val <= -0.10) cellBg = "bg-amber-950/20 text-amber-400";
                      if (Math.abs(val) >= 0.88) ringStyle = "ring-1 ring-[#F59E0B] ring-inset";
                    } else {
                      cellBg = "bg-[#11151E] text-[#8B95A7] font-semibold";
                    }

                    return `
                      <td class="p-2 font-mono text-[11px] ${cellBg} ${ringStyle} transition-colors" title="${rowName} ↔ ${colName}: r = ${val}">
                        ${isDiag ? '1.00' : (val > 0 ? '+' : '') + val.toFixed(2)}
                      </td>
                    `;
                  }).join("")}
                  <td class="p-1.5 bg-[#0D1017]/60 text-center">
                    <button class="btn-visualize-var btn-technical text-[10px] py-0.5 px-2" data-var="${rowName}">
                      View
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
                <span class="text-xs font-mono px-2 py-0.5 rounded font-bold ${isNumeric ? 'bg-[#4F67FF]/10 text-[#4F67FF] border border-[#4F67FF]/20' : 'bg-[#6956E8]/10 text-[#6956E8] border border-[#6956E8]/20'}">
                  ${isNumeric ? 'NUMERIC' : 'CATEGORICAL'}
                </span>
                <h3 class="text-base font-bold text-[#F7F8FA] font-mono">${col.name}</h3>
                ${isTarget ? `<span class="badge-sys text-[10px] px-2 py-0.5 rounded font-bold">TARGET</span>` : ""}
              </div>
              <p class="text-xs text-[#8B95A7] mt-1 font-sans">
                Type: <span class="font-mono text-[#F7F8FA]">${col.dtype}</span> •
                Nulls: <span class="font-mono ${col.null_count > 0 ? 'text-[#F59E0B]' : 'text-[#F7F8FA]'}">${col.null_count} (${((col.null_count / (this.profile.row_count || 1)) * 100).toFixed(1)}%)</span> •
                Unique: <span class="font-mono text-[#F7F8FA]">${col.unique_count != null ? col.unique_count.toLocaleString() : '—'}</span>
                ${col.target_correlation != null ? ` • Target correlation: <span class="font-mono font-bold ${Math.abs(col.target_correlation) >= 0.25 ? 'text-[#22C55E]' : 'text-[#F7F8FA]'}">${col.target_correlation > 0 ? '+' : ''}${col.target_correlation.toFixed(3)}</span>` : ""}
              </p>
            </div>
            <button id="btnCloseVarModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1.5 rounded hover:bg-[#161B26] transition-colors" title="Close">${icon("x", "icon-sm")}</button>
          </div>

          <!-- Modal Tabs -->
          <div class="flex items-center space-x-2 border-b border-[#252C38] pb-2 text-xs font-sans">
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${this.activeModalTab === 'boxplot' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}" data-modal-tab="boxplot">
              Box Plot (IQR Separation)
            </button>
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${this.activeModalTab === 'histogram' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}" data-modal-tab="histogram">
              Histogram & Distribution
            </button>
            <button class="var-tab-btn px-3 py-1.5 rounded-lg border font-medium transition-colors ${this.activeModalTab === 'pattern' ? 'bg-[#4F67FF] text-white border-[#4F67FF]' : 'bg-[#151B26] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}" data-modal-tab="pattern">
              Target Association
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
        <div class="p-8 text-center text-[#8B95A7] space-y-2">
          <div class="flex justify-center text-[#8B95A7]/50">${icon("boxes", "icon-xl", 32)}</div>
          <p class="text-sm font-medium">Box plots are designed for quantitative numerical features.</p>
          <p class="text-xs text-[#8B95A7]/70">For categorical features, inspect the Distribution or Target Association tabs.</p>
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
        const color = idx === 0 ? "#4F67FF" : (idx === 1 ? "#22C55E" : "#F59E0B");
        const xMin = scaleX(t.min);
        const xQ1 = scaleX(t.q25);
        const xMed = scaleX(t.median);
        const xQ3 = scaleX(t.q75);
        const xMax = scaleX(t.max);
        const xMean = t.mean != null ? scaleX(t.mean) : null;

        svgRows += `
          <text x="${padL - 10}" y="${yCenter + 4}" fill="${color}" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">
            Class ${t.class_label}
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
      const color = "#4F67FF";
      const xMin = scaleX(bp.min);
      const xQ1 = scaleX(bp.q25);
      const xMed = scaleX(bp.median);
      const xQ3 = scaleX(bp.q75);
      const xMax = scaleX(bp.max);
      const xMean = bp.mean != null ? scaleX(bp.mean) : null;

      svgRows = `
        <text x="${padL - 10}" y="${yCenter + 4}" fill="${color}" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">
          Distribution
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
        <line x1="${x}" y1="${yAxis - 4}" x2="${x}" y2="${yAxis}" stroke="#384355" stroke-width="1" />
        <text x="${x}" y="${yAxis + 12}" fill="#8B95A7" font-size="9" font-family="monospace" text-anchor="middle">${val.toFixed(1)}</text>
      `;
    }).join("");

    return `
      <div class="space-y-4">
        <div class="p-3 bg-[#090C12] rounded-xl border border-[#252C38] flex flex-col items-center">
          <div class="w-full flex items-center justify-between text-[11px] text-[#8B95A7] font-sans mb-2">
            <span>Tukey Box Plot Dispersion Analysis</span>
            <div class="flex items-center space-x-3 text-xs">
              <span class="flex items-center space-x-1"><span class="w-2.5 h-0.5 bg-white inline-block"></span><span>Median</span></span>
              <span class="flex items-center space-x-1"><span class="w-2 h-2 rotate-45 bg-amber-400 inline-block"></span><span>Mean</span></span>
              <span class="flex items-center space-x-1"><span class="w-3 h-2 bg-[#4F67FF]/30 border border-[#4F67FF] inline-block"></span><span>IQR Range</span></span>
            </div>
          </div>

          <svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="w-full max-w-2xl overflow-visible">
            <line x1="${padL}" y1="${yAxis}" x2="${padL + plotW}" y2="${yAxis}" stroke="#384355" stroke-width="1" />
            ${ticks}
            ${svgRows}
          </svg>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full wb-table text-left text-xs">
            <thead>
              <tr>
                <th>Segment / Class</th>
                <th>Min</th>
                <th>Q1 (25%)</th>
                <th>Median (50%)</th>
                <th>Q3 (75%)</th>
                <th>Max</th>
                <th>IQR</th>
                <th>Mean</th>
              </tr>
            </thead>
            <tbody>
              ${hasTargetSplit ? byTarget.map(t => `
                <tr>
                  <td class="font-mono font-bold text-[#F7F8FA]">${this.profile.target_column} = ${t.class_label}</td>
                  <td class="font-mono text-[#8B95A7]">${t.min != null ? t.min.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${t.q25 != null ? t.q25.toFixed(2) : '—'}</td>
                  <td class="font-mono font-bold text-[#4F67FF]">${t.median != null ? t.median.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${t.q75 != null ? t.q75.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${t.max != null ? t.max.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${t.iqr != null ? t.iqr.toFixed(2) : '—'}</td>
                  <td class="font-mono text-amber-300">${t.mean != null ? t.mean.toFixed(2) : '—'}</td>
                </tr>
              `).join("") : `
                <tr>
                  <td class="font-mono font-bold text-[#F7F8FA]">Global (${col.name})</td>
                  <td class="font-mono text-[#8B95A7]">${bp.min != null ? bp.min.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${bp.q25 != null ? bp.q25.toFixed(2) : '—'}</td>
                  <td class="font-mono font-bold text-[#4F67FF]">${bp.median != null ? bp.median.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${bp.q75 != null ? bp.q75.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${bp.max != null ? bp.max.toFixed(2) : '—'}</td>
                  <td class="font-mono text-[#8B95A7]">${bp.iqr != null ? bp.iqr.toFixed(2) : '—'}</td>
                  <td class="font-mono text-amber-300">${bp.mean != null ? bp.mean.toFixed(2) : '—'}</td>
                </tr>
              `}
            </tbody>
          </table>
        </div>

        ${hasTargetSplit && byTarget.length === 2 ? `
          <div class="p-3 bg-[#151B26] border border-[#252C38] rounded-xl text-xs space-y-1">
            <span class="font-bold text-[#4F67FF] font-sans">Bivariate Target Interpretation:</span>
            <p class="text-[#8B95A7] font-sans">
              Inter-class median delta is <span class="font-mono font-bold text-[#22C55E]">${Math.abs(byTarget[1].median - byTarget[0].median).toFixed(2)}</span>
              ${Math.abs(byTarget[1].median - byTarget[0].median) > (bp.iqr * 0.25) ?
                '(strong distribution shift, high discriminative power for tree ensembles).' :
                '(moderate overlap between distributions; combining with interaction features recommended).'
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
            <span class="text-[#F7F8FA] font-medium font-sans">Frequency Histogram (10 Equidistant Bins)</span>
            ${col.skew != null ? `
              <span class="font-mono text-xs px-2 py-0.5 rounded font-bold ${Math.abs(col.skew) > 1.0 ? 'badge-warn' : 'badge-neutral'}">
                Skewness: ${col.skew.toFixed(2)}
              </span>
            ` : ""}
          </div>

          <div class="p-4 bg-[#090C12] rounded-xl border border-[#252C38] space-y-2">
            <div class="h-44 flex items-end justify-between gap-1.5 pt-6 pb-2 border-b border-[#252C38]">
              ${h.bins.map((binLabel, i) => {
                const count = h.counts[i];
                const pct = h.percentages ? h.percentages[i] : ((count / (col.null_count + count)) * 100).toFixed(1);
                const barHeight = Math.max(4, Math.round((count / maxCount) * 100));

                return `
                  <div class="flex-1 flex flex-col items-center h-full justify-end group relative">
                    <span class="opacity-0 group-hover:opacity-100 transition-opacity absolute -top-5 text-[10px] font-mono text-[#4F67FF] bg-[#151B26] px-1 rounded border border-[#252C38] whitespace-nowrap z-10">
                      ${count.toLocaleString()} (${pct}%)
                    </span>
                    <div style="height: ${barHeight}%;" class="w-full bg-[#4F67FF]/70 hover:bg-[#4F67FF] rounded-t transition-all"></div>
                  </div>
                `;
              }).join("")}
            </div>

            <div class="flex items-center justify-between text-[9px] text-[#8B95A7] font-mono">
              <span class="truncate max-w-[80px]">${h.bins[0]}</span>
              <span class="truncate max-w-[80px]">${h.bins[Math.floor(h.bins.length / 2)]}</span>
              <span class="truncate max-w-[80px]">${h.bins[h.bins.length - 1]}</span>
            </div>
          </div>

          <div class="p-3 bg-[#151B26] border border-[#252C38] rounded-xl text-xs space-y-1">
            <span class="font-bold text-[#F7F8FA] font-sans">Distribution & Skewness Diagnostics:</span>
            <p class="text-[#8B95A7] font-sans">
              ${col.skew == null ? 'Quantitative distribution without computed skewness.' :
                Math.abs(col.skew) <= 0.5 ? 'Approximately symmetric and bell-shaped. Optimal for linear models and tree ensembles without transformation.' :
                col.skew > 0.5 ? `Positive skewness (+${col.skew.toFixed(2)}). Values concentrate on the left with a long right tail. CATML suggests log1p(x) transformation to stabilize variance.` :
                `Negative skewness (${col.skew.toFixed(2)}). Long left tail. Power transformation or robust scaling recommended.`
              }
            </p>
          </div>
        </div>
      `;
    } else if (col.top_categories && col.top_categories.length > 0) {
      return `
        <div class="space-y-4">
          <div class="text-xs text-[#F7F8FA] font-medium font-sans">Categorical Frequency Distribution</div>
          <div class="space-y-2.5 p-4 bg-[#090C12] rounded-xl border border-[#252C38]">
            ${col.top_categories.map(c => `
              <div class="space-y-1">
                <div class="flex items-center justify-between text-xs">
                  <span class="font-mono font-medium text-[#F7F8FA]">${c.value}</span>
                  <span class="font-mono text-[#8B95A7]">${c.count.toLocaleString()} (${c.pct}%)</span>
                </div>
                <div class="h-1.5 w-full bg-[#151B26] rounded-full overflow-hidden">
                  <div class="h-full bg-[#6956E8] rounded-full" style="width: ${c.pct}%;"></div>
                </div>
              </div>
            `).join("")}
          </div>
        </div>
      `;
    }

    return `
      <div class="p-6 text-center text-[#8B95A7] text-xs font-sans">
        No distribution or frequency data available for this feature.
      </div>
    `;
  }

  _renderTargetPatternContent(col) {
    const isTarget = col.name === this.profile.target_column;
    if (isTarget) {
      return `
        <div class="p-6 text-center text-[#8B95A7] text-xs font-sans">
          This column is the primary target variable.
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
              <span class="text-[#F7F8FA] font-medium font-sans">Target Incidence Rate (%) by Category</span>
              <span class="text-[11px] font-mono text-[#4F67FF]">Propensity pattern</span>
            </div>

            <div class="space-y-3 p-4 bg-[#090C12] rounded-xl border border-[#252C38]">
              ${col.top_categories.map(c => {
                const rate = c.target_rate != null ? c.target_rate : 0.0;
                let colorClass = "bg-[#4F67FF]";
                if (rate >= 60) colorClass = "bg-[#22C55E]";
                else if (rate <= 40) colorClass = "bg-[#EF4444]";

                return `
                  <div class="space-y-1">
                    <div class="flex items-center justify-between text-xs">
                      <span class="font-mono font-medium text-[#F7F8FA]">${c.value}</span>
                      <span class="font-mono font-bold ${rate >= 50 ? 'text-[#22C55E]' : 'text-[#8B95A7]'}">
                        ${rate.toFixed(1)}% positive rate (${c.count.toLocaleString()} samples)
                      </span>
                    </div>
                    <div class="h-2 w-full bg-[#151B26] rounded-full overflow-hidden flex">
                      <div class="h-full ${colorClass} rounded-full transition-all" style="width: ${rate}%;"></div>
                    </div>
                  </div>
                `;
              }).join("")}
            </div>

            <div class="p-3 bg-[#151B26] border border-[#252C38] rounded-xl text-xs space-y-1">
              <span class="font-bold text-[#F7F8FA] font-sans">Modeling Utility:</span>
              <p class="text-[#8B95A7] font-sans">
                Variation in positive rate across categories demonstrates discriminative power. CATML will apply out-of-fold Target Encoding to leverage propensity differences without data leakage.
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
            <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-1">
              <span class="text-[#8B95A7] text-[10px] block font-sans uppercase">Pearson Correlation with Target</span>
              <span class="font-mono text-base font-bold ${corr != null && Math.abs(corr) >= 0.25 ? 'text-[#22C55E]' : 'text-[#F7F8FA]'}">
                ${corr != null ? (corr > 0 ? '+' : '') + corr.toFixed(4) : 'Not computed'}
              </span>
              <p class="text-[10px] text-[#8B95A7] font-sans">
                ${corr != null && Math.abs(corr) >= 0.4 ? 'Strong correlation: prime predictive candidate.' :
                  corr != null && Math.abs(corr) >= 0.2 ? 'Moderate correlation: useful predictor.' :
                  'Low linear correlation: tree models will extract non-linear patterns.'
                }
              </p>
            </div>

            <div class="p-3.5 rounded-xl bg-[#151B26] border border-[#252C38] space-y-1">
              <span class="text-[#8B95A7] text-[10px] block font-sans uppercase">Interquartile Separation Across Classes</span>
              <span class="font-mono text-base font-bold text-[#4F67FF]">
                ${byTarget.length === 2 ? Math.abs(byTarget[1].median - byTarget[0].median).toFixed(2) : (bp?.iqr != null ? bp.iqr.toFixed(2) : '—')}
              </span>
              <p class="text-[10px] text-[#8B95A7] font-sans">
                ${byTarget.length === 2 ? 'Absolute delta between class 0 and class 1 medians.' : 'Interquartile range (IQR = Q75 - Q25).'}
              </p>
            </div>
          </div>

          ${byTarget.length >= 2 ? `
            <div class="p-3.5 bg-[#090C12] rounded-xl border border-[#252C38] space-y-2">
              <span class="text-xs font-semibold text-[#F7F8FA] font-sans">Target Class Distribution Comparison</span>
              <div class="space-y-2">
                ${byTarget.map(t => `
                  <div class="flex items-center justify-between text-xs font-mono">
                    <span class="text-[#8B95A7]">${this.profile.target_column} = ${t.class_label}:</span>
                    <span class="font-bold text-[#F7F8FA]">Mean = ${t.mean.toFixed(2)} • Median = ${t.median.toFixed(2)} (${t.count.toLocaleString()} rows)</span>
                  </div>
                `).join("")}
              </div>
            </div>
          ` : ""}
        </div>
      `;
    }

    return `
      <div class="p-6 text-center text-[#8B95A7] text-xs font-sans">
        Insufficient bivariate data to correlate with target.
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
        <div class="p-4 rounded-xl border border-[#252C38] bg-[#151B26] flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="space-y-1">
            <div class="flex items-center space-x-2">
              <span class="text-[#4F67FF]">${icon("sparkles", "icon-sm")}</span>
              <h3 class="text-sm font-semibold text-[#F7F8FA] font-sans">Feature Calculator & Derived Feature Engine</h3>
            </div>
            <p class="text-xs text-[#8B95A7] font-sans">
              Engineers complex tabular features via mathematical formulas or sandboxed Python expressions with division-by-zero isolation and type validation.
            </p>
          </div>

          <div class="flex items-center space-x-2">
            <button id="btnCalcModeFormula" class="px-3 py-1.5 text-xs font-sans rounded-md border transition-colors ${this.calcMode === 'formula' ? 'bg-[#4F67FF] text-white border-[#4F67FF] font-semibold' : 'bg-[#10151E] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">
              Formula Mode
            </button>
            <button id="btnCalcModePython" class="px-3 py-1.5 text-xs font-sans rounded-md border transition-colors ${this.calcMode === 'python_code' ? 'bg-[#4F67FF] text-white border-[#4F67FF] font-semibold' : 'bg-[#10151E] text-[#8B95A7] border-[#252C38] hover:text-[#F7F8FA]'}">
              Python Mode
            </button>
            <button id="btnCalcSuggest" class="btn-technical text-xs text-[#6956E8] border-[#6956E8]/30 hover:border-[#6956E8] flex items-center space-x-1.5">
              <span>${icon("sparkles", "icon-sm")}</span>
              <span>Suggest with AI</span>
            </button>
          </div>
        </div>

        <!-- Inputs Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <!-- Left Column: Name & Expression Editor -->
          <div class="lg:col-span-2 space-y-4">
            <div class="space-y-1.5">
              <label class="text-xs font-sans text-[#8B95A7] uppercase font-semibold tracking-wider">Derived Feature Name:</label>
              <input type="text" id="calcFeatureName" value="${this.calcFeatureName || ''}" placeholder="e.g. debt_to_income_ratio" class="w-full bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs font-mono px-3 py-2 rounded-lg focus:border-[#4F67FF] outline-none">
            </div>

            <div class="space-y-1.5">
              <div class="flex items-center justify-between">
                <label class="text-xs font-sans text-[#8B95A7] uppercase font-semibold tracking-wider">
                  ${this.calcMode === 'formula' ? 'Mathematical Formula:' : 'Sandboxed Python Function (df):'}
                </label>
                <span class="text-[10px] text-[#8B95A7] font-mono">
                  ${this.calcMode === 'formula' ? 'Zero-division guard active' : 'Restricted to numpy & pandas'}
                </span>
              </div>
              <textarea id="calcExpression" rows="${this.calcMode === 'python_code' ? 7 : 4}" placeholder="${this.calcMode === 'python_code' ? 'def compute_feature(df):\n    ratio = df[\'col_a\'] / (df[\'col_b\'] + 1e-5)\n    return np.log1p(ratio)' : 'e.g. debt / (income + 1e-5)'}" class="w-full bg-[#090C12] border border-[#252C38] text-[#F7F8FA] text-xs font-mono p-3 rounded-lg focus:border-[#4F67FF] outline-none">${this.calcExpression || ''}</textarea>
            </div>

            <!-- Quick Operators Toolbar (Formula Mode) -->
            ${this.calcMode === 'formula' ? `
              <div class="space-y-2">
                <span class="text-[11px] font-sans text-[#8B95A7] font-semibold uppercase">Quick Operators & Transforms:</span>
                <div class="flex flex-wrap gap-1.5 text-xs font-mono">
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" + ">+</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" - ">-</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" * ">*</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" / ">/</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op=" ** 2">**2</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="log1p()">log1p()</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="sqrt()">sqrt()</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="clip(col, 0, 100)">clip()</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="zscore()">zscore()</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="if_else(cond, x, y)">if_else()</button>
                  <button class="calc-op-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#8B95A7] hover:text-[#F7F8FA] hover:border-[#4F67FF]" data-op="fillna(col, 0)">fillna()</button>
                </div>
              </div>
            ` : ""}

            <!-- Action CTAs -->
            <div class="flex items-center space-x-3 pt-2">
              <button id="btnCalcValidate" class="btn-technical text-xs font-semibold">
                <span>Test & Preview</span>
              </button>
              <button id="btnCalcApply" class="btn-signal text-xs ${evalRes && evalRes.is_valid ? '' : 'opacity-50 cursor-not-allowed'}" ${evalRes && evalRes.is_valid ? '' : 'disabled'}>
                <span>Apply to Dataset & Save</span>
              </button>
            </div>
          </div>

          <!-- Right Column: Available Column Chips -->
          <div class="space-y-3 p-4 rounded-xl border border-[#252C38] bg-[#090C12]">
            <div class="flex items-center justify-between border-b border-[#252C38] pb-2">
              <span class="text-xs font-sans text-[#F7F8FA] font-semibold uppercase tracking-wider">Available Features</span>
              <span class="text-[10px] font-mono text-[#8B95A7]">Click to insert</span>
            </div>
            <div class="space-y-2 max-h-64 overflow-y-auto pr-1">
              <div class="text-[10px] font-sans text-[#8B95A7] font-semibold uppercase">Numeric:</div>
              <div class="flex flex-wrap gap-1.5">
                ${numCols.map(c => `
                  <button class="calc-col-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#4F67FF] font-mono text-xs hover:border-[#4F67FF] hover:text-white transition-colors" data-col="${c.name}">
                    ${c.name}
                  </button>
                `).join("")}
              </div>

              ${catCols.length > 0 ? `
                <div class="text-[10px] font-sans text-[#8B95A7] font-semibold uppercase pt-2">Categorical / Other:</div>
                <div class="flex flex-wrap gap-1.5">
                  ${catCols.map(c => `
                    <button class="calc-col-chip px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#6956E8] font-mono text-xs hover:border-[#4F67FF] hover:text-white transition-colors" data-col="${c.name}">
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
          <div class="p-5 rounded-xl border ${evalRes.is_valid ? 'border-emerald-800/60 bg-emerald-950/15' : 'border-rose-800/60 bg-rose-950/15'} space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#252C38] pb-3">
              <div class="flex items-center space-x-3">
                <span class="text-xs px-2.5 py-1 rounded-md font-mono font-bold uppercase inline-flex items-center space-x-1.5 ${evalRes.is_valid ? 'badge-gain' : 'badge-err'}">
                  ${evalRes.is_valid ? `${icon("check", "icon-sm")}<span>VALID FORMULA</span>` : `${icon("x", "icon-sm")}<span>VALIDATION ERROR</span>`}
                </span>
                <span class="text-sm font-bold font-mono text-[#F7F8FA]">${evalRes.feature_name}</span>
                <span class="text-xs font-mono text-[#8B95A7]">(${evalRes.dtype || 'unknown'})</span>
              </div>

              <div class="flex items-center space-x-2 text-xs font-mono">
                <span class="px-2 py-0.5 rounded-md bg-[#090C12] border border-[#252C38] text-[#8B95A7]">
                  Rows: <strong class="text-[#F7F8FA]">${evalRes.row_count}</strong>
                </span>
                <span class="px-2 py-0.5 rounded-md bg-[#090C12] border border-[#252C38] ${evalRes.null_percentage > 0.2 ? 'text-amber-400' : 'text-[#8B95A7]'}">
                  Nulls: <strong>${evalRes.null_count} (${(evalRes.null_percentage * 100).toFixed(1)}%)</strong>
                </span>
                <span class="px-2 py-0.5 rounded-md bg-[#090C12] border border-[#252C38] ${evalRes.zero_division_occurred ? 'text-[#22C55E] font-bold' : 'text-[#8B95A7]'}">
                  Zero-div: ${evalRes.zero_division_occurred ? 'Guarded (0.0/eps)' : 'None'}
                </span>
              </div>
            </div>

            ${evalRes.error_message ? `
              <div class="p-3 rounded-lg bg-rose-950/30 border border-rose-800/60 text-rose-300 font-mono text-xs space-y-1">
                <div class="font-bold">Validation Failure Details:</div>
                <div>${evalRes.error_message}</div>
              </div>
            ` : ""}

            ${evalRes.warnings && evalRes.warnings.length > 0 ? `
              <div class="space-y-1">
                ${evalRes.warnings.map(w => `
                  <div class="text-xs font-mono text-amber-300 flex items-center space-x-1.5">
                    <span class="text-[#F59E0B]">${icon("triangle-alert", "icon-sm")}</span>
                    <span>${w}</span>
                  </div>
                `).join("")}
              </div>
            ` : ""}

            ${evalRes.is_valid ? `
              <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono pt-1">
                <!-- Summary Stats -->
                <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#252C38] space-y-2">
                  <div class="text-[10px] text-[#8B95A7] uppercase font-sans font-semibold tracking-wider">Descriptive Statistics:</div>
                  <div class="grid grid-cols-4 gap-2 text-center">
                    <div class="p-1.5 rounded-lg bg-[#151B26]">
                      <div class="text-[10px] text-[#8B95A7]">Min</div>
                      <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.min != null ? evalRes.summary_stats.min.toFixed(3) : '-'}</div>
                    </div>
                    <div class="p-1.5 rounded-lg bg-[#151B26]">
                      <div class="text-[10px] text-[#8B95A7]">Mean</div>
                      <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.mean != null ? evalRes.summary_stats.mean.toFixed(3) : '-'}</div>
                    </div>
                    <div class="p-1.5 rounded-lg bg-[#151B26]">
                      <div class="text-[10px] text-[#8B95A7]">Std</div>
                      <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.std != null ? evalRes.summary_stats.std.toFixed(3) : '-'}</div>
                    </div>
                    <div class="p-1.5 rounded-lg bg-[#151B26]">
                      <div class="text-[10px] text-[#8B95A7]">Max</div>
                      <div class="font-bold text-[#F7F8FA]">${evalRes.summary_stats?.max != null ? evalRes.summary_stats.max.toFixed(3) : '-'}</div>
                    </div>
                  </div>
                </div>

                <!-- Sample Preview -->
                <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#252C38] space-y-2">
                  <div class="text-[10px] text-[#8B95A7] uppercase font-sans font-semibold tracking-wider">Sample Computed Values (First 5 Rows):</div>
                  <div class="flex items-center space-x-2">
                    ${(evalRes.sample_values || []).map((val, idx) => `
                      <span class="px-2 py-1 rounded-md bg-[#151B26] border border-[#252C38] text-[#F7F8FA] font-bold">
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
          <div class="p-4 rounded-xl border border-indigo-900/40 bg-indigo-950/15 space-y-3">
            <div class="flex items-center justify-between border-b border-indigo-900/30 pb-2">
              <span class="text-xs font-sans font-semibold text-indigo-300 uppercase tracking-wider">Feature Hypotheses Proposed by AI</span>
              <span class="text-[10px] font-mono text-[#8B95A7]">${this.calcSuggestions.length} validated proposals</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              ${this.calcSuggestions.map(sugg => `
                <div class="p-3 rounded-xl bg-[#090C12] border border-[#252C38] space-y-2">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-mono font-bold text-indigo-300">${sugg.definition.name}</span>
                    <button class="btn-load-suggestion px-2.5 py-0.5 rounded-md bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-200 border border-indigo-500/40 text-[11px] font-sans font-medium inline-flex items-center space-x-1" data-name="${sugg.definition.name}" data-expr="${encodeURIComponent(sugg.definition.expression)}" data-type="${sugg.definition.expression_type}">
                      <span>Load in Calculator</span>
                      <span>${icon("arrow-right", "icon-sm")}</span>
                    </button>
                  </div>
                  <div class="text-[11px] font-mono text-[#F7F8FA] bg-[#151B26] px-2 py-1 rounded-md border border-[#252C38] overflow-x-auto">
                    ${sugg.definition.expression}
                  </div>
                  <p class="text-[10px] text-[#8B95A7] font-sans">${sugg.definition.description || ''}</p>
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
        alert("Please enter both the feature name and expression.");
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
        alert(`Validation error: ${err.message}`);
      }
    });

    // Apply & Save to Dataset
    this.container.querySelector("#btnCalcApply")?.addEventListener("click", async () => {
      const name = this.calcFeatureName || this.container.querySelector("#calcFeatureName")?.value.trim();
      const expr = this.calcExpression || this.container.querySelector("#calcExpression")?.value.trim();
      if (!name || !expr) return;

      const btn = this.container.querySelector("#btnCalcApply");
      if (btn) btn.textContent = "Applying to Dataset...";

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
        alert(`Failed to apply derived feature: ${err.message}`);
        this.render();
      }
    });

    // AI Suggestions
    this.container.querySelector("#btnCalcSuggest")?.addEventListener("click", async () => {
      const btn = this.container.querySelector("#btnCalcSuggest");
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<span class="animate-spin inline-block mr-1">${icon("refresh-cw", "icon-sm")}</span><span>Analyzing...</span>`;
      }
      try {
        const res = await api.suggestDerivedFeatures(this.activeDatasetId);
        this.calcSuggestions = res.suggestions || [];
        this.render();
      } catch (err) {
        alert(`Failed to generate suggestions: ${err.message}`);
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
      liveCount.innerHTML = `<strong class="text-[#4F67FF]">${this.selectedFeatures.size}</strong> of ${featureCols.length} features selected (${featureCols.length - this.selectedFeatures.size} excluded)`;
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
          <div class="flex justify-center text-slate-600">${icon("database", "icon-xl", 36)}</div>
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
            <span class="text-indigo-400">${icon("plus", "icon-sm")}</span>
            <span class="text-sm font-semibold text-slate-200">Register New Dataset</span>
          </div>
          ${this.profile ? `<button id="btnCloseRegisterForm" class="text-slate-400 hover:text-slate-200 text-xs inline-flex items-center space-x-1">${icon("x", "icon-sm")}<span>Close</span></button>` : ""}
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
    this.container.querySelectorAll(".segmented-pill, .filter-pill").forEach(btn => {
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
