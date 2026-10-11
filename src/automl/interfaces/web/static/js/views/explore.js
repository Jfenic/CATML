/**
 * ExploreView — CATML Explore Visual Interactive Laboratory (Phase E3)
 * Professional analytical workbench view complying strictly with ADR-006 (Tech Minimalista Premium).
 * 
 * Features:
 * - Study and Execution Selector with New Study Modal & Instant Run CTA.
 * - Dynamic Findings Gallery with Category, Severity & Benjamini-Hochberg FDR filters.
 * - Declarative SVG/Table Visualizations:
 *   * Bivariate Correlation Heatmap (Pearson & Spearman toggle).
 *   * Categorical Association Matrix (Bias-corrected Cramér's V).
 *   * Interactive Vector Distribution Histograms.
 *   * Quantile Boxplots with Outlier markers.
 *   * Bivariate Scatter Plots with Linear Trend Lines.
 * - Technical Markdown / JSON Report Export.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml } from "../utils.js";

export class ExploreView {
  constructor() {
    this.container = null;
    this.activeDatasetId = null;
    this.studies = [];
    this.selectedStudyId = null;
    this.studyDetail = null;
    this.findings = [];
    this.visualizations = [];
    this.hypotheses = [];
    this.evidence = [];
    this.verifyingHypothesisId = null;
    this.loading = true;
    this.runningStudy = false;
    this.showCreateModal = false;

    // Filter states
    this.filterCategory = "ALL";
    this.filterSeverity = "ALL";
    this.filterFdrOnly = false;
    this.activeVizType = "correlation_matrix";
    this.matrixMethod = "pearson"; // "pearson" | "spearman"
  }

  async mount(container) {
    this.container = container;
    const state = store.getState();
    this.activeDatasetId = state.activeDatasetId;
    this.renderLoading();
    await this.fetchData();
    this.render();
  }

  destroy() {
    this.container = null;
  }

  renderLoading() {
    if (!this.container) return;
    this.container.innerHTML = `
      <div class="workbench-card p-12 text-center text-[#8B95A7] space-y-3">
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-sans font-medium">Cargando estudios y motor analítico...</div>
      </div>
    `;
  }

  async fetchData() {
    this.loading = true;
    try {
      if (!this.activeDatasetId) {
        this.studies = [];
        this.studyDetail = null;
        return;
      }

      this.studies = await api.getAnalysisStudies(this.activeDatasetId).catch(() => []);

      if (this.studies.length > 0) {
        if (!this.selectedStudyId || !this.studies.some(s => s.id === this.selectedStudyId)) {
          this.selectedStudyId = this.studies[0].id;
        }

        const detail = await api.getAnalysisStudy(this.selectedStudyId).catch(() => null);
        if (detail) {
          this.studyDetail = detail;
          this.findings = detail.findings || [];
          this.visualizations = detail.visualizations || [];
          this.hypotheses = detail.hypotheses || [];
        }
        const evList = await api.getAnalysisEvidence(this.selectedStudyId).catch(() => []);
        this.evidence = evList || [];
      } else {
        this.selectedStudyId = null;
        this.studyDetail = null;
        this.findings = [];
        this.visualizations = [];
        this.hypotheses = [];
        this.evidence = [];
      }
    } catch (err) {
      console.error("ExploreView fetch error:", err);
    } finally {
      this.loading = false;
    }
  }

  _getFilteredFindings() {
    return this.findings.filter(f => {
      // Category filter
      if (this.filterCategory === "COLLINEARITY" && f.finding_type !== "high_collinearity") return false;
      if (this.filterCategory === "DISTRIBUTION" && !["skewness", "normality_violation", "multimodal_distribution", "outlier_density"].includes(f.finding_type)) return false;
      if (this.filterCategory === "CATEGORICAL" && f.finding_type !== "categorical_association") return false;
      if (this.filterCategory === "TARGET" && !["target_class_separation", "target_correlation"].includes(f.finding_type)) return false;

      // Severity filter
      const sev = this._classifySeverity(f);
      if (this.filterSeverity !== "ALL" && sev !== this.filterSeverity) return false;

      // FDR significance toggle
      if (this.filterFdrOnly) {
        const isFdrSig = f.metrics && f.metrics.fdr_significant === true;
        if (!isFdrSig) return false;
      }

      return true;
    });
  }

  _classifySeverity(f) {
    if (["high_collinearity", "multivariate_outliers"].includes(f.finding_type)) return "CRITICAL";
    if (["skewness", "outlier_density", "categorical_association"].includes(f.finding_type)) return "WARNING";
    return "INFO";
  }

  render() {
    if (!this.container) return;

    if (!this.activeDatasetId) {
      this.container.innerHTML = `
        <div class="workbench-card p-12 text-center space-y-4">
          <div class="text-[#8B95A7] inline-block">${icon("database", "icon-lg")}</div>
          <h3 class="text-base font-bold font-sans text-[#F7F8FA]">Sin dataset activo</h3>
          <p class="text-xs text-[#8B95A7]">Selecciona o carga un dataset para realizar estudios exploratorios.</p>
        </div>
      `;
      return;
    }

    const filteredFindings = this._getFilteredFindings();
    const activeViz = this.visualizations.find(v => v.chart_type === this.activeVizType) || this.visualizations[0];

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Header & Study Selector Controls -->
        <div class="workbench-card p-5 space-y-4">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div class="flex items-center space-x-3">
              <span class="text-[#4F67FF]">${icon("compass", "icon-md")}</span>
              <div>
                <h3 class="text-sm font-bold font-sans text-[#F7F8FA]">Laboratorio Exploratorio · CATML Explore</h3>
                <p class="text-[11px] text-[#8B95A7] font-mono">Inferencia matemática determinista sin alucinaciones</p>
              </div>
            </div>

            <!-- Action buttons -->
            <div class="flex items-center space-x-2">
              <button id="btnOpenNewStudyModal" class="btn-technical text-xs flex items-center space-x-1.5">
                ${icon("plus", "icon-sm")}
                <span>Nuevo Estudio</span>
              </button>

              ${this.studyDetail ? `
                <button id="btnRunStudyExecution" ${this.runningStudy ? "disabled" : ""} class="btn-signal text-xs flex items-center space-x-1.5 shadow-md shadow-[#4F67FF]/20 ${this.runningStudy ? 'opacity-60 cursor-not-allowed' : ''}">
                  ${this.runningStudy ? `<span class="animate-spin">${icon("refresh-cw", "icon-sm")}</span>` : icon("play", "icon-sm")}
                  <span>${this.runningStudy ? "Calculando..." : "Ejecutar Análisis"}</span>
                </button>
                <button id="btnExportStudyReport" class="btn-technical text-xs flex items-center space-x-1.5">
                  ${icon("download", "icon-sm")}
                  <span>Exportar Informe</span>
                </button>
              ` : ""}
            </div>
          </div>

          <!-- Study dropdown & quick status -->
          ${this.studies.length > 0 ? `
            <div class="border-t border-[#242A36] pt-3 flex flex-wrap items-center justify-between gap-3">
              <div class="flex items-center space-x-2">
                <span class="text-xs font-sans text-[#8B95A7]">Estudio activo:</span>
                <select id="selectActiveStudy" class="input-base text-xs font-mono py-1 px-2.5 max-w-xs">
                  ${this.studies.map(s => `
                    <option value="${s.id}" ${s.id === this.selectedStudyId ? "selected" : ""}>
                      ${escapeHtml(s.name)} [${s.status}]
                    </option>
                  `).join("")}
                </select>
              </div>

              ${this.studyDetail ? `
                <div class="flex items-center space-x-3 text-xs font-mono">
                  <span class="text-[#8B95A7]">Target: <strong class="text-[#F7F8FA]">${escapeHtml(this.studyDetail.target_column || "Ninguno (Descriptivo)")}</strong></span>
                  <span class="text-[#8B95A7]">Hallazgos: <strong class="text-[#53C8FF]">${this.findings.length}</strong></span>
                  <span class="text-[#8B95A7]">Hipótesis: <strong class="text-[#22C55E]">${this.hypotheses.length}</strong></span>
                </div>
              ` : ""}
            </div>
          ` : `
            <div class="border-t border-[#242A36] pt-4 text-center py-6 space-y-2">
              <p class="text-xs text-[#8B95A7] font-sans">No hay estudios registrados para este dataset.</p>
              <button id="btnCreateInitialStudy" class="btn-signal text-xs inline-flex items-center space-x-1.5">
                ${icon("plus", "icon-sm")}
                <span>Crear Primer Estudio Exploratorio</span>
              </button>
            </div>
          `}
        </div>

        ${this.studyDetail ? `
          <!-- Visualizations Section -->
          ${this.visualizations.length > 0 ? `
            <div class="workbench-card p-5 space-y-4">
              <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#242A36] pb-3 gap-2">
                <div class="flex items-center space-x-2">
                  <span class="text-[#53C8FF]">${icon("bar-chart-2", "icon-sm")}</span>
                  <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Visualizaciones Declarativas</h4>
                </div>

                <!-- Visualization Type Tabs -->
                <div class="flex flex-wrap items-center gap-1.5 text-xs font-sans">
                  ${this._renderVizTypeButtons(activeViz)}
                </div>
              </div>

              <!-- Render Active Visualization -->
              <div id="activeVisualizationContainer">
                ${this._renderVisualizationContent(activeViz)}
              </div>
            </div>
          ` : ""}

          <!-- Findings Gallery Section -->
          <div class="workbench-card p-5 space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#242A36] pb-3 gap-3">
              <div class="flex items-center space-x-2">
                <span class="text-[#F59E0B]">${icon("alert-circle", "icon-sm")}</span>
                <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Galería de Hallazgos Estadísticos</h4>
                <span class="badge-sys text-[10px] font-mono">${filteredFindings.length} mostrados</span>
              </div>

              <!-- Filter chips -->
              <div class="flex flex-wrap items-center gap-2 text-xs font-sans">
                <!-- Category select -->
                <select id="filterCategorySelect" class="input-base text-xs py-1 px-2 font-mono">
                  <option value="ALL" ${this.filterCategory === "ALL" ? "selected" : ""}>Todas las categorías</option>
                  <option value="COLLINEARITY" ${this.filterCategory === "COLLINEARITY" ? "selected" : ""}>Colinealidad</option>
                  <option value="DISTRIBUTION" ${this.filterCategory === "DISTRIBUTION" ? "selected" : ""}>Distribución / Outliers</option>
                  <option value="CATEGORICAL" ${this.filterCategory === "CATEGORICAL" ? "selected" : ""}>Asociación Categórica</option>
                  <option value="TARGET" ${this.filterCategory === "TARGET" ? "selected" : ""}>Separación Objetivo</option>
                </select>

                <!-- Severity select -->
                <select id="filterSeveritySelect" class="input-base text-xs py-1 px-2 font-mono">
                  <option value="ALL" ${this.filterSeverity === "ALL" ? "selected" : ""}>Toda severidad</option>
                  <option value="CRITICAL" ${this.filterSeverity === "CRITICAL" ? "selected" : ""}>Críticos</option>
                  <option value="WARNING" ${this.filterSeverity === "WARNING" ? "selected" : ""}>Advertencias</option>
                  <option value="INFO" ${this.filterSeverity === "INFO" ? "selected" : ""}>Informativos</option>
                </select>

                <!-- FDR toggle -->
                <label class="flex items-center space-x-1.5 cursor-pointer text-[#8B95A7] hover:text-[#F7F8FA] select-none text-[11px] font-mono">
                  <input type="checkbox" id="checkFdrOnly" ${this.filterFdrOnly ? "checked" : ""} class="rounded border-[#242A36] bg-[#090C12] text-[#4F67FF] focus:ring-0">
                  <span>Solo FDR p &lt; 0.05</span>
                </label>
              </div>
            </div>

            <!-- Findings Grid -->
            ${filteredFindings.length === 0 ? `
              <div class="p-8 text-center text-[#8B95A7] font-sans text-xs space-y-1">
                <p>No se encontraron hallazgos que coincidan con los filtros aplicados.</p>
              </div>
            ` : `
              <div class="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                ${filteredFindings.map(f => this._renderFindingCardHtml(f)).join("")}
              </div>
            `}
          </div>

          <!-- Section 3: Hipótesis de ML y Evidencia Empírica (Hypothesis Engine) -->
          ${this._renderHypothesesAndEvidenceHtml()}
        ` : ""}

        <!-- Create Study Modal -->
        ${this.showCreateModal ? this._renderCreateStudyModalHtml() : ""}
      </div>
    `;

    this._bindEvents();
  }

  _renderVizTypeButtons(activeViz) {
    const typesPresent = [...new Set(this.visualizations.map(v => v.chart_type))];
    const labels = {
      correlation_matrix: "Correlación (Pearson/Spearman)",
      categorical_association_matrix: "Asociación Categórica (Cramér's V)",
      histogram: "Histogramas",
      boxplot: "Diagramas de Caja",
      scatter: "Dispersión",
    };

    return typesPresent.map(t => {
      const isSelected = activeViz && activeViz.chart_type === t;
      return `
        <button class="btn-viz-tab px-2.5 py-1 rounded text-xs font-mono transition-colors ${isSelected ? 'bg-[#4F67FF] text-white font-bold' : 'bg-[#161B26] text-[#8B95A7] hover:text-[#F7F8FA]'}" data-viz-type="${t}">
          ${labels[t] || t}
        </button>
      `;
    }).join("");
  }

  _renderVisualizationContent(viz) {
    if (!viz) return `<div class="text-xs text-[#8B95A7] p-4 text-center">Sin visualizaciones disponibles.</div>`;

    if (viz.chart_type === "correlation_matrix") {
      return this._renderCorrelationMatrixHtml(viz);
    } else if (viz.chart_type === "categorical_association_matrix") {
      return this._renderCategoricalMatrixHtml(viz);
    } else if (viz.chart_type === "histogram") {
      return this._renderHistogramSvgHtml(viz);
    } else if (viz.chart_type === "boxplot") {
      return this._renderBoxplotSvgHtml(viz);
    } else if (viz.chart_type === "scatter") {
      return this._renderScatterSvgHtml(viz);
    }

    return `<pre class="text-xs font-mono p-4 bg-[#090C12] rounded border border-[#242A36] overflow-x-auto">${escapeHtml(JSON.stringify(viz, null, 2))}</pre>`;
  }

  _renderCorrelationMatrixHtml(viz) {
    const cols = viz.data?.columns || [];
    const isSpearman = this.matrixMethod === "spearman";
    const matrix = isSpearman ? (viz.data?.spearman_matrix || viz.data?.matrix || []) : (viz.data?.matrix || []);

    return `
      <div class="space-y-3">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#8B95A7]">${viz.title || "Matriz de Correlación"}</span>
          <div class="flex items-center space-x-1 bg-[#090C12] p-0.5 rounded border border-[#242A36]">
            <button class="btn-matrix-toggle px-2 py-0.5 rounded ${!isSpearman ? 'bg-[#4F67FF] text-white font-semibold' : 'text-[#8B95A7]'}" data-method="pearson">Pearson</button>
            <button class="btn-matrix-toggle px-2 py-0.5 rounded ${isSpearman ? 'bg-[#4F67FF] text-white font-semibold' : 'text-[#8B95A7]'}" data-method="spearman">Spearman</button>
          </div>
        </div>

        <div class="overflow-x-auto border border-[#242A36] rounded-lg">
          <table class="w-full text-xs font-mono border-collapse">
            <thead>
              <tr>
                <th class="p-2 text-left text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36]">Variable</th>
                ${cols.map(c => `
                  <th class="p-2 text-center text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[100px]" title="${escapeHtml(c)}">
                    ${escapeHtml(c)}
                  </th>
                `).join("")}
              </tr>
            </thead>
            <tbody>
              ${cols.map((rowName, rIdx) => `
                <tr>
                  <td class="p-2 font-medium text-[#F7F8FA] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[140px]" title="${escapeHtml(rowName)}">
                    ${escapeHtml(rowName)}
                  </td>
                  ${cols.map((colName, cIdx) => {
                    const val = (matrix[rIdx] && matrix[rIdx][cIdx] != null) ? matrix[rIdx][cIdx] : 0.0;
                    const absVal = Math.abs(val);
                    const isDiag = rIdx === cIdx;
                    let bgColor = "rgba(22, 27, 38, 0.4)";
                    let textColor = "#8B95A7";

                    if (!isDiag) {
                      if (val > 0) {
                        bgColor = `rgba(79, 103, 255, ${Math.min(0.85, absVal * 0.9)})`;
                        textColor = absVal > 0.4 ? "#FFFFFF" : "#F7F8FA";
                      } else {
                        bgColor = `rgba(239, 68, 68, ${Math.min(0.85, absVal * 0.9)})`;
                        textColor = absVal > 0.4 ? "#FFFFFF" : "#F7F8FA";
                      }
                    }

                    return `
                      <td style="background-color: ${bgColor}; color: ${textColor};" class="p-2 text-center border-b border-r border-[#242A36] font-mono ${absVal >= 0.70 && !isDiag ? 'font-bold' : ''}" title="${escapeHtml(rowName)} vs ${escapeHtml(colName)}: ${val.toFixed(3)}">
                        ${val.toFixed(2)}
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

  _renderCategoricalMatrixHtml(viz) {
    const cols = viz.data?.columns || [];
    const matrix = viz.data?.matrix || [];

    return `
      <div class="space-y-3">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#8B95A7]">Cramér's V corregido por sesgo muestral [0.0 - 1.0]</span>
        </div>

        <div class="overflow-x-auto border border-[#242A36] rounded-lg">
          <table class="w-full text-xs font-mono border-collapse">
            <thead>
              <tr>
                <th class="p-2 text-left text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36]">Variable</th>
                ${cols.map(c => `
                  <th class="p-2 text-center text-[#8B95A7] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[100px]" title="${escapeHtml(c)}">
                    ${escapeHtml(c)}
                  </th>
                `).join("")}
              </tr>
            </thead>
            <tbody>
              ${cols.map((rowName, rIdx) => `
                <tr>
                  <td class="p-2 font-medium text-[#F7F8FA] bg-[#090C12] border-b border-r border-[#242A36] truncate max-w-[140px]" title="${escapeHtml(rowName)}">
                    ${escapeHtml(rowName)}
                  </td>
                  ${cols.map((colName, cIdx) => {
                    const val = (matrix[rIdx] && matrix[rIdx][cIdx] != null) ? matrix[rIdx][cIdx] : 0.0;
                    const isDiag = rIdx === cIdx;
                    let bgColor = isDiag ? "rgba(22, 27, 38, 0.4)" : `rgba(83, 200, 255, ${Math.min(0.85, val * 0.9)})`;
                    let textColor = val > 0.4 && !isDiag ? "#FFFFFF" : "#F7F8FA";

                    return `
                      <td style="background-color: ${bgColor}; color: ${textColor};" class="p-2 text-center border-b border-r border-[#242A36] font-mono ${val >= 0.40 && !isDiag ? 'font-bold' : ''}" title="${escapeHtml(rowName)} vs ${escapeHtml(colName)}: ${val.toFixed(3)}">
                        ${val.toFixed(2)}
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

  _renderHistogramSvgHtml(viz) {
    const counts = viz.data?.counts || [];
    const edges = viz.data?.bin_edges || [];
    const maxCount = Math.max(...counts, 1);
    const svgWidth = 600;
    const svgHeight = 180;
    const padding = 30;
    const chartWidth = svgWidth - padding * 2;
    const chartHeight = svgHeight - padding * 2;
    const barWidth = counts.length > 0 ? chartWidth / counts.length : 10;

    return `
      <div class="space-y-2 p-3 bg-[#090C12] rounded-lg border border-[#242A36]">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#F7F8FA] font-medium">${viz.title || "Histograma"}</span>
          <span class="text-[#8B95A7]">Media: ${viz.data?.mean || "-"} · Desv: ${viz.data?.std || "-"}</span>
        </div>

        <div class="w-full overflow-x-auto flex justify-center">
          <svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="w-full max-w-2xl h-44">
            <!-- Background grid lines -->
            <line x1="${padding}" y1="${padding}" x2="${svgWidth - padding}" y2="${padding}" stroke="#242A36" stroke-dasharray="2" />
            <line x1="${padding}" y1="${padding + chartHeight / 2}" x2="${svgWidth - padding}" y2="${padding + chartHeight / 2}" stroke="#242A36" stroke-dasharray="2" />
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${svgWidth - padding}" y2="${padding + chartHeight}" stroke="#384355" />

            <!-- Bars -->
            ${counts.map((c, i) => {
              const barHeight = (c / maxCount) * chartHeight;
              const x = padding + i * barWidth;
              const y = padding + chartHeight - barHeight;
              return `
                <rect x="${x + 2}" y="${y}" width="${Math.max(1, barWidth - 4)}" height="${barHeight}" fill="#4F67FF" rx="2" opacity="0.85">
                  <title>Bin [${edges[i]} - ${edges[i+1]}]: ${c} observaciones</title>
                </rect>
              `;
            }).join("")}
          </svg>
        </div>
      </div>
    `;
  }

  _renderBoxplotSvgHtml(viz) {
    const data = viz.data || {};
    const q25 = data.q25 || 0;
    const med = data.median || 0;
    const q75 = data.q75 || 0;
    const wLow = data.whisker_low || 0;
    const wHigh = data.whisker_high || 0;
    const range = Math.max(1e-6, wHigh - wLow);

    return `
      <div class="space-y-2 p-4 bg-[#090C12] rounded-lg border border-[#242A36]">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#F7F8FA] font-medium">${viz.title || "Diagrama de Caja"}</span>
          <span class="text-[#F59E0B] font-semibold">${data.outliers_count || 0} outliers detectados</span>
        </div>

        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 text-xs font-mono text-[#8B95A7]">
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Bigote Inf: <strong class="text-[#F7F8FA]">${wLow}</strong></div>
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Q1 (25%): <strong class="text-[#F7F8FA]">${q25}</strong></div>
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Mediana: <strong class="text-[#53C8FF]">${med}</strong></div>
          <div class="bg-[#11151E] p-2 rounded border border-[#242A36]">Q3 (75%): <strong class="text-[#F7F8FA]">${q75}</strong></div>
        </div>
      </div>
    `;
  }

  _renderScatterSvgHtml(viz) {
    const xPoints = viz.data?.x_points || [];
    const yPoints = viz.data?.y_points || [];
    const n = Math.min(xPoints.length, yPoints.length);
    if (n === 0) return `<div class="text-xs text-[#8B95A7] p-4">Sin datos de dispersión.</div>`;

    const minX = Math.min(...xPoints);
    const maxX = Math.max(...xPoints);
    const minY = Math.min(...yPoints);
    const maxY = Math.max(...yPoints);
    const rangeX = Math.max(1e-6, maxX - minX);
    const rangeY = Math.max(1e-6, maxY - minY);

    const svgWidth = 600;
    const svgHeight = 200;
    const padding = 30;
    const chartWidth = svgWidth - padding * 2;
    const chartHeight = svgHeight - padding * 2;

    return `
      <div class="space-y-2 p-3 bg-[#090C12] rounded-lg border border-[#242A36]">
        <div class="flex items-center justify-between text-xs font-mono">
          <span class="text-[#F7F8FA] font-medium">${viz.title || "Dispersión"}</span>
          <span class="text-[#53C8FF]">Pearson r = ${viz.data?.pearson_r || "-"}</span>
        </div>

        <div class="w-full overflow-x-auto flex justify-center">
          <svg viewBox="0 0 ${svgWidth} ${svgHeight}" class="w-full max-w-2xl h-48">
            <line x1="${padding}" y1="${padding + chartHeight}" x2="${svgWidth - padding}" y2="${padding + chartHeight}" stroke="#384355" />
            <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${padding + chartHeight}" stroke="#384355" />

            <!-- Points -->
            ${xPoints.map((xp, i) => {
              const cx = padding + ((xp - minX) / rangeX) * chartWidth;
              const cy = padding + chartHeight - ((yPoints[i] - minY) / rangeY) * chartHeight;
              return `<circle cx="${cx}" cy="${cy}" r="3" fill="#53C8FF" opacity="0.75" />`;
            }).join("")}
          </svg>
        </div>
      </div>
    `;
  }

  _renderFindingCardHtml(f) {
    const sev = this._classifySeverity(f);
    const sevBadge = sev === "CRITICAL" ? "badge-err" : sev === "WARNING" ? "badge-warn" : "badge-sys";
    const fdrSig = f.metrics && f.metrics.fdr_significant === true;
    const fdrVal = f.metrics && f.metrics.p_value_fdr != null ? f.metrics.p_value_fdr : null;

    return `
      <div class="p-4 rounded-xl bg-[#090C12] border border-[#242A36] space-y-3 hover:border-[#384355] transition-colors flex flex-col justify-between">
        <div class="space-y-2">
          <div class="flex items-start justify-between gap-2">
            <span class="text-xs font-mono font-bold text-[#F7F8FA]">${escapeHtml(f.finding_type || "HALLAZGO")}</span>
            <div class="flex items-center space-x-1.5">
              ${fdrVal != null ? `
                <span class="${fdrSig ? 'text-[#22C55E] bg-[#22C55E]/10 border-[#22C55E]/30' : 'text-amber-400 bg-amber-400/10 border-amber-400/30'} border text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold">
                  ${fdrSig ? 'FDR p < 0.05' : `FDR p=${fdrVal.toFixed(3)}`}
                </span>
              ` : ""}
              <span class="${sevBadge} text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">
                ${sev}
              </span>
            </div>
          </div>

          <p class="text-xs text-[#F7F8FA] font-sans leading-relaxed">
            ${escapeHtml(f.summary)}
          </p>

          ${f.limitations && f.limitations.length > 0 ? `
            <div class="text-[11px] font-sans text-amber-400/90 bg-amber-400/5 p-2 rounded border border-amber-400/20">
              <span class="font-semibold">Nota metodológica:</span> ${escapeHtml(f.limitations[0])}
            </div>
          ` : ""}
        </div>

        <div class="pt-2 border-t border-[#1C2230] flex flex-wrap items-center justify-between gap-2">
          <div class="flex items-center gap-1.5">
            ${f.column_name ? `<span class="text-[11px] font-mono text-[#8B95A7] bg-[#161B26] px-2 py-0.5 rounded border border-[#242A36]">${escapeHtml(f.column_name)}</span>` : ""}
            ${f.secondary_column ? `<span class="text-[11px] font-mono text-[#8B95A7] bg-[#161B26] px-2 py-0.5 rounded border border-[#242A36]">${escapeHtml(f.secondary_column)}</span>` : ""}
          </div>

          ${f.effect_size != null ? `
            <span class="text-[11px] font-mono font-semibold text-[#53C8FF] bg-[#53C8FF]/10 px-2 py-0.5 rounded border border-[#53C8FF]/20">
              Efecto = ${Number(f.effect_size).toFixed(3)}
            </span>
          ` : ""}
        </div>
      </div>
    `;
  }

  _renderHypothesesAndEvidenceHtml() {
    return `
      <div class="workbench-card p-5 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between border-b border-[#242A36] pb-3 gap-3">
          <div class="flex items-center space-x-2">
            <span class="text-[#22C55E]">${icon("zap", "icon-sm")}</span>
            <h4 class="text-sm font-semibold font-sans text-[#F7F8FA]">Hipótesis de ML y Evidencia Empírica</h4>
            <span class="badge-sys text-[10px] font-mono">${this.hypotheses.length} formuladas • ${this.evidence.length} verificadas</span>
          </div>
          <span class="text-[11px] font-mono text-[#8B95A7]">Protocolo Propose ≠ Accept</span>
        </div>

        <p class="text-xs text-[#8B95A7] font-sans">
          Las hipótesis estadísticas son candidatas a experimentos de AutoML. Bajo el principio <em>"Propose ≠ Accept"</em>, una hipótesis solo se acepta si supera empíricamente al baseline en la misma partición exacta de datos.
        </p>

        <!-- Hypotheses Catalog -->
        <div class="space-y-3">
          <div class="flex items-center justify-between">
            <h5 class="text-xs font-mono font-semibold text-[#8B95A7] uppercase tracking-wider">Catálogo de Hipótesis</h5>
            <span class="text-[11px] font-mono text-[#8B95A7]">Total: ${this.hypotheses.length}</span>
          </div>
          ${this.hypotheses.length === 0 ? `
            <div class="p-4 text-center text-[#8B95A7] font-sans text-xs bg-[#090C12] rounded-lg border border-[#242A36]">
              No hay hipótesis formuladas para este estudio.
            </div>
          ` : `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              ${this.hypotheses.map(h => this._renderHypothesisCardHtml(h)).join("")}
            </div>
          `}
        </div>

        <!-- Evidence Links Table -->
        ${this.evidence.length > 0 ? `
          <div class="space-y-3 pt-3 border-t border-[#242A36]">
            <div class="flex items-center justify-between">
              <h5 class="text-xs font-mono font-semibold text-[#8B95A7] uppercase tracking-wider">Enlaces de Evidencia Empírica (EvidenceLink)</h5>
              <span class="text-[11px] font-mono text-[#22C55E] font-medium">${this.evidence.filter(e => e.accepted).length} aceptadas • ${this.evidence.filter(e => !e.accepted).length} rechazadas</span>
            </div>
            <div class="overflow-x-auto border border-[#242A36] rounded-lg">
              <table class="w-full text-xs font-mono border-collapse">
                <thead>
                  <tr class="bg-[#090C12] text-[#8B95A7] border-b border-[#242A36]">
                    <th class="p-2.5 text-left">Link ID</th>
                    <th class="p-2.5 text-left">Hipótesis</th>
                    <th class="p-2.5 text-left">Experimento</th>
                    <th class="p-2.5 text-right">Baseline</th>
                    <th class="p-2.5 text-right">Candidato</th>
                    <th class="p-2.5 text-center">Métrica</th>
                    <th class="p-2.5 text-center">Veredicto</th>
                    <th class="p-2.5 text-left">Detalle / Notas</th>
                  </tr>
                </thead>
                <tbody>
                  ${this.evidence.map(ev => `
                    <tr class="border-b border-[#1C2230] hover:bg-[#11151E]">
                      <td class="p-2.5 text-[#53C8FF]">${escapeHtml(ev.id)}</td>
                      <td class="p-2.5 text-[#F7F8FA]">${escapeHtml(ev.hypothesis_id)}</td>
                      <td class="p-2.5 text-[#8B95A7]">${escapeHtml(ev.experiment_id || "-")}</td>
                      <td class="p-2.5 text-right text-[#F7F8FA]">${ev.baseline_score != null ? Number(ev.baseline_score).toFixed(4) : "-"}</td>
                      <td class="p-2.5 text-right text-[#F7F8FA]">${ev.candidate_score != null ? Number(ev.candidate_score).toFixed(4) : "-"}</td>
                      <td class="p-2.5 text-center text-[#8B95A7]">${escapeHtml(ev.metric || "-")}</td>
                      <td class="p-2.5 text-center">
                        <span class="${ev.accepted ? 'text-[#22C55E] bg-[#22C55E]/10 border-[#22C55E]/30' : 'text-[#EF4444] bg-[#EF4444]/10 border-[#EF4444]/30'} border text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">
                          ${ev.accepted ? "Aceptada" : "Rechazada"}
                        </span>
                      </td>
                      <td class="p-2.5 text-left text-[11px] font-sans text-[#8B95A7] max-w-xs truncate" title="${escapeHtml(ev.notes || "")}">
                        ${escapeHtml(ev.notes || "-")}
                      </td>
                    </tr>
                  `).join("")}
                </tbody>
              </table>
            </div>
          </div>
        ` : ""}
      </div>
    `;
  }

  _renderHypothesisCardHtml(h) {
    const isVerifying = this.verifyingHypothesisId === h.id;
    const statusBadge = h.status === "accepted"
      ? `<span class="badge-gain text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">ACEPTADA</span>`
      : h.status === "rejected"
      ? `<span class="badge-err text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">RECHAZADA</span>`
      : `<span class="badge-warn text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase">PROPUESTA</span>`;

    return `
      <div class="p-3.5 rounded-xl bg-[#090C12] border border-[#242A36] space-y-2.5 flex flex-col justify-between">
        <div class="space-y-1.5">
          <div class="flex items-center justify-between gap-2">
            <span class="text-xs font-mono font-bold text-[#53C8FF]">${escapeHtml(h.proposed_action || "acción")}</span>
            ${statusBadge}
          </div>
          <p class="text-xs text-[#F7F8FA] font-sans leading-relaxed">
            ${escapeHtml(h.description)}
          </p>
        </div>

        <div class="pt-2 border-t border-[#1C2230] flex items-center justify-between gap-2">
          <span class="text-[10px] font-mono text-[#8B95A7]">${escapeHtml(h.id)}</span>
          ${h.status === "proposed" ? `
            <button class="btn-verify-hyp btn-signal text-[11px] py-1 px-2.5 font-mono inline-flex items-center space-x-1" data-hyp-id="${escapeHtml(h.id)}" ${isVerifying ? "disabled" : ""}>
              ${isVerifying ? `<span class="animate-spin mr-1">${icon("refresh-cw", "icon-xs")}</span>` : `${icon("play", "icon-xs")}`}
              <span>${isVerifying ? "Verificando..." : "Verificar"}</span>
            </button>
          ` : `
            <span class="text-[11px] font-mono text-[#8B95A7] flex items-center space-x-1">
              ${icon("check-circle-2", "icon-xs")}
              <span>Evaluada</span>
            </span>
          `}
        </div>
      </div>
    `;
  }

  _renderCreateStudyModalHtml() {
    return `
      <div class="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-md w-full p-6 space-y-5 border border-[#4F67FF]/30 shadow-2xl">
          <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
            <div class="flex items-center space-x-2">
              <span class="text-[#4F67FF]">${icon("compass", "icon-sm")}</span>
              <h3 class="text-sm font-bold font-sans text-[#F7F8FA]">Nuevo Estudio Exploratorio</h3>
            </div>
            <button id="btnCloseStudyModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1">${icon("x", "icon-sm")}</button>
          </div>

          <div class="space-y-4 text-xs font-sans">
            <div class="space-y-1.5">
              <label class="font-medium text-[#F7F8FA]">Nombre del Estudio</label>
              <input id="inputStudyName" type="text" class="input-base w-full text-xs" value="Estudio de Diagnóstico Completo" />
            </div>

            <div class="space-y-1.5">
              <label class="font-medium text-[#F7F8FA]">Columna Objetivo (Opcional)</label>
              <input id="inputTargetCol" type="text" class="input-base w-full text-xs font-mono" placeholder="Dejar vacío para modo no supervisado" />
              <p class="text-[11px] text-[#8B95A7]">Si se especifica, se ejecutarán pruebas inferenciales contra el target (ANOVA / Welch t-test).</p>
            </div>
          </div>

          <div class="flex items-center justify-end space-x-2 border-t border-[#242A36] pt-4">
            <button id="btnCancelStudyModal" class="btn-technical text-xs">Cancelar</button>
            <button id="btnConfirmCreateStudy" class="btn-signal text-xs flex items-center space-x-1.5 shadow-md shadow-[#4F67FF]/20">
              <span>Crear y Configurar</span>
            </button>
          </div>
        </div>
      </div>
    `;
  }

  _bindEvents() {
    // Select active study
    document.getElementById("selectActiveStudy")?.addEventListener("change", async (e) => {
      this.selectedStudyId = e.target.value;
      this.renderLoading();
      await this.fetchData();
      this.render();
    });

    // Run study execution
    document.getElementById("btnRunStudyExecution")?.addEventListener("click", async () => {
      if (!this.selectedStudyId || this.runningStudy) return;
      this.runningStudy = true;
      this.render();
      try {
        await api.runAnalysisStudy(this.selectedStudyId);
        await this.fetchData();
      } catch (err) {
        console.error("Run study error:", err);
      } finally {
        this.runningStudy = false;
        this.render();
      }
    });

    // Modal open / close
    document.getElementById("btnOpenNewStudyModal")?.addEventListener("click", () => {
      this.showCreateModal = true;
      this.render();
    });
    document.getElementById("btnCreateInitialStudy")?.addEventListener("click", () => {
      this.showCreateModal = true;
      this.render();
    });
    document.getElementById("btnCloseStudyModal")?.addEventListener("click", () => {
      this.showCreateModal = false;
      this.render();
    });
    document.getElementById("btnCancelStudyModal")?.addEventListener("click", () => {
      this.showCreateModal = false;
      this.render();
    });

    // Confirm create study
    document.getElementById("btnConfirmCreateStudy")?.addEventListener("click", async () => {
      const name = document.getElementById("inputStudyName")?.value || "Estudio Exploratorio";
      const target = document.getElementById("inputTargetCol")?.value.trim() || null;
      try {
        const res = await api.createAnalysisStudy({
          dataset_id: this.activeDatasetId,
          name,
          target_column: target,
        });
        this.showCreateModal = false;
        if (res && res.study_id) {
          this.selectedStudyId = res.study_id;
        }
        this.renderLoading();
        await this.fetchData();
        this.render();
      } catch (err) {
        console.error("Create study error:", err);
      }
    });

    // Filters
    document.getElementById("filterCategorySelect")?.addEventListener("change", (e) => {
      this.filterCategory = e.target.value;
      this.render();
    });
    document.getElementById("filterSeveritySelect")?.addEventListener("change", (e) => {
      this.filterSeverity = e.target.value;
      this.render();
    });
    document.getElementById("checkFdrOnly")?.addEventListener("change", (e) => {
      this.filterFdrOnly = e.target.checked;
      this.render();
    });

    // Viz tab switches
    document.querySelectorAll(".btn-viz-tab").forEach(btn => {
      btn.addEventListener("click", (e) => {
        this.activeVizType = e.currentTarget.getAttribute("data-viz-type");
        this.render();
      });
    });

    // Matrix method toggle
    document.querySelectorAll(".btn-matrix-toggle").forEach(btn => {
      btn.addEventListener("click", (e) => {
        this.matrixMethod = e.currentTarget.getAttribute("data-method");
        this.render();
      });
    });

    // Export report
    document.getElementById("btnExportStudyReport")?.addEventListener("click", () => {
      this._downloadMarkdownReport();
    });

    // Verify hypothesis action
    document.querySelectorAll(".btn-verify-hyp").forEach(btn => {
      btn.addEventListener("click", async (e) => {
        const hypId = e.currentTarget.getAttribute("data-hyp-id");
        if (!hypId || this.verifyingHypothesisId) return;
        this.verifyingHypothesisId = hypId;
        this.render();
        try {
          await api.verifyHypothesis(hypId, null, 0.0);
          await this.fetchData();
        } catch (err) {
          console.error("Error verifying hypothesis:", err);
          alert("Error verificando hipótesis: " + (err.message || err));
        } finally {
          this.verifyingHypothesisId = null;
          this.render();
        }
      });
    });
  }

  _downloadMarkdownReport() {
    if (!this.studyDetail) return;
    const s = this.studyDetail;
    let md = `# Informe de Exploración Estadística · CATML Explore\n\n`;
    md += `**Estudio:** ${s.name} (\`${s.id}\`)\n`;
    md += `**Dataset:** ${s.data_source?.dataset_id || s.dataset_id}\n`;
    md += `**Variable Objetivo:** ${s.target_column || "Ninguna (No supervisado)"}\n`;
    md += `**Estado:** ${s.status}\n\n`;
    md += `## Hallazgos Estadísticos (${this.findings.length} detectados)\n\n`;

    this.findings.forEach((f, i) => {
      const fdrStr = f.metrics?.p_value_fdr != null ? ` (FDR p=${f.metrics.p_value_fdr})` : "";
      md += `### ${i + 1}. [${f.finding_type.toUpperCase()}] ${f.summary}\n`;
      md += `- **Variables:** \`${f.column_name || "-"}\` ${f.secondary_column ? " / `" + f.secondary_column + "`" : ""}\n`;
      md += `- **Método:** ${f.method_name}\n`;
      if (f.p_value != null) md += `- **P-valor crudo:** ${f.p_value}${fdrStr}\n`;
      if (f.effect_size != null) md += `- **Tamaño del efecto:** ${f.effect_size}\n`;
      md += `\n`;
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `informe_explore_${s.id}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }
}
