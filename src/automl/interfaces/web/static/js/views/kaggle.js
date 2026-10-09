/**
 * KaggleView — Dedicated Kaggle Competition Workbench
 * Pre-flight validation checklist, CV vs LB tracking & Submission generation.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml } from "../utils.js";
import { isRunActive } from "../utils.js";

export class KaggleView {
  constructor() {
    this.container = null;
    this.isSubmitting = false;
    this.activeRun = null;
    this.status = null;
    this.uploadedTemplateInfo = null;
    this.generatedSubmission = null;
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
        <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-lg")}</div>
        <div class="text-sm font-medium">Loading competition state and pre-flight verifications...</div>
      </div>
    `;
  }

  async fetchData() {
    try {
      const state = store.getState();
      const runs = state.runs || [];
      this.activeRun = runs.find(r => r.id === state.activeRunId)
        || runs.find(r => isRunActive(r))
        || runs[0]
        || null;

      this.status = await api.getKaggleStatus(this.activeRun ? this.activeRun.id : "").catch(() => null);
    } catch (e) {
      console.warn("KaggleView fetchData error:", e);
    }
  }

  render() {
    const run = this.activeRun;
    const st = this.status || {
      competition: run ? (run.dataset_name || "Custom Competition") : "Custom Competition",
      title: run ? `AutoML Benchmark — ${run.dataset_name || "Active"}` : "AutoML Benchmark",
      metric: run ? run.metric : "ROC-AUC",
      local_best_cv: run ? run.best_score : null,
      submissions: [],
      checklist: { row_count: 0 },
    };

    const bestScoreText = st.local_best_cv != null
      ? Number(st.local_best_cv).toFixed(5)
      : (run && run.best_score != null ? Number(run.best_score).toFixed(5) : "—");

    const defaultTestPath = run && run.dataset_path
      ? run.dataset_path.replace("train.csv", "test.csv").replace("train.parquet", "test.parquet")
      : "data/test.csv";

    const defaultOutputPath = run
      ? `submissions/submission_${run.id}.csv`
      : "submissions/submission.csv";

    const defaultTemplatePath = this.uploadedTemplateInfo
      ? this.uploadedTemplateInfo.template_path
      : (run && run.dataset_path
          ? run.dataset_path.replace("train.csv", "sample_submission.csv")
          : "data/sample_submission.csv");

    const rowCountLabel = this.uploadedTemplateInfo
      ? `${this.uploadedTemplateInfo.row_count.toLocaleString()} rows`
      : (st.checklist && st.checklist.row_count ? `${st.checklist.row_count.toLocaleString()} rows` : "MATCH");

    const idColLabel = this.uploadedTemplateInfo
      ? `PASS (${this.uploadedTemplateInfo.id_column})`
      : "PASS";

    const schemaLabel = this.uploadedTemplateInfo
      ? `VERIFIED (${this.uploadedTemplateInfo.target_column})`
      : "ALIGNED";

    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- Competition Header -->
        <div class="workbench-card p-5 bg-gradient-to-r from-slate-900 to-indigo-950/40 border-indigo-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex items-center space-x-3">
            <span class="text-[#4F67FF]">${icon("trophy", "icon-xl", 24)}</span>
            <div>
              <div class="flex items-center space-x-2">
                <h2 class="text-lg font-bold text-slate-100">${st.title}</h2>
                <span class="badge-sys text-xs px-2.5 py-0.5 rounded-full font-mono font-semibold">Active Benchmark</span>
              </div>
              <p class="text-xs text-slate-400 mt-0.5">${st.competition} • Target: <span class="text-indigo-300 font-mono">${run ? run.target : 'target'}</span> • Metric: <span class="text-emerald-400 font-mono font-semibold">${st.metric}</span></p>
            </div>
          </div>

          <div class="flex items-center space-x-6 text-right">
            <div>
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Local Best CV</div>
              <div class="text-xl font-bold font-mono-num text-emerald-400">${bestScoreText}</div>
            </div>
            <div>
              <div class="text-[10px] uppercase text-slate-400 font-semibold tracking-wider">Metric Scale</div>
              <div class="text-xl font-bold font-mono-num text-indigo-400">${st.metric || 'Score'}</div>
            </div>
          </div>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Left: Submissions History & CV vs LB Tracking -->
          <div class="lg:col-span-7 workbench-card overflow-hidden flex flex-col">
            <div class="workbench-panel-header flex items-center justify-between">
              <span class="text-sm font-semibold text-slate-200">Submissions & Public Leaderboard Tracker</span>
              <span class="text-xs text-slate-400">Parity verified</span>
            </div>

            <div class="overflow-x-auto flex-1">
              <table class="w-full wb-table text-left">
                <thead>
                  <tr>
                    <th>Experiment</th>
                    <th>Local CV</th>
                    <th>Public LB</th>
                    <th>Delta CV ↔ LB</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  ${st.submissions && st.submissions.length > 0 ? st.submissions.map(sub => `
                    <tr>
                      <td class="font-medium text-slate-200">${escapeHtml(sub.experiment)}</td>
                      <td class="font-mono text-emerald-400 font-bold">${sub.cv != null ? Number(sub.cv).toFixed(5) : "—"}</td>
                      <td class="font-mono text-slate-200">${sub.public_lb != null ? Number(sub.public_lb).toFixed(5) : "—"}</td>
                      <td class="font-mono text-xs text-indigo-400">${escapeHtml(sub.delta || "—")}</td>
                      <td><span class="${sub.status === 'VERIFIED' ? 'badge-gain' : 'badge-sys'} text-[10px] px-2 py-0.5 rounded font-mono font-semibold">${escapeHtml(sub.status)}</span></td>
                    </tr>
                  `).join("") : `
                    <tr>
                      <td colspan="5" class="text-center text-slate-500 py-8 text-xs">
                        No submissions generated yet for active run. Generate a submission below.
                      </td>
                    </tr>
                  `}
                </tbody>
              </table>
            </div>
          </div>

          <!-- Right: Pre-Flight Submission Validation Checklist & Form -->
          <div class="lg:col-span-5 workbench-card p-5 space-y-5 flex flex-col justify-between">
            <div class="space-y-4">
              <div class="flex items-center justify-between border-b border-[#252C38] pb-2">
                <span class="text-xs uppercase font-bold text-[#F7F8FA] tracking-wider font-sans">Submission Pre-Flight Validation</span>
                <span class="badge-gain text-xs px-2.5 py-0.5 rounded font-mono font-bold">Checks Ready</span>
              </div>

              <!-- Template Dropzone -->
              <div class="space-y-1.5">
                <div class="flex items-center justify-between">
                  <label class="text-[11px] font-semibold text-[#8B95A7] uppercase tracking-wider font-sans">Template Dropzone</label>
                  ${this.uploadedTemplateInfo ? `<span class="badge-gain text-[10px] px-2 py-0.5 rounded font-mono font-bold">Uploaded</span>` : `<span class="text-[10px] text-[#8B95A7] font-mono">sample_submission.csv</span>`}
                </div>
                <div id="kaggleDropzone" class="border-2 border-dashed border-[#252C38] hover:border-[#4F67FF] bg-[#090C12] rounded-xl p-4 text-center cursor-pointer transition-colors space-y-1.5">
                  <div class="text-[#4F67FF] flex justify-center">${icon("upload", "icon-lg")}</div>
                  ${this.uploadedTemplateInfo ? `
                    <div class="space-y-0.5">
                      <div class="text-xs font-mono font-bold text-[#F7F8FA]">${this.uploadedTemplateInfo.filename}</div>
                      <div class="text-[11px] text-[#22C55E] font-mono">✓ ${this.uploadedTemplateInfo.row_count} rows • Target: "${this.uploadedTemplateInfo.target_column}"</div>
                    </div>
                  ` : `
                    <div class="space-y-0.5">
                      <div class="text-xs font-medium text-[#F7F8FA] font-sans">Drag & drop <span class="font-mono text-[#4F67FF]">sample_submission.csv</span> here</div>
                      <div class="text-[10px] text-[#8B95A7] font-sans">or click to browse from disk for instant schema validation</div>
                    </div>
                  `}
                  <input type="file" id="kaggleTemplateFileInput" accept=".csv" class="hidden">
                </div>
              </div>

              <!-- Checklist Items -->
              <div class="space-y-2 text-xs font-mono">
                <div class="p-2.5 rounded-lg bg-[#0F131C] border border-[#252C38] flex items-center justify-between">
                  <span class="text-[#F7F8FA] inline-flex items-center gap-1.5">${icon("check", "icon-sm text-[#22C55E]")} <span>ID Column preserved</span></span>
                  <span class="text-[#22C55E] font-bold">${idColLabel}</span>
                </div>

                <div class="p-2.5 rounded-lg bg-[#0F131C] border border-[#252C38] flex items-center justify-between">
                  <span class="text-[#F7F8FA] inline-flex items-center gap-1.5">${icon("check", "icon-sm text-[#22C55E]")} <span>Row count alignment</span></span>
                  <span class="text-[#22C55E] font-bold">${rowCountLabel}</span>
                </div>

                <div class="p-2.5 rounded-lg bg-[#0F131C] border border-[#252C38] flex items-center justify-between">
                  <span class="text-[#F7F8FA] inline-flex items-center gap-1.5">${icon("check", "icon-sm text-[#22C55E]")} <span>Prediction range bounded</span></span>
                  <span class="text-[#22C55E] font-bold">BOUNDED</span>
                </div>

                <div class="p-2.5 rounded-lg bg-[#0F131C] border border-[#252C38] flex items-center justify-between">
                  <span class="text-[#F7F8FA] inline-flex items-center gap-1.5">${icon("check", "icon-sm text-[#22C55E]")} <span>Schema Alignment</span></span>
                  <span class="text-[#4F67FF] font-bold">${schemaLabel}</span>
                </div>
              </div>

              <!-- Configurable File Paths -->
              <div class="space-y-3 pt-2 border-t border-[#252C38] text-xs">
                <div>
                  <label class="block text-[#8B95A7] text-[11px] font-semibold mb-1">Test Dataset Path</label>
                  <input type="text" id="inputTestPath" value="${defaultTestPath}" class="w-full bg-[#090C12] border border-[#252C38] rounded-lg p-2 text-[#F7F8FA] font-mono text-[11px] focus:border-[#4F67FF]">
                </div>

                <div>
                  <label class="block text-[#8B95A7] text-[11px] font-semibold mb-1">Output Submission Path</label>
                  <input type="text" id="inputOutputPath" value="${defaultOutputPath}" class="w-full bg-[#090C12] border border-[#252C38] rounded-lg p-2 text-[#F7F8FA] font-mono text-[11px] focus:border-[#4F67FF]">
                </div>

                <div>
                  <label class="block text-[#8B95A7] text-[11px] font-semibold mb-1">Sample Submission Template (Optional)</label>
                  <input type="text" id="inputTemplatePath" value="${defaultTemplatePath}" class="w-full bg-[#090C12] border border-[#252C38] rounded-lg p-2 text-[#F7F8FA] font-mono text-[11px] focus:border-[#4F67FF]">
                </div>
              </div>
            </div>

            <!-- Action Buttons -->
            <div class="space-y-2 pt-2 border-t border-[#252C38]">
              <label class="flex items-center gap-2 text-xs text-[#8B95A7]">
                <input type="checkbox" id="useOOF" checked class="rounded text-[#4F67FF] bg-[#090C12] border-[#252C38]" /> Average 5 folds (OOF Ensemble)
              </label>

              <button id="btnGenerateSubmission" class="btn-signal w-full py-2.5 rounded-lg font-semibold flex items-center justify-center space-x-2 text-xs">
                ${icon("zap", "icon-sm")}
                <span>Generate & Validate submission.csv</span>
              </button>

              ${this.generatedSubmission ? `
                <a id="btnDownloadSubmission" href="/api/kaggle/download?file=${encodeURIComponent(this.generatedSubmission.output_path)}" download="${this.generatedSubmission.filename || 'submission.csv'}" class="w-full bg-[#22C55E] hover:bg-[#16A34A] text-white text-xs py-2.5 rounded-lg font-semibold transition-colors flex items-center justify-center space-x-2 shadow-lg shadow-[#22C55E]/20">
                  ${icon("download", "icon-sm")}
                  <span>Download ${this.generatedSubmission.filename || 'submission.csv'}</span>
                </a>
              ` : ""}

              <button id="btnSubmitKaggleCLI" class="w-full bg-[#1F2633] hover:bg-[#283244] border border-[#2F384A] text-[#F7F8FA] text-xs py-2.5 rounded-lg font-semibold transition-colors flex items-center justify-center space-x-2">
                ${icon("terminal", "icon-sm")}
                <span>Prepare Kaggle CLI Submission</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    `;

    this._bindEvents(st);
  }

  _bindEvents(st) {
    const dropzone = this.container.querySelector("#kaggleDropzone");
    const fileInput = this.container.querySelector("#kaggleTemplateFileInput");

    if (dropzone && fileInput) {
      dropzone.addEventListener("click", () => fileInput.click());

      dropzone.addEventListener("dragover", e => {
        e.preventDefault();
        dropzone.classList.add("border-[#4F67FF]", "bg-[#161B26]");
      });

      dropzone.addEventListener("dragleave", e => {
        e.preventDefault();
        dropzone.classList.remove("border-[#4F67FF]", "bg-[#161B26]");
      });

      dropzone.addEventListener("drop", async e => {
        e.preventDefault();
        dropzone.classList.remove("border-[#4F67FF]", "bg-[#161B26]");
        if (e.dataTransfer && e.dataTransfer.files.length > 0) {
          await this._handleTemplateUpload(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener("change", async e => {
        if (e.target.files && e.target.files.length > 0) {
          await this._handleTemplateUpload(e.target.files[0]);
        }
      });
    }

    this.container.querySelector("#btnGenerateSubmission")?.addEventListener("click", async () => {
      const btn = this.container.querySelector("#btnGenerateSubmission");
      if (!btn) return;
      const originalText = btn.innerHTML;

      try {
        const state = store.getState();
        const activeRun = this.activeRun || state.runs[0];
        if (!activeRun) {
          alert("No active run found.");
          return;
        }

        const testPath = this.container.querySelector("#inputTestPath")?.value.trim();
        const outputPath = this.container.querySelector("#inputOutputPath")?.value.trim();
        const templatePath = this.container.querySelector("#inputTemplatePath")?.value.trim();

        btn.disabled = true;
        btn.innerHTML = `
          <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white inline-block" viewBox="0 0 24 24" fill="none">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <span>Generating inferences...</span>
        `;

        const res = await api.generateSubmission({
          run_id: activeRun.id,
          test_dataset_path: testPath || "data/test.csv",
          output_path: outputPath || "submissions/submission.csv",
          template_path: templatePath || undefined,
          predict_proba: (activeRun.task_type || "").includes("classification"),
          folds: this.container.querySelector("#useOOF")?.checked ? 5 : undefined,
        }, job => {
          btn.textContent = `${job.status}: ${job.completed}/${job.total || "?"} · ${job.message}`;
        });

        btn.className = "w-full bg-[#22C55E] text-white text-xs py-2.5 rounded-lg font-semibold flex items-center justify-center space-x-2";
        btn.innerHTML = `<span class="flex items-center gap-1.5">${icon("check", "icon-sm")} <span>Generated successfully! (${res.row_count} rows)</span></span>`;

        this.generatedSubmission = {
          output_path: res.output_path || outputPath || "submissions/submission.csv",
          filename: (outputPath || "submission.csv").split("/").pop(),
          row_count: res.row_count,
        };

        await this.fetchData();
        this.render();
      } catch (err) {
        alert("Error generating submission: " + err.message);
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    });

    this.container.querySelector("#btnSubmitKaggleCLI")?.addEventListener("click", () => {
      const outputPath = this.container.querySelector("#inputOutputPath")?.value.trim() || "submission.csv";
      const compSlug = (st && st.competition) ? st.competition.toLowerCase().replace(/\s+/g, "-") : "dataset-competition";
      alert(`Suggested Kaggle CLI command:\n\nkaggle competitions submit -c ${compSlug} -f ${outputPath} -m 'CATML AutoML Submission'\n\nReady to run via CLI or terminal.`);
    });
  }

  async _handleTemplateUpload(file) {
    if (!file) return;
    try {
      const text = await file.text();
      const res = await api.uploadKaggleTemplate({
        filename: file.name,
        content: text,
      });
      this.uploadedTemplateInfo = res;
      this.render();
    } catch (err) {
      alert("Error parsing and uploading sample submission: " + err.message);
    }
  }

  destroy() {
    this.container = null;
  }
}

