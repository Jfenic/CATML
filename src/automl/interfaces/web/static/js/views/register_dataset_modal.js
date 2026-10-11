/**
 * RegisterDatasetModal — Polymorphic Dataset Ingestion Modal
 * Supports:
 * 1. Drag & Drop file upload directly into workspace
 * 2. Interactive file browser in data/ and workspace directories
 * 3. Manual path entry with automated column inspection
 * 4. Smart target and problem type detection
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";
import { icon } from "../icons.js";
import { escapeHtml } from "../utils.js";

export class RegisterDatasetModal {
  constructor() {
    this.container = null;
    this.activeTab = "dragdrop"; // 'dragdrop' | 'browse' | 'manual'
    this.browseDir = "data";
    this.browseData = { current_dir: "", parent_dir: null, folders: [], files: [] };
    this.browseSearch = "";
    this.isUploading = false;
    this.isInspecting = false;
    this.isSubmitting = false;
    this.selectedFile = null; // { path, filename, size_formatted, columns, suggested_target, suggested_name, total_rows }
    this.errorMessage = null;
  }

  async mount(container) {
    this.container = container;
    this.selectedFile = null;
    this.errorMessage = null;
    this.activeTab = "dragdrop";
    this.render();
    if (this.activeTab === "browse") {
      await this.loadBrowseDirectory(this.browseDir);
    }
  }

  async loadBrowseDirectory(dirPath = "", search = "") {
    try {
      this.browseData = await api.browseFiles(dirPath, search);
      this.renderBrowseList();
    } catch (err) {
      console.warn("Could not load directory:", err);
      this.browseData = { current_dir: dirPath, parent_dir: null, folders: [], files: [] };
      this.renderBrowseList();
    }
  }

  render() {
    if (!this.container) return;

    this.container.innerHTML = `
      <div class="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
        <div class="workbench-card max-w-2xl w-full p-6 space-y-5 border-[#242A36] shadow-2xl bg-[#0D1017]">
          <!-- Header -->
          <div class="flex items-center justify-between border-b border-[#242A36] pb-3">
            <div>
              <div class="flex items-center space-x-2">
                <span class="w-1.5 h-1.5 rounded-full bg-[#4F67FF]"></span>
                <span class="text-[10px] font-mono text-[#8B95A7] uppercase tracking-wider">Ingestión de Datos</span>
              </div>
              <h3 class="text-base font-bold text-[#F7F8FA] font-sans mt-0.5">Añadir Dataset al Laboratorio</h3>
            </div>
            <button id="btnCloseRegModal" class="text-[#8B95A7] hover:text-[#F7F8FA] p-1.5 rounded-md hover:bg-[#161B26] transition-colors">
              ${icon("x", "icon-sm")}
            </button>
          </div>

          <!-- Error Alert Banner -->
          <div id="regModalError" class="p-3 rounded-lg bg-[#EF4444]/10 border border-[#EF4444]/30 text-xs text-[#EF4444] font-sans ${this.errorMessage ? '' : 'hidden'}">
            ${escapeHtml(this.errorMessage || "")}
          </div>

          <!-- Tabs de Método de Ingestión -->
          <div class="flex items-center space-x-2 border-b border-[#242A36] pb-2 text-xs font-sans">
            <button class="ingest-tab-btn px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center space-x-1.5 ${this.activeTab === 'dragdrop' ? 'bg-[#161B26] text-[#F7F8FA] border border-[#242A36]' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="dragdrop">
              ${icon("download", "icon-sm")}
              <span>Arrastrar archivo</span>
            </button>
            <button class="ingest-tab-btn px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center space-x-1.5 ${this.activeTab === 'browse' ? 'bg-[#161B26] text-[#F7F8FA] border border-[#242A36]' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="browse">
              ${icon("search", "icon-sm")}
              <span>Explorar en data/</span>
            </button>
            <button class="ingest-tab-btn px-3 py-1.5 rounded-lg font-medium transition-colors flex items-center space-x-1.5 ${this.activeTab === 'manual' ? 'bg-[#161B26] text-[#F7F8FA] border border-[#242A36]' : 'text-[#8B95A7] hover:text-[#F7F8FA]'}" data-tab="manual">
              ${icon("scroll-text", "icon-sm")}
              <span>Ruta manual</span>
            </button>
          </div>

          <!-- Contenido de la pestaña activa -->
          <div id="ingestTabBody">
            ${this._renderTabBody()}
          </div>

          <!-- Paso 2: Configuración del Dataset (aparece al seleccionar un archivo) -->
          <div id="step2ConfigBlock" class="${this.selectedFile ? '' : 'hidden'} border-t border-[#242A36] pt-4 space-y-4">
            ${this._renderStep2Config()}
          </div>

          <!-- Footer Actions -->
          <div class="flex items-center justify-between pt-3 border-t border-[#242A36] text-xs">
            <button id="btnCancelRegModal" class="text-[#8B95A7] hover:text-[#F7F8FA] px-3 py-1.5 font-sans transition-colors">
              Cancelar
            </button>
            <button id="btnConfirmRegisterDataset" class="btn-signal text-xs flex items-center space-x-1.5 ${this.selectedFile ? '' : 'opacity-50 pointer-events-none'}" ${this.selectedFile ? '' : 'disabled'}>
              ${icon("plus", "icon-sm")}
              <span>Registrar en el Laboratorio</span>
            </button>
          </div>
        </div>
      </div>
    `;

    this._bindEvents();
    if (this.activeTab === "browse") {
      this.loadBrowseDirectory(this.browseDir, this.browseSearch);
    }
  }

  _renderTabBody() {
    if (this.activeTab === "dragdrop") {
      return `
        <div class="space-y-3">
          <div id="dropZoneArea" class="border-2 border-dashed border-[#242A36] hover:border-[#4F67FF]/60 hover:bg-[#11151E] rounded-xl p-8 text-center transition-all cursor-pointer space-y-3">
            <div class="inline-block p-3 rounded-full bg-[#161B26] text-[#4F67FF]">
              ${icon("download", "icon-lg")}
            </div>
            <div class="space-y-1">
              <div class="text-sm font-semibold font-sans text-[#F7F8FA]">
                Arrastra y suelta tu dataset aquí
              </div>
              <p class="text-xs text-[#8B95A7] font-sans max-w-sm mx-auto">
                Soporta archivos <span class="font-mono text-[#F7F8FA]">.csv</span>, <span class="font-mono text-[#F7F8FA]">.parquet</span>, <span class="font-mono text-[#F7F8FA]">.tsv</span>, <span class="font-mono text-[#F7F8FA]">.json</span> y <span class="font-mono text-[#F7F8FA]">.jsonl</span>
              </p>
            </div>
            <div>
              <button id="btnTriggerFileInput" type="button" class="bg-[#161B26] hover:bg-[#1E2536] text-[#F7F8FA] border border-[#242A36] px-3 py-1.5 rounded-lg text-xs font-sans font-medium transition-colors">
                Examinar archivos de tu equipo
              </button>
              <input type="file" id="hiddenFileInput" class="hidden" accept=".csv,.parquet,.pq,.tsv,.json,.jsonl">
            </div>
          </div>

          <!-- Uploading progress -->
          <div id="uploadProgress" class="${this.isUploading ? '' : 'hidden'} p-3 rounded-lg bg-[#11151E] border border-[#242A36] text-xs font-sans text-center space-y-2">
            <div class="animate-spin text-[#4F67FF] inline-block">${icon("refresh-cw", "icon-sm")}</div>
            <div class="text-[#8B95A7]">Subiendo e inspeccionando dataset...</div>
          </div>
        </div>
      `;
    }

    if (this.activeTab === "browse") {
      return `
        <div class="space-y-3">
          <div class="flex items-center space-x-2">
            <div class="relative flex-1">
              <input id="browseSearchInput" type="text" placeholder="Filtrar archivos en data/... (ej: train.csv)" value="${escapeHtml(this.browseSearch)}" class="w-full bg-[#11151E] border border-[#242A36] text-[#F7F8FA] rounded-lg pl-8 pr-3 py-1.5 text-xs font-mono outline-none focus:border-[#4F67FF]">
              <span class="absolute left-2.5 top-2 text-[#8B95A7]">${icon("search", "icon-sm")}</span>
            </div>
            <button id="btnBrowseSearch" class="btn-technical text-xs py-1.5 px-3">Buscar</button>
          </div>

          <!-- Directory header & Parent nav -->
          <div class="flex items-center justify-between text-[11px] font-mono text-[#8B95A7] px-1">
            <div class="truncate flex items-center space-x-1.5">
              <span>Directorio:</span>
              <span class="text-[#F7F8FA] font-medium" id="currentBrowseDirLabel">${escapeHtml(this.browseData.rel_dir || this.browseDir || "data")}</span>
            </div>
            <button id="btnBrowseUpDir" class="text-[#4F67FF] hover:underline flex items-center space-x-1 ${this.browseData.parent_dir ? '' : 'opacity-40 pointer-events-none'}">
              <span>↑ Subir nivel</span>
            </button>
          </div>

          <!-- Items list container -->
          <div id="browseItemsContainer" class="max-h-56 overflow-y-auto space-y-1.5 pr-1 text-xs">
            <div class="p-4 text-center text-[#8B95A7] font-sans text-xs">Cargando archivos del workspace...</div>
          </div>
        </div>
      `;
    }

    if (this.activeTab === "manual") {
      return `
        <div class="space-y-3">
          <div class="space-y-1.5">
            <label class="block text-xs font-sans font-medium text-[#8B95A7]">Ruta del archivo en el sistema</label>
            <div class="flex items-center space-x-2">
              <input id="inputManualPath" type="text" placeholder="/ruta/absoluta/o/relativa/a/datos.csv" class="flex-1 bg-[#11151E] border border-[#242A36] text-[#F7F8FA] rounded-lg px-3 py-2 text-xs font-mono outline-none focus:border-[#4F67FF]">
              <button id="btnInspectManualPath" class="btn-technical text-xs py-2 px-3 flex items-center space-x-1.5">
                <span>Inspeccionar</span>
              </button>
            </div>
            <p class="text-[11px] text-[#8B95A7] font-sans">
              Introduce la ruta exacta del archivo (ejemplo: <code class="font-mono text-[#4F67FF]">data/train.csv</code> o <code class="font-mono text-[#4F67FF]">competitions/playground-series-s6e9/data/train.csv</code>).
            </p>
          </div>
        </div>
      `;
    }

    return "";
  }

  _renderStep2Config() {
    if (!this.selectedFile) return "";

    const sf = this.selectedFile;
    const cols = sf.columns || [];
    const suggestedTarget = sf.suggested_target || "";
    const suggestedName = sf.suggested_name || "dataset";

    return `
      <div class="p-3.5 rounded-xl bg-[#11151E] border border-[#242A36] space-y-3">
        <!-- Selected file pill -->
        <div class="flex items-center justify-between border-b border-[#242A36] pb-2.5">
          <div class="flex items-center space-x-2.5 truncate">
            <span class="text-[#22C55E]">${icon("check-circle-2", "icon-sm")}</span>
            <div class="truncate">
              <div class="font-sans font-semibold text-xs text-[#F7F8FA] truncate">${escapeHtml(sf.filename || sf.path)}</div>
              <div class="text-[10px] font-mono text-[#8B95A7] truncate">${escapeHtml(sf.path)}</div>
            </div>
          </div>
          <div class="text-right text-[11px] font-mono text-[#8B95A7] whitespace-nowrap pl-2">
            ${sf.size_formatted ? `<span>${sf.size_formatted}</span> • ` : ''}
            <span class="text-[#4F67FF]">${cols.length} columnas</span>
          </div>
        </div>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-sans">
          <!-- Nombre del Dataset -->
          <div class="space-y-1">
            <label class="block text-[11px] font-medium text-[#8B95A7]">Nombre en el Laboratorio</label>
            <input id="inputDatasetName" type="text" value="${escapeHtml(suggestedName)}" class="w-full bg-[#161B26] border border-[#242A36] text-[#F7F8FA] rounded-lg px-3 py-1.5 font-sans outline-none focus:border-[#4F67FF]">
          </div>

          <!-- Variable Objetivo (Target) -->
          <div class="space-y-1">
            <label class="block text-[11px] font-medium text-[#8B95A7]">Variable Objetivo (Target)</label>
            <select id="selectTargetCol" class="w-full bg-[#161B26] border border-[#242A36] text-[#F7F8FA] rounded-lg px-3 py-1.5 font-mono outline-none focus:border-[#4F67FF]">
              <option value="">— Sin target (Aprendizaje No Supervisado) —</option>
              ${cols.map(c => `
                <option value="${escapeHtml(c)}" ${c === suggestedTarget ? 'selected' : ''}>${escapeHtml(c)} ${c === suggestedTarget ? '(Sugerido)' : ''}</option>
              `).join("")}
            </select>
          </div>

          <!-- Tipo de Problema -->
          <div class="space-y-1 sm:col-span-2">
            <label class="block text-[11px] font-medium text-[#8B95A7]">Tipo de Tarea ML</label>
            <select id="selectTaskType" class="w-full bg-[#161B26] border border-[#242A36] text-[#F7F8FA] rounded-lg px-3 py-1.5 font-sans outline-none focus:border-[#4F67FF]">
              <option value="">Auto-inferir tipo de problema automáticamente</option>
              <option value="binary_classification">Clasificación Binaria</option>
              <option value="multiclass_classification">Clasificación Multiclase</option>
              <option value="regression">Regresión</option>
              <option value="clustering">Clustering / No supervisado</option>
            </select>
          </div>
        </div>
      </div>
    `;
  }

  renderBrowseList() {
    const container = this.container?.querySelector("#browseItemsContainer");
    if (!container) return;

    const folders = this.browseData.folders || [];
    const files = this.browseData.files || [];

    if (folders.length === 0 && files.length === 0) {
      container.innerHTML = `
        <div class="p-6 text-center text-[#8B95A7] font-sans text-xs bg-[#11151E] rounded-xl border border-[#242A36]">
          No se encontraron carpetas ni archivos compatibles (.csv, .parquet, .tsv, .json) en esta ruta.
        </div>
      `;
      return;
    }

    let html = "";

    // Folders first
    folders.forEach(f => {
      html += `
        <div class="browse-folder-row p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] hover:border-[#384355] hover:bg-[#161B26] transition-colors flex items-center justify-between cursor-pointer" data-folder-path="${escapeHtml(f.path)}">
          <div class="flex items-center space-x-2.5 truncate">
            <span class="text-[#F59E0B]">${icon("workflow", "icon-sm")}</span>
            <span class="font-sans font-medium text-xs text-[#F7F8FA] truncate">${escapeHtml(f.name)}/</span>
          </div>
          <span class="text-[10px] font-mono text-[#8B95A7]">Abrir carpeta →</span>
        </div>
      `;
    });

    // Files next
    files.forEach(f => {
      html += `
        <div class="browse-file-row p-2.5 rounded-lg bg-[#11151E] border border-[#242A36] hover:border-[#4F67FF]/60 hover:bg-[#161B26] transition-colors flex items-center justify-between cursor-pointer" data-file-path="${escapeHtml(f.path)}" data-file-name="${escapeHtml(f.name)}">
          <div class="flex items-center space-x-2.5 truncate">
            <span class="text-[#4F67FF]">${icon("database", "icon-sm")}</span>
            <div class="truncate">
              <span class="font-sans font-medium text-xs text-[#F7F8FA] truncate">${escapeHtml(f.name)}</span>
              <span class="text-[10px] font-mono text-[#8B95A7] ml-2">${escapeHtml(f.size_formatted || "")}</span>
            </div>
          </div>
          <button class="btn-select-file px-2.5 py-1 rounded-md text-[11px] font-sans font-medium bg-[#161B26] text-[#4F67FF] border border-[#4F67FF]/30 hover:bg-[#4F67FF] hover:text-white transition-colors" data-file-path="${escapeHtml(f.path)}">
            Seleccionar
          </button>
        </div>
      `;
    });

    container.innerHTML = html;

    // Bind clicks
    container.querySelectorAll(".browse-folder-row").forEach(row => {
      row.addEventListener("click", () => {
        const p = row.getAttribute("data-folder-path");
        if (p) this.loadBrowseDirectory(p, this.browseSearch);
      });
    });

    container.querySelectorAll(".browse-file-row, .btn-select-file").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const p = btn.getAttribute("data-file-path");
        if (p) this.inspectAndSelectPath(p);
      });
    });
  }

  async inspectAndSelectPath(path) {
    this.errorMessage = null;
    this.isInspecting = true;
    try {
      const resp = await api.inspectFile(path);
      this.selectedFile = {
        path: resp.path,
        filename: resp.suggested_name,
        size_formatted: resp.size_formatted,
        columns: resp.columns || [],
        suggested_target: resp.suggested_target,
        suggested_name: resp.suggested_name,
        total_rows: resp.total_rows,
      };
      this.render();
    } catch (err) {
      this.errorMessage = `Error al inspeccionar archivo: ${err.message || err}`;
      this.render();
    } finally {
      this.isInspecting = false;
    }
  }

  async handleFileUpload(file) {
    if (!file) return;
    this.errorMessage = null;
    this.isUploading = true;
    const progressEl = this.container?.querySelector("#uploadProgress");
    if (progressEl) progressEl.classList.remove("hidden");

    try {
      const reader = new FileReader();
      const base64Promise = new Promise((resolve, reject) => {
        reader.onload = () => {
          const res = reader.result;
          const base64 = res.split(",")[1];
          resolve(base64);
        };
        reader.onerror = reject;
      });
      reader.readAsDataURL(file);
      const b64 = await base64Promise;

      const uploadResp = await api.uploadDataset(file.name, b64);
      this.selectedFile = {
        path: uploadResp.path,
        filename: uploadResp.filename,
        size_formatted: uploadResp.size_formatted,
        columns: uploadResp.columns || [],
        suggested_target: uploadResp.suggested_target,
        suggested_name: uploadResp.suggested_name,
      };
      this.render();
    } catch (err) {
      this.errorMessage = `Error al subir archivo: ${err.message || err}`;
      this.render();
    } finally {
      this.isUploading = false;
    }
  }

  async submitRegistration() {
    if (!this.selectedFile) return;

    const nameInput = this.container.querySelector("#inputDatasetName");
    const targetSelect = this.container.querySelector("#selectTargetCol");
    const taskSelect = this.container.querySelector("#selectTaskType");

    const name = (nameInput?.value || this.selectedFile.suggested_name || "dataset").trim();
    const target = targetSelect?.value || null;
    const task_type = taskSelect?.value || null;

    const submitBtn = this.container.querySelector("#btnConfirmRegisterDataset");
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.innerHTML = `${icon("refresh-cw", "icon-sm animate-spin")} <span>Registrando y perfilando...</span>`;
    }

    try {
      const resp = await api.registerDataset({
        name,
        path: this.selectedFile.path,
        target: target || null,
        task_type: task_type || null,
      });

      if (resp && resp.dataset_id) {
        store.setActiveDataset(resp.dataset_id);
        this.close();
        bus.emit("dataset:switched", resp.dataset_id);
        store.setNav("dataset");
      } else {
        throw new Error(resp.error || "No se recibió ID de dataset");
      }
    } catch (err) {
      this.errorMessage = `Error al registrar dataset: ${err.message || err}`;
      this.render();
    }
  }

  _bindEvents() {
    const close = () => this.close();
    this.container.querySelector("#btnCloseRegModal")?.addEventListener("click", close);
    this.container.querySelector("#btnCancelRegModal")?.addEventListener("click", close);

    // Tab buttons
    this.container.querySelectorAll(".ingest-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const tab = btn.getAttribute("data-tab");
        if (tab && tab !== this.activeTab) {
          this.activeTab = tab;
          this.render();
        }
      });
    });

    // Drag and Drop Zone
    const dropZone = this.container.querySelector("#dropZoneArea");
    if (dropZone) {
      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("border-[#4F67FF]", "bg-[#4F67FF]/10");
      });
      dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-[#4F67FF]", "bg-[#4F67FF]/10");
      });
      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-[#4F67FF]", "bg-[#4F67FF]/10");
        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
          this.handleFileUpload(files[0]);
        }
      });

      // File input trigger
      const hiddenInput = this.container.querySelector("#hiddenFileInput");
      this.container.querySelector("#btnTriggerFileInput")?.addEventListener("click", () => {
        hiddenInput?.click();
      });
      hiddenInput?.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // Browse search
    this.container.querySelector("#btnBrowseSearch")?.addEventListener("click", () => {
      const q = this.container.querySelector("#browseSearchInput")?.value || "";
      this.browseSearch = q;
      this.loadBrowseDirectory(this.browseDir, q);
    });

    this.container.querySelector("#browseSearchInput")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        this.browseSearch = e.target.value || "";
        this.loadBrowseDirectory(this.browseDir, this.browseSearch);
      }
    });

    this.container.querySelector("#btnBrowseUpDir")?.addEventListener("click", () => {
      if (this.browseData.parent_dir) {
        this.loadBrowseDirectory(this.browseData.parent_dir, this.browseSearch);
      }
    });

    // Manual inspect
    this.container.querySelector("#btnInspectManualPath")?.addEventListener("click", () => {
      const p = this.container.querySelector("#inputManualPath")?.value?.trim();
      if (p) this.inspectAndSelectPath(p);
    });

    this.container.querySelector("#inputManualPath")?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const p = e.target.value?.trim();
        if (p) this.inspectAndSelectPath(p);
      }
    });

    // Submit button
    this.container.querySelector("#btnConfirmRegisterDataset")?.addEventListener("click", () => {
      this.submitRegistration();
    });
  }

  close() {
    if (this.container) {
      this.container.innerHTML = "";
    }
  }
}
