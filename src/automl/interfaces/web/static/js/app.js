/**
 * App — Main Application Orchestrator & Router
 * Implements View Controller, Observer, and EventBus patterns.
 */
import { store } from "./store.js";
import { bus } from "./bus.js";
import { api } from "./api.js";
import { JobsPanel } from "./views/jobs.js";
import { isRunActive, isJobActive } from "./utils.js";

import { OverviewView } from "./views/overview.js";
import { StudioView } from "./views/studio.js";
import { DatasetsView } from "./views/datasets.js";
import { CompareView } from "./views/compare.js";
import { PipelineView } from "./views/pipeline.js";
import { KnowledgeView } from "./views/knowledge.js";
import { KaggleView } from "./views/kaggle.js";
import { AgentDrawer } from "./views/agent.js";
import { NewExperimentModal } from "./views/new_experiment.js";
import { HomeView } from "./views/home.js";
import { SwitchDatasetModal } from "./views/switch_dataset_modal.js";
import { RegisterDatasetModal } from "./views/register_dataset_modal.js";
import { ExploreView } from "./views/explore.js";

class App {
  constructor() {
    this.currentViewInstance = null;
    this.mainCanvas = document.getElementById("mainContent");
    this.agentDrawerContainer = document.getElementById("agentDrawer");
    this.modalContainer = document.getElementById("modalContainer");

    this.agentDrawer = new AgentDrawer();
    this.newExperimentModal = new NewExperimentModal();
    this.switchDatasetModal = new SwitchDatasetModal();
    this.registerDatasetModal = new RegisterDatasetModal();
  }

  async init() {
    this.jobsPanel = new JobsPanel();
    this.jobsPanel.mount();
    this._bindNavigation();
    this._bindGlobalEvents();
    this.agentDrawer.mount(this.agentDrawerContainer);

    // Initial load from backend
    await this._fetchInitialState();

    // Subscribe to store updates
    store.subscribe((state, prevState) => {
      if (state.currentNav !== prevState.currentNav) {
        this._routeTo(state.currentNav);
      }
      if (state.agentDrawerOpen !== prevState.agentDrawerOpen) {
        this._updateAgentDrawer(state.agentDrawerOpen);
      }
      if (state.activeDatasetId !== prevState.activeDatasetId || state.activeDataset !== prevState.activeDataset) {
        this._updateSidebarActiveDataset(state.activeDataset);
      }
    });

    // Initial render
    this._routeTo(store.getState().currentNav);
  }

  _updateSidebarActiveDataset(activeDataset) {
    const nameEl = document.getElementById("sidebarActiveDatasetName");
    const badgeEl = document.getElementById("sidebarDatasetBadge");
    if (nameEl) {
      if (activeDataset) {
        nameEl.textContent = activeDataset.name || activeDataset.id;
        nameEl.title = `${activeDataset.name} (${activeDataset.id})`;
        if (badgeEl) badgeEl.className = "w-1.5 h-1.5 rounded-full bg-[#22C55E]";
      } else {
        nameEl.textContent = "Sin dataset seleccionado";
        nameEl.title = "Ninguno";
        if (badgeEl) badgeEl.className = "w-1.5 h-1.5 rounded-full bg-[#8B95A7]";
      }
    }
  }

  async _fetchInitialState() {
    try {
      const [overview, runs, datasets] = await Promise.all([
        api.getOverview().catch(() => null),
        api.getRuns().catch(() => []),
        api.getDatasets().catch(() => []),
      ]);

      const { activeDataset, activeRun } = store.resolveContext({ datasets, runs, overview });
      this._updateSidebarActiveDataset(activeDataset);

      if (overview) {
        if (overview.workspace) {
          const wsEl = document.getElementById("workspaceLabel");
          if (wsEl) wsEl.textContent = overview.workspace;
        }
        if (overview.version) {
          const vLabel = document.getElementById("sidebarVersionLabel");
          if (vLabel) vLabel.textContent = `v${overview.version}`;
          const aBadge = document.getElementById("sidebarAgentBadge");
          if (aBadge) aBadge.textContent = `v${overview.version}`;
        }
      }

      // Dynamic Project Name Label
      const projectEl = document.getElementById("projectNameLabel");
      if (projectEl) {
        if (activeDataset) {
          projectEl.textContent = activeRun ? `${activeDataset.name} (${activeRun.id})` : activeDataset.name;
        } else {
          projectEl.textContent = "Espacio AutoML";
        }
      }

      // Dynamic Kaggle Nav Item & Badge
      const kaggleLabel = document.getElementById("kaggleNavLabel");
      const kaggleBadge = document.getElementById("kaggleNavBadge");
      if (kaggleLabel) {
        kaggleLabel.textContent = activeDataset ? activeDataset.name : "Kaggle & Envíos";
      }
      if (kaggleBadge) {
        if (activeRun && activeRun.best_score != null) {
          kaggleBadge.textContent = Number(activeRun.best_score).toFixed(4);
        } else {
          kaggleBadge.textContent = "—";
        }
      }

      const isAnyRunning = (this.jobsPanel?.jobs || []).some(j => isJobActive(j))
        || (runs || []).some(r => isRunActive(r))
        || Boolean(overview && overview.is_active);
      const statusBadge = document.getElementById("globalStatusBadge");
      const sysHw = document.getElementById("sysHardwareHeader");
      if (sysHw) {
        sysHw.textContent = isAnyRunning ? "Entrenamiento AutoML en curso" : "Motor AutoML listo";
      }

      if (statusBadge) {
        if (isAnyRunning) {
          statusBadge.className = "badge-warn px-2.5 py-0.5 rounded-md font-mono text-[11px] font-medium flex items-center space-x-1.5";
          statusBadge.title = "Hay un entrenamiento o trabajo activo";
          statusBadge.innerHTML = `<svg class="animate-spin h-3 w-3 text-[#F59E0B] inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span>ENTRENANDO</span>`;
        } else {
          // Global badge = workspace readiness, not experiment entity status.
          // Avoid showing run COMPLETED next to experiment CREATED (different entities).
          const runStatus = activeRun && activeRun.status ? String(activeRun.status).toUpperCase() : "";
          let label = "LISTO";
          let tip = "Motor AutoML listo";
          if (runStatus === "FAILED") {
            label = "ERROR";
            tip = `Ejecución ${activeRun.id}: fallida (estado de run, no de experimento)`;
          } else if (runStatus === "COMPLETED" || runStatus === "READY") {
            label = "LISTO";
            tip = activeRun
              ? `Ejecución ${activeRun.id}: completada · los experimentos tienen su propio estado`
              : tip;
          } else if (runStatus) {
            label = runStatus;
            tip = `Ejecución ${activeRun.id}: ${runStatus}`;
          }
          statusBadge.title = tip;
          statusBadge.className = "badge-gain px-2.5 py-0.5 rounded-md font-mono text-[11px] font-medium flex items-center space-x-1.5";
          statusBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span><span>${label}</span>`;
        }
      }
    } catch (e) {
      console.warn("Could not load initial server state, using offline defaults:", e);
    }
  }

  _bindNavigation() {
    const sidebar = document.getElementById("sidebarNav");
    const toggleMobileBtn = document.getElementById("btnToggleMobileSidebar");

    if (toggleMobileBtn && sidebar) {
      toggleMobileBtn.addEventListener("click", () => {
        sidebar.classList.toggle("hidden");
        sidebar.classList.toggle("absolute");
        sidebar.classList.toggle("z-30");
        sidebar.classList.toggle("h-[calc(100vh-53px)]");
      });
    }

    document.querySelectorAll(".nav-item").forEach(item => {
      item.addEventListener("click", e => {
        e.preventDefault();
        const navId = item.getAttribute("data-nav");
        if (sidebar && sidebar.classList.contains("absolute")) {
          sidebar.classList.add("hidden");
          sidebar.classList.remove("absolute", "z-30", "h-[calc(100vh-53px)]");
        }
        if (navId === "agent") {
          store.toggleAgentDrawer(true);
        } else {
          store.setNav(navId);
        }
      });
    });

    document.getElementById("btnSidebarSwitchDataset")?.addEventListener("click", () => {
      this.switchDatasetModal.mount(this.modalContainer);
    });

    document.getElementById("btnToggleAgent")?.addEventListener("click", () => {
      store.toggleAgentDrawer();
    });

    document.getElementById("btnRefresh")?.addEventListener("click", async () => {
      await this._fetchInitialState();
      this._routeTo(store.getState().currentNav);
    });
  }

  _bindGlobalEvents() {
    bus.on("modal:new-experiment", (opts) => {
      this.newExperimentModal.mount(this.modalContainer, opts);
    });

    bus.on("modal:switch-dataset", () => {
      this.switchDatasetModal.mount(this.modalContainer);
    });

    bus.on("modal:register-dataset", () => {
      this.registerDatasetModal.mount(this.modalContainer);
    });

    bus.on("dataset:switched", (datasetId) => {
      store.setActiveDataset(datasetId);
      const activeDataset = store.getActiveDataset();
      this._updateSidebarActiveDataset(activeDataset);
      if (this.agentDrawer) this.agentDrawer.refresh();
      this._routeTo(store.getState().currentNav);
    });

    bus.on("run:select", runId => {
      store.setActiveRun(runId);
      const activeDataset = store.getActiveDataset();
      this._updateSidebarActiveDataset(activeDataset);
      if (this.agentDrawer) this.agentDrawer.refresh();
      store.setNav("experiments");
    });
  }

  _updateAgentDrawer(isOpen) {
    if (isOpen) {
      this.agentDrawerContainer.classList.remove("translate-x-full");
      this.agentDrawerContainer.classList.add("translate-x-0");
    } else {
      this.agentDrawerContainer.classList.add("translate-x-full");
      this.agentDrawerContainer.classList.remove("translate-x-0");
    }
  }

  _routeTo(navId) {
    const canonicalNav = (navId === "datasets") ? "dataset"
      : (navId === "studio") ? "experiments"
      : (navId === "overview") ? "evidence"
      : navId;

    // Update active class on sidebar items
    document.querySelectorAll(".nav-item").forEach(item => {
      const id = item.getAttribute("data-nav");
      if (id === canonicalNav) {
        item.classList.add("active");
      } else if (id !== "agent") {
        item.classList.remove("active");
      }
    });

    if (this.currentViewInstance && typeof this.currentViewInstance.destroy === "function") {
      this.currentViewInstance.destroy();
    }

    switch (navId) {
      case "home":
        this.currentViewInstance = new HomeView();
        break;
      case "dataset":
      case "datasets":
        this.currentViewInstance = new DatasetsView();
        break;
      case "explore":
        this.currentViewInstance = new ExploreView();
        break;
      case "experiments":
      case "studio":
        this.currentViewInstance = new StudioView();
        break;
      case "evidence":
      case "overview":
        this.currentViewInstance = new OverviewView();
        break;
      case "compare":
        this.currentViewInstance = new CompareView();
        break;
      case "pipelines":
        this.currentViewInstance = new PipelineView();
        break;
      case "knowledge":
        this.currentViewInstance = new KnowledgeView();
        break;
      case "kaggle":
        this.currentViewInstance = new KaggleView();
        break;
      default:
        this.currentViewInstance = new HomeView();
        break;
    }

    this.currentViewInstance.mount(this.mainCanvas);
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }
}

function boot() {
  const app = new App();
  app.init();
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
