/**
 * App — Main Application Orchestrator & Router
 * Implements View Controller, Observer, and EventBus patterns.
 */
import { store } from "./store.js";
import { bus } from "./bus.js";
import { api } from "./api.js";
import { JobsPanel } from "./views/jobs.js";

import { OverviewView } from "./views/overview.js";
import { StudioView } from "./views/studio.js";
import { DatasetsView } from "./views/datasets.js";
import { CompareView } from "./views/compare.js";
import { PipelineView } from "./views/pipeline.js";
import { KnowledgeView } from "./views/knowledge.js";
import { KaggleView } from "./views/kaggle.js";
import { AgentDrawer } from "./views/agent.js";
import { NewExperimentModal } from "./views/new_experiment.js";

class App {
  constructor() {
    this.currentViewInstance = null;
    this.mainCanvas = document.getElementById("mainContent");
    this.agentDrawerContainer = document.getElementById("agentDrawer");
    this.modalContainer = document.getElementById("modalContainer");

    this.agentDrawer = new AgentDrawer();
    this.newExperimentModal = new NewExperimentModal();
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
    });

    // Initial render
    this._routeTo(store.getState().currentNav);
  }

  async _fetchInitialState() {
    try {
      const [overview, runs] = await Promise.all([
        api.getOverview().catch(() => null),
        api.getRuns().catch(() => []),
      ]);

      store.setState({
        overview,
        runs,
        activeRunId: runs.length ? runs[0].id : null,
      });

      if (overview && overview.workspace) {
        const wsEl = document.getElementById("workspaceLabel");
        if (wsEl) wsEl.textContent = overview.workspace;
      }

      // Dynamic Project Name Label
      const projectEl = document.getElementById("projectNameLabel");
      if (projectEl) {
        if (runs && runs.length && runs[0].dataset_name && runs[0].dataset_name !== "-") {
          projectEl.textContent = `${runs[0].dataset_name} (${runs[0].id})`;
        } else if (overview && overview.recent_datasets && overview.recent_datasets.length) {
          projectEl.textContent = overview.recent_datasets[0].name;
        } else {
          projectEl.textContent = "AutoML Workspace";
        }
      }

      // Dynamic Kaggle Nav Item & Badge
      const kaggleLabel = document.getElementById("kaggleNavLabel");
      const kaggleBadge = document.getElementById("kaggleNavBadge");
      if (kaggleLabel) {
        if (runs && runs.length && runs[0].dataset_name && runs[0].dataset_name !== "-") {
          kaggleLabel.textContent = runs[0].dataset_name;
        } else if (overview && overview.recent_datasets && overview.recent_datasets.length) {
          kaggleLabel.textContent = overview.recent_datasets[0].name;
        } else {
          kaggleLabel.textContent = "Kaggle & Submissions";
        }
      }
      if (kaggleBadge) {
        if (runs && runs.length && runs[0].best_score != null) {
          kaggleBadge.textContent = Number(runs[0].best_score).toFixed(4);
        } else if (overview && overview.best_score) {
          kaggleBadge.textContent = Number(overview.best_score).toFixed(4);
        } else {
          kaggleBadge.textContent = "—";
        }
      }

      const isAnyRunning = (runs || []).some(r => r.status === "RUNNING");
      const statusBadge = document.getElementById("globalStatusBadge");
      const sysHw = document.getElementById("sysHardwareHeader");
      if (sysHw) {
        sysHw.textContent = isAnyRunning ? "AutoML Training Folds Active" : "AutoML Engine Ready";
      }

      if (statusBadge) {
        if (isAnyRunning) {
          statusBadge.className = "badge-warn px-2.5 py-0.5 rounded-md font-mono text-[11px] font-medium flex items-center space-x-1.5";
          statusBadge.innerHTML = `<svg class="animate-spin h-3 w-3 text-[#F59E0B] inline" viewBox="0 0 24 24" fill="none"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg><span>TRAINING</span>`;
        } else {
          const lastStatus = runs && runs[0] ? runs[0].status : "READY";
          statusBadge.className = "badge-gain px-2.5 py-0.5 rounded-md font-mono text-[11px] font-medium flex items-center space-x-1.5";
          statusBadge.innerHTML = `<span class="w-1.5 h-1.5 rounded-full bg-[#22C55E]"></span><span>${lastStatus}</span>`;
        }
      }
    } catch (e) {
      console.warn("Could not load initial server state, using offline defaults:", e);
    }
  }

  _bindNavigation() {
    document.querySelectorAll(".nav-item").forEach(item => {
      item.addEventListener("click", e => {
        e.preventDefault();
        const navId = item.getAttribute("data-nav");
        if (navId === "agent") {
          store.toggleAgentDrawer(true);
        } else {
          store.setNav(navId);
        }
      });
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
    bus.on("modal:new-experiment", () => {
      this.newExperimentModal.mount(this.modalContainer);
    });

    bus.on("run:select", runId => {
      store.setState({ activeRunId: runId });
      store.setNav("studio");
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
    // Update active class on sidebar items
    document.querySelectorAll(".nav-item").forEach(item => {
      const id = item.getAttribute("data-nav");
      if (id === navId) {
        item.classList.add("active");
      } else if (id !== "agent") {
        item.classList.remove("active");
      }
    });

    if (this.currentViewInstance && typeof this.currentViewInstance.destroy === "function") {
      this.currentViewInstance.destroy();
    }

    switch (navId) {
      case "overview":
        this.currentViewInstance = new OverviewView();
        break;
      case "datasets":
        this.currentViewInstance = new DatasetsView();
        break;
      case "studio":
      case "experiments":
        this.currentViewInstance = new StudioView();
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
        this.currentViewInstance = new OverviewView();
        break;
    }

    this.currentViewInstance.mount(this.mainCanvas);
    if (window.lucide && typeof window.lucide.createIcons === "function") {
      window.lucide.createIcons();
    }
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new App();
  app.init();
});
