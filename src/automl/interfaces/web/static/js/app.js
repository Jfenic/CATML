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
        document.getElementById("workspaceLabel").textContent = overview.workspace;
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
        item.classList.add("bg-indigo-600/20", "text-indigo-400", "border-indigo-600");
        item.classList.remove("text-slate-400", "hover:bg-slate-900", "border-transparent");
      } else if (id !== "agent") {
        item.classList.remove("bg-indigo-600/20", "text-indigo-400", "border-indigo-600");
        item.classList.add("text-slate-400", "hover:bg-slate-900", "border-transparent");
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
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new App();
  app.init();
});
