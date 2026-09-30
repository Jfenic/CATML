import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";

export class LiveView {
  mount(container) {
    this.container = container;
    this.unsubscribe = store.subscribe(() => this.render());
    this.render();
  }
  bindCommon() {
    this.container.querySelector("#runSelector")?.addEventListener("change", event => {
      store.setState({ activeRunId: event.target.value, activeExperimentId: null, snapshotRunId: null, experiments: [] });
      bus.emit("workspace:refresh");
    });
    this.container.querySelectorAll("[data-open-experiment]").forEach(node => node.addEventListener("click", () => {
      store.setState({ activeRunId: node.dataset.run, activeExperimentId: node.dataset.openExperiment });
      store.setNav("studio");
      bus.emit("workspace:refresh");
    }));
    this.container.querySelectorAll("[data-control]").forEach(node => node.addEventListener("click", async () => {
      node.disabled = true;
      try { await api.controlJob(node.dataset.jobId, node.dataset.control); bus.emit("workspace:refresh"); }
      catch (error) { store.setState({ error: error.message }); }
    }));
    this.container.querySelector("[data-new-experiment]")?.addEventListener("click", () => bus.emit("modal:new-experiment"));
  }
  destroy() { this.unsubscribe?.(); this.container = null; }
}
