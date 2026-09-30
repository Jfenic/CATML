import { store } from "../store.js";
import { LiveView } from "./live_view.js";
import { pendingJobStatuses, jobCard } from "../ui.js";
export class JobsPanel extends LiveView {
  mount() {
    this.panel = document.createElement("details");
    this.panel.className = "fixed bottom-4 right-4 z-40 bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-slate-200 shadow-xl";
    this.panel.style.width = "min(380px, calc(100vw - 32px))";
    this.summary = document.createElement("summary");
    this.container = document.createElement("div");
    this.container.className = "space-y-3 mt-3";
    this.container.style.maxHeight = "50vh";
    this.container.style.overflowY = "auto";
    this.panel.append(this.summary, this.container);
    document.body.append(this.panel);
    this.unsubscribe = store.subscribe(() => this.render());
    this.render();
  }
  render() {
    const jobs = store.getState().jobs;
    const pending = jobs.filter(job => pendingJobStatuses.includes(job.status));
    const recent = jobs.filter(job => !pending.includes(job)).slice(0, 5);
    this.summary.textContent = `Trabajos · ${pending.length} pendientes`;
    this.container.innerHTML = [...pending, ...recent].map(jobCard).join("") || "No hay trabajos guardados.";
    this.bindCommon();
  }
  destroy() { super.destroy(); this.panel.remove(); }
}
