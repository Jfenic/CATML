import { api } from "../api.js";

const labels = {
  queued: "En cola", running: "En ejecución", pause_requested: "Pausa solicitada",
  cancel_requested: "Cancelación solicitada", paused: "Pausado", cancelled: "Cancelado",
  completed: "Completado", failed: "Fallido", interrupted: "Interrumpido",
};
const actions = {
  queued: ["pause", "cancel"], running: ["pause", "cancel"], pause_requested: ["cancel"],
  paused: ["resume", "cancel"], failed: ["retry", "cancel"], interrupted: ["retry", "cancel"],
};
const actionLabels = { pause: "Pausar", cancel: "Cancelar", resume: "Continuar", retry: "Reintentar" };

/** Persistent activity is recovered from the server on every page load. */
export class JobsPanel {
  mount() {
    this.panel = document.createElement("details");
    this.panel.className = "fixed bottom-4 left-4 z-50 bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-slate-200 shadow-xl";
    this.panel.style.width = "min(360px, calc(100vw - 32px))";
    this.summary = document.createElement("summary");
    this.summary.textContent = "Trabajos";
    this.content = document.createElement("div");
    this.content.style.maxHeight = "45vh";
    this.content.style.overflowY = "auto";
    this.panel.append(this.summary, this.content);
    document.body.append(this.panel);
    this.refresh();
    this.timer = setInterval(() => {
      if (!document.hidden) this.refresh();
    }, 1500);
  }

  async refresh() {
    if (this.refreshing) return;
    this.refreshing = true;
    try {
      const jobs = await api.getJobs();
      const active = jobs.filter(job => !["completed", "cancelled", "failed", "interrupted"].includes(job.status));
      this.summary.textContent = `Trabajos (${active.length} pendientes)`;
      this.content.replaceChildren();
      if (!jobs.length) this.content.textContent = "Todavía no hay trabajos.";
      const recent = jobs.filter(job => !active.includes(job)).slice(0, 10);
      for (const job of [...active, ...recent]) this.content.append(this._renderJob(job));
    } catch (error) {
      this.summary.textContent = "Trabajos: conexión pendiente";
    } finally {
      this.refreshing = false;
    }
  }

  _renderJob(job) {
    const row = document.createElement("div");
    row.className = "border-t border-slate-700 mt-3 pt-3 space-y-2";
    const heading = document.createElement("p");
    heading.textContent = `${job.operation} · ${labels[job.status] || job.status} · intento ${job.attempt}`;
    const identity = document.createElement("p");
    identity.className = "text-slate-400 break-all";
    identity.textContent = job.id;
    const detail = document.createElement("p");
    detail.textContent = job.error || job.message;
    row.append(heading, identity, detail);
    if (job.total) {
      const progress = document.createElement("progress");
      progress.max = job.total;
      progress.value = job.completed;
      progress.className = "w-full";
      const counts = document.createElement("span");
      counts.textContent = `${job.completed}/${job.total} pasos terminados`;
      row.append(progress, counts);
    }
    if (job.status === "completed") {
      const result = document.createElement("p");
      result.textContent = job.result.output_path || `Experimento: ${job.result.experiment_id}`;
      row.append(result);
    }
    for (const action of actions[job.status] || []) {
      const button = document.createElement("button");
      button.className = "bg-slate-700 rounded px-2 py-1 mr-2 hover:bg-slate-600";
      button.textContent = actionLabels[action];
      button.addEventListener("click", async () => {
        button.disabled = true;
        try {
          await api.controlJob(job.id, action);
          await this.refresh();
        } catch (error) {
          detail.textContent = error.message;
          button.disabled = false;
        }
      });
      row.append(button);
    }
    return row;
  }

  destroy() {
    clearInterval(this.timer);
    this.panel.remove();
  }
}
