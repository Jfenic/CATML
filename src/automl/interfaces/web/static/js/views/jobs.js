import { api } from "../api.js";

const labels = {
  queued: "Queued", running: "Running", pause_requested: "Pause requested",
  cancel_requested: "Cancel requested", paused: "Paused", cancelled: "Cancelled",
  completed: "Completed", failed: "Failed", interrupted: "Interrupted",
};
const actions = {
  queued: ["pause", "cancel"], running: ["pause", "cancel"], pause_requested: ["cancel"],
  paused: ["resume", "cancel"], failed: ["retry", "cancel"], interrupted: ["retry", "cancel"],
};
const actionLabels = { pause: "Pause", cancel: "Cancel", resume: "Resume", retry: "Retry" };

/** Persistent activity is recovered from the server on every page load. */
export class JobsPanel {
  mount() {
    this.panel = document.createElement("details");
    this.panel.className = "fixed bottom-4 left-4 z-50 bg-[#10151E] border border-[#252C38] rounded-xl p-3 text-xs text-slate-200 shadow-xl";
    this.panel.style.width = "min(360px, calc(100vw - 32px))";
    this.summary = document.createElement("summary");
    this.summary.className = "cursor-pointer font-semibold text-slate-300 hover:text-white select-none";
    this.summary.textContent = "Jobs";
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
      this.summary.textContent = `Jobs (${active.length} pending)`;
      this.content.replaceChildren();
      if (!jobs.length) this.content.textContent = "No background jobs running.";
      const recent = jobs.filter(job => !active.includes(job)).slice(0, 10);
      for (const job of [...active, ...recent]) this.content.append(this._renderJob(job));
    } catch (error) {
      this.summary.textContent = "Jobs: connection pending";
    } finally {
      this.refreshing = false;
    }
  }

  _renderJob(job) {
    const row = document.createElement("div");
    row.className = "border-t border-[#252C38] mt-3 pt-3 space-y-2";
    const heading = document.createElement("p");
    heading.className = "font-medium text-slate-200";
    heading.textContent = `${job.operation} · ${labels[job.status] || job.status} · attempt ${job.attempt}`;
    const identity = document.createElement("p");
    identity.className = "text-slate-400 break-all font-mono text-[11px]";
    identity.textContent = job.id;
    const detail = document.createElement("p");
    detail.className = "text-slate-400 text-xs";
    detail.textContent = job.error || job.message;
    row.append(heading, identity, detail);
    if (job.total) {
      const progress = document.createElement("progress");
      progress.max = job.total;
      progress.value = job.completed;
      progress.className = "w-full";
      const counts = document.createElement("span");
      counts.className = "text-[11px] text-slate-400 font-mono";
      counts.textContent = `${job.completed}/${job.total} steps completed`;
      row.append(progress, counts);
    }
    if (job.status === "completed") {
      const result = document.createElement("p");
      result.className = "text-xs font-mono text-emerald-400";
      result.textContent = job.result.output_path || `Experiment: ${job.result.experiment_id}`;
      row.append(result);
    }
    for (const action of actions[job.status] || []) {
      const button = document.createElement("button");
      button.className = "bg-[#151B26] border border-[#252C38] rounded px-2.5 py-1 mr-2 hover:bg-[#1A2230] text-slate-300 hover:text-white transition-colors text-xs font-medium";
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
