export const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const score = value => typeof value === "number" && Number.isFinite(value) ? value.toFixed(5) : "—";
export const selectedRun = state => state.runs.find(run => run.id === state.activeRunId) || null;
export const selectedExperiment = state => state.snapshotRunId === state.activeRunId ? state.experiments.find(exp => exp.id === state.activeExperimentId) || null : null;
export const activeJobStatuses = ["running", "pause_requested", "cancel_requested"];
export const pendingJobStatuses = [...activeJobStatuses, "queued", "paused"];
export const jobLabels = { queued: "En cola", running: "En ejecución", pause_requested: "Pausa solicitada", cancel_requested: "Cancelación solicitada", paused: "Pausado", cancelled: "Cancelado", completed: "Completado", failed: "Fallido", interrupted: "Interrumpido" };
export const jobActions = { queued: ["pause", "cancel"], running: ["pause", "cancel"], pause_requested: ["cancel"], paused: ["resume", "cancel"], failed: ["retry", "cancel"], interrupted: ["retry", "cancel"] };
export const actionLabels = { pause: "Pausar", resume: "Continuar", cancel: "Cancelar", retry: "Reintentar" };
export const button = (label, attrs = "") => `<button ${attrs} class="bg-indigo-600 hover:bg-indigo-500 rounded-lg px-3 py-2 text-xs text-white disabled:opacity-40">${escapeHtml(label)}</button>`;
export const empty = text => `<p class="p-5 text-sm text-slate-400">${escapeHtml(text)}</p>`;
export const card = (title, content) => `<section class="workbench-card p-5 space-y-4"><h2 class="text-base font-semibold text-slate-100">${escapeHtml(title)}</h2>${content}</section>`;
export function runSelector(state) {
  return `<label class="text-xs text-slate-400">Dataset / ejecución<select id="runSelector" class="block w-full mt-2 bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200">${state.runs.map(run => `<option value="${escapeHtml(run.id)}" ${run.id === state.activeRunId ? "selected" : ""}>${escapeHtml(run.dataset_name)} · ${escapeHtml(run.id)}</option>`).join("")}</select></label>`;
}
export function jobCard(job) {
  return `<article class="border border-slate-700 rounded-lg p-4 space-y-2" data-job="${escapeHtml(job.id)}">
    <div class="flex justify-between gap-3"><strong>${escapeHtml(job.payload.name || job.operation)}</strong><span class="text-indigo-300">${escapeHtml(jobLabels[job.status])}</span></div>
    <p class="text-xs text-slate-400 break-all">${escapeHtml(job.id)} · intento ${job.attempt}</p>
    <p class="text-sm">${escapeHtml(job.error || job.message)}</p>
    ${job.total ? `<progress class="w-full" value="${job.completed}" max="${job.total}"></progress><p class="text-xs text-slate-400">${job.completed}/${job.total} pasos terminados</p>` : ""}
    ${job.result.experiment_id ? button("Ver experimento", `data-open-experiment="${escapeHtml(job.result.experiment_id)}" data-run="${escapeHtml(job.run_id)}"`) : ""}
    ${(jobActions[job.status] || []).map(action => button(actionLabels[action], `data-control="${action}" data-job-id="${escapeHtml(job.id)}"`)).join(" ")}
  </article>`;
}
