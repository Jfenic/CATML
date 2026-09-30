import { store } from "../store.js";
import { LiveView } from "./live_view.js";
import { escapeHtml as e, score, card, empty, button, selectedRun, activeJobStatuses, jobCard, runSelector } from "../ui.js";

export class OverviewView extends LiveView {
  render() {
    if (!this.container) return;
    const state = store.getState();
    const run = selectedRun(state);
    const running = state.jobs.filter(job => activeJobStatuses.includes(job.status));
    const queued = state.jobs.filter(job => job.status === "queued");
    this.container.innerHTML = `<div class="space-y-5">
      <div class="flex justify-between"><div><h1 class="text-xl font-bold">Resumen del workspace</h1><p class="text-sm text-slate-400">Experimentos guardados y actividad actual.</p></div>${button("Nuevo experimento", "data-new-experiment")}</div>
      ${state.error ? `<p role="alert" class="text-amber-300">${e(state.error)} · Se conserva la última lectura.</p>` : ""}
      <div class="grid md:grid-cols-3 gap-4">${card("En ejecución", `<p class="text-3xl font-semibold">${running.length}</p><p class="text-xs text-slate-400">${queued.length} en cola</p>`)}${card("Trials guardados", `<p class="text-3xl font-semibold">${state.overview?.total_trials ?? 0}</p>`)}${card("Última lectura", `<p class="text-sm">${state.lastUpdated ? e(new Date(state.lastUpdated).toLocaleTimeString()) : "Esperando datos"}</p>`)}</div>
      ${card("Trabajos en ejecución", running.length ? running.map(jobCard).join("") : empty("No hay entrenamientos en ejecución."))}
      ${card("Ejecución seleccionada", run ? `${runSelector(state)}<p class="text-sm">${e(run.task_type)} · target: ${e(run.target)} · ${e(run.metric)}</p><p>Mejor validación guardada: <strong>${score(run.best_score)}</strong> · ${e(run.best_model || "Sin resultados")}</p><p class="text-xs text-slate-400">${run.experiments_count} experimentos · ${run.trials_count} trials. Las métricas son locales, no puntuaciones de Kaggle.</p>` : empty("No hay datasets registrados."))}
      ${card("Experimentos", state.snapshotRunId === state.activeRunId && state.experiments.length ? `<div class="space-y-3">${state.experiments.map(exp => `<div class="flex justify-between items-center border-b border-slate-800 pb-3"><div><strong>${e(exp.name)}</strong><p class="text-xs text-slate-400">${e(exp.id)} · ${e(exp.status)} · ${e(exp.metric)} ${score(exp.best_score)}</p></div>${button("Abrir", `data-open-experiment="${e(exp.id)}" data-run="${e(run.id)}"`)}</div>`).join("")}</div>` : empty("Sin experimentos para esta ejecución."))}
    </div>`;
    this.bindCommon();
  }
}
