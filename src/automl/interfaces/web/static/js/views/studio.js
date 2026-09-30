import { store } from "../store.js";
import { LiveView } from "./live_view.js";
import { escapeHtml as e, score, card, empty, button, selectedRun, selectedExperiment, jobCard, runSelector, pendingJobStatuses } from "../ui.js";

export class StudioView extends LiveView {
  render() {
    if (!this.container) return;
    const state = store.getState(), run = selectedRun(state), exp = selectedExperiment(state);
    const jobs = state.jobs.filter(job => job.run_id === run?.id);
    const related = exp ? jobs.filter(job => job.result.experiment_id === exp.id || job.payload.experiment_id === exp.id) : [];
    const pending = jobs.filter(job => pendingJobStatuses.includes(job.status));
    const trials = exp?.trials || [];
    this.container.innerHTML = `<div class="space-y-5">
      <div class="flex items-center justify-between"><div><h1 class="text-xl font-bold">Experimentos</h1><p class="text-sm text-slate-400">Selecciona un experimento para ver su configuración y sus resultados.</p></div>${button("Nuevo experimento", "data-new-experiment")}</div>
      ${state.error ? `<p role="alert" class="text-amber-300">${e(state.error)}</p>` : ""}
      ${card("Seleccionar", run ? `<div class="grid md:grid-cols-2 gap-4">${runSelector(state)}<label class="text-xs text-slate-400">Experimento<select id="experimentSelector" class="block w-full mt-2 bg-slate-950 border border-slate-700 rounded-lg p-2 text-slate-200">${state.experiments.map(item => `<option value="${e(item.id)}" ${item.id === exp?.id ? "selected" : ""}>${e(item.name)} · ${e(item.id)} · ${e(item.status)}</option>`).join("")}</select></label></div>` : empty("No hay una ejecución seleccionada."))}
      ${card("Trabajos pendientes de esta ejecución", pending.length ? pending.map(jobCard).join("") : empty("No hay trabajos en ejecución, en cola o pausados."))}
      ${exp ? `${card(exp.name, `<div class="grid md:grid-cols-4 gap-4"><div><p class="text-xs text-slate-400">Estado del experimento</p><strong>${e(exp.status)}</strong></div><div><p class="text-xs text-slate-400">Mejor ${e(exp.metric)}</p><strong>${score(exp.best_score)}</strong></div><div><p class="text-xs text-slate-400">Validación</p><strong>${e(exp.validation_strategy || "Sin registrar")}</strong></div><div><p class="text-xs text-slate-400">Trials guardados</p><strong>${trials.length}</strong></div></div><p class="text-xs text-slate-400 break-all">${e(exp.id)}</p><p class="text-sm">${e(exp.hypothesis || "Sin hipótesis registrada")}</p><p class="text-sm">Modelos: ${e(exp.model_ids.join(", "))}</p><details><summary class="cursor-pointer text-sm">${exp.features_count} features utilizadas</summary><p class="mt-3 text-xs">${e(exp.feature_names.join(", "))}</p></details>`)}
      ${card("Resultados por trial", trials.length ? `<div class="overflow-x-auto"><table class="w-full wb-table text-left"><thead><tr><th>Trial</th><th>Modelo</th><th>${e(exp.metric)}</th><th>Duración</th><th>Resultado</th><th>Parámetros guardados</th></tr></thead><tbody>${trials.map(trial => `<tr><td class="text-xs">${e(trial.trial_id)}</td><td>${e(trial.model_id)}</td><td>${trial.succeeded ? score(trial.score) : "—"}</td><td>${e(trial.time_s)} s</td><td>${trial.succeeded ? "Correcto" : e(trial.failure_reason || "Fallido")}</td><td><code class="text-xs">${e(JSON.stringify(trial.params || {}))}</code></td></tr>`).join("")}</tbody></table></div>` : empty("Todavía no hay resultados guardados."))}
      ${card("Historial del trabajo asociado", related.length ? related.map(jobCard).join("") : empty("Este experimento se ejecutó fuera de la cola de trabajos o no tiene un job asociado."))}` : empty("No hay experimentos guardados en esta ejecución.")}
      ${card("Eventos registrados de la ejecución", state.events.length ? `<div class="space-y-2 max-h-64 overflow-auto">${state.events.map(event => `<details class="text-xs border-b border-slate-800 pb-2"><summary>${e(event.timestamp)} · ${e(event.event_type)}</summary><pre class="whitespace-pre-wrap mt-2">${e(JSON.stringify(event.payload, null, 2))}</pre></details>`).join("")}</div>` : empty("Sin eventos registrados."))}
      <p class="text-xs text-slate-400">Un fit en curso termina antes de aplicar pausa o cancelación. CPU/RAM e importancia de hiperparámetros no se miden en esta vista.</p>
    </div>`;
    this.bindCommon();
    this.container.querySelector("#experimentSelector")?.addEventListener("change", event => store.setState({ activeExperimentId: event.target.value }));
  }
}
