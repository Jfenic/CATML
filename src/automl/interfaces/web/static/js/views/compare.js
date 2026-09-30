import { store } from "../store.js";
import { LiveView } from "./live_view.js";
import { card, empty, escapeHtml as e, score, runSelector } from "../ui.js";
export class CompareView extends LiveView {
  render() {
    if (!this.container) return;
    const state = store.getState();
    this.container.innerHTML = card("Comparar experimentos guardados", `${runSelector(state)}<p class="text-xs text-slate-400">Comparar solo métricas y estrategias de validación equivalentes. No se estima una mejora sin evaluación independiente.</p>${state.experiments.length ? `<table class="w-full wb-table"><thead><tr><th>Experimento</th><th>Modelos</th><th>Validación</th><th>Métrica</th><th>Resultado</th></tr></thead><tbody>${state.experiments.map(exp => `<tr><td>${e(exp.name)}<p class="text-xs text-slate-400">${e(exp.id)}</p></td><td>${e(exp.model_ids.join(", "))}</td><td>${e(exp.validation_strategy)}</td><td>${e(exp.metric)}</td><td>${score(exp.best_score)}</td></tr>`).join("")}</tbody></table>` : empty("No hay resultados para comparar.")}`);
    this.bindCommon();
  }
}
