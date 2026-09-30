import { store } from "../store.js";
import { api } from "../api.js";
import { bus } from "../bus.js";
import { escapeHtml as e, selectedRun, score, card, empty, button } from "../ui.js";
export class KaggleView {
  mount(container) {
    this.container = container;
    const run = selectedRun(store.getState());
    const folder = run?.dataset_path?.replace(/[^/]+$/, "") || "";
    container.innerHTML = `<div class="space-y-5">${card("Predicciones y submission", `<p class="text-sm text-slate-400">Ejecución seleccionada: ${e(run?.dataset_name || "Ninguna")}. Puedes revisar los archivos antes de enviarlos a Kaggle.</p><form id="submissionForm" class="space-y-4">${[["test_dataset_path", "CSV de test", folder + "test.csv"], ["template_path", "Plantilla (opcional)", folder + "sample_submission.csv"], ["output_path", "Archivo de salida", folder + "submission_workbench.csv"]].map(([name,label,value]) => `<label class="block text-sm">${label}<input name="${name}" value="${e(value)}" ${name !== "template_path" ? "required" : ""} class="block w-full mt-2 p-2 bg-slate-950 border border-slate-700 rounded-lg"></label>`).join("")}<label class="block text-sm"><input type="checkbox" name="oof"> Evaluar OOF binario antes de generar</label><label class="block text-sm">Folds OOF<input name="folds" type="number" min="2" value="5" class="ml-3 w-20 p-2 bg-slate-950 border border-slate-700 rounded"></label>${button("Generar en segundo plano", `type="submit" ${run ? "" : "disabled"}`)}<p id="submissionMessage" role="status"></p></form>`)}<div id="submissionResults"></div><p class="text-xs text-slate-400">El envío a Kaggle y su puntuación pública no están integrados. No se muestran puntuaciones ni comprobaciones inventadas.</p></div>`;
    container.querySelector("#submissionForm").addEventListener("submit", async event => {
      event.preventDefault();
      const form = new FormData(event.target), submit = event.target.querySelector("button"), message = container.querySelector("#submissionMessage");
      const payload = { test_dataset_path: form.get("test_dataset_path"), output_path: form.get("output_path"), predict_proba: true };
      if (form.get("template_path")) payload.template_path = form.get("template_path");
      if (form.get("oof")) payload.folds = Number(form.get("folds"));
      submit.disabled = true;
      try {
        const result = await api.submitJob(payload.folds ? "oof" : "submission", run.id, payload);
        message.textContent = `Trabajo enviado: ${result.job_id}. Sigue su estado en el panel Trabajos.`;
        bus.emit("workspace:refresh");
      } catch (error) { message.textContent = error.message; }
      finally { submit.disabled = false; }
    });
    this.unsubscribe = store.subscribe(() => this.renderResults(run?.id));
    this.renderResults(run?.id);
  }
  renderResults(runId) {
    const results = store.getState().jobs.filter(job => job.run_id === runId && job.status === "completed" && ["oof", "submission"].includes(job.operation));
    this.container.querySelector("#submissionResults").innerHTML = card("Archivos generados por trabajos completados", results.length ? results.map(job => `<article class="border-b border-slate-800 p-3"><p class="text-sm break-all">${e(job.result.output_path)}</p><p class="text-xs text-slate-400">${job.result.row_count} filas · ${e(job.updated_at)}</p>${job.result.oof ? `<p>OOF ${e(job.result.oof.metric)}: ${score(job.result.oof.score)} · delta frente al baseline: ${score(job.result.oof.delta)}</p>` : ""}</article>`).join("") : empty("Todavía no hay submissions terminadas en la cola de esta ejecución."));
  }
  destroy() { this.unsubscribe?.(); this.container = null; }
}
