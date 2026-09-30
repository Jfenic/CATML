import { store } from "../store.js";
import { api } from "../api.js";
import { bus } from "../bus.js";
import { escapeHtml as e, selectedRun, button } from "../ui.js";
export class NewExperimentModal {
  async mount(container) {
    this.container = container;
    const run = selectedRun(store.getState());
    container.innerHTML = `<div class="fixed inset-0 bg-black/75 z-50 flex items-center justify-center p-4"><div class="workbench-card p-6 max-w-xl w-full space-y-4"><div class="flex justify-between"><h2 class="text-lg font-bold">Nuevo experimento</h2>${button("Cerrar", 'id="closeExperiment"')}</div><p class="text-sm text-slate-400">${e(run?.dataset_name || "Selecciona una ejecución en el Studio")}</p><p id="formMessage" role="status"></p><form id="experimentForm" class="space-y-4"><label class="block text-sm">Nombre<input name="name" required value="Nuevo experimento" class="block w-full mt-2 p-2 rounded bg-slate-950 border border-slate-700"></label><fieldset id="modelOptions" class="space-y-2"><legend class="text-sm mb-2">Modelos compatibles</legend><p>Cargando modelos…</p></fieldset><p class="text-xs text-slate-400">Se utilizan las features no identificadoras del dataset, excluyendo el target. El entrenamiento queda en la cola local.</p>${button("Enviar a la cola", 'type="submit" id="submitExperiment" disabled')}</form></div></div>`;
    container.querySelector("#closeExperiment").addEventListener("click", () => this.destroy());
    if (!run) return;
    try {
      const plugins = await api.getPlugins();
      if (this.container !== container) return;
      const models = plugins.filter(plugin => plugin.plugin_type === "model" && plugin.capabilities.supported_tasks.some(task => task === run.task_type || task === "*"));
      container.querySelector("#modelOptions").innerHTML = `<legend class="text-sm mb-2">Modelos compatibles</legend>${models.map((model, index) => `<label class="block text-sm"><input type="checkbox" name="model" value="${e(model.plugin_id)}" ${index === 0 ? "checked" : ""}> ${e(model.name)} <span class="text-xs text-slate-400">${e(model.plugin_id)}</span></label>`).join("")}`;
      container.querySelector("#submitExperiment").disabled = !models.length;
      container.querySelector("#experimentForm").addEventListener("submit", async event => {
        event.preventDefault();
        const form = new FormData(event.target), models = form.getAll("model");
        const message = container.querySelector("#formMessage"), submit = container.querySelector("#submitExperiment");
        if (!models.length) { message.textContent = "Selecciona al menos un modelo."; return; }
        submit.disabled = true;
        try {
          await api.submitJob("experiment", run.id, { name: form.get("name"), model_ids: models });
          store.setState({ activeRunId: run.id, activeExperimentId: null });
          store.setNav("studio");
          bus.emit("workspace:refresh");
          this.destroy();
        } catch (error) { message.textContent = error.message; submit.disabled = false; }
      });
    } catch (error) { if (this.container === container) container.querySelector("#formMessage").textContent = error.message; }
  }
  destroy() { if (this.container) this.container.innerHTML = ""; this.container = null; }
}
