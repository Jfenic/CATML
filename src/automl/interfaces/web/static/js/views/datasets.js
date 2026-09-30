import { store } from "../store.js";
import { api } from "../api.js";
import { bus } from "../bus.js";
import { card, empty, escapeHtml as e, selectedRun, runSelector, button } from "../ui.js";
export class DatasetsView {
  mount(container) {
    this.container = container;
    this.unsubscribe = store.subscribe((state, previous) => { if (state.activeRunId !== previous.activeRunId) this.load(); });
    this.load();
  }
  async load() {
    const run = selectedRun(store.getState());
    const id = run?.dataset_id;
    this.container.innerHTML = card("Dataset", empty(id ? "Cargando perfil guardado…" : "Sin dataset seleccionado."));
    if (!id) return;
    try {
      const profile = await api.getDatasetProfile(id);
      if (!this.container || selectedRun(store.getState())?.dataset_id !== id) return;
      this.container.innerHTML = card(run.dataset_name, `${runSelector(store.getState())}<p>${profile.row_count.toLocaleString()} filas · ${profile.column_count} columnas · target: ${e(profile.target_column)}</p><p class="text-xs text-slate-400">Los indicadores de identificación son heurísticas del perfil, no evidencia de mejora predictiva.</p><table class="w-full wb-table"><thead><tr><th>Columna</th><th>Tipo</th><th>Nulos</th><th>Valores únicos</th><th>Identificador</th></tr></thead><tbody>${profile.columns.map(col => `<tr><td>${e(col.name)}${col.name === profile.target_column ? " (target)" : ""}</td><td>${e(col.dtype)}</td><td>${col.null_count}</td><td>${col.unique_count}</td><td>${col.is_identifier ? "Sí" : "No"}</td></tr>`).join("")}</tbody></table>${button("Crear experimento", 'id="createDatasetExperiment"')}`);
      this.container.querySelector("#runSelector").addEventListener("change", event => { store.setState({ activeRunId: event.target.value, activeExperimentId: null, experiments: [], snapshotRunId: null }); bus.emit("workspace:refresh"); });
      this.container.querySelector("#createDatasetExperiment").addEventListener("click", () => bus.emit("modal:new-experiment"));
    } catch (error) { if (this.container) this.container.innerHTML = card("Dataset", empty(error.message)); }
  }
  destroy() { this.unsubscribe?.(); this.container = null; }
}
