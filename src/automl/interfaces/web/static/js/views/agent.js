import { store } from "../store.js";
import { card, empty, button } from "../ui.js";
export class AgentDrawer {
  mount(container) {
    this.container = container;
    container.innerHTML = `<div class="h-full bg-slate-900 p-5">${button("Cerrar", 'id="closeAgent"')}${card("Agente · pendiente de integración", empty("Todavía no hay un agente operativo que proponga, evalúe o promueva experimentos. Los resultados reales están en el Studio."))}</div>`;
    container.querySelector("#closeAgent").addEventListener("click", () => store.toggleAgentDrawer(false));
  }
}
