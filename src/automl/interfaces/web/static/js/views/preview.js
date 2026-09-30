import { card, empty } from "../ui.js";
export class PreviewView {
  constructor(title) { this.title = title; }
  mount(container) { container.innerHTML = card(this.title, empty("Esta interfaz está pendiente de integración. No hay resultados ni acciones automáticas disponibles aquí.")); }
  destroy() {}
}
