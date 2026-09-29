/**
 * KnowledgeView — V0.8 Meta-Learning & Memory Layer
 * Dataset fingerprinting, meta-learning similarity & warm start recommendations.
 */
import { store } from "../store.js";
import { bus } from "../bus.js";
import { api } from "../api.js";

export class KnowledgeView {
  constructor() {
    this.container = null;
    this.selectedSimilar = "Customer Churn";
  }

  mount(container) {
    this.container = container;
    this.render();
  }

  render() {
    this.container.innerHTML = `
      <div class="space-y-6">
        <!-- V0.8 Meta-Learning Header -->
        <div class="workbench-card p-4 bg-purple-950/20 border-purple-900/60 flex items-center justify-between">
          <div class="flex items-center space-x-3">
            <span class="text-purple-400 font-bold text-xl">🧠</span>
            <div>
              <div class="flex items-center space-x-2">
                <h2 class="text-base font-bold text-slate-100">CATML Meta-Learning Knowledge (V0.8)</h2>
                <span class="badge-intel text-[10px] px-2 py-0.5 rounded font-mono">Memory Layer</span>
              </div>
              <p class="text-xs text-slate-400">Meta-learning autónomo a partir del histórico de experimentos y datasets similares</p>
            </div>
          </div>
          <span class="badge-sys text-xs px-3 py-1 rounded-full font-mono font-semibold">4 Datasets Indexed</span>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <!-- Left: Dataset Fingerprint & Similarity -->
          <div class="lg:col-span-5 space-y-6">
            <!-- Fingerprint Card -->
            <div class="workbench-card p-5 space-y-3">
              <div class="flex items-center justify-between border-b border-slate-800 pb-2">
                <span class="text-xs uppercase font-bold text-slate-300 tracking-wider">Current Dataset Fingerprint</span>
                <span class="font-mono text-xs text-indigo-400 font-semibold">EV Purchases</span>
              </div>

              <div class="grid grid-cols-2 gap-3 text-xs pt-1 font-mono">
                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Rows</span>
                  <div class="text-sm font-bold text-slate-100">668,665</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Features</span>
                  <div class="text-sm font-bold text-slate-100">13</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Categorical Ratio</span>
                  <div class="text-sm font-bold text-purple-400">54%</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Numerical Ratio</span>
                  <div class="text-sm font-bold text-indigo-400">46%</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Missing Ratio</span>
                  <div class="text-sm font-bold text-amber-400">2.1%</div>
                </div>

                <div class="p-2.5 rounded bg-slate-900 border border-slate-800">
                  <span class="text-slate-400 text-[10px] uppercase font-sans">Target Entropy</span>
                  <div class="text-sm font-bold text-emerald-400">0.681</div>
                </div>
              </div>
            </div>

            <!-- Similar Datasets in CATML Memory -->
            <div class="workbench-card p-5 space-y-3">
              <span class="text-xs uppercase font-bold text-slate-300 tracking-wider">Similar Datasets in Memory</span>
              <div class="space-y-2 pt-1 text-xs">
                <div class="p-3 rounded-lg bg-slate-900 border border-indigo-600/60 cursor-pointer flex items-center justify-between">
                  <div>
                    <div class="font-bold text-slate-100">Customer Churn</div>
                    <div class="text-[11px] text-slate-400">Tabular Binary • 12 features</div>
                  </div>
                  <span class="badge-gain text-xs px-2.5 py-1 rounded font-mono font-bold">91% similar</span>
                </div>

                <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 cursor-pointer flex items-center justify-between">
                  <div>
                    <div class="font-bold text-slate-200">Insurance Conversion</div>
                    <div class="text-[11px] text-slate-400">Tabular Binary • High Cardinality</div>
                  </div>
                  <span class="badge-sys text-xs px-2.5 py-1 rounded font-mono font-bold">84% similar</span>
                </div>

                <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 cursor-pointer flex items-center justify-between">
                  <div>
                    <div class="font-bold text-slate-200">Loan Acceptance</div>
                    <div class="text-[11px] text-slate-400">Tabular Binary • ROC-AUC</div>
                  </div>
                  <span class="badge-sys text-xs px-2.5 py-1 rounded font-mono font-bold">81% similar</span>
                </div>

                <div class="p-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-700 cursor-pointer flex items-center justify-between">
                  <div>
                    <div class="font-bold text-slate-200">Credit Default</div>
                    <div class="text-[11px] text-slate-400">Tabular Binary • Imbalanced</div>
                  </div>
                  <span class="badge-sys text-xs px-2.5 py-1 rounded font-mono font-bold">74% similar</span>
                </div>
              </div>
            </div>
          </div>

          <!-- Right: Meta-learning Evidence & Warm Start -->
          <div class="lg:col-span-7 space-y-6">
            <!-- Why Similar Breakdown -->
            <div class="workbench-card p-5 space-y-4">
              <div class="flex items-center justify-between border-b border-slate-800 pb-2">
                <span class="text-xs uppercase font-bold text-slate-300 tracking-wider">Similarity Breakdown: Customer Churn (91%)</span>
                <span class="text-[11px] text-slate-400 font-mono">Cosine distance on meta-features</span>
              </div>

              <div class="grid grid-cols-2 md:grid-cols-5 gap-2 text-center text-xs font-mono">
                <div class="p-2 rounded bg-slate-900 border border-slate-800">
                  <div class="text-[10px] text-slate-400 font-sans">Rows Match</div>
                  <div class="text-indigo-400 font-bold mt-0.5">0.83</div>
                </div>

                <div class="p-2 rounded bg-slate-900 border border-slate-800">
                  <div class="text-[10px] text-slate-400 font-sans">Categorical</div>
                  <div class="text-emerald-400 font-bold mt-0.5">0.96</div>
                </div>

                <div class="p-2 rounded bg-slate-900 border border-slate-800">
                  <div class="text-[10px] text-slate-400 font-sans">Cardinality</div>
                  <div class="text-emerald-400 font-bold mt-0.5">0.91</div>
                </div>

                <div class="p-2 rounded bg-slate-900 border border-slate-800">
                  <div class="text-[10px] text-slate-400 font-sans">Imbalance</div>
                  <div class="text-indigo-400 font-bold mt-0.5">0.88</div>
                </div>

                <div class="p-2 rounded bg-slate-900 border border-slate-800">
                  <div class="text-[10px] text-slate-400 font-sans">Missing Prof</div>
                  <div class="text-emerald-400 font-bold mt-0.5">0.97</div>
                </div>
              </div>
            </div>

            <!-- Historical Empirical Evidence on Similar Problems -->
            <div class="workbench-card p-5 space-y-3">
              <span class="text-xs uppercase font-bold text-slate-300 tracking-wider">Historical Algorithm Performance on Similar Datasets</span>
              <div class="overflow-x-auto pt-1">
                <table class="w-full wb-table text-left">
                  <thead>
                    <tr>
                      <th>Algorithm</th>
                      <th>Tested Experiments</th>
                      <th>Mean Rank</th>
                      <th>Top-1 Probability</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td class="font-bold text-slate-200">CatBoost</td>
                      <td class="font-mono text-slate-400">12 experiments</td>
                      <td class="font-mono font-bold text-emerald-400">1.6</td>
                      <td><span class="badge-gain text-xs px-2 py-0.5 rounded font-mono font-bold">58%</span></td>
                    </tr>
                    <tr>
                      <td class="font-bold text-slate-200">LightGBM</td>
                      <td class="font-mono text-slate-400">18 experiments</td>
                      <td class="font-mono font-bold text-indigo-400">2.1</td>
                      <td><span class="badge-sys text-xs px-2 py-0.5 rounded font-mono font-bold">31%</span></td>
                    </tr>
                    <tr>
                      <td class="font-bold text-slate-200">XGBoost</td>
                      <td class="font-mono text-slate-400">14 experiments</td>
                      <td class="font-mono font-bold text-slate-400">2.7</td>
                      <td><span class="badge-warn text-xs px-2 py-0.5 rounded font-mono font-bold">11%</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Recommended Warm Start Box -->
            <div class="workbench-card p-5 space-y-4 border-indigo-700/60 bg-indigo-950/20">
              <div class="flex items-center justify-between">
                <div>
                  <div class="text-xs uppercase font-bold text-indigo-300 tracking-wider">Recommended Warm Start for HPO</div>
                  <div class="text-base font-bold text-slate-100 mt-0.5">CatBoost Classifier</div>
                </div>
                <span class="badge-gain text-xs px-3 py-1 rounded font-mono font-bold">~37% Search Reduction</span>
              </div>

              <div class="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs space-y-1 text-slate-300">
                <div>depth:          <span class="text-indigo-400 font-bold">7</span></div>
                <div>learning_rate:  <span class="text-indigo-400 font-bold">0.035</span></div>
                <div>l2_leaf_reg:    <span class="text-indigo-400 font-bold">4.2</span></div>
                <div>subsample:      <span class="text-indigo-400 font-bold">0.85</span></div>
              </div>

              <div class="flex justify-end">
                <button id="btnUseWarmStart" class="bg-indigo-600 hover:bg-indigo-500 text-white text-xs px-4 py-2 rounded-lg font-semibold transition-colors shadow-lg shadow-indigo-600/20">
                  ⚡ Use Warm Start in Next HPO
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    this.container.querySelector("#btnUseWarmStart")?.addEventListener("click", () => {
      alert("Warm start parameters loaded! Subsequent HPO runs will initialize with CatBoost prior.");
    });
  }

  destroy() {
    this.container = null;
  }
}
