/**
 * Utility functions for CATML Workbench UI.
 * Standardized across all views to enforce robust input sanitization,
 * XSS prevention, and strict Tech Minimalista UI rendering.
 */

/**
 * Escapes special HTML characters in a string or primitive value to prevent XSS.
 * @param {any} val - The input value to escape.
 * @returns {string} Safe HTML-escaped string.
 */
export function escapeHtml(val) {
  if (val === null || val === undefined) return "";
  return String(val)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export const ACTIVE_RUN_STATUSES = new Set([
  "EXPERIMENTING",
  "OPTIMIZING",
  "FINALIZING",
  "RUNNING",
  "EVALUATING",
  "QUEUED",
]);

/** Spanish labels for experiment entity status (not run/job). */
export const EXPERIMENT_STATUS_LABELS = {
  CREATED: "Creado",
  QUEUED: "En cola",
  RUNNING: "En ejecución",
  EVALUATING: "Evaluando",
  COMPLETED: "Completado",
  PARTIAL: "Parcial (presupuesto)",
  FAILED: "Fallido",
  CANCELLED: "Cancelado",
  PAUSED: "Pausado",
};

/**
 * Format experiment status for display. Distinguishes from run-level badges.
 * @param {string|undefined} status
 * @returns {string}
 */
export function formatExperimentStatus(status) {
  if (!status) return "Creado";
  const key = String(status).toUpperCase();
  return EXPERIMENT_STATUS_LABELS[key] || key;
}

/**
 * Resolve completed-trial count from live experiments/leaderboard before stale run.trials_count.
 * @param {{trials_count?: number}|null} activeRun
 * @param {Array<{trials_count?: number, trials?: Array}>} experiments
 * @param {Array} leaderboard
 * @returns {number}
 */
export function resolveTrialsCount(activeRun, experiments = [], leaderboard = []) {
  const fromExperiments = (experiments || []).reduce((sum, e) => {
    const n = e.trials_count != null ? Number(e.trials_count) : (e.trials ? e.trials.length : 0);
    return sum + (Number.isFinite(n) ? n : 0);
  }, 0);
  const fromLeaderboard = Array.isArray(leaderboard) ? leaderboard.length : 0;
  const fromRun = activeRun && activeRun.trials_count != null ? Number(activeRun.trials_count) : 0;
  return Math.max(fromExperiments, fromLeaderboard, Number.isFinite(fromRun) ? fromRun : 0);
}

/**
 * Checks if a run is actively running or queued.
 * @param {object|string} run - The run object or status string.
 * @returns {boolean} True if the run status is active.
 */
export function isRunActive(run) {
  if (!run) return false;
  if (typeof run === "object" && typeof run.is_active === "boolean") {
    return run.is_active;
  }
  const status = typeof run === "string" ? run : run.status;
  if (!status) return false;
  return ACTIVE_RUN_STATUSES.has(String(status).toUpperCase());
}

export const ACTIVE_JOB_STATUSES = new Set([
  "queued",
  "running",
  "pause_requested",
  "cancel_requested",
]);

/**
 * Checks if a background job is actively running or queued.
 * @param {object|string} job - The job object or status string.
 * @returns {boolean} True if the job status is active.
 */
export function isJobActive(job) {
  if (!job) return false;
  const status = typeof job === "string" ? job : job.status;
  if (!status) return false;
  return ACTIVE_JOB_STATUSES.has(String(status).toLowerCase());
}

export const JOB_STATUS_LABELS = {
  queued: "En cola",
  running: "En ejecución",
  pause_requested: "Pausando...",
  cancel_requested: "Cancelando...",
  paused: "Pausado",
  cancelled: "Cancelado",
  completed: "Completado",
  failed: "Fallido",
  interrupted: "Interrumpido",
};

export const JOB_ACTION_LABELS = {
  pause: "Pausar",
  cancel: "Cancelar",
  resume: "Reanudar",
  retry: "Reintentar",
};

export function getJobStatusLabel(status) {
  if (!status) return "Desconocido";
  const s = String(status).toLowerCase();
  return JOB_STATUS_LABELS[s] || status;
}

/**
 * Determines whether a metric should be minimized (lower is better) or maximized (higher is better).
 * @param {string} metric - Metric name (e.g. 'rmse', 'roc_auc', 'mae', 'log_loss', etc.)
 * @returns {boolean} True if lower score is better.
 */
export function isMinimizingMetric(metric) {
  if (!metric) return false;
  const m = String(metric).toLowerCase().trim();
  return [
    "mae",
    "rmse",
    "mse",
    "loss",
    "log_loss",
    "logloss",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "neg_mean_absolute_error",
    "neg_mean_squared_error",
    "neg_root_mean_squared_error",
    "neg_log_loss",
  ].some(k => m === k || m.includes(k));
}

/**
 * Returns human-readable metric direction badge text and hint.
 * @param {string} metric - Metric name.
 * @returns {{ isMinimizing: boolean, label: string, arrow: string, short: string }}
 */
export function getMetricDirectionInfo(metric) {
  const isMin = isMinimizingMetric(metric);
  return {
    isMinimizing: isMin,
    label: isMin ? "menor es mejor" : "mayor es mejor",
    arrow: isMin ? "↓" : "↑",
    short: isMin ? "min" : "max",
  };
}

/**
 * Human-readable p-value (avoids p=0.000e+00).
 * @param {number|null|undefined} p
 * @returns {string}
 */
export function formatPValue(p) {
  if (p == null || !Number.isFinite(Number(p))) return "—";
  const v = Number(p);
  if (v <= 0) return "< 1e-300";
  if (v < 0.0001) return `< 0.0001`;
  if (v < 0.001) return v.toFixed(4);
  if (v < 0.01) return v.toFixed(3);
  return v.toFixed(3);
}

/**
 * Formats a metric delta comparing two scores.
 * @param {number} currentScore - Score of candidate model.
 * @param {number} baselineScore - Score of baseline model.
 * @param {string} metric - Metric name.
 * @returns {{ diff: number, isImprovement: boolean, isNeutral: boolean, text: string, badgeClass: string }}
 */
export function formatMetricComparison(currentScore, baselineScore, metric) {
  if (currentScore == null || baselineScore == null) {
    return { diff: 0, isImprovement: false, isNeutral: true, text: "—", badgeClass: "badge-sys" };
  }
  const diff = currentScore - baselineScore;
  const isMin = isMinimizingMetric(metric);
  // Tiny deltas on [0,1] metrics are noise for demos — treat as inconclusive.
  const eps = 0.001;
  const isNeutral = Math.abs(diff) < eps;
  const isImprovement = isMin ? (diff < -eps) : (diff > eps);
  const sign = diff >= 0 ? "+" : "";
  const diffStr = sign + diff.toFixed(5);

  if (isNeutral) {
    return {
      diff,
      isImprovement: false,
      isNeutral: true,
      text: `Δ: ${diffStr} (diferencia no concluyente)`,
      badgeClass: "badge-sys",
    };
  }
  return {
    diff,
    isImprovement,
    isNeutral: false,
    text: `Δ: ${diffStr} (${isImprovement ? "diferencia observada favorable" : "degradación observada"})`,
    badgeClass: isImprovement ? "badge-gain" : "badge-warn",
  };
}

/**
 * 5 explicit transformation and recipe lifecycle stages (ADR-009 & Section 8):
 * Propuesta -> Verificada -> Seleccionada -> Aplicada -> Promovida.
 */
export const TRANSFORMATION_LIFECYCLE_STAGES = {
  PROPOSED: { key: "proposed", label: "Propuesta", badgeClass: "text-[#8B95A7] bg-[#161B26] border-[#242A36]" },
  VERIFIED: { key: "verified", label: "Verificada", badgeClass: "text-[#53C8FF] bg-[#53C8FF]/10 border-[#53C8FF]/30" },
  SELECTED: { key: "selected", label: "Seleccionada", badgeClass: "text-[#6956E8] bg-[#6956E8]/10 border-[#6956E8]/30" },
  APPLIED: { key: "applied", label: "Aplicada", badgeClass: "text-[#F59E0B] bg-[#F59E0B]/10 border-[#F59E0B]/30" },
  PROMOTED: { key: "promoted", label: "Promovida", badgeClass: "text-[#22C55E] bg-[#22C55E]/10 border-[#22C55E]/30" },
};

export function getTransformationStageInfo(status) {
  const s = String(status || "").toLowerCase().trim();
  if (s === "promoted") return TRANSFORMATION_LIFECYCLE_STAGES.PROMOTED;
  if (s === "applied" || s === "submitted") return TRANSFORMATION_LIFECYCLE_STAGES.APPLIED;
  if (s === "selected") return TRANSFORMATION_LIFECYCLE_STAGES.SELECTED;
  if (s === "verified" || s === "accepted") return TRANSFORMATION_LIFECYCLE_STAGES.VERIFIED;
  return TRANSFORMATION_LIFECYCLE_STAGES.PROPOSED;
}


