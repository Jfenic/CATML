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
  const isNeutral = Math.abs(diff) < 0.00005;
  const isImprovement = isMin ? (diff < -0.00005) : (diff > 0.00005);
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


