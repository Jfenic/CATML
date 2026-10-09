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

