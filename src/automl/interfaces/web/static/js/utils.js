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
