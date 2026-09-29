/**
 * CATMLApiClient — Repository / Adapter Pattern
 * Encapsulates all backend REST calls through Hexagonal Architecture ports.
 */
export class CATMLApiClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl;
  }

  async _fetch(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    try {
      const response = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        ...options,
      });

      if (!response.ok) {
        let errMessage = `HTTP ${response.status}: ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData.error || errData.message) {
            errMessage = errData.error || errData.message;
          }
        } catch (_) {}
        throw new Error(errMessage);
      }

      return await response.json();
    } catch (err) {
      console.error(`[API Error] ${endpoint}:`, err);
      throw err;
    }
  }

  // GET queries
  async getOverview() {
    return this._fetch("/api/overview");
  }

  async getRuns() {
    return this._fetch("/api/runs");
  }

  async getExperiments(runId) {
    return this._fetch(`/api/experiments?run_id=${encodeURIComponent(runId)}`);
  }

  async compareExperiments(experimentIds) {
    const ids = Array.isArray(experimentIds) ? experimentIds.join(",") : experimentIds;
    return this._fetch(`/api/experiments/compare?ids=${encodeURIComponent(ids)}`);
  }

  async getLeaderboard(runId) {
    return this._fetch(`/api/leaderboard?run_id=${encodeURIComponent(runId)}`);
  }

  async getDatasetProfile(datasetId) {
    return this._fetch(`/api/dataset/profile?dataset_id=${encodeURIComponent(datasetId)}`);
  }

  async getPlan(runId, datasetId) {
    const q = runId ? `run_id=${encodeURIComponent(runId)}` : `dataset_id=${encodeURIComponent(datasetId)}`;
    return this._fetch(`/api/plan?${q}`);
  }

  async getKnowledge(datasetId) {
    const q = datasetId ? `?dataset_id=${encodeURIComponent(datasetId)}` : "";
    return this._fetch(`/api/knowledge${q}`);
  }

  async getAgentHypotheses(runId) {
    const q = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    return this._fetch(`/api/agent/hypotheses${q}`);
  }

  async getKaggleStatus(runId) {
    const q = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
    return this._fetch(`/api/kaggle/status${q}`);
  }

  async getPlugins() {
    return this._fetch("/api/plugins");
  }

  // POST commands
  async registerDataset(payload) {
    return this._fetch("/api/dataset/register", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async pauseRun(runId) {
    return this._fetch("/api/run/pause", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    });
  }

  async resumeRun(runId) {
    return this._fetch("/api/run/resume", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    });
  }

  async cancelRun(runId) {
    return this._fetch("/api/run/cancel", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    });
  }

  async cloneRun(runId, newName) {
    return this._fetch("/api/run/clone", {
      method: "POST",
      body: JSON.stringify({ run_id: runId, new_name: newName }),
    });
  }

  async runExperiment(payload) {
    return this._fetch("/api/experiment/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async createAndRunExperiment(payload) {
    return this._fetch("/api/experiment/create_and_run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async optimizeExperiment(payload) {
    return this._fetch("/api/optimize", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async generateSubmission(payload) {
    return this._fetch("/api/predict", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async sendAgentAction(hypothesisId, action) {
    return this._fetch("/api/agent/action", {
      method: "POST",
      body: JSON.stringify({ hypothesis_id: hypothesisId, action }),
    });
  }
}

export const api = new CATMLApiClient();
