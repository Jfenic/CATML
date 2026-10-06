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

  async getDatasets() {
    return this._fetch("/api/datasets");
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

  getMediaPreviewUrl(path, datasetId = "") {
    let url = `/api/media/preview?path=${encodeURIComponent(path)}`;
    if (datasetId) {
      url += `&dataset_id=${encodeURIComponent(datasetId)}`;
    }
    return url;
  }

  // POST commands
  async registerDataset(payload) {
    return this._fetch("/api/dataset/register", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async _controlRunJobs(runId, action, statuses) {
    const jobs = (await this.getJobs(runId)).filter(job => statuses.includes(job.status));
    if (!jobs.length) return null;
    await Promise.all(jobs.map(job => this.controlJob(job.id, action)));
    return { status: "success", run_id: runId };
  }

  async pauseRun(runId) {
    const jobs = await this._controlRunJobs(runId, "pause", ["queued", "running"]);
    if (jobs) return jobs;
    return this._fetch("/api/run/pause", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    });
  }

  async resumeRun(runId) {
    const jobs = await this._controlRunJobs(runId, "resume", ["paused"]);
    if (jobs) return jobs;
    return this._fetch("/api/run/resume", {
      method: "POST",
      body: JSON.stringify({ run_id: runId }),
    });
  }

  async cancelRun(runId) {
    const jobs = await this._controlRunJobs(runId, "cancel", ["queued", "running", "pause_requested", "paused", "failed", "interrupted"]);
    if (jobs) return jobs;
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

  async runExperiment(payload, onProgress) {
    return this._runJob("experiment", payload.run_id, {
      name: payload.name || `exp_${payload.model_id || "lightgbm"}`,
      model_ids: [payload.model_id || "lightgbm"],
      feature_names: payload.feature_names,
    }, onProgress);
  }

  async createAndRunExperiment(payload, onProgress) {
    return this._runJob("experiment", payload.run_id, {
      name: payload.name || "Workbench experiment",
      model_ids: payload.models || ["lightgbm"],
    }, onProgress);
  }

  async optimizeExperiment(payload) {
    return this._fetch("/api/optimize", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async generateSubmission(payload, onProgress) {
    const { run_id, ...arguments_ } = payload;
    if (arguments_.folds === undefined) delete arguments_.folds;
    return this._runJob(arguments_.folds === undefined ? "submission" : "oof", run_id, arguments_, onProgress);
  }

  async getJobs(runId) {
    return this._fetch(`/api/jobs${runId ? `?run_id=${encodeURIComponent(runId)}` : ""}`);
  }

  async getJob(jobId) {
    return this._fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
  }

  async controlJob(jobId, action) {
    return this._fetch(`/api/jobs/${encodeURIComponent(jobId)}/${action}`, { method: "POST", body: "{}" });
  }

  async submitJob(operation, runId, payload, key = crypto.randomUUID()) {
    return this._fetch("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ operation, run_id: runId, payload, idempotency_key: key }),
    });
  }

  async _runJob(operation, runId, payload, onProgress) {
    const { job_id } = await this.submitJob(operation, runId, payload);
    for (;;) {
      const job = await this.getJob(job_id);
      onProgress?.(job);
      if (job.status === "completed") return job.result;
      if (["failed", "cancelled", "interrupted"].includes(job.status)) {
        throw new Error(`${job.id}: ${job.error || job.message}`);
      }
      await new Promise(resolve => setTimeout(resolve, 750));
    }
  }

  async sendAgentAction(hypothesisId, action) {
    return this._fetch("/api/agent/action", {
      method: "POST",
      body: JSON.stringify({ hypothesis_id: hypothesisId, action }),
    });
  }

  async calculateDerivedFeature(payload) {
    return this._fetch("/api/features/calculate", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async applyDerivedFeature(payload) {
    return this._fetch("/api/features/apply", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async suggestDerivedFeatures(datasetId) {
    return this._fetch("/api/features/suggest", {
      method: "POST",
      body: JSON.stringify({ dataset_id: datasetId }),
    });
  }

  getModelExportUrl(runId, experimentId, trialId) {
    let url = `/api/models/export?run_id=${encodeURIComponent(runId || '')}`;
    if (experimentId) url += `&experiment_id=${encodeURIComponent(experimentId)}`;
    if (trialId) url += `&trial_id=${encodeURIComponent(trialId)}`;
    return url;
  }

  async getModelExportInfo(runId, experimentId, trialId) {
    let url = `/api/models/export-info?run_id=${encodeURIComponent(runId || '')}`;
    if (experimentId) url += `&experiment_id=${encodeURIComponent(experimentId)}`;
    if (trialId) url += `&trial_id=${encodeURIComponent(trialId)}`;
    return this._fetch(url);
  }

  async buildEnsemble(payload) {
    return this._fetch("/api/ensemble/build", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }

  async uploadKaggleTemplate(payload) {
    return this._fetch("/api/kaggle/upload-template", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  }
}

export const api = new CATMLApiClient();
