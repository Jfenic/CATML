/** One owner refreshes backend snapshots; views never invent missing data. */
export class WorkspaceSync {
  constructor(api, store) { this.api = api; this.store = store; this.refreshing = false; }
  async refresh() {
    if (this.refreshing) return;
    this.refreshing = true;
    try {
      const [overview, runs, jobs] = await Promise.all([this.api.getOverview(), this.api.getRuns(), this.api.getJobs()]);
      const requestedRun = this.store.getState().activeRunId;
      const runId = runs.some(run => run.id === requestedRun) ? requestedRun : runs[0]?.id || null;
      const [experiments, events] = runId ? await Promise.all([this.api.getExperiments(runId), this.api.getEvents(runId)]) : [[], []];
      if (this.store.getState().activeRunId !== requestedRun) return;
      const requestedExperiment = this.store.getState().activeExperimentId;
      const running = jobs.find(job => job.run_id === runId && ["running", "pause_requested", "cancel_requested"].includes(job.status));
      const experimentId = experiments.some(exp => exp.id === requestedExperiment) ? requestedExperiment : running?.result.experiment_id || experiments[0]?.id || null;
      this.store.setState({ overview, runs, jobs, experiments, events, activeRunId: runId, activeExperimentId: experimentId, snapshotRunId: runId, error: null, lastUpdated: new Date().toISOString() });
    } catch (error) {
      this.store.setState({ error: `No se pudo actualizar: ${error.message}` });
    } finally { this.refreshing = false; }
  }
  start() { this.timer = setInterval(() => { if (!document.hidden) this.refresh(); }, 3000); }
  stop() { clearInterval(this.timer); }
}
