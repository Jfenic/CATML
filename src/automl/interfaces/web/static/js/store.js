/**
 * WorkbenchStore — State / Observable Pattern
 * Manages the reactive state of the CATML Workbench.
 * Enforces centralized context: activeDatasetId and activeRunId as single source of truth.
 * Rules:
 * 1. If a dataset is selected, active run MUST belong to that dataset.
 * 2. If the active dataset has no runs, activeRunId MUST be null (NEVER silently select a run from another dataset).
 * 3. Selecting a run from another dataset automatically switches activeDatasetId to match run.dataset_id.
 * 4. Stored local IDs are validated against existing datasets/runs upon resolution.
 * 5. Dataset changes increment contextVersion and invalidate stale run/experiment state.
 */
export class WorkbenchStore {
  constructor() {
    let savedDatasetId = null;
    let savedRunId = null;
    try {
      savedDatasetId = localStorage.getItem("catml_active_dataset_id");
      savedRunId = localStorage.getItem("catml_active_run_id");
    } catch (_) {}

    this._state = {
      currentNav: "home",
      activeRunId: savedRunId,
      activeDatasetId: savedDatasetId,
      activeDataset: null,
      activeRun: null,
      contextVersion: 1,
      datasets: [],
      overview: null,
      runs: [],
      experiments: [],
      leaderboard: [],
      datasetProfile: null,
      plan: null,
      knowledge: null,
      agentHypotheses: [],
      kaggleStatus: null,
      selectedExperimentsForCompare: [],
      agentDrawerOpen: false,
      newExperimentModalOpen: false,
      loading: false,
      error: null,
    };
    this._listeners = new Set();
  }

  getState() {
    return this._state;
  }

  setState(partialState) {
    const prevState = { ...this._state };
    this._state = { ...this._state, ...partialState };
    this._notify(this._state, prevState);
  }

  subscribe(listener) {
    this._listeners.add(listener);
    return () => this._listeners.delete(listener);
  }

  _notify(state, prevState) {
    for (const listener of this._listeners) {
      try {
        listener(state, prevState);
      } catch (err) {
        console.error("[WorkbenchStore] Listener error:", err);
      }
    }
  }

  // Navigation and Drawer actions
  setNav(navId) {
    this.setState({ currentNav: navId });
  }

  toggleAgentDrawer(open) {
    const nextState = open !== undefined ? open : !this._state.agentDrawerOpen;
    this.setState({ agentDrawerOpen: nextState });
  }

  toggleCompareExperiment(expId) {
    const current = [...this._state.selectedExperimentsForCompare];
    const index = current.indexOf(expId);
    if (index > -1) {
      current.splice(index, 1);
    } else {
      current.push(expId);
    }
    this.setState({ selectedExperimentsForCompare: current });
  }

  getContextVersion() {
    return this._state.contextVersion;
  }

  /**
   * Centralized Context Resolution
   * Validates activeDatasetId and activeRunId against live datasets and runs.
   */
  resolveContext({ datasets = [], runs = [], overview = null } = {}) {
    const prevDatasetId = this._state.activeDatasetId;
    let targetDatasetId = prevDatasetId;

    // Validate that targetDatasetId actually exists in backend datasets
    if (targetDatasetId && !datasets.some(d => d.id === targetDatasetId)) {
      targetDatasetId = null;
    }

    // If no valid dataset selected, select first available dataset (or null if empty)
    if (!targetDatasetId && datasets.length > 0) {
      targetDatasetId = datasets[0].id;
    }

    const activeDataset = datasets.find(d => d.id === targetDatasetId) || null;

    // Persist or clean storage
    try {
      if (targetDatasetId) {
        localStorage.setItem("catml_active_dataset_id", targetDatasetId);
      } else {
        localStorage.removeItem("catml_active_dataset_id");
      }
    } catch (_) {}

    // Resolve activeRunId strictly within targetDatasetId
    const datasetRuns = targetDatasetId ? runs.filter(r => r.dataset_id === targetDatasetId) : [];
    let targetRunId = this._state.activeRunId;

    if (targetRunId && !datasetRuns.some(r => r.id === targetRunId)) {
      // Current activeRunId does not belong to this dataset! Must not leak runs from other datasets!
      targetRunId = null;
    }

    if (!targetRunId && datasetRuns.length > 0) {
      // Pick active run if one is running, else latest run for THIS dataset
      const running = datasetRuns.find(r => r.status === "RUNNING" || r.status === "TRAINING");
      targetRunId = running ? running.id : datasetRuns[0].id;
    }

    try {
      if (targetRunId) {
        localStorage.setItem("catml_active_run_id", targetRunId);
      } else {
        localStorage.removeItem("catml_active_run_id");
      }
    } catch (_) {}

    const activeRun = targetRunId ? datasetRuns.find(r => r.id === targetRunId) || null : null;

    const datasetChanged = prevDatasetId !== targetDatasetId;
    const nextVersion = datasetChanged ? this._state.contextVersion + 1 : this._state.contextVersion;

    const patch = {
      datasets,
      runs,
      overview: overview !== undefined ? overview : this._state.overview,
      activeDatasetId: targetDatasetId,
      activeDataset,
      activeRunId: targetRunId,
      activeRun,
      contextVersion: nextVersion,
    };

    if (datasetChanged) {
      // Invalidate stale data from previous dataset
      patch.experiments = [];
      patch.leaderboard = [];
      patch.datasetProfile = null;
      patch.plan = null;
      patch.agentHypotheses = [];
      patch.selectedExperimentsForCompare = [];
    }

    this.setState(patch);
    return { activeDatasetId: targetDatasetId, activeRunId: targetRunId, activeDataset, activeRun };
  }

  /**
   * Switches active dataset and strictly aligns activeRunId
   */
  setActiveDataset(datasetId) {
    if (!datasetId) {
      try {
        localStorage.removeItem("catml_active_dataset_id");
        localStorage.removeItem("catml_active_run_id");
      } catch (_) {}
      this.setState({
        activeDatasetId: null,
        activeDataset: null,
        activeRunId: null,
        activeRun: null,
        experiments: [],
        leaderboard: [],
        datasetProfile: null,
        plan: null,
        agentHypotheses: [],
        selectedExperimentsForCompare: [],
        contextVersion: this._state.contextVersion + 1,
      });
      return;
    }

    const datasets = this._state.datasets || [];
    const ds = datasets.find(d => d.id === datasetId) || null;
    try {
      localStorage.setItem("catml_active_dataset_id", datasetId);
    } catch (_) {}

    const datasetRuns = (this._state.runs || []).filter(r => r.dataset_id === datasetId);
    const matchingRun = datasetRuns.find(r => r.status === "RUNNING" || r.status === "TRAINING") || datasetRuns[0] || null;

    try {
      if (matchingRun) {
        localStorage.setItem("catml_active_run_id", matchingRun.id);
      } else {
        localStorage.removeItem("catml_active_run_id");
      }
    } catch (_) {}

    this.setState({
      activeDatasetId: datasetId,
      activeDataset: ds,
      activeRunId: matchingRun ? matchingRun.id : null,
      activeRun: matchingRun,
      experiments: [],
      leaderboard: [],
      datasetProfile: null,
      plan: null,
      agentHypotheses: [],
      selectedExperimentsForCompare: [],
      contextVersion: this._state.contextVersion + 1,
    });
  }

  /**
   * Switches active run.
   * If run belongs to a different dataset, switches dataset context coherently.
   */
  setActiveRun(runId) {
    if (!runId) {
      try {
        localStorage.removeItem("catml_active_run_id");
      } catch (_) {}
      this.setState({ activeRunId: null, activeRun: null });
      return;
    }

    const runs = this._state.runs || [];
    const run = runs.find(r => r.id === runId) || null;
    if (!run) return;

    try {
      localStorage.setItem("catml_active_run_id", runId);
    } catch (_) {}

    if (run.dataset_id && run.dataset_id !== this._state.activeDatasetId) {
      // Switching to a run that belongs to a different dataset
      const datasets = this._state.datasets || [];
      const ds = datasets.find(d => d.id === run.dataset_id) || null;
      try {
        localStorage.setItem("catml_active_dataset_id", run.dataset_id);
      } catch (_) {}

      this.setState({
        activeDatasetId: run.dataset_id,
        activeDataset: ds,
        activeRunId: run.id,
        activeRun: run,
        experiments: [],
        leaderboard: [],
        datasetProfile: null,
        plan: null,
        agentHypotheses: [],
        selectedExperimentsForCompare: [],
        contextVersion: this._state.contextVersion + 1,
      });
    } else {
      this.setState({
        activeRunId: run.id,
        activeRun: run,
        experiments: [],
        leaderboard: [],
        plan: null,
      });
    }
  }

  getActiveDatasetId() {
    return this._state.activeDatasetId;
  }

  getActiveRunId() {
    return this._state.activeRunId;
  }

  getActiveDataset() {
    if (this._state.activeDataset) return this._state.activeDataset;
    if (this._state.activeDatasetId) {
      return (this._state.datasets || []).find(d => d.id === this._state.activeDatasetId) || null;
    }
    return null;
  }

  /**
   * Returns active run strictly verified against active dataset.
   * Never leaks a run from another dataset.
   */
  getActiveRun() {
    const { activeDatasetId, activeRunId, runs = [] } = this._state;
    if (!activeDatasetId || !activeRunId) return null;
    const run = runs.find(r => r.id === activeRunId);
    if (run && run.dataset_id === activeDatasetId) {
      return run;
    }
    return null;
  }

  getRunsForActiveDataset() {
    const { activeDatasetId, runs = [] } = this._state;
    if (!activeDatasetId) return [];
    return runs.filter(r => r.dataset_id === activeDatasetId);
  }
}

export const store = new WorkbenchStore();
