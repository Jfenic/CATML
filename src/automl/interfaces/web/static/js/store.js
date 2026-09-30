/**
 * WorkbenchStore — State / Observable Pattern
 * Manages the reactive state of the CATML Workbench.
 */
export class WorkbenchStore {
  constructor() {
    this._state = {
      currentNav: "overview",
      activeRunId: null,
      activeExperimentId: null,
      snapshotRunId: null,
      jobs: [],
      events: [],
      lastUpdated: null,
      activeDatasetId: null,
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

  // Helper actions
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
}

export const store = new WorkbenchStore();
