import { test } from "node:test";
import assert from "node:assert/strict";
import { load } from "./module_loader.mjs";
const { store, WorkbenchStore } = await load("store.js");
const { OverviewView } = await load("views/overview.js");
const { StudioView } = await load("views/studio.js");
const { WorkspaceSync } = await load("workspace_sync.js");
const { jobCard, score } = await load("ui.js");
const run = { id: "run_real", dataset_name: "Real dataset", metric: "rmse", task_type: "regression", target: "price", best_score: 0, best_model: "real_model", experiments_count: 1, trials_count: 1 };
const exp = { id: "exp_real", name: "Actual experiment", status: "COMPLETED", model_ids: ["real_model"], feature_names: ["a", "b"], features_count: 2, metric: "rmse", best_score: 0, validation_strategy: "holdout", trials: [{ trial_id: "trial_real", model_id: "real_model", score: 0, succeeded: true, time_s: 1, params: { depth: 3 } }] };
const container = () => ({ innerHTML: "", querySelector: () => null, querySelectorAll: () => [] });
function seed(partial = {}) { store.setState({ runs: [run], activeRunId: run.id, activeExperimentId: exp.id, snapshotRunId: run.id, experiments: [exp], jobs: [], events: [], overview: { total_trials: 1 }, error: null, ...partial }); }

test("overview uses saved records, preserves zero, and shows no fabricated running job", () => {
  seed(); const target = container(), view = new OverviewView(); view.mount(target);
  assert.match(target.innerHTML, /No hay entrenamientos en ejecución/);
  assert.match(target.innerHTML, /Actual experiment/);
  assert.match(target.innerHTML, /0\.00000/);
  assert.doesNotMatch(target.innerHTML, /CatBoost|Ensemble #7|CPU 67|0\.94621/);
  view.destroy();
});

test("studio shows selected experiment, actual validation, parameters and failed trial message", () => {
  seed({ experiments: [exp, { ...exp, id: "exp_other", name: "Other experiment" }], activeExperimentId: "exp_other" });
  const target = container(), view = new StudioView(); view.mount(target);
  assert.match(target.innerHTML, /Other experiment/);
  assert.match(target.innerHTML, /holdout/);
  assert.match(target.innerHTML, /depth/);
  assert.doesNotMatch(target.innerHTML, /Experiment #42|5-Fold Stratified|0\.94621/);
  store.setState({ experiments: [{ ...exp, trials: [{ ...exp.trials[0], succeeded: false, failure_reason: "Fit failed" }] }], activeExperimentId: exp.id });
  assert.match(target.innerHTML, /Fit failed/);
  view.destroy();
});

test("HTML values are escaped and snapshots from another run are not shown", () => {
  seed({ experiments: [{ ...exp, name: '<img src=x onerror="bad()">' }] });
  const target = container(), view = new StudioView(); view.mount(target);
  assert.match(target.innerHTML, /&lt;img/);
  assert.doesNotMatch(target.innerHTML, /<img src=x/);
  store.setState({ activeRunId: "run_other", runs: [run, { ...run, id: "run_other" }], snapshotRunId: run.id });
  assert.match(target.innerHTML, /No hay experimentos guardados/);
  view.destroy();
});

test("job cards derive controls and progress from durable job state", () => {
  const job = { id: "job_real", run_id: run.id, payload: {}, result: {}, status: "running", attempt: 1, completed: 1, total: 2, message: "Finished model A" };
  assert.match(jobCard(job), /1\/2 pasos terminados/);
  assert.match(jobCard(job), /data-control="pause"/);
  assert.doesNotMatch(jobCard({ ...job, status: "completed" }), /data-control=/);
  assert.equal(score(null), "—"); assert.equal(score(0), "0.00000");
});

test("polling keeps selected run and experiment and does not replace failed reads with mock results", async () => {
  const local = new WorkbenchStore(); local.setState({ activeRunId: run.id, activeExperimentId: exp.id });
  const api = { getOverview: async () => ({ total_trials: 1 }), getRuns: async () => [run], getJobs: async () => [], getExperiments: async () => [exp], getEvents: async () => [{ event_type: "TrialCompleted" }] };
  const sync = new WorkspaceSync(api, local); await sync.refresh();
  assert.equal(local.getState().activeExperimentId, exp.id);
  assert.equal(local.getState().snapshotRunId, run.id);
  api.getRuns = async () => { throw Error("connection lost"); };
  await sync.refresh();
  assert.match(local.getState().error, /connection lost/);
  assert.deepEqual(local.getState().experiments, [exp]);
});

test("late responses do not overwrite a newly selected run", async () => {
  const local = new WorkbenchStore(); local.setState({ activeRunId: run.id });
  let release;
  const waiting = new Promise(resolve => { release = resolve; });
  const api = { getOverview: async () => ({}), getRuns: async () => [run], getJobs: async () => [], getExperiments: () => waiting, getEvents: async () => [] };
  const sync = new WorkspaceSync(api, local), refresh = sync.refresh();
  await new Promise(resolve => setImmediate(resolve));
  local.setState({ activeRunId: "new_run", experiments: [] });
  release([exp]); await refresh;
  assert.equal(local.getState().activeRunId, "new_run");
  assert.deepEqual(local.getState().experiments, []);
});
