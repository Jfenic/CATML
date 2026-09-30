import { readFile } from "node:fs/promises";
import { test } from "node:test";
import assert from "node:assert/strict";

const source = await readFile(new URL("../../src/automl/interfaces/web/static/js/api.js", import.meta.url), "utf8");
const { CATMLApiClient } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

test("experiment submission polls persisted result and reports actual progress", async () => {
  const api = new CATMLApiClient();
  const calls = [];
  const snapshot = { id: "job_1", status: "completed", completed: 2, total: 2, result: { primary_score: 0.8 } };
  api._fetch = async (path, options) => {
    calls.push({ path, body: options?.body && JSON.parse(options.body) });
    return path === "/api/jobs" ? { job_id: "job_1" } : snapshot;
  };
  const progress = [];
  const result = await api.createAndRunExperiment({ run_id: "run", models: ["lightgbm", "xgboost"] }, job => progress.push(job));
  assert.deepEqual(result, snapshot.result);
  assert.deepEqual(progress, [snapshot]);
  assert.equal(calls[0].body.operation, "experiment");
  assert.deepEqual(calls[0].body.payload.model_ids, ["lightgbm", "xgboost"]);
  assert.equal(calls[1].path, "/api/jobs/job_1");
});

test("OOF and ordinary submissions use distinct allowlisted operations", async () => {
  const api = new CATMLApiClient();
  const jobs = [];
  api._runJob = async (...args) => { jobs.push(args); return {}; };
  await api.generateSubmission({ run_id: "run", test_dataset_path: "test.csv", folds: 5 });
  await api.generateSubmission({ run_id: "run", test_dataset_path: "test.csv", folds: undefined });
  assert.equal(jobs[0][0], "oof");
  assert.equal(jobs[0][2].folds, 5);
  assert.equal(jobs[1][0], "submission");
  assert.equal("folds" in jobs[1][2], false);
  assert.equal("run_id" in jobs[0][2], false);
});

test("lost-response retries can reuse a caller-provided idempotency key", async () => {
  const api = new CATMLApiClient();
  const payloads = [];
  api._fetch = async (_path, options) => { payloads.push(JSON.parse(options.body)); return { job_id: "job_1" }; };
  await api.submitJob("experiment", "run", { model_ids: ["lightgbm"] }, "stable-key");
  await api.submitJob("experiment", "run", { model_ids: ["lightgbm"] }, "stable-key");
  assert.deepEqual(payloads[0], payloads[1]);
  assert.equal(payloads[0].idempotency_key, "stable-key");
});

test("terminal job errors surface with their durable ID", async () => {
  const api = new CATMLApiClient();
  api._fetch = async path => path === "/api/jobs" ? { job_id: "job_1" } : { id: "job_1", status: "failed", error: "training failed" };
  await assert.rejects(api.runExperiment({ run_id: "run" }), /job_1: training failed/);
});
