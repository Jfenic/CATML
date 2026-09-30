"""Job validation and controls: commands return IDs, queries return snapshots."""
from dataclasses import asdict
import json
import uuid

from automl.application.commands.workspace_commands import GenerateOOFSubmissionCommand, GenerateSubmissionCommand
from automl.domain.jobs.job import Job, JobStatus
from automl.domain.ports import JobRepositoryPort
from automl.domain.runs.states import RunStatus, RunPhase


class JobService:
    def __init__(self, workspace, repository: JobRepositoryPort):
        self.workspace = workspace
        self.repository = repository

    def submit(self, command) -> str:
        if not isinstance(command.idempotency_key, str) or not command.idempotency_key.strip():
            raise ValueError("A nonempty idempotency_key is required")
        if not isinstance(command.payload, dict):
            raise ValueError("payload must be an object")
        payload = json.loads(json.dumps(command.payload, allow_nan=False))
        run = self.workspace._get_run(command.run_id)
        if command.operation == "experiment":
            allowed = {"experiment_id", "name", "model_ids", "feature_names", "hypothesis"}
            if set(payload) - allowed:
                raise ValueError("Unknown experiment job arguments")
            if payload.get("experiment_id"):
                experiment = self.workspace.repository.get_experiment(payload["experiment_id"])
                if experiment is None or experiment.run_id != run.id:
                    raise ValueError("Experiment does not belong to this run")
                if len(payload) != 1:
                    raise ValueError("Existing experiments only accept experiment_id")
            else:
                models = payload.get("model_ids")
                if not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models):
                    raise ValueError("model_ids must be a nonempty list")
                dataset = self.workspace._get_dataset(run.dataset_id)
                self.workspace.model_registry.validate_for_task(models, dataset.task_type)
        elif command.operation in {"oof", "submission"}:
            cls = GenerateOOFSubmissionCommand if command.operation == "oof" else GenerateSubmissionCommand
            if "run_id" in payload:
                raise ValueError("Specify run_id outside payload")
            # Persist defaults as well, so equality remains stable across requests.
            payload = asdict(cls(run_id=run.id, **payload))
            payload.pop("run_id")
            experiment_id = payload.get("experiment_id")
            if experiment_id:
                experiment = self.workspace.repository.get_experiment(experiment_id)
                if experiment is None or experiment.run_id != run.id:
                    raise ValueError("Experiment does not belong to this run")
            if payload.get("trial_id"):
                trial = self.workspace.repository.get_trial(payload["trial_id"])
                experiment = self.workspace.repository.get_experiment(trial.experiment_id) if trial else None
                if experiment is None or experiment.run_id != run.id:
                    raise ValueError("Trial does not belong to this run")
            if command.operation == "oof":
                dataset = self.workspace._get_dataset(run.dataset_id)
                if dataset.task_type != "binary_classification":
                    raise ValueError("OOF currently supports binary classification only")
        else:
            raise ValueError("Supported job operations: experiment, oof, submission")
        return self.repository.enqueue(Job(id=f"job_{uuid.uuid4().hex}", operation=command.operation,
                                           run_id=run.id, payload=payload,
                                           idempotency_key=command.idempotency_key))

    def control(self, command) -> str:
        job = self.repository.get(command.job_id)
        action = command.action
        if action == "pause":
            target = JobStatus.PAUSED if job.status == JobStatus.QUEUED else JobStatus.PAUSE_REQUESTED
            allowed = {JobStatus.QUEUED, JobStatus.RUNNING}
        elif action == "cancel":
            target = JobStatus.CANCEL_REQUESTED if job.status in {JobStatus.RUNNING, JobStatus.PAUSE_REQUESTED} else JobStatus.CANCELLED
            allowed = {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.PAUSE_REQUESTED, JobStatus.PAUSED, JobStatus.INTERRUPTED, JobStatus.FAILED}
        elif action == "resume":
            target, allowed = JobStatus.QUEUED, {JobStatus.PAUSED}
        elif action == "retry":
            target, allowed = JobStatus.QUEUED, {JobStatus.FAILED, JobStatus.INTERRUPTED}
        else:
            raise ValueError("Supported actions: pause, resume, cancel, retry")
        changes = {"status": target, "message": f"{action.capitalize()} requested", "error": None}
        if action == "retry" and job.status == JobStatus.FAILED and job.operation == "experiment":
            changes["retry_failed_models"] = True
        self.repository.change(job.id, allowed & {job.status}, **changes)
        if action == "resume":
            run = self.workspace.repository.get_run(job.run_id)
            if run.status == RunStatus.PAUSED:
                run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
                self.workspace.repository.save_run(run)
        return job.id

    def get(self, job_id: str) -> dict:
        return self.repository.get(job_id).to_dict()

    def list(self, run_id: str | None) -> list[dict]:
        return [job.to_dict() for job in self.repository.list(run_id)]
