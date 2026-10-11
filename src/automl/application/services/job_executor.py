"""Execute the allowlisted operations through the same application commands."""

from automl.application.commands.workspace_commands import (
    CreateExperimentCommand, GenerateOOFSubmissionCommand, GenerateSubmissionCommand,
)
from automl.application.queries.workspace_queries import GetOOFResultQuery
from automl.domain.jobs.job import Job, JobStatus, ACTIVE_STATUSES
from automl.domain.ports import JobRepositoryPort
from automl.domain.experiments.trial import ExperimentStatus


class JobStopped(RuntimeError):
    def __init__(self, status: JobStatus):
        super().__init__(f"Job {status.value}")
        self.status = status


class JobExecutor:
    def __init__(self, workspace, command_bus, query_bus, repository: JobRepositoryPort, stopping=lambda: False):
        self.workspace = workspace
        self.commands = command_bus
        self.queries = query_bus
        self.repository = repository
        self.stopping = stopping

    def execute(self, job: Job) -> dict:
        def check():
            current = self.repository.get(job.id)
            if current.status == JobStatus.CANCEL_REQUESTED:
                raise JobStopped(JobStatus.CANCELLED)
            if current.status == JobStatus.PAUSE_REQUESTED:
                raise JobStopped(JobStatus.PAUSED)
            if self.stopping():
                raise JobStopped(JobStatus.INTERRUPTED)

        def progress(completed: int, total: int, message: str):
            self.repository.change(job.id, ACTIVE_STATUSES, completed=completed, total=total, message=message)
            check()

        self.workspace.execution_check = check
        self.workspace.execution_progress = progress
        check()
        if job.operation == "experiment":
            experiment_id = job.result.get("experiment_id") or job.payload.get("experiment_id")
            if experiment_id is None:
                args = dict(job.payload)
                budget = args.pop("budget", None)
                args.pop("mode", None)
                if budget:
                    budget_map = {"quick": 60.0, "balanced": 300.0, "thorough": 1200.0}
                    time_sec = budget_map.get(budget, 300.0) if isinstance(budget, str) else float(budget)
                    r = self.workspace._get_run(job.run_id)
                    r.config.time_budget_seconds = time_sec
                    self.workspace.repository.save_run(r)
                if args.get("feature_names") is None:
                    run = self.workspace._get_run(job.run_id)
                    dataset = self.workspace._get_dataset(run.dataset_id)
                    profile = self.workspace.repository.get_dataset_profile(dataset.id)
                    if profile and hasattr(profile, "resolve_safe_feature_names"):
                        args["feature_names"] = profile.resolve_safe_feature_names(
                            exclude_columns=[run.config.group_column] if run.config.group_column else None,
                        )
                    else:
                        args["feature_names"] = [c.name for c in profile.columns
                                                 if not c.is_identifier and c.name != dataset.target_column]
                args.setdefault("name", "Background experiment")
                valid_cmd_keys = {
                    "name", "feature_names", "feature_set_id", "model_ids",
                    "hypothesis", "priority", "validation_strategy", "group_column", "allow_leakage"
                }
                cmd_args = {k: v for k, v in args.items() if k in valid_cmd_keys}
                experiment = self.commands.dispatch(CreateExperimentCommand(run_id=job.run_id, **cmd_args))
                experiment_id = experiment.id
                self.repository.change(job.id, ACTIVE_STATUSES, result={"experiment_id": experiment_id})
            else:
                experiment = self.workspace.repository.get_experiment(experiment_id)
            if experiment is None or experiment.run_id != job.run_id:
                raise ValueError("Experiment does not belong to this run")
            # A checkpoint is written after each durable model result, before control checks.
            checkpoint = self.workspace.repository.get_checkpoint(job.run_id)
            start_index = checkpoint[1] if checkpoint and checkpoint[0] == experiment_id else job.completed
            successful_models = set()
            if job.retry_failed_models:
                start_index = 0
                successful_models = {r.model_id for r in self.workspace.repository.list_trial_results(experiment_id)
                                     if r.succeeded}
            try:
                self.workspace.run_experiment(self.workspace._get_run(job.run_id), experiment,
                                              start_index=start_index, skip_model_ids=successful_models)
            except JobStopped as exc:
                experiment.status = ExperimentStatus.CANCELLED if exc.status == JobStatus.CANCELLED else ExperimentStatus.PAUSED
                self.workspace.repository.save_experiment(experiment)
                raise
            if experiment.status == ExperimentStatus.PAUSED:
                raise JobStopped(JobStatus.PAUSED)
            if experiment.status == ExperimentStatus.CANCELLED:
                raise JobStopped(JobStatus.CANCELLED)
            check()
            all_results = self.workspace.repository.list_trial_results(experiment_id)
            if not all_results or not any(r.succeeded for r in all_results):
                raise RuntimeError("All experiment trials failed")
            best = self.workspace.get_best_trial(experiment_id)
            return {"status": "success", "experiment_id": experiment_id, "succeeded": True,
                    "model_id": best["model_id"], "primary_score": best["best_score"],
                    "primary_metric": best["metric"], "training_time_seconds": best["training_time_seconds"]}
        if job.operation == "oof":
            # Partial OOF fits are intentionally restarted; no fold model serialization yet.
            self.repository.change(job.id, ACTIVE_STATUSES, completed=0, total=0)
            experiment_id = self.commands.dispatch(GenerateOOFSubmissionCommand(run_id=job.run_id, **job.payload))
            report = self.queries.dispatch(GetOOFResultQuery(job.run_id, experiment_id))
            return {"status": "success", **report, "oof": report}
        progress(0, 1, "Fitting and exporting submission")
        result = self.commands.dispatch(GenerateSubmissionCommand(run_id=job.run_id, **job.payload))
        progress(1, 1, "Submission exported")
        return {"status": "success", **result, "oof": None}
