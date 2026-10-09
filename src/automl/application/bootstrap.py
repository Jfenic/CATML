from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.application.registries import (
    register_analysis_handlers,
    register_core_handlers,
    register_experiment_handlers,
    register_feature_handlers,
    register_inference_handlers,
    register_job_handlers,
)
from automl.application.services.workspace import AutoMLWorkspace

if TYPE_CHECKING:
    from automl.application.services.jobs import JobService


def register_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
    job_service: JobService | None = None,
) -> None:
    """
    Registers all CQRS command and query handlers by domain context.
    
    Sub-context registries:
      - Job registry: asynchronous task queue and job management
      - Core registry: workspaces, runs, datasets, dataset profiles, task types, plugins
      - Experiment registry: candidate models, trials, training executions, priority queues
      - Feature registry: feature discovery, selection, ablation, metadata lineages
      - Inference registry: predictions, submissions, ensembles, OOF, and pipelines
      - Analysis registry: exploratory statistical studies, runs, findings, visualizations
    """
    register_job_handlers(workspace, command_bus, query_bus, job_service=job_service)
    register_core_handlers(workspace, command_bus, query_bus)
    register_experiment_handlers(workspace, command_bus, query_bus)
    register_feature_handlers(workspace, command_bus, query_bus)
    register_inference_handlers(workspace, command_bus, query_bus)
    register_analysis_handlers(workspace, command_bus, query_bus)


def build_application(
    root_dir: str | None = None,
    job_service: JobService | None = None,
) -> tuple[AutoMLWorkspace, CommandBus, QueryBus]:
    """
    Application composition root.
    
    Constructs the AutoMLWorkspace, instantiates CommandBus and QueryBus,
    and registers all modular handlers.
    """
    workspace = AutoMLWorkspace.load_or_create("default", root_dir=root_dir)
    command_bus = CommandBus()
    query_bus = QueryBus()
    register_handlers(workspace, command_bus, query_bus, job_service=job_service)
    return workspace, command_bus, query_bus
