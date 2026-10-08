from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.commands.workspace_commands import (
    BuildEnsembleCommand,
    ExecutePipelineCommand,
    GenerateOOFSubmissionCommand,
    GenerateSubmissionCommand,
)
from automl.application.queries.workspace_queries import (
    GetOOFResultQuery,
    GetPipelineExecutionOrderQuery,
    PredictDatasetQuery,
    ValidatePipelineGraphQuery,
)

if TYPE_CHECKING:
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.services.workspace import AutoMLWorkspace


def register_inference_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
) -> None:
    """Registers inference, prediction, pipeline DAG and ensembling handlers."""
    # Commands
    command_bus.register(
        GenerateSubmissionCommand,
        lambda cmd: workspace.generate_submission(
            run_id=cmd.run_id,
            test_dataset_path=cmd.test_dataset_path,
            output_path=cmd.output_path,
            id_column=cmd.id_column,
            template_path=cmd.template_path,
            experiment_id=cmd.experiment_id,
            trial_id=cmd.trial_id,
            predict_proba=cmd.predict_proba,
        ),
    )
    command_bus.register(
        ExecutePipelineCommand,
        lambda cmd: workspace.execute_pipeline(cmd.graph, cmd.inputs),
    )
    command_bus.register(
        BuildEnsembleCommand,
        lambda cmd: workspace.build_ensemble(
            run_id=cmd.run_id,
            model_ids=cmd.model_ids,
            method=cmd.method,
            meta_model=cmd.meta_model,
            folds=cmd.folds,
            name=cmd.name,
        ).id,
    )
    command_bus.register(GenerateOOFSubmissionCommand, workspace.generate_oof_submission)

    # Queries
    query_bus.register(GetOOFResultQuery, lambda q: workspace.get_oof_result(q.run_id, q.experiment_id))
    query_bus.register(
        PredictDatasetQuery,
        lambda q: workspace.predict(
            run_id=q.run_id,
            test_dataset_path=q.test_dataset_path,
            experiment_id=q.experiment_id,
            trial_id=q.trial_id,
            predict_proba=q.predict_proba,
            template_path=q.template_path,
            id_column=q.id_column,
        ),
    )
    query_bus.register(
        ValidatePipelineGraphQuery,
        lambda q: workspace.validate_pipeline_graph(q.graph),
    )
    query_bus.register(
        GetPipelineExecutionOrderQuery,
        lambda q: workspace.get_pipeline_execution_order(q.graph),
    )
