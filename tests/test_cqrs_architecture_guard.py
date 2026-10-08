import re
from pathlib import Path
import pytest
import pandas as pd

from automl.application.bootstrap import build_application
from automl.application.queries.workspace_queries import (
    GetRunQuery,
    ListRunsQuery,
    GetDatasetQuery,
    ListDatasetsQuery,
    GetTrialQuery,
    GetExperimentQuery,
    ListTrialResultsQuery,
    GetDatasetProfileQuery,
)


def test_no_repository_or_private_state_leaks_in_interfaces_and_facade():
    repo_root = Path(__file__).resolve().parent.parent
    interfaces_dir = repo_root / "src" / "automl" / "interfaces"
    facade_file = repo_root / "src" / "automl" / "facade.py"

    forbidden_patterns = [
        re.compile(r"\bws\.repository\b"),
        re.compile(r"\bworkspace\.repository\b"),
        re.compile(r"\bws\._runs\b"),
        re.compile(r"\bws\._datasets\b"),
    ]

    files_to_check = list(interfaces_dir.glob("**/*.py")) + [facade_file]

    violations = []
    for f in files_to_check:
        lines = f.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in forbidden_patterns:
                if pattern.search(line):
                    violations.append(f"{f.relative_to(repo_root)}:{idx} -> {line.strip()}")

    assert not violations, f"Architectural CQRS leak detected:\n" + "\n".join(violations)


def test_cqrs_queries_dispatch_via_query_bus(tmp_path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path))

    df = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [10, 20, 30, 40], "target": [0, 1, 0, 1]})
    dataset = ws.register_dataset(name="cqrs_test", path=df, target="target")
    run = ws.create_run(dataset)

    # 1. Test GetRunQuery & ListRunsQuery
    found_run = qry.dispatch(GetRunQuery(run.id))
    assert found_run is not None
    assert found_run.id == run.id

    all_runs = qry.dispatch(ListRunsQuery())
    assert any(r.id == run.id for r in all_runs)

    runs_by_ds = qry.dispatch(ListRunsQuery(dataset_id=dataset.id))
    assert len(runs_by_ds) == 1
    assert runs_by_ds[0].id == run.id

    # 2. Test GetDatasetQuery & ListDatasetsQuery
    found_ds = qry.dispatch(GetDatasetQuery(dataset.id))
    assert found_ds is not None
    assert found_ds.id == dataset.id

    all_datasets = qry.dispatch(ListDatasetsQuery())
    assert any(d.id == dataset.id for d in all_datasets)

    # 3. Test GetDatasetProfileQuery
    profile = qry.dispatch(GetDatasetProfileQuery(dataset.id))
    assert profile is not None
    assert profile.dataset_id == dataset.id

    # 4. Create an experiment and test GetExperimentQuery, GetTrialQuery, ListTrialResultsQuery
    exp = ws.create_experiment(run, name="exp_test", feature_names=["feat1", "feat2"], model_ids=["random_forest"])
    found_exp = qry.dispatch(GetExperimentQuery(exp.id))
    assert found_exp is not None
    assert found_exp.id == exp.id

    # Run experiment
    results = ws.run_experiment(run, exp)
    assert len(results) > 0

    trial_results = qry.dispatch(ListTrialResultsQuery(exp.id))
    assert len(trial_results) == len(results)

    trial_id = trial_results[0].trial_id
    found_trial = qry.dispatch(GetTrialQuery(trial_id))
    assert found_trial is not None
    assert found_trial.id == trial_id


def test_modular_registries_can_be_composed_independently(tmp_path):
    """
    Validates that each sub-context registry (core, job, experiment, feature, inference)
    can be composed modularly on clean buses without monolithic coupling.
    """
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.queries.workspace_queries import ListTaskTypesQuery
    from automl.application.registries import (
        register_core_handlers,
        register_experiment_handlers,
        register_feature_handlers,
        register_inference_handlers,
        register_job_handlers,
    )
    from automl.application.services.workspace import AutoMLWorkspace

    ws = AutoMLWorkspace.load_or_create("clean_modular", root_dir=tmp_path / "mod_ws")
    cb = CommandBus()
    qb = QueryBus()

    # Core only
    register_core_handlers(ws, cb, qb)
    task_types = qb.dispatch(ListTaskTypesQuery())
    assert len(task_types) > 0

    # Independent composition of other registries
    register_job_handlers(ws, cb, qb)
    register_experiment_handlers(ws, cb, qb)
    register_feature_handlers(ws, cb, qb)
    register_inference_handlers(ws, cb, qb)

