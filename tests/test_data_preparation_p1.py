"""Data Preparation P1: scope guards, preview purity, apply immutability."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.preparation.commands import (
    AnalyzeDataQualityCommand,
    ApplyPreparationPlanCommand,
    CreatePreparationPlanCommand,
    PreviewPreparationPlanCommand,
    SuggestPreparationPlanCommand,
    ValidatePreparationPlanCommand,
)
from automl.application.preparation.queries import GetPreparationPlanQuery, ListDatasetVersionsQuery
from automl.domain.preparation.models import (
    PreparationOperation,
    PreparationScope,
    PreparationStep,
)
from automl.domain.preparation.validation import PreparationValidationError, assert_step_scope_legal


def _register_csv(tmp_path: Path, name: str = "prep.csv") -> tuple:
    df = pd.DataFrame(
        {
            "a": [1, 2, 2, None],
            "b": [" x ", "y", "y", "z"],
            "const": [1, 1, 1, 1],
            "target": [0, 1, 0, 1],
        }
    )
    path = tmp_path / name
    df.to_csv(path, index=False)
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))
    ds = ws.register_dataset(name="prep_ds", path=str(path), target="target", task_type="binary_classification")
    return ws, cmd, qry, ds, path


def test_learning_op_rejects_dataset_scope():
    step = PreparationStep(
        operation=PreparationOperation.MARK_IMPUTE,
        scope=PreparationScope.DATASET,
        columns=("a",),
    )
    with pytest.raises(PreparationValidationError, match="training_pipeline"):
        assert_step_scope_legal(step)


def test_mark_impute_ok_on_training_pipeline():
    step = PreparationStep(
        operation=PreparationOperation.MARK_IMPUTE,
        scope=PreparationScope.TRAINING_PIPELINE,
        columns=("a",),
    )
    assert_step_scope_legal(step)


def test_analyze_suggest_preview_apply_parity(tmp_path: Path):
    _ws, cmd, qry, ds, original = _register_csv(tmp_path)
    original_bytes = original.read_bytes()

    report = cmd.dispatch(AnalyzeDataQualityCommand(dataset_id=ds.id))
    assert report["row_count"] == 4
    assert any(i["code"] == "nulls" for i in report["issues"])
    assert any(i["code"] == "constant_column" for i in report["issues"])

    suggested = cmd.dispatch(SuggestPreparationPlanCommand(dataset_id=ds.id))
    plan_id = suggested["id"]
    assert suggested["status"] == "suggested"

    preview = cmd.dispatch(PreviewPreparationPlanCommand(plan_id=plan_id))
    assert preview["mutated_stored_dataset"] is False
    assert original.read_bytes() == original_bytes

    validated = cmd.dispatch(ValidatePreparationPlanCommand(plan_id=plan_id))
    assert validated["ok"] is True

    # Drop constant requires approval
    with pytest.raises(PreparationValidationError, match="approval"):
        cmd.dispatch(ApplyPreparationPlanCommand(plan_id=plan_id, approved=False))

    applied = cmd.dispatch(ApplyPreparationPlanCommand(plan_id=plan_id, approved=True))
    assert applied["original_path_intact"] is True
    assert original.read_bytes() == original_bytes
    assert applied["registered_dataset_id"]
    assert Path(applied["path"]).is_file()

    versions = qry.dispatch(ListDatasetVersionsQuery(parent_dataset_id=ds.id))
    assert len(versions) == 1
    assert versions[0]["id"] == applied["version_id"]

    plan = qry.dispatch(GetPreparationPlanQuery(plan_id=plan_id))
    assert plan["status"] == "applied"


def test_create_plan_illegal_impute_on_dataset(tmp_path: Path):
    _ws, cmd, _qry, ds, _path = _register_csv(tmp_path)
    with pytest.raises(PreparationValidationError):
        cmd.dispatch(
            CreatePreparationPlanCommand(
                dataset_id=ds.id,
                steps=(
                    {
                        "operation": "mark_impute",
                        "scope": "dataset",
                        "columns": ["a"],
                    },
                ),
            )
        )


def test_apply_drop_columns_creates_new_dataset(tmp_path: Path):
    ws, cmd, _qry, ds, original = _register_csv(tmp_path)
    plan_id = cmd.dispatch(
        CreatePreparationPlanCommand(
            dataset_id=ds.id,
            steps=(
                {
                    "operation": "drop_columns",
                    "scope": "dataset",
                    "columns": ["const"],
                    "requires_approval": True,
                },
                {
                    "operation": "normalize_whitespace",
                    "scope": "dataset",
                    "columns": ["b"],
                },
            ),
        )
    )
    applied = cmd.dispatch(ApplyPreparationPlanCommand(plan_id=plan_id, approved=True))
    new_ds = ws.get_dataset(applied["registered_dataset_id"])
    prepared = pd.read_csv(new_ds.path)
    assert "const" not in prepared.columns
    assert prepared["b"].iloc[0] == "x"
    assert Path(original).exists()
