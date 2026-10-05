"""Comprehensive unit and integration test suite for Temporal Dynamics Engine (Phase 2)."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import GenerateTemporalFeaturesCommand
from automl.application.queries.workspace_queries import DetectTemporalStructureQuery
from automl.domain.features.feature_set import FeatureSet
from automl.domain.features.temporal import (
    CyclicalSpec,
    DeltaSpec,
    GeneratedTemporalFeature,
    LagSpec,
    RollingWindowSpec,
    TemporalPeriodicity,
    TemporalStructure,
)
from automl.engine.features.generation.temporal_generator import TemporalDynamicsGenerator
from automl.engine.profiling.dataset_profiler import detect_sequential_structure, profile_dataset
from automl.interfaces.cli.main import features_temporal_cli


class TestTemporalDomainContracts:
    """Verify pure domain contracts, hexagonal boundaries, and DTO roundtrips."""

    def test_domain_hexagonal_boundary_purity(self):
        """Domain contracts must not import pandas, numpy, scikit-learn, or external libraries."""
        import automl.domain.features.temporal as temporal_mod

        source = inspect.getsource(temporal_mod)
        for forbidden in ("pandas", "numpy", "sklearn", "scipy", "sqlite3"):
            assert forbidden not in source, f"Forbidden framework import '{forbidden}' found in pure domain module"

    def test_temporal_periodicity_roundtrip(self):
        p = TemporalPeriodicity(name="hourly", period=24.0, column="hour", description="24h diurnal")
        d = p.to_dict()
        assert d["name"] == "hourly"
        assert d["period"] == 24.0
        assert d["column"] == "hour"

        reconstructed = TemporalPeriodicity.from_dict(d)
        assert reconstructed == p

    def test_specs_roundtrip(self):
        lag = LagSpec(column="sensor_a", lag=2, feature_name="temp_lag_2_sensor_a", description="lag 2")
        assert LagSpec.from_dict(lag.to_dict()) == lag

        delta = DeltaSpec(column="sensor_a", lag=1, feature_name="temp_delta_1_sensor_a", description="delta 1")
        assert DeltaSpec.from_dict(delta.to_dict()) == delta

        cyc = CyclicalSpec(
            column="month",
            period=12.0,
            sin_feature_name="temp_sin_12_month",
            cos_feature_name="temp_cos_12_month",
        )
        assert CyclicalSpec.from_dict(cyc.to_dict()) == cyc

        roll = RollingWindowSpec(
            column="sensor_a",
            window=5,
            agg="mean",
            feature_name="temp_roll_mean_5_sensor_a",
        )
        assert RollingWindowSpec.from_dict(roll.to_dict()) == roll

    def test_temporal_structure_roundtrip(self):
        ts = TemporalStructure(
            is_sequential=True,
            order_column="timestamp",
            detected_periodicities=[
                TemporalPeriodicity(name="hourly", period=24.0, column="hour"),
                TemporalPeriodicity(name="weekly", period=7.0, column="dow"),
            ],
            temporal_columns=["timestamp", "hour", "dow"],
            metadata={"source": "telemetry"},
        )
        d = ts.to_dict()
        assert d["is_sequential"] is True
        assert d["order_column"] == "timestamp"
        assert len(d["detected_periodicities"]) == 2

        reconstructed = TemporalStructure.from_dict(d)
        assert reconstructed.is_sequential == ts.is_sequential
        assert reconstructed.order_column == ts.order_column
        assert len(reconstructed.detected_periodicities) == 2
        assert reconstructed.temporal_columns == ["timestamp", "hour", "dow"]

    def test_generated_temporal_feature_roundtrip(self):
        feat = GeneratedTemporalFeature(
            name="temp_delta_1_pressure",
            feature_type="delta",
            source_column="pressure",
            lag=1,
            description="Pressure rate of change",
        )
        d = feat.to_dict()
        assert d["feature_type"] == "delta"
        assert GeneratedTemporalFeature.from_dict(d) == feat


class TestTemporalProfilingDetection:
    """Test automated discovery of chronological sequences and cyclical periodicities."""

    def test_detects_datetime_iso_strings(self):
        df = pd.DataFrame({
            "recorded_at": [
                "2026-01-01 00:00:00",
                "2026-01-01 01:00:00",
                "2026-01-01 02:00:00",
                "2026-01-01 03:00:00",
                "2026-01-01 04:00:00",
            ],
            "temperature": [18.2, 18.0, 17.5, 17.1, 16.9],
            "target": [0, 0, 1, 0, 1],
        })
        ts = detect_sequential_structure(df, target_column="target")
        assert ts.is_sequential is True
        assert ts.order_column == "recorded_at"
        assert "recorded_at" in ts.temporal_columns

    def test_detects_sequential_monotonic_integer_step(self):
        df = pd.DataFrame({
            "step_id": [101, 102, 103, 104, 105, 106, 107],
            "sensor_vibration": [0.12, 0.15, 0.14, 0.19, 0.22, 0.25, 0.21],
            "churn": [0, 0, 0, 1, 1, 1, 0],
        })
        ts = detect_sequential_structure(df, target_column="churn")
        assert ts.is_sequential is True
        assert ts.order_column == "step_id"
        assert "step_id" in ts.temporal_columns

    def test_detects_cyclical_periodicities(self):
        df = pd.DataFrame({
            "hour_of_day": [0, 6, 12, 18, 23],
            "weekday": [0, 1, 2, 3, 4],
            "month": [1, 3, 6, 9, 12],
            "load_mw": [120.5, 180.2, 340.1, 290.4, 150.0],
            "failure": [0, 0, 0, 0, 1],
        })
        ts = detect_sequential_structure(df, target_column="failure")
        assert ts.is_sequential is True
        p_names = {p.name: p.period for p in ts.detected_periodicities}
        assert "hourly" in p_names and p_names["hourly"] == 24.0
        assert "weekly" in p_names and p_names["weekly"] == 7.0
        assert "monthly" in p_names and p_names["monthly"] == 12.0

    def test_non_sequential_dataset_yields_false(self):
        df = pd.DataFrame({
            "customer_age": [45, 23, 67, 34, 52],
            "credit_score": [720, 610, 805, 590, 680],
            "geo_category": ["FR", "ES", "DE", "FR", "ES"],
            "churn": [0, 1, 0, 1, 0],
        })
        ts = detect_sequential_structure(df, target_column="churn")
        assert ts.is_sequential is False
        assert ts.order_column is None
        assert len(ts.detected_periodicities) == 0

    def test_dataset_profile_integration_and_recommendation(self, tmp_path: Path):
        csv_path = tmp_path / "telemetry.csv"
        df = pd.DataFrame({
            "timestamp": ["2026-01-01 01:00", "2026-01-01 02:00", "2026-01-01 03:00", "2026-01-01 04:00"],
            "sensor_1": [10.0, 12.0, 15.0, 14.0],
            "hour": [1, 2, 3, 4],
            "alarm": [0, 0, 1, 0],
        })
        df.to_csv(csv_path, index=False)

        from automl.domain.datasets.profile import Dataset
        ds = Dataset(id="ds-telemetry", workspace_id="ws", name="Telemetry", path=str(csv_path), target_column="alarm", task_type="binary_classification")
        profile = profile_dataset(ds, df=df)

        assert "temporal_structure" in profile.to_dict()
        assert profile.temporal_structure["is_sequential"] is True
        # Check that sequential recommendation badge is created
        assert any(r.get("badge") == "Sequential Dynamics" for r in profile.recommendations)


class TestTemporalDynamicsGenerator:
    """Test feature proposal, mathematical precision, immutability, and candidate packaging."""

    @pytest.fixture
    def sequential_df(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 30
        return pd.DataFrame({
            "step": np.arange(n),
            "hour": np.tile(np.arange(24), 2)[:n],
            "pressure": np.linspace(100.0, 200.0, n) + np.random.normal(0, 2, n),
            "temperature": np.sin(np.linspace(0, 4 * np.pi, n)) * 20.0 + 50.0,
            "target": np.random.choice([0, 1], size=n),
        })

    def test_propose_features_creates_lags_deltas_cycles(self, sequential_df: pd.DataFrame):
        gen = TemporalDynamicsGenerator(max_lags=2, include_lags=True, include_deltas=True, include_cyclical=True)
        feats = gen.propose_features(
            df=sequential_df,
            feature_names=["step", "hour", "pressure", "temperature"],
            target_column="target",
        )
        types = {f.feature_type for f in feats}
        assert "lag" in types
        assert "delta" in types
        assert "cyclical_sin" in types
        assert "cyclical_cos" in types

        names = [f.name for f in feats]
        assert "temp_lag_1_pressure" in names
        assert "temp_lag_2_pressure" in names
        assert "temp_delta_1_pressure" in names
        assert "temp_sin_24_hour" in names
        assert "temp_cos_24_hour" in names

    def test_transform_is_strictly_immutable(self, sequential_df: pd.DataFrame):
        original_copy = sequential_df.copy(deep=True)
        gen = TemporalDynamicsGenerator(max_lags=1)
        feats = gen.propose_features(sequential_df, ["pressure", "temperature"], target_column="target")

        transformed = gen.transform(sequential_df, feats)

        # Original dataframe is completely unchanged
        pd.testing.assert_frame_equal(sequential_df, original_copy)
        # New columns are added
        assert len(transformed.columns) > len(sequential_df.columns)
        assert "temp_lag_1_pressure" in transformed.columns
        assert "temp_delta_1_pressure" in transformed.columns

    def test_mathematical_precision_of_lags_and_deltas(self):
        df = pd.DataFrame({
            "val": [10.0, 20.0, 35.0, 50.0, 70.0],
            "target": [0, 0, 1, 1, 0],
        })
        gen = TemporalDynamicsGenerator(max_lags=2, include_cyclical=False)
        feats = gen.propose_features(df, ["val"], target_column="target")
        res = gen.transform(df, feats)

        # Lag 1: shifted by 1, initial filled with bfill/median without NaN
        assert not res["temp_lag_1_val"].isna().any()
        # From row 1 onwards, lag_1 must equal previous val
        assert list(res["temp_lag_1_val"].iloc[1:]) == [10.0, 20.0, 35.0, 50.0]

        # Delta 1: X_t - X_{t-1}
        # Row 1: 20 - 10 = 10, Row 2: 35 - 20 = 15, Row 3: 50 - 35 = 15, Row 4: 70 - 50 = 20
        assert list(res["temp_delta_1_val"].iloc[1:]) == [10.0, 15.0, 15.0, 20.0]

    def test_mathematical_precision_of_cyclical_sin_cos(self):
        df = pd.DataFrame({
            "hour": [0.0, 6.0, 12.0, 18.0, 24.0],
            "target": [0, 1, 0, 1, 0],
        })
        ts = TemporalStructure(
            is_sequential=True,
            detected_periodicities=[TemporalPeriodicity(name="hourly", period=24.0, column="hour")],
        )
        gen = TemporalDynamicsGenerator(include_lags=False, include_deltas=False, include_cyclical=True)
        feats = gen.propose_features(df, ["hour"], target_column="target", temporal_structure=ts)
        res = gen.transform(df, feats)

        # Hour 0: sin(0) = 0.0, cos(0) = 1.0
        assert np.isclose(res["temp_sin_24_hour"].iloc[0], 0.0, atol=1e-5)
        assert np.isclose(res["temp_cos_24_hour"].iloc[0], 1.0, atol=1e-5)

        # Hour 6: sin(2pi * 6/24) = sin(pi/2) = 1.0, cos(pi/2) = 0.0
        assert np.isclose(res["temp_sin_24_hour"].iloc[1], 1.0, atol=1e-5)
        assert np.isclose(res["temp_cos_24_hour"].iloc[1], 0.0, atol=1e-5)

        # Hour 12: sin(pi) = 0.0, cos(pi) = -1.0
        assert np.isclose(res["temp_sin_24_hour"].iloc[2], 0.0, atol=1e-5)
        assert np.isclose(res["temp_cos_24_hour"].iloc[2], -1.0, atol=1e-5)

    def test_test_set_continuation_transformation(self):
        train_df = pd.DataFrame({"sensor": [10.0, 20.0, 30.0, 40.0, 50.0]})
        test_df = pd.DataFrame({"sensor": [60.0, 70.0, 80.0]})

        gen = TemporalDynamicsGenerator(max_lags=1, include_cyclical=False)
        feats = gen.propose_features(train_df, ["sensor"])
        gen.fit(train_df, feats)

        test_transformed = gen.transform(test_df, feats, is_continuation=True)
        # First row of test set should have lag 1 equal to last row of train set (50.0)
        assert test_transformed["temp_lag_1_sensor"].iloc[0] == 50.0
        assert test_transformed["temp_delta_1_sensor"].iloc[0] == 10.0  # 60.0 - 50.0

    def test_propose_candidate_feature_sets_packaging(self, sequential_df: pd.DataFrame):
        gen = TemporalDynamicsGenerator(max_lags=1)
        feats = gen.propose_features(sequential_df, ["pressure", "temperature", "hour"], target_column="target")
        candidate_sets = gen.propose_candidate_feature_sets(
            base_features=["pressure", "temperature", "hour"],
            generated_features=feats,
            dataset_id="ds_test",
        )

        names = {cs.name for cs in candidate_sets}
        assert "interactions_temporal_all" in names
        assert "interactions_temporal_lags" in names
        assert "interactions_temporal_deltas" in names
        assert "interactions_temporal_cyclical" in names

        # Candidate sets are valid FeatureSet instances with lineage
        all_set = next(cs for cs in candidate_sets if cs.name == "interactions_temporal_all")
        assert len(all_set.feature_names) == 3 + len(feats)
        assert all_set.created_by == "temporal_dynamics_generator"
        assert "temporal features" in all_set.lineage


class TestTemporalDynamicsWorkspaceAndCQRS:
    """Test full integration with AutoMLWorkspace, CommandBus and QueryBus."""

    @pytest.fixture
    def workspace_env(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_file = data_dir / "time_series_data.csv"

        df = pd.DataFrame({
            "step": list(range(20)),
            "hour": [i % 24 for i in range(20)],
            "vibration": [0.5 + 0.1 * i for i in range(20)],
            "pressure": [100.0 - 0.5 * i for i in range(20)],
            "target": [0 if i < 15 else 1 for i in range(20)],
        })
        df.to_csv(csv_file, index=False)

        ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))
        dataset = ws.register_dataset(name="ts_data", path=str(csv_file), target="target")
        run = ws.create_run(dataset)
        return ws, cmd, qry, dataset, run

    def test_detect_temporal_structure_query(self, workspace_env):
        ws, cmd, qry, dataset, run = workspace_env

        res = qry.dispatch(DetectTemporalStructureQuery(dataset_id=dataset.id))
        assert isinstance(res, dict)
        assert res["is_sequential"] is True
        assert res["order_column"] == "step"
        assert any(p["name"] == "hourly" for p in res["detected_periodicities"])

    def test_generate_temporal_features_command(self, workspace_env):
        ws, cmd, qry, dataset, run = workspace_env

        command = GenerateTemporalFeaturesCommand(
            run_id=run.id,
            dataset_id=dataset.id,
            max_lags=1,
            include_lags=True,
            include_deltas=True,
            include_cyclical=True,
        )
        created_sets = cmd.dispatch(command)
        assert len(created_sets) >= 3

        # Candidate sets are saved in repository and in workspace candidate registry
        for fs in created_sets:
            loaded = ws.repository.get_feature_set(fs.id)
            assert loaded is not None
            assert loaded.name == fs.name

        candidate_list = ws.list_candidate_feature_sets(run.id)
        assert len(candidate_list) >= 3
        assert any("interactions_temporal" in c["name"] for c in candidate_list)

    def test_cli_features_temporal(self, workspace_env, capsys):
        ws, cmd, qry, dataset, run = workspace_env

        args = argparse.Namespace(
            dataset=dataset.path,
            workspace=str(ws.root_dir),
            target="target",
            max_lags=1,
            no_lags=False,
            no_deltas=False,
            no_cyclical=False,
            json=True,
        )
        exit_code = features_temporal_cli(args)
        assert exit_code == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["temporal_structure"]["is_sequential"] is True
        assert len(data["candidate_feature_sets"]) >= 3
