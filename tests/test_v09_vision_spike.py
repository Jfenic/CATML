from __future__ import annotations

from pathlib import Path
import struct
import zlib
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.artifacts.model_artifact import ModelArtifact
from automl.domain.datasets.profile import Dataset
from automl.domain.modalities.modality import Modality
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.domain.experiments.trial import Experiment, Trial
from automl.domain.ports import TrialExecution
from automl.engine.profiling.dataset_profiler import profile_dataset
from automl.engine.training.sklearn_trainer import SklearnTrainer, _build_pipeline
from automl.engine.vision.image_encoder import ImageEncoderNode
from automl.facade import AutoML
from automl.plugins.modalities.image_plugin import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageModalityPlugin,
    is_image_column,
)
from automl.plugins.models.vision_plugin import TimmVisionPlugin


def create_test_png(path: Path, width: int = 4, height: int = 4, color: tuple[int, int, int] = (100, 150, 200)) -> Path:
    """Creates a tiny valid PNG image using only Python stdlib struct and zlib."""
    png = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
    png += struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc

    raw_data = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    compressed = zlib.compress(raw_data)
    idat_crc = struct.pack(">I", zlib.crc32(b"IDAT" + compressed))
    png += struct.pack(">I", len(compressed)) + b"IDAT" + compressed + idat_crc

    iend_crc = struct.pack(">I", zlib.crc32(b"IEND"))
    png += struct.pack(">I", 0) + b"IEND" + iend_crc

    path.write_bytes(png)
    return path


@pytest.fixture
def vision_dataset_dir(tmp_path: Path):
    """Generates synthetic image files and returns dataframe and image dir."""
    img_dir = tmp_path / "spike_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(20):
        color = ((i * 12) % 256, (i * 25 + 30) % 256, (i * 45 + 70) % 256)
        img_path = img_dir / f"img_{i:03d}.png"
        create_test_png(img_path, width=4, height=4, color=color)

        rows.append({
            "image_path": str(img_path),
            "age": 20 + i * 2,
            "category": "A" if i % 2 == 0 else "B",
            "notes": f"Medical clinical observation note for patient sequence {i} with symptoms",
            "target": 1 if i >= 10 else 0,
        })

    df = pd.DataFrame(rows)
    return df, img_dir


class TestVisionSpikeHeuristics:
    def test_is_image_column_heuristic(self, tmp_path: Path):
        # Valid image path series
        paths = pd.Series([f"/data/img_{i}.jpg" for i in range(10)])
        assert is_image_column(paths) is True

        # Valid png/webp mixed
        mixed_paths = pd.Series(["a.png", "b.PNG", "c.webp", "d.jpeg", "e.tiff"])
        assert is_image_column(mixed_paths) is True

        # Regular categorical strings
        categories = pd.Series(["apple", "banana", "cherry", "date"] * 5)
        assert is_image_column(categories) is False

        # Natural language text
        sentences = pd.Series(["This is a long sentence describing a test sample." for _ in range(5)])
        assert is_image_column(sentences) is False

        # Numeric series
        numbers = pd.Series([1.2, 3.4, 5.6])
        assert is_image_column(numbers) is False

    def test_profiler_does_not_flag_image_column_as_identifier(self, vision_dataset_dir):
        df, _ = vision_dataset_dir
        # Replicate 60 rows so row_count >= 50 triggers cardinality heuristics
        df_large = pd.concat([df] * 3, ignore_index=True)
        # Give each row a unique image path to test high cardinality
        df_large["image_path"] = [f"/path/to/unique_img_{i}.png" for i in range(len(df_large))]

        ds = Dataset(
            id="ds_spike",
            workspace_id="ws_1",
            name="spike_data",
            path="",
            target_column="target",
            task_type="binary_classification",
        )
        profile = profile_dataset(ds, df_large)

        # image_path must be recognized as image, not as an identifier to be excluded
        img_col_profile = next(c for c in profile.columns if c.name == "image_path")
        assert img_col_profile.is_image is True
        assert img_col_profile.is_identifier is False

        # DatasetProfile property
        assert "image_path" in profile.image_column_names

        # Recommendations should include Image Feature
        rec = next((r for r in profile.recommendations if r.get("column") == "image_path"), None)
        assert rec is not None
        assert rec["badge"] == "Image Feature"
        assert rec["type"] == "vision"


class TestImageEncoderNodeCompatibility:
    def test_encoder_handles_various_input_shapes(self, vision_dataset_dir):
        df, _ = vision_dataset_dir
        encoder = ImageEncoderNode(output_dim=32, handle_missing="zero")

        # 1D Series
        emb_series = encoder.fit_transform(df["image_path"])
        assert isinstance(emb_series, np.ndarray)
        assert emb_series.shape == (20, 32)
        assert emb_series.dtype == np.float32

        # 2D DataFrame (single column slice from ColumnTransformer)
        emb_df = encoder.transform(df[["image_path"]])
        assert emb_df.shape == (20, 32)
        np.testing.assert_allclose(emb_series, emb_df)

        # 2D numpy array
        emb_arr = encoder.transform(df[["image_path"]].values)
        assert emb_arr.shape == (20, 32)
        np.testing.assert_allclose(emb_series, emb_arr)

        # Python list
        emb_list = encoder.transform(df["image_path"].tolist())
        assert emb_list.shape == (20, 32)
        np.testing.assert_allclose(emb_series, emb_list)

    def test_encoder_handles_missing_values_gracefully(self):
        encoder = ImageEncoderNode(output_dim=16, handle_missing="zero")
        dirty_input = [None, "", "non_existent_file.png", np.nan]
        emb = encoder.transform(dirty_input)

        assert emb.shape == (4, 16)
        # All invalid inputs should produce zero vectors
        np.testing.assert_allclose(emb, np.zeros((4, 16), dtype=np.float32))

    def test_encoder_scikit_learn_pipeline_compatibility(self, vision_dataset_dir):
        df, _ = vision_dataset_dir
        encoder = ImageEncoderNode(node_id="test_vision", output_dim=16)

        # get_params and set_params from BaseEstimator
        params = encoder.get_params()
        assert "output_dim" in params
        assert params["output_dim"] == 16

        # get_feature_names_out
        feature_names = encoder.get_feature_names_out()
        assert len(feature_names) == 16
        assert feature_names[0] == "test_vision_0"


class TestVisionPluginsCapabilityLayer:
    def test_image_modality_plugin_capabilities(self):
        plugin = ImageModalityPlugin()
        assert plugin.plugin_id == "image_modality"
        assert plugin.modality == Modality.IMAGE
        assert plugin.requirements() == ("catml[vision]", "torch", "torchvision", "timm", "pillow")
        assert "pip install 'catml[vision]'" in plugin.install_instructions()
        assert isinstance(plugin.available(), bool)
        assert plugin.is_available == plugin.available()

        # Callable capabilities check
        caps = plugin.capabilities()
        assert isinstance(caps, PluginCapability)
        assert caps.is_compatible_with_modality("image")

    def test_timm_vision_plugin_capabilities(self):
        plugin = TimmVisionPlugin()
        assert plugin.plugin_id == "timm_vision"
        assert plugin.plugin_type == PluginType.MODEL
        assert "catml[vision]" in plugin.requirements()
        assert "pip install 'catml[vision]'" in plugin.install_instructions()
        assert isinstance(plugin.available(), bool)

        caps = plugin.capabilities()
        assert caps.is_compatible_with_task("binary_classification")
        assert caps.is_compatible_with_modality("image")

        # Search space check
        space = plugin.get_search_space("binary_classification")
        assert "backbone" in space.parameters
        assert "learning_rate" in space.parameters

        # Estimator build fallback
        est = plugin.build_estimator(task_type="binary_classification")
        assert hasattr(est, "fit")
        assert hasattr(est, "predict")


class TestMultimodalTrainingAndPipeline:
    def test_build_pipeline_with_image_columns(self, vision_dataset_dir):
        from sklearn.linear_model import LogisticRegression

        df, _ = vision_dataset_dir
        X = df[["image_path", "age", "category", "notes"]]
        y = df["target"]

        pipeline = _build_pipeline(X, LogisticRegression(), image_columns=["image_path"])
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        assert len(preds) == len(df)
        assert set(preds).issubset({0, 1})

    def test_pure_image_pipeline_training(self, vision_dataset_dir):
        from sklearn.ensemble import HistGradientBoostingClassifier

        df, _ = vision_dataset_dir
        # Only image column and target
        X = df[["image_path"]]
        y = df["target"]

        pipeline = _build_pipeline(X, HistGradientBoostingClassifier(), image_columns=["image_path"])
        pipeline.fit(X, y)

        preds = pipeline.predict(X)
        assert len(preds) == len(df)

    def test_cooperative_cancellation_during_run(self, vision_dataset_dir, tmp_path: Path):
        df, _ = vision_dataset_dir
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        run = AutoMLRun(
            id="run_cancelled",
            workspace_id="ws_1",
            dataset_id="ds_1",
            config=RunConfig(task_type="binary_classification", target="target"),
            status=RunStatus.CANCELLED,
            current_phase=RunPhase.EXPERIMENT_EXECUTION,
        )

        exp = Experiment(
            id="exp_1",
            run_id="run_cancelled",
            name="cancelled_exp",
            hypothesis="",
            model_ids=["logistic_regression"],
            metric="accuracy",
            feature_names=["image_path", "age"],
        )
        trial = Trial(id="trial_1", experiment_id="exp_1", model_id="logistic_regression")

        execution = TrialExecution(
            trial=trial,
            experiment=exp,
            run=run,
            feature_names=["image_path", "age"],
            dataset_path=str(csv_path),
            target_column="target",
            task_type="binary_classification",
            metric="accuracy",
            validation_strategy="holdout",
            test_size=0.2,
            cv_folds=5,
            random_seed=42,
        )

        trainer = SklearnTrainer()
        result = trainer.run(execution)
        assert result.succeeded is False
        assert "cancelled" in (result.failure_reason or "").lower()


class TestAutoMLFacadeVisionSpike:
    def test_automl_fit_predict_and_artifact_export(self, vision_dataset_dir, tmp_path: Path):
        df, _ = vision_dataset_dir

        automl = AutoML(
            task="binary_classification",
            metric="accuracy",
            models=["logistic_regression"],
            workspace_dir=tmp_path / "automl_ws",
            random_state=42,
        )

        result = automl.fit(
            data=df,
            target="target",
            image_columns=["image_path"],
            text_columns=["notes"],
        )

        assert result is not None
        assert result.best_score > 0.0
        assert len(result.leaderboard()) >= 1

        # Predict with AutoML facade
        preds = automl.predict(df[["image_path", "age", "category", "notes"]])
        assert len(preds) == len(df)

        # Standalone ModelArtifact verification
        artifact = result.best_model
        assert isinstance(artifact, ModelArtifact)
        assert "image_path" in artifact.feature_names
        assert "provenance" in artifact.describe()

        # Serialize artifact to disk and reload independently
        artifact_path = tmp_path / "vision_model.pkl"
        artifact.save(artifact_path)
        assert artifact_path.is_file()

        loaded_artifact = ModelArtifact.load(artifact_path)
        loaded_preds = loaded_artifact.predict(df[["image_path", "age", "category", "notes"]])
        np.testing.assert_array_equal(preds, loaded_preds)
