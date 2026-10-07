import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand, GenerateOOFSubmissionCommand, PauseRunCommand, RunExperimentCommand,
)
from automl.application.queries.workspace_queries import GetOOFResultQuery, PredictDatasetQuery
from automl.domain.experiments.trial import ExperimentStatus
from automl.engine.ensemble.oof import evaluate_oof
from automl import __version__
from automl.interfaces.cli.main import main


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    y = pd.Series(["no", "yes"] * 30)
    X = pd.DataFrame({"x": rng.normal(size=60) + (y == "yes") * 1.5,
                      "category": ["a", "b", "c"] * 20})
    return X, y, X.iloc[:8].copy()


def factories():
    return {"logistic": lambda: LogisticRegression(max_iter=200),
            "forest": lambda: RandomForestClassifier(n_estimators=5, max_depth=3)}


def test_oof_coverage_equal_weights_and_reproducibility(data):
    X, y, test = data
    first = evaluate_oof(X, y, test, factories())
    second = evaluate_oof(X, y, test, factories())
    assert np.bincount(first.fold_ids).tolist() == [12] * 5
    assert np.isfinite(first.blended_oof).all()
    np.testing.assert_allclose(first.blended_oof, (first.oof["logistic"] + first.oof["forest"]) / 2)
    np.testing.assert_allclose(first.blended_test, (first.test["logistic"] + first.test["forest"]) / 2)
    np.testing.assert_allclose(first.blended_test, second.blended_test)
    np.testing.assert_allclose(first.blended_oof, second.blended_oof)
    assert first.classes == ["no", "yes"]
    assert all(len(scores) == 5 for scores in first.fold_scores.values())
    assert all(0 <= score <= 1 for score in first.scores.values())


class Memorizer(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        # A scaler trained on all data would not give each training fold mean zero.
        assert abs(X[:, 0].mean()) < 1e-10
        self.rows_ = {tuple(row): label for row, label in zip(X, y)}
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.array([self.rows_.get(tuple(row), 0.5) for row in X])
        return np.column_stack([1 - p, p])


def test_validation_rows_and_preprocessing_are_not_seen_during_fit():
    X = pd.DataFrame({"unique": np.arange(40, dtype=float) ** 2})
    result = evaluate_oof(X, pd.Series([0, 1] * 20), X.iloc[:3], {"memory": Memorizer})
    np.testing.assert_allclose(result.blended_oof, 0.5)
    assert result.scores["memory"] == 0.5


@pytest.mark.parametrize("folds", [0, 1, True, 2.5, 31])
def test_invalid_fold_counts(data, folds):
    with pytest.raises(ValueError, match="folds"):
        evaluate_oof(*data, factories(), folds=folds)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_budget(data, seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        evaluate_oof(*data, factories(), max_seconds=seconds)


def test_timeout_and_model_capabilities(data):
    with pytest.raises(TimeoutError, match="budget"):
        evaluate_oof(*data, factories(), max_seconds=1e-12)
    with pytest.raises(ValueError, match="one or two"):
        evaluate_oof(*data, {})
    from sklearn.svm import LinearSVC
    with pytest.raises(ValueError, match="predict_proba"):
        evaluate_oof(*data, {"svc": LinearSVC})


def test_invalid_target_and_feature_alignment(data):
    X, y, test = data
    with pytest.raises(ValueError, match="binary"):
        evaluate_oof(X, pd.Series([0, 1, 2] * 20), test, factories())
    with pytest.raises(ValueError, match="columns"):
        evaluate_oof(X, y, test.rename(columns={"x": "other"}), factories())
    y.iloc[0] = None
    with pytest.raises(ValueError, match="complete aligned"):
        evaluate_oof(X, y, test, factories())


class InvalidProbabilities(Memorizer):
    def predict_proba(self, X):
        return np.ones((len(X), 2))


def test_invalid_probabilities_are_rejected():
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    with pytest.raises(ValueError, match="bounded and sum"):
        evaluate_oof(X, pd.Series([0, 1] * 10), X.iloc[:2], {"invalid": InvalidProbabilities})


class ReversedClasses(Memorizer):
    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, X):
        return np.tile([0.2, 0.8], (len(X), 1))


def test_probability_columns_follow_class_labels():
    X = pd.DataFrame({"x": np.arange(20, dtype=float)})
    result = evaluate_oof(X, pd.Series([0, 1] * 10), X.iloc[:2], {"reversed": ReversedClasses})
    np.testing.assert_allclose(result.blended_oof, 0.2)


@pytest.fixture
def app(tmp_path, data):
    X, y, test = data
    train_path, test_path, template_path = (tmp_path / name for name in ("train.csv", "test.csv", "template.csv"))
    pd.DataFrame({"id": range(len(X)), **X, "churn": y}).to_csv(train_path, index=False)
    pd.DataFrame({"id": range(100, 108), **test}).to_csv(test_path, index=False)
    pd.DataFrame({"id": list(range(107, 99, -1)), "Probability": 0.0}).to_csv(template_path, index=False)
    workspace, commands, queries = build_application(root_dir=str(tmp_path / "workspace"))
    dataset = workspace.register_dataset(name="churn", path=train_path, target="churn")
    run = workspace.create_run(dataset, metric="roc_auc")
    source = commands.dispatch(CreateExperimentCommand(run_id=run.id, name="source",
                              feature_names=["x", "category"], model_ids=["logistic_regression", "random_forest"]))
    return workspace, commands, queries, run, source, test_path, template_path


def command(app, tmp_path, **kwargs):
    _, _, _, run, source, test, template = app
    return GenerateOOFSubmissionCommand(run_id=run.id, experiment_id=source.id,
                                       test_dataset_path=str(test), output_path=str(tmp_path / "submission.csv"),
                                       template_path=str(template), **kwargs)


def test_cqrs_submission_artifacts_reload_and_queries_have_no_writes(app, tmp_path):
    workspace, commands, queries, run, source, test, template = app
    experiment_id = commands.dispatch(command(app, tmp_path))
    assert isinstance(experiment_id, str)
    events_before = workspace.repository.list_events()
    _, _, queries = build_application(root_dir=str(workspace.root_dir))
    report = queries.dispatch(GetOOFResultQuery(run.id, experiment_id))
    assert workspace.repository.list_events() == events_before
    assert report["source_experiment_id"] == source.id
    assert report["baseline_model"] == "logistic_regression"
    assert report["delta"] == pytest.approx(report["score"] - report["baseline_score"])
    assert report["promoted"] is False
    assert report["backends"]["random_forest"].endswith("RandomForestClassifier")
    assert report["plugin_versions"]["random_forest"]
    assert len(report["train_hash"]) == len(report["config_hash"]) == 64
    assert report["versions"]["automl"] == __version__
    submission = pd.read_csv(report["output_path"])
    assert list(submission.columns) == ["id", "Probability"]
    assert submission["id"].tolist() == pd.read_csv(template)["id"].tolist()
    assert submission["Probability"].between(0, 1).all()
    oof = pd.read_csv(report["oof_predictions_path"])
    assert oof["source_row"].tolist() == list(range(60))
    cached = queries.dispatch(PredictDatasetQuery(run.id, str(test), experiment_id=experiment_id, predict_proba=True, template_path=str(template)))
    np.testing.assert_allclose(cached, submission["Probability"])
    labels = queries.dispatch(PredictDatasetQuery(run.id, str(test), experiment_id=experiment_id))
    assert set(labels).issubset({"no", "yes"})
    assert workspace.repository.list_events() == events_before
    with pytest.raises(ValueError, match="requested run"):
        queries.dispatch(GetOOFResultQuery("another-run", experiment_id))
    artifact = Path(report["test_predictions_path"])
    original = artifact.read_bytes()
    artifact.write_text("oof_blend\n0.5\n")
    with pytest.raises(ValueError, match="Stored OOF predictions changed"):
        queries.dispatch(PredictDatasetQuery(run.id, str(test), experiment_id=experiment_id))
    artifact.write_bytes(original)
    pd.read_csv(test).assign(x=0).to_csv(test, index=False)
    with pytest.raises(ValueError, match="different test"):
        queries.dispatch(PredictDatasetQuery(run.id, str(test), experiment_id=experiment_id))


def test_best_source_selection_and_cli(app, tmp_path, capsys):
    workspace, commands, _, run, source, test, template = app
    commands.dispatch(RunExperimentCommand(run.id, source.id))
    assert main(["predict", "--workspace", str(workspace.root_dir), "--run-id", run.id,
                 "--test-dataset", str(test), "--template", str(template), "--folds", "5",
                 "--models", "logistic_regression", "--proba", "--output", str(tmp_path / "cli.csv"), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["folds"] == 5
    assert result["models"] == ["logistic_regression"]
    assert Path(result["output_path"]).exists()


@pytest.mark.parametrize("kwargs", [{"folds": 1}, {"max_seconds": 1e-12}])
def test_failed_evaluation_records_failure_and_does_not_write_submission(app, tmp_path, kwargs):
    workspace, commands, queries, run, _, _, _ = app
    with pytest.raises((ValueError, TimeoutError)):
        commands.dispatch(command(app, tmp_path, **kwargs))
    failed = next(e for e in workspace.repository.list_experiments(run.id) if e.validation_strategy == "oof")
    assert failed.status == ExperimentStatus.FAILED
    assert not workspace.repository.list_trial_results(failed.id)[0].succeeded
    assert not (tmp_path / "submission.csv").exists()
    with pytest.raises(ValueError, match="no completed report"):
        queries.dispatch(GetOOFResultQuery(run.id, failed.id))


@pytest.mark.parametrize("models", [["voting_ensemble"], ["ridge"], ["logistic_regression"] * 2, []])
def test_invalid_models_rejected_before_experiment(app, tmp_path, models):
    workspace, commands, _, run, _, _, _ = app
    before = len(workspace.repository.list_experiments(run.id))
    with pytest.raises(ValueError):
        commands.dispatch(command(app, tmp_path, model_ids=models))
    assert len(workspace.repository.list_experiments(run.id)) == before


def test_paused_run_and_cross_run_source(app, tmp_path):
    workspace, commands, _, run, _, _, _ = app
    other = workspace.create_run(workspace._get_dataset(run.dataset_id))
    with pytest.raises(ValueError, match="belonging"):
        commands.dispatch(GenerateOOFSubmissionCommand(other.id, str(app[5]), str(tmp_path / "bad.csv"), experiment_id=app[4].id))
    commands.dispatch(PauseRunCommand(run.id))
    with pytest.raises(RuntimeError, match="paused"):
        commands.dispatch(command(app, tmp_path))


def test_cli_flags_and_evaluation_failure(app, tmp_path, capsys):
    workspace, _, _, run, source, test, _ = app
    args = ["predict", "--workspace", str(workspace.root_dir), "--run-id", run.id,
            "--test-dataset", str(test), "--experiment-id", source.id]
    assert main(args + ["--models", "logistic_regression"]) == 1
    assert "requires --folds" in capsys.readouterr().err
    assert main(args + ["--folds", "1"]) == 1
    assert "folds" in capsys.readouterr().err


def test_label_submission_and_unaligned_output(app, tmp_path):
    workspace, commands, queries, run, source, test, _ = app
    experiment_id = commands.dispatch(GenerateOOFSubmissionCommand(
        run.id, str(test), str(tmp_path / "labels.csv"), experiment_id=source.id,
        model_ids=["logistic_regression"], predict_proba=False,
    ))
    report = queries.dispatch(GetOOFResultQuery(run.id, experiment_id))
    rows = pd.read_csv(report["output_path"])
    assert rows["id"].tolist() == list(range(100, 108))
    assert set(rows["churn"]).issubset({"no", "yes"})
    assert report["positive_class"] == "yes"
    assert workspace.repository.get_run(run.id).status.value == "COMPLETED"


@pytest.mark.parametrize("action", ["pause_run", "cancel_run"])
def test_run_controls_abort_at_fold_boundaries(app, tmp_path, monkeypatch, action):
    workspace, commands, _, run, _, _, _ = app
    plugin = workspace.plugin_registry.get_model_plugin("logistic_regression")
    build = plugin.build_estimator

    def controlled_build(**kwargs):
        getattr(workspace, action)(run.id)
        return build(**kwargs)

    monkeypatch.setattr(plugin, "build_estimator", controlled_build)
    with pytest.raises(RuntimeError, match="restart OOF"):
        commands.dispatch(command(app, tmp_path))
    assert workspace.repository.get_run(run.id).status.value in {"PAUSED", "CANCELLED"}
    assert not (tmp_path / "submission.csv").exists()


def test_input_changes_during_training_invalidate_result(app, tmp_path, monkeypatch):
    from automl.application.services import oof_submission
    evaluate = oof_submission.evaluate_oof

    def changing_input(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        pd.read_csv(app[5]).assign(x=1).to_csv(app[5], index=False)
        return result

    monkeypatch.setattr(oof_submission, "evaluate_oof", changing_input)
    with pytest.raises(ValueError, match="Dataset changed"):
        app[1].dispatch(command(app, tmp_path))
    assert not (tmp_path / "submission.csv").exists()


def test_missing_source_and_nonbinary_task_rejected(app, tmp_path):
    workspace, commands, _, run, source, test, _ = app
    with pytest.raises(ValueError, match="source experiment"):
        commands.dispatch(GenerateOOFSubmissionCommand(run.id, str(test), str(tmp_path / "out.csv")))
    dataset = workspace._get_dataset(run.dataset_id)
    dataset.task_type = "regression"
    with pytest.raises(ValueError, match="binary"):
        commands.dispatch(command(app, tmp_path))


def test_composite_source_uses_individual_default_models(app, tmp_path):
    workspace, commands, queries, run, source, _, _ = app
    source.model_ids = ["voting_ensemble"]
    workspace.repository.save_experiment(source)
    experiment_id = commands.dispatch(command(app, tmp_path))
    assert queries.dispatch(GetOOFResultQuery(run.id, experiment_id))["models"] == ["logistic_regression", "random_forest"]


def test_web_oof_submission_uses_same_command_and_report(app, tmp_path, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from automl.interfaces.web.server import AutoMLWebHandler

    workspace, _, _, run, source, test, template = app
    monkeypatch.setattr(AutoMLWebHandler, "workspace_dir", str(workspace.root_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {"run_id": run.id, "experiment_id": source.id,
                   "test_dataset_path": str(test), "template_path": str(template),
                   "output_path": str(tmp_path / "web.csv"), "predict_proba": True,
                   "folds": 5, "model_ids": ["logistic_regression"]}
        request = Request(f"http://127.0.0.1:{server.server_port}/api/predict",
                          data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            result = json.load(response)
        assert result["status"] == "success"
        assert result["oof"]["folds"] == 5
        assert result["row_count"] == 8
        assert result["oof"]["promoted"] is False
        assert Path(result["output_path"]).exists()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
