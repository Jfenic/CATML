"""10 — Multi-model fit / ensemble-capable model list."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import run_fire_test  # noqa: E402


def main() -> None:
    def body():
        from sklearn.datasets import load_breast_cancer

        from catml import AutoML

        df = load_breast_cancer(as_frame=True).frame
        target = "target"
        ws = Path(__file__).resolve().parent / "results" / "ws_10"
        models = ["logistic_regression", "random_forest", "extra_trees"]
        # Try voting ensemble if registered; ignore if unavailable.
        try:
            automl = AutoML(
                metric="roc_auc",
                models=models + ["voting_ensemble"],
                workspace_dir=str(ws),
                random_state=42,
            )
            used = models + ["voting_ensemble"]
        except Exception:
            automl = AutoML(
                metric="roc_auc",
                models=models,
                workspace_dir=str(ws),
                random_state=42,
            )
            used = models
        result = automl.fit(df, target=target)
        board = result.leaderboard()
        assert board is not None and not board.empty
        assert len(board) >= 2
        return {
            "models_requested": used,
            "n_leaderboard": int(len(board)),
            "top_score": float(board.iloc[0]["score"]),
            "top_model": str(board.iloc[0].get("model_name", board.iloc[0].get("model", ""))),
        }

    r = run_fire_test("10_ensemble_compare", "Ensemble / multi-model compare", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
