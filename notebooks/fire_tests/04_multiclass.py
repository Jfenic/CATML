"""04 — Multiclass (sklearn wine)."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import load_multiclass, run_fire_test  # noqa: E402


def main() -> None:
    def body():
        from catml import AutoML

        df, target = load_multiclass()
        ws = Path(__file__).resolve().parent / "results" / "ws_04"
        automl = AutoML(
            metric="log_loss",
            models=["logistic_regression", "random_forest"],
            workspace_dir=str(ws),
            random_state=42,
        )
        result = automl.fit(df, target=target)
        board = result.leaderboard()
        assert board is not None and not board.empty
        score = float(board.iloc[0]["score"])
        assert score < 1.0
        return {"rows": len(df), "top_log_loss": score, "n_models": int(len(board))}

    r = run_fire_test("04_multiclass", "Multiclass wine", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
