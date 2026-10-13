"""07 — Short HPO / time_budget smoke."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import load_titanic_like, run_fire_test  # noqa: E402


def main() -> None:
    def body():
        from catml import AutoML

        df, target = load_titanic_like()
        ws = Path(__file__).resolve().parent / "results" / "ws_07"
        automl = AutoML(
            metric="roc_auc",
            models=["random_forest"],
            time_budget=30.0,
            workspace_dir=str(ws),
            random_state=42,
        )
        result = automl.fit(df, target=target)
        board = result.leaderboard()
        assert board is not None and not board.empty
        return {"top_score": float(board.iloc[0]["score"]), "time_budget": 30.0}

    r = run_fire_test("07_hpo_short", "Short HPO time_budget", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
