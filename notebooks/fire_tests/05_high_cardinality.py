"""05 — High-cardinality categoricals / near-id columns."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import load_high_cardinality, run_fire_test  # noqa: E402


def main() -> None:
    def body():
        from catml import AutoML

        df, target = load_high_cardinality()
        ws = Path(__file__).resolve().parent / "results" / "ws_05"
        automl = AutoML(
            metric="roc_auc",
            models=["logistic_regression", "random_forest"],
            workspace_dir=str(ws),
            random_state=42,
        )
        result = automl.fit(df, target=target)
        board = result.leaderboard()
        assert board is not None and not board.empty
        score = float(board.iloc[0]["score"])
        return {"rows": len(df), "top_score": score, "n_models": int(len(board))}

    r = run_fire_test("05_high_cardinality", "High-cardinality cats", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
