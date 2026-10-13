"""06 — Ordered / time-index aware classification."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import load_ordered_split, run_fire_test  # noqa: E402


def main() -> None:
    def body():
        from catml import AutoML

        df, target = load_ordered_split()
        ws = Path(__file__).resolve().parent / "results" / "ws_06"
        # Prefer explicit time column if facade/workspace accepts validation config via models only.
        automl = AutoML(
            metric="roc_auc",
            models=["logistic_regression", "random_forest"],
            workspace_dir=str(ws),
            random_state=42,
        )
        # Chronological holdout: fit on early rows only for a smoke check of ordering.
        cut = int(len(df) * 0.8)
        train = df.iloc[:cut].copy()
        result = automl.fit(train, target=target)
        board = result.leaderboard()
        assert board is not None and not board.empty
        preds = result.predict(df.iloc[cut:].drop(columns=[target]))
        assert len(preds) == len(df) - cut
        return {
            "train_rows": len(train),
            "test_rows": len(df) - cut,
            "top_score": float(board.iloc[0]["score"]),
            "time_col": "time_idx",
        }

    r = run_fire_test("06_ordered_time_split", "Ordered/time split", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
