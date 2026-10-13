"""02 — House-prices-like regression."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import load_house_prices_like, run_fire_test  # noqa: E402


def main() -> None:
    def body():
        from catml import AutoML

        df, target = load_house_prices_like()
        # Keep pilot small
        df = df.sample(n=min(2500, len(df)), random_state=42)
        ws = Path(__file__).resolve().parent / "results" / "ws_02"
        automl = AutoML(
            metric="rmse",
            models=["ridge", "random_forest"],
            workspace_dir=str(ws),
            random_state=42,
        )
        result = automl.fit(df, target=target)
        board = result.leaderboard()
        assert board is not None and not board.empty
        score = float(board.iloc[0]["score"])
        assert score < 1.5  # california housing RMSE scale
        return {"rows": len(df), "top_rmse": score, "n_models": int(len(board))}

    r = run_fire_test("02_house_prices_like", "House prices-like regression", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
