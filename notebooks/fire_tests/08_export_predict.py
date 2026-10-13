"""08 — Fit → save_model → load → predict."""
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
        ws = Path(__file__).resolve().parent / "results" / "ws_08"
        automl = AutoML(
            metric="roc_auc",
            models=["logistic_regression"],
            workspace_dir=str(ws),
            random_state=42,
        )
        result = automl.fit(df, target=target)
        model_path = ws / "model.joblib"
        saved = result.save_model(model_path)
        assert Path(saved).exists()
        loaded = AutoML.load_model(saved)
        X = df.drop(columns=[target]).head(10)
        preds = loaded.predict(X)
        assert len(preds) == 10
        return {"saved": str(saved), "n_preds": int(len(preds))}

    r = run_fire_test("08_export_predict", "Export → predict", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
