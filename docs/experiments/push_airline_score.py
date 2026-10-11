"""Push ROC-AUC on airline satisfaction sample (no external solutions)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/kagle/train.csv"
OUT = ROOT / "docs/experiments"
OUT.mkdir(parents=True, exist_ok=True)
RS = 42


def stratified_sample(n: int, seed: int = RS) -> pd.DataFrame:
    pos_need = n // 2
    neg_need = n - pos_need
    pos_buf, neg_buf = [], []
    for chunk in pd.read_csv(SRC, chunksize=100_000):
        chunk["satisfaction"] = chunk["satisfaction"].astype(int)
        pos_buf.append(chunk[chunk.satisfaction == 1])
        neg_buf.append(chunk[chunk.satisfaction == 0])
        if sum(map(len, pos_buf)) >= pos_need and sum(map(len, neg_buf)) >= neg_need:
            break
    pos = pd.concat(pos_buf).sample(pos_need, random_state=seed)
    neg = pd.concat(neg_buf).sample(neg_need, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def split_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat


def preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    transformers = []
    if num:
        transformers.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", make_ohe()),
                    ]
                ),
                cat,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def oof_auc(model, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> float:
    num, cat = split_cols(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RS)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in cv.split(X, y):
        pipe = Pipeline(
            [
                ("pre", preprocessor(num, cat)),
                ("model", model.__class__(**model.get_params())),
            ]
        )
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(y, oof))


def main() -> None:
    results: list[dict] = []
    print("=== 50k stratified sample (drop id only) ===", flush=True)
    df = stratified_sample(50_000)
    y = df["satisfaction"].to_numpy()
    X = df.drop(columns=["satisfaction", "id"])
    print(f"n={len(df)} cols={len(X.columns)}", flush=True)

    models = {
        "lgbm_default": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RS,
            n_jobs=4,
            verbose=-1,
        ),
        "xgb_default": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="auc",
            random_state=RS,
            n_jobs=4,
            tree_method="hist",
        ),
    }

    print("=== Benchmark defaults ===", flush=True)
    for name, model in models.items():
        t0 = time.time()
        auc = oof_auc(model, X, y)
        dt = time.time() - t0
        print(f"{name}: {auc:.6f} ({dt:.1f}s)", flush=True)
        results.append({"name": name, "auc": auc, "seconds": round(dt, 2)})

    # log1p skewed non-negative numerics
    X_fe = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            continue
        s = pd.to_numeric(X[c], errors="coerce")
        if abs(float(s.skew(skipna=True))) > 2.0 and (s.fillna(0) >= 0).all():
            X_fe[f"{c}__log1p"] = np.log1p(s.fillna(0).clip(lower=0))
    t0 = time.time()
    auc = oof_auc(models["lgbm_default"], X_fe, y)
    print(f"lgbm_log1p: {auc:.6f} ({time.time()-t0:.1f}s)", flush=True)
    results.append({"name": "lgbm_log1p", "auc": auc, "seconds": round(time.time() - t0, 2)})

    best_params = None
    best_hpo = -1.0
    if optuna is not None:
        print("=== Optuna LightGBM (25 trials) ===", flush=True)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 900),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 24, 96),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
                "random_state": RS,
                "n_jobs": 4,
                "verbose": -1,
            }
            return oof_auc(LGBMClassifier(**params), X, y)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS)
        )
        study.optimize(objective, n_trials=25, show_progress_bar=False)
        best_hpo = float(study.best_value)
        best_params = dict(study.best_params)
        print(f"lgbm_hpo25: {best_hpo:.6f}", best_params, flush=True)
        results.append({"name": "lgbm_hpo25", "auc": best_hpo, "params": best_params})

        print("=== Optuna XGB (15 trials) ===", flush=True)

        def objective_xgb(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 9),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
                "eval_metric": "auc",
                "random_state": RS,
                "n_jobs": 4,
                "tree_method": "hist",
            }
            return oof_auc(XGBClassifier(**params), X, y)

        st_x = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=RS)
        )
        st_x.optimize(objective_xgb, n_trials=15, show_progress_bar=False)
        print(f"xgb_hpo15: {st_x.best_value:.6f}", st_x.best_params, flush=True)
        results.append(
            {"name": "xgb_hpo15", "auc": float(st_x.best_value), "params": dict(st_x.best_params)}
        )

        # Blend OOF
        print("=== Blend best LGBM + XGB ===", flush=True)
        lgbm = LGBMClassifier(
            **{**best_params, "random_state": RS, "n_jobs": 4, "verbose": -1}
        )
        xgb = XGBClassifier(
            **{
                **st_x.best_params,
                "eval_metric": "auc",
                "random_state": RS,
                "n_jobs": 4,
                "tree_method": "hist",
            }
        )
        num, cat = split_cols(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
        oof_l = np.zeros(len(y))
        oof_x = np.zeros(len(y))
        for tr, te in cv.split(X, y):
            pl = Pipeline([("pre", preprocessor(num, cat)), ("m", lgbm.__class__(**lgbm.get_params()))])
            px = Pipeline([("pre", preprocessor(num, cat)), ("m", xgb.__class__(**xgb.get_params()))])
            pl.fit(X.iloc[tr], y[tr])
            px.fit(X.iloc[tr], y[tr])
            oof_l[te] = pl.predict_proba(X.iloc[te])[:, 1]
            oof_x[te] = px.predict_proba(X.iloc[te])[:, 1]
        best_b, best_w = 0.0, 0.5
        for w in np.linspace(0, 1, 21):
            a = roc_auc_score(y, w * oof_l + (1 - w) * oof_x)
            if a > best_b:
                best_b, best_w = a, float(w)
        print(f"blend: {best_b:.6f} (w_lgbm={best_w:.2f})", flush=True)
        results.append({"name": "blend_lgbm_xgb", "auc": float(best_b), "w_lgbm": best_w})

        # Confirm 150k
        print("=== Confirm LGBM HPO on 150k ===", flush=True)
        df2 = stratified_sample(150_000)
        y2 = df2["satisfaction"].to_numpy()
        X2 = df2.drop(columns=["satisfaction", "id"])
        t0 = time.time()
        auc150 = oof_auc(
            LGBMClassifier(**{**best_params, "random_state": RS, "n_jobs": 4, "verbose": -1}),
            X2,
            y2,
        )
        print(f"lgbm_hpo @150k: {auc150:.6f} ({time.time()-t0:.1f}s)", flush=True)
        results.append({"name": "lgbm_hpo_confirm_150k", "auc": float(auc150)})

    summary = {
        "protocol": {
            "metric": "roc_auc",
            "cv": "StratifiedKFold(5)",
            "drop": ["id"],
            "keep_customer_type": True,
            "external_solutions": False,
        },
        "results": sorted(results, key=lambda r: -r["auc"]),
        "best": max(results, key=lambda r: r["auc"]),
    }
    path = OUT / "airline_push_scores.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== LEADERBOARD ===", flush=True)
    for r in summary["results"]:
        print(f"{r['auc']:.6f}  {r['name']}", flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
