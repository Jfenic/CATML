from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from automl.domain.datasets.profile import ColumnProfile, Dataset, DatasetProfile


def detect_column_cardinality_and_role(
    name: str,
    series: pd.Series,
    row_count: int,
    target_column: str | None = None,
) -> tuple[float, bool, bool]:
    """
    Evaluates a dataset column to calculate its cardinality ratio and determine
    if it acts as a non-predictive identifier or high-cardinality feature.

    Returns:
        tuple[float, bool, bool]: (cardinality_ratio, is_identifier, is_high_cardinality)
    """
    if target_column and name == target_column:
        return 0.0, False, False

    unique_count = int(series.nunique(dropna=True))
    cardinality_ratio = float(unique_count / row_count) if row_count > 0 else 0.0

    clean_name = name.strip().lower()
    dtype_str = str(series.dtype).lower()
    is_object_or_string = (
        series.dtype.kind in {"O", "S", "U"}
        or "str" in dtype_str
        or "object" in dtype_str
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    # 1. High cardinality detection (primarily for string/text/categorical features)
    is_high_cardinality = False
    if is_object_or_string and row_count >= 50:
        if cardinality_ratio > 0.70 or (unique_count > 100 and cardinality_ratio > 0.50):
            is_high_cardinality = True

    # 2. Identifier detection:
    # A) Name-based ID heuristic
    name_matches_id = (
        clean_name in {"id", "guid", "uuid", "hash", "pk", "key"}
        or clean_name.endswith(("_id", ".id", "-id", "_guid", "_uuid", "_hash", "_key", "_pk"))
        or clean_name.startswith(("id_", "guid_", "uuid_", "pk_"))
        or name.endswith(("Id", "ID", "Guid", "GUID", "Uuid", "UUID"))
    )

    is_identifier = False
    if name_matches_id:
        # Check that it actually has identifier-like cardinality (avoid small enums like status_id with 2 values)
        if row_count <= 20:
            is_identifier = unique_count >= max(2, int(row_count * 0.7))
        else:
            is_identifier = unique_count >= 20 or cardinality_ratio >= 0.25

    # B) Content-based identifier heuristic (no ID name required)
    if not is_identifier and row_count >= 50:
        # High cardinality text/hash columns with near-unique values
        if is_object_or_string and cardinality_ratio > 0.70:
            is_identifier = True
        # Monotonically increasing sequential index integers (e.g. 0, 1, 2, ... N-1 or row counter)
        elif series.dtype.kind in {"i", "u"} and cardinality_ratio == 1.0:
            clean_series = series.dropna()
            if clean_series.is_monotonic_increasing:
                diffs = clean_series.diff().iloc[1:]
                if len(diffs) > 0 and (diffs == 1).all():
                    is_identifier = True

    return cardinality_ratio, is_identifier, is_high_cardinality


def profile_dataset(dataset: Dataset) -> DatasetProfile:
    path = Path(dataset.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if dataset.target_column not in df.columns:
        raise ValueError(f"Target column '{dataset.target_column}' not in dataset")

    row_count = len(df)
    target_series = df[dataset.target_column]

    # Target numeric proxy for calculating target correlation
    target_numeric: pd.Series | None = None
    if pd.api.types.is_numeric_dtype(target_series):
        target_numeric = target_series
    else:
        unique_targets = target_series.dropna().unique()
        if len(unique_targets) == 2:
            target_numeric = (target_series == unique_targets[1]).astype(float)
        elif len(unique_targets) > 2:
            target_numeric = target_series.astype("category").cat.codes.astype(float)

    columns: list[ColumnProfile] = []
    recommendations: list[dict[str, Any]] = []

    for name in df.columns:
        series = df[name]
        sample = series.dropna().head(3).tolist()
        unique_count = int(series.nunique(dropna=True))
        null_count = int(series.isna().sum())
        card_ratio, is_id, is_high_card = detect_column_cardinality_and_role(
            name=name,
            series=series,
            row_count=row_count,
            target_column=dataset.target_column,
        )

        mean_val = None
        std_val = None
        min_val = None
        max_val = None
        median_val = None
        q25_val = None
        q75_val = None
        skew_val = None
        target_corr = None
        top_cats: list[dict[str, Any]] = []

        is_numeric = pd.api.types.is_numeric_dtype(series)

        if is_numeric and unique_count > 0:
            clean_s = series.dropna()
            if len(clean_s) > 0:
                mean_val = round(float(clean_s.mean()), 4)
                std_val = round(float(clean_s.std()), 4) if len(clean_s) > 1 else 0.0
                min_val = round(float(clean_s.min()), 4)
                max_val = round(float(clean_s.max()), 4)
                median_val = round(float(clean_s.median()), 4)
                q25_val = round(float(clean_s.quantile(0.25)), 4)
                q75_val = round(float(clean_s.quantile(0.75)), 4)
                if len(clean_s) > 2 and std_val > 0:
                    try:
                        skew_val = round(float(clean_s.skew()), 4)
                    except Exception:
                        skew_val = None

            if target_numeric is not None and name != dataset.target_column and len(clean_s) > 1:
                try:
                    c = float(series.corr(target_numeric))
                    if not pd.isna(c):
                        target_corr = round(c, 4)
                except Exception:
                    pass
        else:
            vc = series.value_counts(dropna=False).head(5)
            for val, cnt in vc.items():
                top_cats.append({
                    "value": str(val) if not pd.isna(val) else "<NULL>",
                    "count": int(cnt),
                    "pct": round(float(cnt / row_count) * 100, 1) if row_count > 0 else 0.0,
                })

        # Generate smart actionable recommendations
        if name != dataset.target_column:
            if is_id:
                recommendations.append({
                    "column": name,
                    "type": "exclude",
                    "badge": "Identifier",
                    "severity": "danger",
                    "title": f"Excluir identificador '{name}'",
                    "description": f"Tiene alta cardinalidad ({unique_count} valores únicos / {card_ratio:.1%}). Debe excluirse para prevenir memorización.",
                    "action": "exclude",
                })
            elif unique_count <= 1 and row_count > 10:
                recommendations.append({
                    "column": name,
                    "type": "exclude",
                    "badge": "Varianza Cero",
                    "severity": "danger",
                    "title": f"Excluir columna constante '{name}'",
                    "description": "Todos los valores son idénticos o nulos; no aporta varianza predictiva.",
                    "action": "exclude",
                })
            elif null_count > 0 and (null_count / row_count) > 0.50:
                recommendations.append({
                    "column": name,
                    "type": "warn",
                    "badge": "Muchos Nulos",
                    "severity": "warning",
                    "title": f"Alta tasa de nulos en '{name}' ({null_count / row_count:.1%})",
                    "description": f"Contiene {null_count} valores ausentes. Se recomienda imputación o descartar.",
                    "action": "impute",
                })
            elif target_corr is not None and abs(target_corr) >= 0.25:
                recommendations.append({
                    "column": name,
                    "type": "recommend",
                    "badge": "Señal Fuerte",
                    "severity": "success",
                    "title": f"Fuerte correlación con target en '{name}' (r = {target_corr:+.3f})",
                    "description": "Característica candidata prioritaria para modelos lineales y árboles.",
                    "action": "keep",
                })
            elif is_high_card:
                recommendations.append({
                    "column": name,
                    "type": "transform",
                    "badge": "Alta Cardinalidad",
                    "severity": "info",
                    "title": f"Alta cardinalidad categórica en '{name}' ({unique_count} categorías)",
                    "description": "Se recomienda Target Encoding o Frequency Encoding para evitar explosión de One-Hot.",
                    "action": "keep",
                })

        columns.append(
            ColumnProfile(
                name=name,
                dtype=str(series.dtype),
                null_count=null_count,
                unique_count=unique_count,
                sample_values=sample,
                cardinality_ratio=card_ratio,
                is_identifier=is_id,
                is_high_cardinality=is_high_card,
                mean=mean_val,
                std=std_val,
                min=min_val,
                max=max_val,
                median=median_val,
                q25=q25_val,
                q75=q75_val,
                skew=skew_val,
                target_correlation=target_corr,
                top_categories=top_cats,
            )
        )

    # Multi-collinearity detection between numeric features
    numeric_cols = [c.name for c in columns if c.mean is not None and c.name != dataset.target_column and not c.is_identifier]
    if len(numeric_cols) >= 2:
        try:
            corr_mat = df[numeric_cols].corr()
            checked_pairs = set()
            for c1 in numeric_cols:
                for c2 in numeric_cols:
                    if c1 < c2 and (c1, c2) not in checked_pairs:
                        checked_pairs.add((c1, c2))
                        val = float(corr_mat.loc[c1, c2])
                        if not pd.isna(val) and abs(val) >= 0.88:
                            recommendations.append({
                                "column": c2,
                                "type": "collinear",
                                "badge": "Colinealidad",
                                "severity": "warning",
                                "title": f"Colinealidad entre '{c1}' y '{c2}' (r = {val:+.2f})",
                                "description": f"Alta redundancia detectada. CATML recomienda evaluar descartar '{c2}'.",
                                "action": "warn",
                            })
        except Exception:
            pass

    # First 8 preview rows (sanitizing NaN to None for clean JSON)
    preview_rows: list[dict[str, Any]] = []
    try:
        sample_df = df.head(8)
        for _, row in sample_df.iterrows():
            clean_row = {}
            for col_k, col_v in row.items():
                if pd.isna(col_v):
                    clean_row[col_k] = None
                elif isinstance(col_v, (int, float, str, bool)):
                    clean_row[col_k] = col_v
                else:
                    clean_row[col_k] = str(col_v)
            preview_rows.append(clean_row)
    except Exception:
        pass

    return DatasetProfile(
        dataset_id=dataset.id,
        row_count=row_count,
        column_count=len(df.columns),
        target_column=dataset.target_column,
        task_type=dataset.task_type,
        columns=columns,
        preview_rows=preview_rows,
        recommendations=recommendations,
    )


def infer_task_type(target_series: pd.Series) -> str:
    from automl.engine.planning.task_planner import infer_task_type as _infer

    return _infer(target_series).value


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
