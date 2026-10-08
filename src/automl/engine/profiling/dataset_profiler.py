from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from automl.domain.datasets.profile import ColumnProfile, Dataset, DatasetProfile
from automl.domain.features.temporal import TemporalPeriodicity, TemporalStructure


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
        # High cardinality text/hash columns with near-unique values (excluding natural language text and image paths)
        if is_object_or_string and cardinality_ratio > 0.70:
            from automl.engine.features.text import is_text_column
            from automl.plugins.modalities.image_plugin import is_image_column

            if not is_text_column(series) and not is_image_column(series):
                is_identifier = True
        # Monotonically increasing sequential index integers (e.g. 0, 1, 2, ... N-1 or row counter)
        elif series.dtype.kind in {"i", "u"} and cardinality_ratio == 1.0:
            clean_series = series.dropna()
            if clean_series.is_monotonic_increasing:
                diffs = clean_series.diff().iloc[1:]
                if len(diffs) > 0 and (diffs == 1).all():
                    is_identifier = True

    return cardinality_ratio, is_identifier, is_high_cardinality


def detect_is_group_candidate(
    name: str,
    series: pd.Series,
    row_count: int,
    target_column: str | None = None,
) -> bool:
    """
    Evaluates whether a dataset column represents an entity or group grouping identifier
    (e.g., patient_id, user_id, device_id, hospital_id) where multiple observations
    belong to the same entity.
    """
    if target_column and name == target_column:
        return False

    if row_count < 10:
        return False

    unique_count = int(series.nunique(dropna=True))
    if unique_count <= 1:
        return False

    if unique_count >= row_count:
        return False

    clean_name = name.strip().lower()
    group_hints = {
        "patient", "subject", "client", "customer", "user", "device", "machine",
        "sensor", "hospital", "center", "centre", "site", "cluster", "group",
        "account", "household", "member", "person", "participant", "session",
        "store", "building", "doctor", "facility", "batch"
    }

    name_has_group_hint = any(hint in clean_name for hint in group_hints)
    name_is_id = (
        clean_name.endswith(("_id", ".id", "-id", "_key", "_pk"))
        or name.endswith(("Id", "ID"))
        or clean_name in {"group", "cluster", "patient", "subject", "entity"}
    )

    cardinality_ratio = unique_count / row_count if row_count > 0 else 0.0

    if name_has_group_hint and unique_count >= 2:
        return True

    if name_is_id and unique_count >= 2 and cardinality_ratio <= 0.90:
        return True

    return False


def detect_group_leakage(
    df: pd.DataFrame,
    group_column: str,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any] | None:
    """
    Simulates a standard IID train/test split to assess if entity groups overlap
    between training and validation partitions.
    """
    if group_column not in df.columns:
        return None

    clean_series = df[group_column].dropna()
    row_count = len(clean_series)
    if row_count < 10:
        return None

    total_unique = int(clean_series.nunique())
    if total_unique <= 1:
        return None

    from sklearn.model_selection import train_test_split

    indices = np.arange(row_count)
    train_idx, val_idx = train_test_split(
        indices, test_size=test_size, random_state=random_seed
    )

    train_groups = set(clean_series.iloc[train_idx])
    val_groups = set(clean_series.iloc[val_idx])
    overlapping = train_groups.intersection(val_groups)

    val_data = clean_series.iloc[val_idx]
    affected_rows = int(val_data.isin(overlapping).sum())
    total_val_rows = len(val_idx)

    leakage_detected = len(overlapping) > 0
    overlapping_ratio = len(overlapping) / len(val_groups) if val_groups else 0.0
    affected_ratio = affected_rows / total_val_rows if total_val_rows else 0.0

    if leakage_detected:
        description = (
            f"Group entity leakage detected in '{group_column}'. "
            f"{affected_rows} validation sample(s) ({affected_ratio:.1%}) share entity identifiers "
            f"across {len(overlapping)} group(s) with the training partition under standard IID splitting. "
            f"Standard validation will produce overly optimistic performance. Enforce GroupKFold('{group_column}')."
        )
    else:
        description = f"No entity group overlap detected in '{group_column}' under simulated split."

    return {
        "column": group_column,
        "leakage_detected": leakage_detected,
        "overlapping_groups_count": len(overlapping),
        "overlapping_groups_ratio": round(overlapping_ratio, 4),
        "affected_rows_count": affected_rows,
        "affected_rows_ratio": round(affected_ratio, 4),
        "total_unique_groups": total_unique,
        "total_val_rows": total_val_rows,
        "strategy_recommendation": f"GroupKFold({group_column})" if leakage_detected else None,
        "description": description,
    }


def detect_sequential_structure(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> TemporalStructure:
    """
    Analyzes dataframe columns to detect temporal sequences, chronological ordering,
    and cyclical periodicities (hourly, daily/weekly, monthly, annual).

    Returns:
        TemporalStructure: Detailed detection report containing order column,
        detected periodicities, and list of temporal/sequential columns.
    """
    valid_cols = [c for c in df.columns if c != target_column]
    temporal_columns: list[str] = []
    detected_periodicities: list[TemporalPeriodicity] = []
    order_column: str | None = None

    row_count = len(df)
    if row_count == 0:
        return TemporalStructure(is_sequential=False)

    # 1. Detect explicit datetime and timestamp columns
    for col in valid_cols:
        series = df[col]
        clean_name = str(col).strip().lower()
        is_datetime_type = pd.api.types.is_datetime64_any_dtype(series)

        # Content-based datetime heuristic for strings/objects
        is_parsable_datetime = False
        if not is_datetime_type and (series.dtype.kind in {"O", "S", "U"} or str(series.dtype) == "object"):
            sample = series.dropna().head(20)
            if len(sample) >= 3:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if parsed.notna().mean() >= 0.8:
                        is_parsable_datetime = True
                except Exception:
                    pass

        if is_datetime_type or is_parsable_datetime:
            if col not in temporal_columns:
                temporal_columns.append(col)
            # Check if this column is ordered monotonically
            try:
                dt_series = series if is_datetime_type else pd.to_datetime(series.dropna(), errors="coerce", format="mixed")
                clean_dt = dt_series.dropna()
                if len(clean_dt) >= 2 and clean_dt.is_monotonic_increasing:
                    if order_column is None:
                        order_column = col
            except Exception:
                pass

    # 2. Sequential integer / index columns (if no explicit datetime order column found yet)
    if order_column is None:
        seq_name_hints = ("step", "seq", "sequence", "time", "timestamp", "epoch", "iteration", "tick", "t_index")
        for col in valid_cols:
            series = df[col]
            clean_name = str(col).strip().lower()
            if series.dtype.kind in {"i", "u", "f"}:
                clean_s = series.dropna()
                if len(clean_s) >= 3 and clean_s.is_monotonic_increasing:
                    if any(hint in clean_name for hint in seq_name_hints):
                        order_column = col
                        if col not in temporal_columns:
                            temporal_columns.append(col)
                        break

    # 3. Cyclical / Periodic column detection
    for col in valid_cols:
        series = df[col]
        clean_name = str(col).strip().lower()
        if not pd.api.types.is_numeric_dtype(series):
            continue

        clean_s = series.dropna()
        if len(clean_s) < 3:
            continue

        min_val = float(clean_s.min())
        max_val = float(clean_s.max())
        nunique = int(clean_s.nunique())

        # 1. Hourly (T = 24.0): by name or 0..23 hour range
        if ("hour" in clean_name or clean_name.startswith("hr") or clean_name.endswith("_hr")) and 0.0 <= min_val and max_val <= 24.0:
            if not any(p.column == col for p in detected_periodicities):
                detected_periodicities.append(
                    TemporalPeriodicity(name="hourly", period=24.0, column=col, description=f"24-hour diurnal cycle detected on '{col}'")
                )
            if col not in temporal_columns:
                temporal_columns.append(col)

        # 2. Weekly / Day of week (T = 7.0): by name or 0..6 / 1..7 range
        elif any(k in clean_name for k in ("weekday", "day_of_week", "dayofweek", "dow")) and 0.0 <= min_val and max_val <= 7.0:
            if not any(p.column == col for p in detected_periodicities):
                detected_periodicities.append(
                    TemporalPeriodicity(name="weekly", period=7.0, column=col, description=f"7-day weekly cycle detected on '{col}'")
                )
            if col not in temporal_columns:
                temporal_columns.append(col)

        # 3. Monthly (T = 12.0): by name or 1..12 month range
        elif ("month" in clean_name or clean_name.startswith("mes") or clean_name.endswith("_month")) and 0.0 <= min_val and max_val <= 12.0:
            if not any(p.column == col for p in detected_periodicities):
                detected_periodicities.append(
                    TemporalPeriodicity(name="monthly", period=12.0, column=col, description=f"12-month annual cycle detected on '{col}'")
                )
            if col not in temporal_columns:
                temporal_columns.append(col)

        # 4. Annual Day of Year (T = 365.25): name contains 'dayofyear', 'doy', 'day_of_year' or range 1..366
        elif any(k in clean_name for k in ("dayofyear", "day_of_year", "doy")) and 1.0 <= min_val and max_val <= 366.0:
            if not any(p.column == col for p in detected_periodicities):
                detected_periodicities.append(
                    TemporalPeriodicity(name="annual", period=365.25, column=col, description=f"365.25-day annual cycle detected on '{col}'")
                )
            if col not in temporal_columns:
                temporal_columns.append(col)

        # 5. Content-based fallback when column name is generic
        elif 0.0 <= min_val and max_val <= 23.0 and max_val >= 20.0 and nunique >= 10:
            if not any(p.column == col for p in detected_periodicities):
                detected_periodicities.append(
                    TemporalPeriodicity(name="hourly", period=24.0, column=col, description=f"24-hour diurnal cycle detected on '{col}'")
                )
            if col not in temporal_columns:
                temporal_columns.append(col)

    is_seq = bool(order_column is not None or len(detected_periodicities) > 0 or len(temporal_columns) > 0)

    return TemporalStructure(
        is_sequential=is_seq,
        order_column=order_column,
        detected_periodicities=detected_periodicities,
        temporal_columns=temporal_columns,
        metadata={
            "detected_periodicities_count": len(detected_periodicities),
            "temporal_columns_count": len(temporal_columns),
        },
    )


def profile_dataset(dataset: Dataset, df: pd.DataFrame | None = None) -> DatasetProfile:
    if df is None:
        path = Path(dataset.path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = load_dataframe(path)

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
    group_leakage_reports: list[dict[str, Any]] = []

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
        is_group_cand = detect_is_group_candidate(
            name=name,
            series=series,
            row_count=row_count,
            target_column=dataset.target_column,
        )
        group_report = None
        if is_group_cand:
            group_report = detect_group_leakage(df, name)
            if group_report is not None:
                group_leakage_reports.append(group_report)

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
        from automl.engine.features.text import is_text_column
        from automl.plugins.modalities.image_plugin import is_image_column

        is_image = False if is_numeric else is_image_column(series)
        is_text = False if (is_numeric or is_image) else is_text_column(series)

        box_plot_data: dict[str, Any] = {}
        histogram_data: dict[str, Any] = {}

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
                iqr_val = round(float(q75_val - q25_val), 4)
                if len(clean_s) > 2 and std_val > 0:
                    try:
                        skew_val = round(float(clean_s.skew()), 4)
                    except Exception:
                        skew_val = None

                box_plot_data = {
                    "min": min_val,
                    "q25": q25_val,
                    "median": median_val,
                    "q75": q75_val,
                    "max": max_val,
                    "mean": mean_val,
                    "iqr": iqr_val,
                    "by_target": [],
                }

                # Histogram calculation (10 bins)
                try:
                    hist_counts, bin_edges = np.histogram(clean_s, bins=10)
                    bin_labels = [f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}" for i in range(len(hist_counts))]
                    histogram_data = {
                        "bins": bin_labels,
                        "counts": [int(c) for c in hist_counts],
                        "percentages": [round(float(c / len(clean_s)) * 100, 1) for c in hist_counts],
                    }
                except Exception:
                    pass

                # If dataset target column is present, calculate box plot by target class
                if dataset.target_column in df.columns and name != dataset.target_column:
                    try:
                        tgt_series = df[dataset.target_column]
                        target_classes = tgt_series.dropna().unique().tolist()
                        if len(target_classes) <= 6:
                            by_target_list = []
                            for tc in sorted(target_classes, key=lambda x: str(x)):
                                mask = (tgt_series == tc)
                                sub_s = series[mask].dropna()
                                if len(sub_s) > 0:
                                    tc_min = round(float(sub_s.min()), 4)
                                    tc_q25 = round(float(sub_s.quantile(0.25)), 4)
                                    tc_med = round(float(sub_s.median()), 4)
                                    tc_q75 = round(float(sub_s.quantile(0.75)), 4)
                                    tc_max = round(float(sub_s.max()), 4)
                                    tc_mean = round(float(sub_s.mean()), 4)
                                    by_target_list.append({
                                        "class_label": str(tc),
                                        "count": int(len(sub_s)),
                                        "min": tc_min,
                                        "q25": tc_q25,
                                        "median": tc_med,
                                        "q75": tc_q75,
                                        "max": tc_max,
                                        "mean": tc_mean,
                                        "iqr": round(tc_q75 - tc_q25, 4),
                                    })
                            box_plot_data["by_target"] = by_target_list
                    except Exception:
                        pass

            if target_numeric is not None and name != dataset.target_column and len(clean_s) > 1:
                try:
                    c = float(series.corr(target_numeric))
                    if not pd.isna(c):
                        target_corr = round(c, 4)
                except Exception:
                    pass
        else:
            vc = series.value_counts(dropna=False).head(10)
            tgt_series = df[dataset.target_column] if dataset.target_column in df.columns and name != dataset.target_column else None
            for val, cnt in vc.items():
                val_str = str(val) if not pd.isna(val) else "<NULL>"
                cat_info: dict[str, Any] = {
                    "value": val_str,
                    "count": int(cnt),
                    "pct": round(float(cnt / row_count) * 100, 1) if row_count > 0 else 0.0,
                }
                if tgt_series is not None and target_numeric is not None:
                    try:
                        mask = (series == val) if not pd.isna(val) else series.isna()
                        sub_tgt = target_numeric[mask].dropna()
                        if len(sub_tgt) > 0:
                            cat_info["target_rate"] = round(float(sub_tgt.mean()) * 100, 1)
                    except Exception:
                        pass
                top_cats.append(cat_info)

        # Generate smart actionable recommendations
        if name != dataset.target_column:
            if is_group_cand and group_report and group_report["leakage_detected"]:
                recommendations.append({
                    "column": name,
                    "type": "group_leakage",
                    "badge": "Group Leakage",
                    "severity": "danger",
                    "title": f"Group entity leakage in '{name}' ({group_report['affected_rows_count']} samples shared)",
                    "description": group_report["description"],
                    "action": "enforce_group_split",
                    "group_column": name,
                    "strategy": group_report["strategy_recommendation"],
                })
            elif is_id:
                recommendations.append({
                    "column": name,
                    "type": "exclude",
                    "badge": "Identifier",
                    "severity": "danger",
                    "title": f"Exclude identifier '{name}'",
                    "description": f"High cardinality ({unique_count} unique values · {card_ratio:.1%}). Exclude to prevent memorization.",
                    "action": "exclude",
                })
            elif unique_count <= 1 and row_count > 10:
                recommendations.append({
                    "column": name,
                    "type": "exclude",
                    "badge": "Zero Variance",
                    "severity": "danger",
                    "title": f"Exclude constant feature '{name}'",
                    "description": "Constant or null values across all rows. No predictive variance.",
                    "action": "exclude",
                })
            elif null_count > 0 and (null_count / row_count) > 0.50:
                recommendations.append({
                    "column": name,
                    "type": "warn",
                    "badge": "High Nulls",
                    "severity": "warning",
                    "title": f"High missing rate in '{name}' ({null_count / row_count:.1%})",
                    "description": f"Contains {null_count} missing entries. Imputation or exclusion recommended.",
                    "action": "impute",
                })
            elif target_corr is not None and abs(target_corr) >= 0.999:
                recommendations.append({
                    "column": name,
                    "type": "leakage",
                    "badge": "Target Leakage",
                    "severity": "danger",
                    "title": f"Critical target leakage in '{name}' (|r| = {abs(target_corr):.4f})",
                    "description": f"Near-perfect correlation (|r| >= 0.999) with target '{dataset.target_column}'. High risk of target duplicate, label leak, or future data contamination. Exclude feature.",
                    "action": "exclude",
                })
            elif target_corr is not None and abs(target_corr) >= 0.25:
                recommendations.append({
                    "column": name,
                    "type": "recommend",
                    "badge": "High Relevance",
                    "severity": "success",
                    "title": f"High relevance in '{name}' (r = {target_corr:+.3f})",
                    "description": "Strong target association. Prime candidate feature.",
                    "action": "keep",
                })
            elif is_image:
                recommendations.append({
                    "column": name,
                    "type": "vision",
                    "badge": "Image Feature",
                    "severity": "info",
                    "title": f"Image file references in '{name}'",
                    "description": f"Image path references detected ({unique_count} distinct entries). Processed via ImageEncoder.",
                    "action": "image_encode",
                })
            elif is_text:
                recommendations.append({
                    "column": name,
                    "type": "nlp",
                    "badge": "Text Feature",
                    "severity": "info",
                    "title": f"Natural language text in '{name}'",
                    "description": f"Freeform text detected ({unique_count} distinct entries). Processed via TF-IDF n-grams.",
                    "action": "nlp_encode",
                })
            elif is_high_card:
                recommendations.append({
                    "column": name,
                    "type": "transform",
                    "badge": "High Cardinality",
                    "severity": "info",
                    "title": f"High categorical cardinality in '{name}' ({unique_count} categories)",
                    "description": "Target or frequency encoding recommended to prevent high-dimensional sparsity.",
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
                is_text=is_text,
                is_image=is_image,
                is_group_candidate=is_group_cand,
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
                histogram=histogram_data,
                box_plot=box_plot_data,
            )
        )

    # Multi-collinearity detection and full correlation matrix
    numeric_cols = [c.name for c in columns if c.mean is not None and c.name != dataset.target_column and not c.is_identifier]
    correlation_matrix_data: dict[str, Any] = {}
    if len(numeric_cols) >= 2:
        try:
            corr_cols = list(numeric_cols)
            corr_df = df[corr_cols].copy()
            if target_numeric is not None and dataset.target_column not in corr_cols:
                corr_df[dataset.target_column] = target_numeric
                corr_cols.append(dataset.target_column)

            corr_mat = corr_df.corr().round(4)
            clean_matrix = []
            for r in corr_mat.values:
                clean_matrix.append([round(float(v), 4) if not pd.isna(v) else 0.0 for v in r])

            correlation_matrix_data = {
                "columns": corr_cols,
                "matrix": clean_matrix,
            }

            checked_pairs = set()
            for c1 in numeric_cols:
                for c2 in numeric_cols:
                    if c1 < c2 and (c1, c2) not in checked_pairs:
                        checked_pairs.add((c1, c2))
                        val = float(corr_mat.loc[c1, c2]) if c1 in corr_mat.index and c2 in corr_mat.columns else 0.0
                        if not pd.isna(val) and abs(val) >= 0.88:
                            recommendations.append({
                                "column": c2,
                                "type": "collinear",
                                "badge": "Collinearity",
                                "severity": "warning",
                                "title": f"Collinearity between '{c1}' and '{c2}' (r = {val:+.2f})",
                                "description": f"High feature redundancy detected. Consider pruning '{c2}' to reduce variance.",
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

    # Temporal & Sequential Structure Analysis
    temporal_struct = detect_sequential_structure(df, target_column=dataset.target_column)
    if temporal_struct.is_sequential:
        recommendations.append({
            "column": temporal_struct.order_column or (temporal_struct.temporal_columns[0] if temporal_struct.temporal_columns else dataset.target_column),
            "type": "temporal",
            "badge": "Sequential Dynamics",
            "severity": "info",
            "title": f"Sequential structure detected ({temporal_struct.order_column or 'cyclical periodicities'})",
            "description": f"Dataset exhibits chronological or cyclical dependencies ({len(temporal_struct.detected_periodicities)} periodicities). Candidate lags and trend deltas can be generated.",
            "action": "generate_temporal_features",
        })

    # Anti-Leakage Guardian: Sequential / Ordering Leakage Detection
    # Detect if target values are monotonically sorted or strongly correlated with row position
    if target_numeric is not None and row_count >= 20:
        row_indices = pd.Series(np.arange(row_count), index=df.index, dtype=float)
        try:
            pos_corr = float(target_numeric.corr(row_indices))
            if not pd.isna(pos_corr) and abs(pos_corr) >= 0.95:
                recommendations.append({
                    "column": dataset.target_column,
                    "type": "leakage",
                    "badge": "Sequential Leakage",
                    "severity": "danger",
                    "title": f"Target correlates with row order (|r| = {abs(pos_corr):.3f})",
                    "description": "Target is sorted or strongly ordered by row position. Standard random CV splits will cause severe optimistic data leakage. Grouped or time-series validation required.",
                    "action": "warn",
                })
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
        correlation_matrix=correlation_matrix_data,
        temporal_structure=temporal_struct.to_dict(),
        group_leakage_reports=group_leakage_reports,
    )


def infer_task_type(target_series: pd.Series) -> str:
    from automl.engine.planning.task_planner import infer_task_type as _infer

    return _infer(target_series).value


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from disk supporting Parquet, JSON/JSONL, TSV, and CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(p)
        except ImportError as exc:
            raise ImportError(
                f"Reading parquet file '{p.name}' requires 'pyarrow' or 'fastparquet'. "
                "Install with `pip install pyarrow` or `pip install catml[parquet]`."
            ) from exc
    elif suffix in {".json", ".jsonl"}:
        return pd.read_json(p, lines=(suffix == ".jsonl"))
    elif suffix in {".tsv", ".tab"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p)
