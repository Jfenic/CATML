"""Shared validation-strategy contracts (single source of truth for CV aliases)."""
from __future__ import annotations

VALIDATION_ALIASES: dict[str, str] = {
    "k_fold": "kfold",
    "k-fold": "kfold",
    "kfolds": "kfold",
    "cv": "kfold",
    "cross_val": "cross_validation",
    "cross-validation": "cross_validation",
    "stratifiedkfold": "stratified_kfold",
    "stratified-kfold": "stratified_kfold",
    "timeseries": "time_series",
    "timeseriessplit": "time_series_split",
    "groupcv": "group_kfold",
    "grouped_cv": "group_kfold",
}

# Strategies the training engine implements (post-normalization).
KNOWN_VALIDATION_STRATEGIES: frozenset[str] = frozenset({
    "holdout",
    "oof",
    "cross_validation",
    "kfold",
    "stratified_kfold",
    "time_series",
    "time_series_split",
    "group_kfold",
    "group_cv",
    "grouped_kfold",
})


def normalize_validation_strategy(name: str | None, *, default: str = "holdout") -> str:
    """Normalize aliases (e.g. k_fold → kfold) to a canonical strategy name."""
    if name is None or not str(name).strip():
        return default
    key = str(name).lower().strip()
    return VALIDATION_ALIASES.get(key, key)


def require_known_validation_strategy(name: str | None, *, default: str = "holdout") -> str:
    """Normalize and reject unknown strategies (fail closed — no silent holdout)."""
    canonical = normalize_validation_strategy(name, default=default)
    if canonical not in KNOWN_VALIDATION_STRATEGIES:
        raise ValueError(
            f"Unknown validation_strategy {name!r} (normalized={canonical!r}). "
            f"Known: {sorted(KNOWN_VALIDATION_STRATEGIES)}"
        )
    return canonical
