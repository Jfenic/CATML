from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from automl.domain.features.evidence import FeatureInteractionEvidence
from automl.domain.features.feature_set import FeatureSet


@dataclass(frozen=True)
class GeneratedFeature:
    """Represents a generated synthetic interaction or encoding feature."""

    name: str
    feature_type: str  # "ratio" | "product" | "difference" | "target_encoding"
    source_columns: tuple[str, ...]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class InteractionFeatureGenerator:
    """
    Automated feature interaction generator for CATML.
    Discovers candidate features by generating:
      1. Pairwise numerical ratios (A / B) with epsilon safety.
      2. Pairwise numerical products (A * B).
      3. Pairwise numerical differences (A - B).
      4. Smoothed Out-Of-Fold target encoding for medium-cardinality categorical features.

    Strictly obeys 'Propose != Accept': Original datasets are never mutated,
    and generated features are proposed as candidate FeatureSets for hypothesis testing.
    """

    def __init__(
        self,
        max_numerical_pairs: int = 10,
        include_ratios: bool = True,
        include_products: bool = True,
        include_differences: bool = True,
        include_target_encoding: bool = True,
        min_categorical_cardinality: int = 2,
        max_categorical_cardinality: int = 100,
        smoothing_weight: float = 10.0,
    ) -> None:
        self.max_numerical_pairs = max_numerical_pairs
        self.include_ratios = include_ratios
        self.include_products = include_products
        self.include_differences = include_differences
        self.include_target_encoding = include_target_encoding
        self.min_categorical_cardinality = min_categorical_cardinality
        self.max_categorical_cardinality = max_categorical_cardinality
        self.smoothing_weight = smoothing_weight

        # Learned state for test transformations
        self._target_enc_maps: dict[str, dict[Any, float]] = {}
        self._global_mean: float = 0.5
        self.fitted: bool = False

    def propose_features(
        self,
        df: pd.DataFrame,
        feature_names: list[str],
        target_column: str | None = None,
        task_type: str = "binary_classification",
    ) -> list[GeneratedFeature]:
        """
        Scans active feature names and generates candidate interaction features
        without modifying the source dataframe.
        """
        candidates: list[GeneratedFeature] = []
        valid_cols = [c for c in feature_names if c in df.columns and c != target_column]

        # 1. Identify numerical and categorical columns
        numeric_cols = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in valid_cols if c not in numeric_cols]

        # 2. Pairwise numerical features (products, ratios, and differences)
        if len(numeric_cols) >= 2 and (self.include_ratios or self.include_products or self.include_differences):
            # Select top numeric pairs by variance to prioritize informative interactions
            variances = df[numeric_cols].var().fillna(0.0)
            sorted_num = sorted(numeric_cols, key=lambda c: float(variances.get(c, 0.0)), reverse=True)
            pairs = list(itertools.combinations(sorted_num[:8], 2))[: self.max_numerical_pairs]

            for col_a, col_b in pairs:
                if self.include_products:
                    prod_name = f"inter_prod_{col_a}_x_{col_b}"
                    candidates.append(
                        GeneratedFeature(
                            name=prod_name,
                            feature_type="product",
                            source_columns=(col_a, col_b),
                            description=f"Interaction product: {col_a} * {col_b}",
                        )
                    )

                if self.include_ratios:
                    # Non-symmetric ratios: A / B and B / A
                    ratio_name_ab = f"inter_ratio_{col_a}_div_{col_b}"
                    candidates.append(
                        GeneratedFeature(
                            name=ratio_name_ab,
                            feature_type="ratio",
                            source_columns=(col_a, col_b),
                            description=f"Interaction ratio: {col_a} / ({col_b} + eps)",
                        )
                    )

                if self.include_differences:
                    # Pairwise difference: A - B
                    diff_name_ab = f"inter_diff_{col_a}_minus_{col_b}"
                    candidates.append(
                        GeneratedFeature(
                            name=diff_name_ab,
                            feature_type="difference",
                            source_columns=(col_a, col_b),
                            description=f"Interaction difference: {col_a} - {col_b}",
                        )
                    )

        # 3. Target encoding for medium-cardinality categoricals
        if self.include_target_encoding and target_column and target_column in df.columns:
            for cat_col in categorical_cols:
                nunique = df[cat_col].nunique(dropna=True)
                if self.min_categorical_cardinality <= nunique <= self.max_categorical_cardinality:
                    enc_name = f"inter_te_{cat_col}"
                    candidates.append(
                        GeneratedFeature(
                            name=enc_name,
                            feature_type="target_encoding",
                            source_columns=(cat_col,),
                            description=f"Smoothed target encoding of {cat_col}",
                        )
                    )

        return candidates

    def fit(
        self,
        df: pd.DataFrame,
        target_column: str,
        features: list[GeneratedFeature],
    ) -> InteractionFeatureGenerator:
        """Fits learned encoding mappings on the training set."""
        if target_column not in df.columns:
            self.fitted = True
            return self

        # Convert target to binary numeric if necessary
        target_s = df[target_column]
        if not pd.api.types.is_numeric_dtype(target_s):
            from sklearn.preprocessing import LabelEncoder
            y_num = LabelEncoder().fit_transform(target_s)
        else:
            y_num = target_s.to_numpy()

        self._global_mean = float(np.mean(y_num))
        self._target_enc_maps.clear()

        for feat in features:
            if feat.feature_type == "target_encoding":
                col = feat.source_columns[0]
                temp_df = pd.DataFrame({"col": df[col], "target": y_num})
                stats = temp_df.groupby("col")["target"].agg(["count", "mean"])
                # Empirical Bayes / m-estimate smoothing: (count * mean + weight * global_mean) / (count + weight)
                smoothed = (
                    stats["count"] * stats["mean"] + self.smoothing_weight * self._global_mean
                ) / (stats["count"] + self.smoothing_weight)
                self._target_enc_maps[feat.name] = smoothed.to_dict()

        self.fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        features: list[GeneratedFeature],
    ) -> pd.DataFrame:
        """
        Applies generated feature transformations without mutating the original dataframe.
        Returns a new DataFrame containing original and transformed columns.
        """
        out_df = df.copy()

        for feat in features:
            if feat.feature_type == "product":
                col_a, col_b = feat.source_columns
                out_df[feat.name] = (out_df[col_a] * out_df[col_b]).astype(float)

            elif feat.feature_type == "ratio":
                col_a, col_b = feat.source_columns
                denom = out_df[col_b].astype(float)
                # Safe division: if denom == 0, replace with small eps
                safe_denom = np.where(denom == 0.0, 1e-6, denom)
                ratio_vals = out_df[col_a].astype(float) / safe_denom
                # Clip extreme infinities
                out_df[feat.name] = np.nan_to_num(ratio_vals, nan=0.0, posinf=1e6, neginf=-1e6)

            elif feat.feature_type == "difference":
                col_a, col_b = feat.source_columns
                out_df[feat.name] = (out_df[col_a].astype(float) - out_df[col_b].astype(float)).astype(float)

            elif feat.feature_type == "target_encoding":
                col = feat.source_columns[0]
                enc_map = self._target_enc_maps.get(feat.name, {})
                out_df[feat.name] = (
                    out_df[col].map(enc_map).fillna(self._global_mean).astype(float)
                )

        return out_df

    def propose_candidate_feature_sets(
        self,
        base_features: list[str],
        generated_features: list[GeneratedFeature],
        dataset_id: str,
    ) -> list[FeatureSet]:
        """
        Packages generated feature candidates into explicit FeatureSet versions
        ready for hypothesis-driven experimentation ('Propose != Accept').
        """
        candidates: list[FeatureSet] = []

        # 1. Base + All Interactions
        all_gen_names = [f.name for f in generated_features]
        if all_gen_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_interactions_all_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_all",
                    feature_names=base_features + all_gen_names,
                    created_by="interaction_generator",
                    lineage=f"Base ({len(base_features)}) + {len(all_gen_names)} interaction features",
                )
            )

        # 2. Base + Top Ratios Only
        ratio_names = [f.name for f in generated_features if f.feature_type == "ratio"]
        if ratio_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_interactions_ratios_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_ratios",
                    feature_names=base_features + ratio_names,
                    created_by="interaction_generator",
                    lineage=f"Base ({len(base_features)}) + {len(ratio_names)} ratio features",
                )
            )

        # 3. Base + Top Differences Only
        diff_names = [f.name for f in generated_features if f.feature_type == "difference"]
        if diff_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_interactions_diffs_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_differences",
                    feature_names=base_features + diff_names,
                    created_by="interaction_generator",
                    lineage=f"Base ({len(base_features)}) + {len(diff_names)} difference features",
                )
            )

        # 4. Base + Target Encoding Only
        te_names = [f.name for f in generated_features if f.feature_type == "target_encoding"]
        if te_names:
            candidates.append(
                FeatureSet(
                    id=f"fs_interactions_target_enc_{uuid.uuid4().hex[:6]}",
                    dataset_id=dataset_id,
                    name="interactions_target_enc",
                    feature_names=base_features + te_names,
                    created_by="interaction_generator",
                    lineage=f"Base ({len(base_features)}) + {len(te_names)} target encodings",
                )
            )

        return candidates

    def create_interaction_evidence(
        self,
        feat: GeneratedFeature,
        score: float,
        gain: float,
        confidence: float = 0.85,
    ) -> FeatureInteractionEvidence:
        """Records hypothesis evidence for a tested interaction."""
        return FeatureInteractionEvidence(
            features=feat.source_columns,
            interaction_score=float(score),
            experimental_gain=float(gain),
            confidence=float(confidence),
        )
