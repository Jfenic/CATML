from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.domain.features.evidence import FeatureInteractionEvidence
from automl.domain.features.feature_set import FeatureSet
from automl.engine.features.generation.interaction_generator import (
    GeneratedFeature,
    InteractionFeatureGenerator,
)


@pytest.fixture
def sample_tabular_data() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    age = np.random.randint(18, 70, size=n)
    income = np.random.uniform(20000, 120000, size=n)
    commute = np.random.uniform(1, 50, size=n)
    city = np.random.choice(["Urban", "Suburban", "Rural"], size=n)
    target = (income * 0.0001 + (age > 40) * 1.5 + (city == "Urban") * 1.0 > 4.0).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Income": income,
        "Commute_km": commute,
        "City_Type": city,
        "Bought_EV": target,
    })


def test_propose_features_immutability(sample_tabular_data: pd.DataFrame) -> None:
    original_cols = list(sample_tabular_data.columns)
    generator = InteractionFeatureGenerator(max_numerical_pairs=5)

    features = generator.propose_features(
        df=sample_tabular_data,
        feature_names=["Age", "Income", "Commute_km", "City_Type"],
        target_column="Bought_EV",
    )

    # 1. Original dataframe is never mutated
    assert list(sample_tabular_data.columns) == original_cols

    # 2. Features were proposed
    assert len(features) > 0
    feature_types = {f.feature_type for f in features}
    assert "ratio" in feature_types
    assert "product" in feature_types
    assert "difference" in feature_types
    assert "target_encoding" in feature_types


def test_fit_and_transform_differences(sample_tabular_data: pd.DataFrame) -> None:
    generator = InteractionFeatureGenerator()
    features = [
        GeneratedFeature("diff_Income_Commute", "difference", ("Income", "Commute_km")),
    ]

    out_df = generator.transform(sample_tabular_data, features)

    # Verify input not modified
    assert "diff_Income_Commute" not in sample_tabular_data.columns

    # Verify output contains new column
    assert "diff_Income_Commute" in out_df.columns

    # Verify mathematical accuracy (A - B)
    expected_diff = sample_tabular_data["Income"] - sample_tabular_data["Commute_km"]
    np.testing.assert_allclose(out_df["diff_Income_Commute"], expected_diff)


def test_disable_differences_flag(sample_tabular_data: pd.DataFrame) -> None:
    generator = InteractionFeatureGenerator(include_differences=False)
    features = generator.propose_features(
        df=sample_tabular_data,
        feature_names=["Age", "Income", "Commute_km"],
        target_column="Bought_EV",
    )
    feature_types = {f.feature_type for f in features}
    assert "difference" not in feature_types
    assert "ratio" in feature_types or "product" in feature_types


def test_fit_and_transform_ratios_and_products(sample_tabular_data: pd.DataFrame) -> None:
    generator = InteractionFeatureGenerator()
    features = [
        GeneratedFeature("prod_Age_Income", "product", ("Age", "Income")),
        GeneratedFeature("ratio_Income_Age", "ratio", ("Income", "Age")),
    ]

    out_df = generator.transform(sample_tabular_data, features)

    # Verify input not modified
    assert "prod_Age_Income" not in sample_tabular_data.columns

    # Verify output contains new columns
    assert "prod_Age_Income" in out_df.columns
    assert "ratio_Income_Age" in out_df.columns

    # Verify mathematical accuracy
    expected_prod = sample_tabular_data["Age"] * sample_tabular_data["Income"]
    np.testing.assert_allclose(out_df["prod_Age_Income"], expected_prod)

    expected_ratio = sample_tabular_data["Income"] / sample_tabular_data["Age"]
    np.testing.assert_allclose(out_df["ratio_Income_Age"], expected_ratio)


def test_zero_division_safety() -> None:
    generator = InteractionFeatureGenerator()
    df = pd.DataFrame({"A": [10.0, 20.0], "B": [0.0, 5.0]})
    features = [GeneratedFeature("ratio_A_B", "ratio", ("A", "B"))]

    out_df = generator.transform(df, features)
    assert not np.isnan(out_df["ratio_A_B"]).any()
    assert not np.isinf(out_df["ratio_A_B"]).any()
    assert out_df["ratio_A_B"].iloc[0] > 0  # Safely handled division by zero


def test_smoothed_target_encoding(sample_tabular_data: pd.DataFrame) -> None:
    generator = InteractionFeatureGenerator(smoothing_weight=5.0)
    features = [GeneratedFeature("te_City", "target_encoding", ("City_Type",))]

    generator.fit(sample_tabular_data, target_column="Bought_EV", features=features)
    assert generator.fitted is True

    out_df = generator.transform(sample_tabular_data, features)
    assert "te_City" in out_df.columns
    assert not out_df["te_City"].isna().any()

    # Test set transformation with unseen category
    test_df = pd.DataFrame({"City_Type": ["Urban", "Unknown_City"]})
    out_test = generator.transform(test_df, features)
    assert len(out_test) == 2
    # Unseen category gets global mean fallback
    assert out_test["te_City"].iloc[1] == generator._global_mean


def test_propose_candidate_feature_sets() -> None:
    generator = InteractionFeatureGenerator()
    features = [
        GeneratedFeature("prod_1", "product", ("A", "B")),
        GeneratedFeature("ratio_1", "ratio", ("A", "B")),
        GeneratedFeature("diff_1", "difference", ("A", "B")),
        GeneratedFeature("te_1", "target_encoding", ("Cat",)),
    ]

    base_features = ["A", "B", "Cat"]
    candidates = generator.propose_candidate_feature_sets(
        base_features=base_features,
        generated_features=features,
        dataset_id="ds_123",
    )

    assert len(candidates) == 4
    names = {c.name for c in candidates}
    assert "interactions_all" in names
    assert "interactions_ratios" in names
    assert "interactions_differences" in names
    assert "interactions_target_enc" in names

    all_cand = next(c for c in candidates if c.name == "interactions_all")
    assert len(all_cand.feature_names) == 7
    assert all_cand.created_by == "interaction_generator"


def test_interaction_evidence_model() -> None:
    generator = InteractionFeatureGenerator()
    feat = GeneratedFeature("ratio_A_B", "ratio", ("A", "B"))
    evidence = generator.create_interaction_evidence(feat, score=0.88, gain=0.03, confidence=0.90)

    assert isinstance(evidence, FeatureInteractionEvidence)
    assert evidence.features == ("A", "B")
    assert evidence.interaction_score == 0.88
    assert evidence.experimental_gain == 0.03
    assert evidence.confidence == 0.90
