from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    PlanAblationExperimentsCommand,
    PromoteCandidateFeatureSetCommand,
    RunExperimentCommand,
    SelectFeaturesCommand,
)
from automl.application.queries.workspace_queries import (
    GetFeatureEvidenceQuery,
    GetFeatureRankingQuery,
    ListCandidateFeatureSetsQuery,
)
from automl.domain.features.evidence import FeatureEvidence, FeatureInteractionEvidence
from automl.domain.features.selection_strategy import (
    FeatureRank,
    FeatureSelectionStrategy,
    FeatureSetCandidate,
)
from automl.domain.ports import FeatureAnalysisContext
from automl.engine.features.reduction.pca import PCAReducer
from automl.engine.features.selection.correlation import CorrelationSelector
from automl.engine.features.selection.ensemble import EnsembleRankSelector
from automl.engine.features.selection.importance import L1Selector, TreeImportanceSelector
from automl.engine.features.selection.mutual_information import MutualInformationSelector
from automl.engine.features.selection.variance import VarianceSelector
from automl.engine.planning.ablation_planner import AblationPlanner


@pytest.fixture
def sample_dataset_csv(tmp_path: Path) -> str:
    csv_path = tmp_path / "test_data.csv"
    np.random.seed(42)
    n = 120
    # salary and debt are strongly informative, noise1 and noise2 are random noise
    salary = np.random.uniform(20000, 100000, size=n)
    debt = np.random.uniform(1000, 50000, size=n)
    noise1 = np.random.normal(0, 1, size=n)
    noise2 = np.random.uniform(-10, 10, size=n)
    # Binary target strongly correlated with salary and debt
    churn = ((salary / 100000) * 0.6 - (debt / 50000) * 0.4 + np.random.normal(0, 0.1, size=n) > 0.1).astype(int)

    df = pd.DataFrame({
        "salary": salary,
        "debt": debt,
        "noise1": noise1,
        "noise2": noise2,
        "churn": churn,
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_domain_feature_dataclasses():
    rank = FeatureRank(feature_name="salary", score=0.85, rank=1, method="mi")
    assert rank.feature_name == "salary"
    d = rank.to_dict()
    assert FeatureRank.from_dict(d).rank == 1

    candidate = FeatureSetCandidate.create("top_2", ["salary", "debt"], method="mi", k=2)
    assert candidate.name == "top_2"
    assert candidate.k == 2
    cd = candidate.to_dict()
    assert FeatureSetCandidate.from_dict(cd).feature_names == ["salary", "debt"]

    strat = FeatureSelectionStrategy(methods=["mutual_information"], top_k=[2, 3])
    assert strat.combine_method == "weighted_rank"
    assert FeatureSelectionStrategy.from_dict(strat.to_dict()).top_k == [2, 3]

    evidence = FeatureEvidence(feature_id="salary", mutual_information=0.42, confidence=0.9)
    ed = evidence.to_dict()
    assert FeatureEvidence.from_dict(ed).mutual_information == 0.42

    interaction = FeatureInteractionEvidence(features=("salary", "debt"), interaction_score=0.75)
    idict = interaction.to_dict()
    assert FeatureInteractionEvidence.from_dict(idict).interaction_score == 0.75


def test_mutual_information_selector(sample_dataset_csv: str):
    context = FeatureAnalysisContext(
        dataset_path=sample_dataset_csv,
        target_column="churn",
        task_type="binary_classification",
        active_feature_names=["salary", "debt", "noise1", "noise2"],
        random_seed=42,
    )
    selector = MutualInformationSelector()
    selector.fit(context)
    ranks = selector.rank_features()

    assert len(ranks) == 4
    assert ranks[0].rank == 1
    # salary and debt should be top ranked features over pure noise
    top_two = [r.feature_name for r in ranks[:2]]
    assert "salary" in top_two or "debt" in top_two

    candidate = selector.select(k=2)
    assert len(candidate.feature_names) == 2
    assert candidate.method == "mutual_information"


def test_tree_importance_and_l1_selectors(sample_dataset_csv: str):
    context = FeatureAnalysisContext(
        dataset_path=sample_dataset_csv,
        target_column="churn",
        task_type="binary_classification",
        active_feature_names=["salary", "debt", "noise1", "noise2"],
        random_seed=42,
    )
    tree_sel = TreeImportanceSelector(n_estimators=30)
    tree_sel.fit(context)
    tree_ranks = tree_sel.rank_features()
    assert len(tree_ranks) == 4
    assert sum(r.score for r in tree_ranks) > 0.0

    l1_sel = L1Selector(alpha=0.01)
    l1_sel.fit(context)
    l1_ranks = l1_sel.rank_features()
    assert len(l1_ranks) == 4


def test_correlation_and_variance_selectors(sample_dataset_csv: str):
    context = FeatureAnalysisContext(
        dataset_path=sample_dataset_csv,
        target_column="churn",
        task_type="binary_classification",
        active_feature_names=["salary", "debt", "noise1", "noise2"],
        random_seed=42,
    )
    corr_sel = CorrelationSelector()
    corr_sel.fit(context)
    corr_ranks = corr_sel.rank_features()
    assert len(corr_ranks) == 4

    var_sel = VarianceSelector()
    var_sel.fit(context)
    var_ranks = var_sel.rank_features()
    assert len(var_ranks) == 4


def test_ensemble_rank_selector(sample_dataset_csv: str):
    context = FeatureAnalysisContext(
        dataset_path=sample_dataset_csv,
        target_column="churn",
        task_type="binary_classification",
        active_feature_names=["salary", "debt", "noise1", "noise2"],
        random_seed=42,
    )
    mi_sel = MutualInformationSelector()
    tree_sel = TreeImportanceSelector(n_estimators=20)

    # Weighted rank combination
    ensemble_weighted = EnsembleRankSelector(
        selectors=[mi_sel, tree_sel],
        combine_method="weighted_rank",
    )
    ensemble_weighted.fit(context)
    ranks_w = ensemble_weighted.rank_features()
    assert len(ranks_w) == 4
    assert ranks_w[0].rank == 1

    # Borda count combination
    ensemble_borda = EnsembleRankSelector(
        selectors=[mi_sel, tree_sel],
        combine_method="borda",
    )
    ensemble_borda.fit(context)
    ranks_b = ensemble_borda.rank_features()
    assert len(ranks_b) == 4
    candidate = ensemble_borda.select(k=2)
    assert len(candidate.feature_names) == 2


def test_pca_reducer(sample_dataset_csv: str):
    context = FeatureAnalysisContext(
        dataset_path=sample_dataset_csv,
        target_column="churn",
        task_type="binary_classification",
        active_feature_names=["salary", "debt", "noise1", "noise2"],
        random_seed=42,
    )
    pca = PCAReducer()
    pca.fit(context, n_components=2)
    assert pca.n_components() == 2
    assert len(pca.explained_variance_ratio()) == 2
    assert sum(pca.explained_variance_ratio()) <= 1.0


def test_ablation_planner():
    planner = AblationPlanner()
    from automl.domain.runs.run import AutoMLRun, RunConfig

    run = AutoMLRun(
        id="test_run",
        workspace_id="test_ws",
        dataset_id="ds_1",
        config=RunConfig(task_type="binary_classification", target="churn", models_include=["logistic_regression"]),
    )
    candidates = planner.propose_ablation(
        run=run,
        base_feature_names=["salary", "debt", "tenure"],
        features_to_ablate=["salary", "debt"],
    )

    assert len(candidates) == 2
    cand_names = [c.name for c in candidates]
    assert "ablation_without_salary" in cand_names
    assert "ablation_without_debt" in cand_names

    # Check that each ablation candidate omitted the targeted feature
    for c in candidates:
        if c.name == "ablation_without_salary":
            assert "salary" not in c.feature_names
            assert "debt" in c.feature_names
            assert "tenure" in c.feature_names


def test_feature_selection_workspace_flow(tmp_path: Path, sample_dataset_csv: str):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "automl_ws"))

    dataset = ws.register_dataset(
        name="churn_data",
        path=sample_dataset_csv,
        target="churn",
    )
    run = ws.create_run(dataset)

    # 1. Dispatch SelectFeaturesCommand
    strat = FeatureSelectionStrategy(
        methods=["mutual_information", "importance"],
        top_k=[2, 3],
        combine_method="weighted_rank",
    )
    candidates = cmd.dispatch(SelectFeaturesCommand(run_id=run.id, strategy=strat))
    assert len(candidates) >= 2

    # 2. Check candidate feature sets query
    queried_candidates = qry.dispatch(ListCandidateFeatureSetsQuery(run_id=run.id))
    assert len(queried_candidates) >= 2

    # 3. Check rankings query
    rankings = qry.dispatch(GetFeatureRankingQuery(run_id=run.id))
    assert len(rankings) == 4

    # 4. Check feature evidence query & persistence
    evidence = qry.dispatch(GetFeatureEvidenceQuery(run_id=run.id))
    assert len(evidence) == 4
    salary_ev = next(e for e in evidence if e.feature_id == "salary")
    assert salary_ev.mutual_information is not None

    # 5. Promote candidate feature set to formal FeatureSet
    promoted_fs = cmd.dispatch(
        PromoteCandidateFeatureSetCommand(
            run_id=run.id,
            candidate_id=candidates[0].id,
            new_name="curated_top_features",
        )
    )
    assert promoted_fs.name == "curated_top_features"
    assert len(promoted_fs.feature_names) == candidates[0].k

    # 6. Plan ablation experiments
    ablation_candidates = cmd.dispatch(
        PlanAblationExperimentsCommand(
            run_id=run.id,
            base_feature_names=["salary", "debt", "noise1"],
            max_features=2,
            auto_enqueue=True,
        )
    )
    assert len(ablation_candidates) == 2
    queue = ws.get_experiment_queue(run.id)
    assert len(queue) == 2

    # 7. Execute baseline experiment vs promoted feature set experiment ("Propose != Accept")
    exp_all = ws.create_experiment(
        run=run,
        name="exp_all_features",
        feature_names=["salary", "debt", "noise1", "noise2"],
        model_ids=["logistic_regression"],
    )
    res_all = ws.run_experiment(run, exp_all)
    assert len(res_all) == 1
    assert res_all[0].primary_score > 0.0

    exp_selected = ws.create_experiment(
        run=run,
        name="exp_selected_features",
        feature_set_id=promoted_fs.id,
        model_ids=["logistic_regression"],
    )
    res_selected = ws.run_experiment(run, exp_selected)
    assert len(res_selected) == 1
    assert res_selected[0].primary_score > 0.0
