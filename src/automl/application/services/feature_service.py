from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.features.evidence import FeatureEvidence
from automl.domain.features.feature_set import FeatureSet
from automl.domain.features.selection_strategy import (
    FeatureRank,
    FeatureSelectionStrategy,
    FeatureSetCandidate,
)
from automl.domain.ports import FeatureAnalysisContext, FeatureSelectorPort
from automl.domain.runs.states import RunStatus
from automl.engine.features.reduction.pca import PCAReducer
from automl.engine.features.selection.correlation import CorrelationSelector
from automl.engine.features.selection.ensemble import EnsembleRankSelector
from automl.engine.features.selection.importance import TreeImportanceSelector
from automl.engine.features.selection.mutual_information import MutualInformationSelector
from automl.engine.features.selection.variance import VarianceSelector
from automl.engine.planning.ablation_planner import AblationPlanner
from automl.engine.profiling.dataset_profiler import load_dataframe, profile_dataset

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class FeatureEngineeringService:
    """Application service for feature discovery, derived features, temporal features, selection, and ablation."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def validate_derived_feature(
        self,
        dataset_id: str,
        name: str,
        expression: str,
        expression_type: str = "formula",
    ) -> Any:
        """Validate and evaluate a derived feature definition on dataset without modifying it."""
        from automl.domain.features.derived_feature import (
            DerivedFeatureDefinition,
            DerivedFeatureType,
        )
        from automl.engine.features.generation.derived_feature_engine import DerivedFeatureEngine

        dataset = self.workspace._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)

        def_obj = DerivedFeatureDefinition(
            name=name,
            expression_type=DerivedFeatureType(expression_type),
            expression=expression,
        )
        engine = DerivedFeatureEngine()
        result, _ = engine.evaluate(df, def_obj)
        return result

    def apply_derived_feature(
        self,
        dataset_id: str,
        name: str,
        expression: str,
        expression_type: str = "formula",
        description: str = "",
    ) -> tuple[Any, Any]:
        """Evaluate and apply a derived feature to the dataset, re-profiling the dataset and saving changes."""
        from automl.domain.features.derived_feature import (
            DerivedFeatureDefinition,
            DerivedFeatureType,
        )
        from automl.engine.features.generation.derived_feature_engine import DerivedFeatureEngine

        dataset = self.workspace._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)

        def_obj = DerivedFeatureDefinition(
            name=name,
            expression_type=DerivedFeatureType(expression_type),
            expression=expression,
            description=description,
        )
        engine = DerivedFeatureEngine()
        updated_df, result = engine.apply(df, def_obj)

        # Save updated dataframe to dataset path
        updated_df.to_csv(dataset.path, index=False)

        # Re-profile dataset so new feature is part of the profile schema and available for experiments
        profile = profile_dataset(dataset, updated_df)
        self.workspace.repository.save_dataset_profile(profile)

        return result, profile

    def suggest_derived_features(
        self,
        dataset_id: str,
        llm_provider: Any = None,
    ) -> list[dict[str, Any]]:
        """Suggest candidate derived features tailored to dataset profile and validate them."""
        from automl.application.agents.specialists.feature_advisor import FeatureAdvisor

        dataset = self.workspace._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)
        profile = self.workspace.repository.get_dataset_profile(dataset_id)

        all_cols = [c.name for c in profile.columns] if profile else list(df.columns)
        numeric_cols = [
            c.name for c in (profile.columns if profile else [])
            if any(term in str(c.dtype).lower() for term in ("numeric", "float", "int"))
        ]
        if not numeric_cols:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        target_col = getattr(dataset, "target_column", getattr(dataset, "target", ""))
        questionnaire = self.workspace.repository.get_dataset_questionnaire(dataset_id)
        domain_hint = questionnaire.domain_hint if questionnaire else "general_tabular"

        advisor = FeatureAdvisor(llm_provider=llm_provider)
        candidates = advisor.propose_features(
            column_names=all_cols,
            numeric_columns=numeric_cols,
            target_column=target_col,
            domain_hint=domain_hint,
        )
        valid_defs, results = advisor.validate_and_filter(df, candidates)

        suggestions = []
        for prop, res in zip(valid_defs, results):
            suggestions.append({
                "definition": prop.to_dict(),
                "evaluation": res.to_dict(),
            })
        return suggestions

    def detect_temporal_structure(self, dataset_id: str) -> dict[str, Any]:
        """Inspects dataset and returns detected temporal periodicities and sequential columns."""
        from automl.engine.profiling.dataset_profiler import detect_sequential_structure

        dataset = self.workspace._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)
        target_col = getattr(dataset, "target_column", getattr(dataset, "target", None))
        structure = detect_sequential_structure(df, target_column=target_col)
        return structure.to_dict()

    def generate_temporal_features(
        self,
        run_id: str,
        dataset_id: str | None = None,
        max_lags: int = 1,
        include_lags: bool = True,
        include_deltas: bool = True,
        include_cyclical: bool = True,
        include_rolling: bool = False,
    ) -> list[FeatureSet]:
        """
        Discovers temporal dynamics (lags, deltas, cyclical periodicities) and registers
        candidate FeatureSet instances obeying 'Propose != Accept'.
        """
        from automl.engine.features.generation.temporal_generator import TemporalDynamicsGenerator
        from automl.engine.profiling.dataset_profiler import detect_sequential_structure

        run = self.workspace._get_run(run_id)
        effective_dataset_id = dataset_id or run.dataset_id
        dataset = self.workspace._get_dataset(effective_dataset_id)
        df = load_dataframe(dataset.path)
        target_col = getattr(dataset, "target_column", getattr(dataset, "target", None))

        base_features = self.workspace.get_feature_registry(effective_dataset_id).active_feature_names(target_col)
        structure = detect_sequential_structure(df, target_column=target_col)

        generator = TemporalDynamicsGenerator(
            max_lags=max_lags,
            include_lags=include_lags,
            include_deltas=include_deltas,
            include_cyclical=include_cyclical,
            include_rolling=include_rolling,
        )
        proposed_features = generator.propose_features(
            df=df,
            feature_names=base_features,
            target_column=target_col,
            temporal_structure=structure,
        )

        candidate_feature_sets = generator.propose_candidate_feature_sets(
            base_features=base_features,
            generated_features=proposed_features,
            dataset_id=effective_dataset_id,
        )

        candidates_as_fsc = generator.propose_candidate_sets_as_candidates(
            base_features=base_features,
            generated_features=proposed_features,
        )
        if run.id not in self.workspace._candidate_feature_sets:
            self.workspace._candidate_feature_sets[run.id] = []
        self.workspace._candidate_feature_sets[run.id].extend(candidates_as_fsc)

        created_sets: list[FeatureSet] = []
        for fs in candidate_feature_sets:
            self.workspace.repository.save_feature_set(fs)
            created_sets.append(fs)
            self.workspace._emit("FeatureSetCreated", fs.to_dict(), run_id=run.id)

        self.workspace._emit(
            "TemporalFeaturesGenerated",
            {
                "run_id": run.id,
                "dataset_id": effective_dataset_id,
                "generated_features_count": len(proposed_features),
                "candidate_sets_count": len(created_sets),
            },
            run_id=run.id,
        )
        return created_sets

    def select_features(
        self,
        run_id: str,
        strategy: FeatureSelectionStrategy | None = None,
    ) -> list[FeatureSetCandidate]:
        run = self.workspace._get_run(run_id)
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run_id} is cancelled")

        strat = strategy or FeatureSelectionStrategy()
        dataset = self.workspace._get_dataset(run.dataset_id)
        feature_registry = self.workspace.get_feature_registry(run.dataset_id)
        active_features = feature_registry.active_feature_names(run.config.target)

        task_type_str = (
            run.config.task_type.value
            if hasattr(run.config.task_type, "value")
            else str(run.config.task_type)
        )
        context = FeatureAnalysisContext(
            dataset_path=dataset.path,
            target_column=dataset.target_column,
            task_type=task_type_str,
            active_feature_names=active_features,
            random_seed=run.config.random_seed,
        )

        selectors: list[FeatureSelectorPort] = []
        for method in strat.methods:
            if method == "mutual_information":
                selectors.append(MutualInformationSelector())
            elif method in ("importance", "tree_importance"):
                selectors.append(TreeImportanceSelector())
            elif method == "correlation":
                selectors.append(CorrelationSelector())
            elif method == "variance":
                selectors.append(VarianceSelector())

        if not selectors:
            selectors = [MutualInformationSelector(), TreeImportanceSelector()]

        all_ranks: dict[str, list[FeatureRank]] = {}
        for sel in selectors:
            sel.fit(context)
            all_ranks[sel.method_id] = sel.rank_features()

        for feature_name in active_features:
            evidence = (
                self.workspace.repository.get_feature_evidence(run.id, feature_name)
                or FeatureEvidence(feature_id=feature_name)
            )
            if "mutual_information" in all_ranks:
                for r in all_ranks["mutual_information"]:
                    if r.feature_name == feature_name:
                        evidence.mutual_information = r.score
            if "importance" in all_ranks:
                for r in all_ranks["importance"]:
                    if r.feature_name == feature_name:
                        evidence.shap_importance = r.score
            evidence.confidence = 0.8
            self.workspace.repository.save_feature_evidence(run.id, evidence)

        if len(selectors) > 1:
            ensemble_sel = EnsembleRankSelector(
                selectors=selectors,
                combine_method=strat.combine_method,
            )
            combined_ranks = ensemble_sel.combine_rankings(
                all_ranks, combine_method=strat.combine_method
            )
        else:
            combined_ranks = all_ranks[selectors[0].method_id]

        self.workspace._feature_ranks[run.id] = combined_ranks

        candidates: list[FeatureSetCandidate] = []
        for k in strat.top_k:
            if k <= len(combined_ranks):
                selected_names = [r.feature_name for r in combined_ranks[:k]]
                candidates.append(
                    FeatureSetCandidate.create(
                        name=f"selected_top_{k}",
                        feature_names=selected_names,
                        method=strat.combine_method if len(selectors) > 1 else selectors[0].method_id,
                        k=k,
                        metadata={
                            "strategy": strat.to_dict(),
                            "features": selected_names,
                        },
                    )
                )

        if strat.include_reduction:
            pca = PCAReducer()
            for var in strat.reduction_variances:
                try:
                    pca.fit(context, n_components=var)
                    candidates.append(
                        FeatureSetCandidate.create(
                            name=f"pca_var_{int(var*100)}",
                            feature_names=[f"PC{i+1}" for i in range(pca.n_components())],
                            method="pca",
                            k=pca.n_components(),
                            metadata={
                                "variance_threshold": var,
                                "explained_variance": pca.explained_variance_ratio(),
                            },
                        )
                    )
                except Exception:
                    pass

        self.workspace._candidate_feature_sets[run.id] = candidates

        self.workspace._emit(
            "FeaturesSelected",
            {
                "run_id": run_id,
                "strategy": strat.to_dict(),
                "candidates": [c.name for c in candidates],
                "top_features": [r.feature_name for r in combined_ranks[:5]],
            },
            run_id=run.id,
        )
        return candidates

    def plan_ablation_experiments(
        self,
        run_id: str,
        base_feature_names: list[str] | None = None,
        model_ids: list[str] | None = None,
        max_features: int = 5,
        auto_enqueue: bool = True,
    ) -> list[ExperimentCandidate]:
        run = self.workspace._get_run(run_id)
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run_id} is cancelled")

        feature_registry = self.workspace.get_feature_registry(run.dataset_id)
        all_active = feature_registry.active_feature_names(run.config.target)

        features_base = base_feature_names or all_active
        if not features_base:
            return []

        ranked = self.workspace._feature_ranks.get(run.id, [])
        if ranked:
            ranked_names = [r.feature_name for r in ranked if r.feature_name in features_base]
            targets = ranked_names[:max_features]
        else:
            targets = features_base[:max_features]

        chosen_model = (
            model_ids[0]
            if model_ids
            else (run.config.models_include[0] if run.config.models_include else None)
        )

        ablation_planner = AblationPlanner()
        candidates = ablation_planner.propose_ablation(
            run=run,
            base_feature_names=features_base,
            features_to_ablate=targets,
            model_id=chosen_model,
        )

        profile = self.workspace.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = self.workspace._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)
            self.workspace.repository.save_dataset_profile(profile)

        for candidate in candidates:
            score = self.workspace.scorer.score(
                candidate=candidate,
                run=run,
                profile=profile,
            )
            candidate.priority = score

        if auto_enqueue:
            queue = self.workspace.get_experiment_queue(run.id)
            queue.enqueue_all(candidates)

        self.workspace._emit(
            "AblationExperimentsPlanned",
            {
                "run_id": run_id,
                "candidate_count": len(candidates),
                "ablated_features": targets,
            },
            run_id=run.id,
        )
        return candidates

    def promote_candidate_feature_set(
        self,
        run_id: str,
        candidate_id: str,
        new_name: str | None = None,
    ) -> FeatureSet:
        run = self.workspace._get_run(run_id)
        candidates = self.workspace._candidate_feature_sets.get(run.id, [])
        candidate = next(
            (c for c in candidates if c.id == candidate_id or c.name == candidate_id),
            None,
        )
        if candidate is None:
            raise KeyError(f"Candidate feature set not found: {candidate_id}")

        feature_set_name = new_name or candidate.name
        feature_set = self.workspace.create_feature_set(
            dataset_id=run.dataset_id,
            name=feature_set_name,
            feature_names=candidate.feature_names,
            lineage=f"promoted_from_{candidate.method}_{candidate.id}",
        )
        self.workspace._emit(
            "FeatureSetPromoted",
            {
                "run_id": run.id,
                "candidate_id": candidate.id,
                "feature_set_id": feature_set.id,
                "feature_names": feature_set.feature_names,
            },
            run_id=run.id,
        )
        return feature_set

    def get_feature_evidence(
        self,
        run_id: str,
        feature_id: str | None = None,
    ) -> FeatureEvidence | list[FeatureEvidence] | None:
        if feature_id:
            return self.workspace.repository.get_feature_evidence(run_id, feature_id)
        return self.workspace.repository.list_feature_evidence(run_id)

    def list_candidate_feature_sets(self, run_id: str) -> list[dict]:
        candidates = self.workspace._candidate_feature_sets.get(run_id, [])
        return [c.to_dict() for c in candidates]

    def get_feature_ranking(self, run_id: str, method: str | None = None) -> list[dict]:
        ranks = self.workspace._feature_ranks.get(run_id, [])
        if method:
            ranks = [r for r in ranks if r.method == method]
        return [r.to_dict() for r in ranks]
