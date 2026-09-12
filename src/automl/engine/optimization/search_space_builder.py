from __future__ import annotations

from automl.domain.optimization.search_space import ParameterSpec, ParameterType, SearchSpace


class SearchSpaceBuilder:
    """
    Constructs model-specific search spaces for hyperparameter optimization.
    """

    @classmethod
    def build(cls, model_id: str, task_type: str | None = None) -> SearchSpace:
        space = SearchSpace()

        if model_id == "logistic_regression":
            space.add(
                ParameterSpec(
                    name="C",
                    type=ParameterType.FLOAT,
                    low=1e-3,
                    high=1e2,
                    log=True,
                    default=1.0,
                )
            )
            space.add(
                ParameterSpec.categorical(
                    name="solver",
                    choices=["lbfgs", "liblinear"],
                    default="lbfgs",
                )
            )
        elif model_id == "random_forest":
            space.add(
                ParameterSpec(
                    name="n_estimators",
                    type=ParameterType.INT,
                    low=20,
                    high=200,
                    step=20,
                    default=100,
                )
            )
            space.add(
                ParameterSpec(
                    name="max_depth",
                    type=ParameterType.INT,
                    low=3,
                    high=15,
                    default=8,
                )
            )
            space.add(
                ParameterSpec(
                    name="min_samples_split",
                    type=ParameterType.INT,
                    low=2,
                    high=10,
                    default=2,
                )
            )
        elif model_id == "ridge":
            space.add(
                ParameterSpec(
                    name="alpha",
                    type=ParameterType.FLOAT,
                    low=1e-3,
                    high=1e3,
                    log=True,
                    default=1.0,
                )
            )
        elif model_id in {"svc", "svr"}:
            space.add(
                ParameterSpec(
                    name="C",
                    type=ParameterType.FLOAT,
                    low=1e-2,
                    high=10.0,
                    log=True,
                    default=1.0,
                )
            )
            space.add(
                ParameterSpec.categorical(
                    name="kernel",
                    choices=["linear", "rbf"],
                    default="rbf",
                )
            )
        elif model_id == "kmeans":
            space.add(
                ParameterSpec(
                    name="n_clusters",
                    type=ParameterType.INT,
                    low=2,
                    high=8,
                    default=3,
                )
            )
            space.add(
                ParameterSpec(
                    name="n_init",
                    type=ParameterType.INT,
                    low=5,
                    high=15,
                    default=10,
                )
            )
        else:
            raise ValueError(f"No default search space for model: {model_id}")

        return space
