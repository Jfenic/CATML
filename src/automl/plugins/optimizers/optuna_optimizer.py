from __future__ import annotations

from typing import Any

import optuna
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from automl.domain.optimization.search_space import ParameterType, SearchSpace
from automl.domain.ports import OptimizerPort
from automl.engine.optimization.early_stopping import EarlyStoppingPolicy

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer(OptimizerPort):
    """
    Optuna-based bayesian optimizer implementing OptimizerPort via ask-and-tell interface.
    """

    def __init__(
        self,
        seed: int = 42,
        direction: str = "maximize",
        patience: int = 5,
        min_delta: float = 0.0001,
    ) -> None:
        self.direction = direction
        sampler = TPESampler(seed=seed)
        self._study = optuna.create_study(direction=direction, sampler=sampler)
        self._active_trials: dict[int, optuna.trial.Trial] = {}
        self.early_stopping = EarlyStoppingPolicy(
            patience=patience,
            min_delta=min_delta,
            mode="max" if direction == "maximize" else "min",
        )

    def suggest(self, trial_number: int, search_space: SearchSpace) -> dict[str, Any]:
        optuna_trial = self._study.ask()
        params: dict[str, Any] = {}

        for spec in search_space.list():
            if spec.type == ParameterType.INT:
                low = int(spec.low if spec.low is not None else 1)
                high = int(spec.high if spec.high is not None else 100)
                step = int(spec.step) if spec.step else 1
                params[spec.name] = optuna_trial.suggest_int(
                    spec.name,
                    low=low,
                    high=high,
                    step=step,
                    log=spec.log,
                )
            elif spec.type == ParameterType.FLOAT:
                low = float(spec.low if spec.low is not None else 1e-3)
                high = float(spec.high if spec.high is not None else 1.0)
                params[spec.name] = optuna_trial.suggest_float(
                    spec.name,
                    low=low,
                    high=high,
                    step=float(spec.step) if spec.step else None,
                    log=spec.log,
                )
            elif spec.type == ParameterType.CATEGORICAL:
                choices = spec.choices or ([spec.default] if spec.default is not None else [])
                if choices:
                    params[spec.name] = optuna_trial.suggest_categorical(spec.name, choices)

        self._active_trials[trial_number] = optuna_trial
        return params

    def observe(
        self,
        trial_number: int,
        parameters: dict[str, Any],
        score: float,
        succeeded: bool = True,
    ) -> None:
        optuna_trial = self._active_trials.pop(trial_number, None)
        if optuna_trial is None:
            return

        if not succeeded:
            self._study.tell(optuna_trial, state=TrialState.FAIL)
            return

        self._study.tell(optuna_trial, values=score)
        self.early_stopping.update(score)

    def should_stop(self) -> bool:
        return self.early_stopping.should_stop()

    def best_parameters(self) -> dict[str, Any]:
        try:
            return dict(self._study.best_params)
        except ValueError:
            return {}

    def best_score(self) -> float:
        try:
            return float(self._study.best_value)
        except ValueError:
            return 0.0
