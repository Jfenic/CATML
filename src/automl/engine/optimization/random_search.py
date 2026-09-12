from __future__ import annotations

import math
import random
from typing import Any

from automl.domain.optimization.search_space import ParameterType, SearchSpace
from automl.domain.ports import OptimizerPort
from automl.engine.optimization.early_stopping import EarlyStoppingPolicy


class RandomSearchOptimizer(OptimizerPort):
    """
    Reference random search optimizer implementing OptimizerPort.
    """

    def __init__(
        self,
        seed: int = 42,
        patience: int = 5,
        min_delta: float = 0.0001,
        mode: str = "max",
    ) -> None:
        self.rng = random.Random(seed)
        self.early_stopping = EarlyStoppingPolicy(
            patience=patience,
            min_delta=min_delta,
            mode=mode,
        )
        self._best_params: dict[str, Any] = {}
        self._best_score: float = -float("inf") if mode == "max" else float("inf")
        self.mode = mode

    def suggest(self, trial_number: int, search_space: SearchSpace) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for spec in search_space.list():
            if spec.type == ParameterType.INT:
                low = int(spec.low if spec.low is not None else 1)
                high = int(spec.high if spec.high is not None else 100)
                step = int(spec.step if spec.step is not None else 1)
                params[spec.name] = self.rng.randrange(low, high + 1, step)
            elif spec.type == ParameterType.FLOAT:
                low = float(spec.low if spec.low is not None else 1e-3)
                high = float(spec.high if spec.high is not None else 1.0)
                if spec.log and low > 0 and high > 0:
                    log_val = self.rng.uniform(math.log10(low), math.log10(high))
                    params[spec.name] = round(10.0 ** log_val, 6)
                else:
                    val = self.rng.uniform(low, high)
                    if spec.step:
                        val = round(val / spec.step) * spec.step
                    params[spec.name] = round(val, 6)
            elif spec.type == ParameterType.CATEGORICAL:
                choices = spec.choices or []
                params[spec.name] = self.rng.choice(choices) if choices else spec.default
        return params

    def observe(
        self,
        trial_number: int,
        parameters: dict[str, Any],
        score: float,
        succeeded: bool = True,
    ) -> None:
        if not succeeded:
            return

        is_better = (
            score > self._best_score if self.mode == "max" else score < self._best_score
        )
        if is_better:
            self._best_score = score
            self._best_params = dict(parameters)

        self.early_stopping.update(score)

    def should_stop(self) -> bool:
        return self.early_stopping.should_stop()

    def best_parameters(self) -> dict[str, Any]:
        return dict(self._best_params)

    def best_score(self) -> float:
        return self._best_score
