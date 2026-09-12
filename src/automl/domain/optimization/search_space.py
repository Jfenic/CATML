from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterType(str, Enum):
    INT = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"


@dataclass
class ParameterSpec:
    name: str
    type: ParameterType
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list[Any] | None = None
    default: Any | None = None

    @property
    def param_type(self) -> ParameterType:
        return self.type

    @classmethod
    def float(
        cls,
        name: str,
        low: float,
        high: float,
        step: float | None = None,
        log: bool = False,
        default: float | None = None,
    ) -> ParameterSpec:
        return cls(
            name=name,
            type=ParameterType.FLOAT,
            low=low,
            high=high,
            step=step,
            log=log,
            default=default,
        )

    @classmethod
    def int(
        cls,
        name: str,
        low: int,
        high: int,
        step: int | None = None,
        default: int | None = None,
    ) -> ParameterSpec:
        return cls(
            name=name,
            type=ParameterType.INT,
            low=low,
            high=high,
            step=step,
            default=default,
        )

    @classmethod
    def categorical(
        cls,
        name: str,
        choices: list[Any],
        default: Any | None = None,
    ) -> ParameterSpec:
        return cls(
            name=name,
            type=ParameterType.CATEGORICAL,
            choices=choices,
            default=default,
        )

    def validate(self, value: Any) -> bool:
        if self.type == ParameterType.INT:
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            if self.low is not None and value < self.low:
                return False
            if self.high is not None and value > self.high:
                return False
            return True
        elif self.type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False
            if self.low is not None and value < self.low:
                return False
            if self.high is not None and value > self.high:
                return False
            return True
        elif self.type == ParameterType.CATEGORICAL:
            if self.choices is not None:
                return value in self.choices
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "low": self.low,
            "high": self.high,
            "step": self.step,
            "log": self.log,
            "choices": self.choices,
            "default": self.default,
        }


@dataclass
class SearchSpace:
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)

    def add(self, spec: ParameterSpec) -> SearchSpace:
        self.parameters[spec.name] = spec
        return self

    def get(self, name: str) -> ParameterSpec | None:
        return self.parameters.get(name)

    def list(self) -> list[ParameterSpec]:
        return list(self.parameters.values())

    def validate_params(self, params: dict[str, Any]) -> dict[str, bool]:
        return {
            name: self.parameters[name].validate(val)
            for name, val in params.items()
            if name in self.parameters
        }

    def sample(self, seed: int | None = None) -> dict[str, Any]:
        rng = random.Random(seed)
        sampled: dict[str, Any] = {}
        for name, spec in self.parameters.items():
            if spec.type == ParameterType.INT:
                low = int(spec.low if spec.low is not None else 1)
                high = int(spec.high if spec.high is not None else 100)
                step = int(spec.step) if spec.step else 1
                sampled[name] = rng.randrange(low, high + 1, step)
            elif spec.type == ParameterType.FLOAT:
                low = float(spec.low if spec.low is not None else 1e-3)
                high = float(spec.high if spec.high is not None else 1.0)
                if spec.log:
                    val = math.exp(rng.uniform(math.log(low), math.log(high)))
                else:
                    val = rng.uniform(low, high)
                sampled[name] = val
            elif spec.type == ParameterType.CATEGORICAL:
                choices = spec.choices or ([spec.default] if spec.default is not None else [])
                if choices:
                    sampled[name] = rng.choice(choices)
        return sampled

    def __contains__(self, name: str) -> bool:
        return name in self.parameters

    def __len__(self) -> int:
        return len(self.parameters)

    def __getitem__(self, name: str) -> ParameterSpec:
        return self.parameters[name]

    def __iter__(self):
        return iter(self.parameters.values())

    def to_dict(self) -> dict[str, Any]:
        return {name: spec.to_dict() for name, spec in self.parameters.items()}
