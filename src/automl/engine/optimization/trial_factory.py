from __future__ import annotations

import uuid
from typing import Any

from automl.domain.experiments.trial import Trial, TrialStatus


class TrialFactory:
    """
    Materializes a Trial entity populated with model_id and suggested parameters.
    """

    @classmethod
    def create(
        cls,
        experiment_id: str,
        model_id: str,
        parameters: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> Trial:
        trial_id = f"trial_{uuid.uuid4().hex[:8]}"
        return Trial(
            id=trial_id,
            experiment_id=experiment_id,
            model_id=model_id,
            parameters=dict(parameters or {}),
            seed=seed,
            status=TrialStatus.CREATED,
        )
