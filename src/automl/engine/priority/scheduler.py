from __future__ import annotations

from typing import Iterable

from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import Priority
from automl.domain.policies.budget import BudgetPolicy


class ExperimentQueue:
    """
    Priority-ordered queue for ExperimentCandidates.
    Pinned candidates are placed ahead of all unpinned candidates,
    followed by candidates ordered by effective_score descending.
    """

    def __init__(self, candidates: Iterable[ExperimentCandidate] | None = None) -> None:
        self._items: dict[str, ExperimentCandidate] = {}
        if candidates:
            self.enqueue_all(candidates)

    def enqueue(self, candidate: ExperimentCandidate) -> None:
        self._items[candidate.id] = candidate

    def enqueue_all(self, candidates: Iterable[ExperimentCandidate]) -> None:
        for candidate in candidates:
            self.enqueue(candidate)

    def _sorted_items(self) -> list[ExperimentCandidate]:
        pinned: list[ExperimentCandidate] = []
        regular: list[ExperimentCandidate] = []

        for item in self._items.values():
            if item.priority is not None and item.priority.is_pinned:
                pinned.append(item)
            else:
                regular.append(item)

        # Regular candidates sorted by effective_score descending
        regular.sort(
            key=lambda c: (c.priority.effective_score if c.priority else 0.0),
            reverse=True,
        )
        return pinned + regular

    def peek(self) -> ExperimentCandidate | None:
        ordered = self._sorted_items()
        return ordered[0] if ordered else None

    def pop_next(self) -> ExperimentCandidate | None:
        ordered = self._sorted_items()
        if not ordered:
            return None
        selected = ordered[0]
        del self._items[selected.id]
        return selected

    def reprioritize(self, candidate_id: str, new_priority: Priority) -> bool:
        candidate = self._items.get(candidate_id)
        if candidate is None:
            return False
        candidate.priority = new_priority
        return True

    def cancel(self, candidate_id: str) -> bool:
        if candidate_id in self._items:
            del self._items[candidate_id]
            return True
        return False

    def get(self, candidate_id: str) -> ExperimentCandidate | None:
        return self._items.get(candidate_id)

    def list_queued(self) -> list[ExperimentCandidate]:
        return self._sorted_items()

    def __len__(self) -> int:
        return len(self._items)


class Scheduler:
    """
    Selects executable experiment candidates from the queue considering budget policy.
    """

    def select_next(
        self,
        queue: ExperimentQueue,
        budget: BudgetPolicy,
        executed_experiments: int = 0,
        executed_trials: int = 0,
    ) -> ExperimentCandidate | None:
        ordered = queue.list_queued()
        for candidate in ordered:
            if budget.is_candidate_eligible(
                candidate,
                executed_experiments=executed_experiments,
                executed_trials=executed_trials,
            ):
                # Remove candidate from queue and return it
                queue.cancel(candidate.id)
                return candidate
        return None

    def select_all_eligible(
        self,
        queue: ExperimentQueue,
        budget: BudgetPolicy,
        executed_experiments: int = 0,
        executed_trials: int = 0,
    ) -> list[ExperimentCandidate]:
        selected: list[ExperimentCandidate] = []
        current_exp = executed_experiments
        current_tri = executed_trials

        while len(queue) > 0:
            candidate = self.select_next(
                queue,
                budget,
                executed_experiments=current_exp,
                executed_trials=current_tri,
            )
            if candidate is None:
                break
            selected.append(candidate)
            current_exp += 1
            current_tri += len(candidate.model_ids)

        return selected
