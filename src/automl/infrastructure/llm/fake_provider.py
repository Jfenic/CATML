"""Deterministic mock LLM provider for reproducible tests and bounded execution."""
from __future__ import annotations

import json
import re
from typing import Any

from automl.application.agents.contracts import LLMResponse
from automl.application.agents.ports import LLMProviderPort


class FakeLLMProvider:
    """Mock LLM provider implementing LLMProviderPort with deterministic responses."""

    def __init__(
        self,
        default_responses: dict[str, Any] | None = None,
        simulated_latency: float = 0.0,
    ) -> None:
        self._canned_responses: list[tuple[re.Pattern[str], dict[str, Any] | str]] = []
        self._simulated_latency = simulated_latency
        self._simulate_error: Exception | None = None
        self.calls: list[dict[str, Any]] = []
        self.total_tokens_consumed: int = 0

        if default_responses:
            for pattern, resp in default_responses.items():
                self.register_response(pattern, resp)

    def register_response(self, pattern: str, response: dict[str, Any] | str) -> None:
        """Register a canned response triggered when prompt matches regex pattern."""
        self._canned_responses.append((re.compile(pattern, re.IGNORECASE), response))

    def set_simulate_error(self, error: Exception | None) -> None:
        """Configure provider to raise an exception upon generation."""
        self._simulate_error = error

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate deterministic response, tracking token usage and call history."""
        if self._simulate_error is not None:
            raise self._simulate_error

        # 1. Match against canned responses
        matched_content: str | None = None
        matched_parsed: dict[str, Any] | None = None

        for pattern, canned in self._canned_responses:
            if pattern.search(prompt) or (system_prompt and pattern.search(system_prompt)):
                if isinstance(canned, dict):
                    matched_parsed = canned
                    matched_content = json.dumps(canned)
                else:
                    matched_content = str(canned)
                    try:
                        matched_parsed = json.loads(matched_content)
                    except Exception:
                        matched_parsed = None
                break

        # 2. Fallback default response if no pattern matched
        if matched_content is None:
            if response_schema:
                # Synthesize minimal valid dict conforming to schema
                fallback_dict: dict[str, Any] = {}
                properties = response_schema.get("properties", {})
                for prop, spec in properties.items():
                    prop_type = spec.get("type", "string")
                    if prop_type == "string":
                        fallback_dict[prop] = f"fake_{prop}"
                    elif prop_type in ("integer", "number"):
                        fallback_dict[prop] = 1
                    elif prop_type == "boolean":
                        fallback_dict[prop] = True
                    elif prop_type == "object":
                        fallback_dict[prop] = {}
                    elif prop_type == "array":
                        fallback_dict[prop] = []
                matched_parsed = fallback_dict
                matched_content = json.dumps(fallback_dict)
            else:
                matched_content = "Fake LLM completion output."
                matched_parsed = None

        prompt_tokens = max(1, len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0))
        completion_tokens = max(1, len(matched_content.split()))
        total_tokens = prompt_tokens + completion_tokens

        self.total_tokens_consumed += total_tokens
        self.calls.append({
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response_schema": response_schema,
            "tokens": total_tokens,
        })

        return LLMResponse(
            content=matched_content,
            provider="fake_provider",
            model="fake-deterministic-v1",
            parsed=matched_parsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_seconds=self._simulated_latency,
        )
