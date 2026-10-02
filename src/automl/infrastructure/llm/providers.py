"""Optional HTTP LLM adapters. No SDK dependency or network calls at import time."""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from automl.application.agents.contracts import LLMResponse
from automl.infrastructure.llm.fake_provider import FakeLLMProvider


class LLMProviderError(RuntimeError):
    """Sanitized provider failure; never includes remote bodies or credentials."""


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str = "fake"
    model: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 30.0
    max_retries: int = 1
    max_output_tokens: int = 1024
    max_input_chars: int = 32000
    max_response_bytes: int = 1048576
    sensitive_values: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if self.provider not in {"fake", "openai", "anthropic", "ollama", "openai-compatible"}:
            raise ValueError("Unknown LLM provider")
        if self.provider != "fake" and not self.model.strip():
            raise ValueError("An explicit model is required")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be in (0, 300]")
        for value, lower, upper in ((self.max_retries, 0, 3), (self.max_output_tokens, 1, 32768),
                                    (self.max_input_chars, 1, 1000000),
                                    (self.max_response_bytes, 1, 10485760)):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("Invalid provider limit")
        if self.provider in {"openai", "anthropic"} and not self.api_key:
            raise ValueError("Provider API key is required")
        if self.provider == "openai-compatible" and not self.base_url:
            raise ValueError("Compatible provider requires base_url including /v1")
        if self.base_url:
            parts = urlsplit(self.base_url)
            if (not parts.hostname or parts.username or parts.password or parts.query or parts.fragment
                    or parts.scheme not in {"http", "https"}
                    or (parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"})):
                raise ValueError("Use HTTPS or a loopback HTTP URL without embedded credentials")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMProviderConfig:
        env = os.environ if env is None else env
        provider = env.get("CATML_LLM_PROVIDER", "fake")
        key_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "CATML_LLM_API_KEY")
        return cls(provider=provider, model=env.get("CATML_LLM_MODEL", ""),
                   base_url=env.get("CATML_LLM_BASE_URL", ""), api_key=env.get(key_name, ""),
                   timeout_seconds=float(env.get("CATML_LLM_TIMEOUT_SECONDS", "30")),
                   max_retries=int(env.get("CATML_LLM_MAX_RETRIES", "1")),
                   max_output_tokens=int(env.get("CATML_LLM_MAX_OUTPUT_TOKENS", "1024")))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        return None


def _http_post(url: str, headers: dict[str, str], payload: dict[str, Any],
               timeout: float, limit: int) -> tuple[int, dict[str, Any]]:
    request = Request(url, data=json.dumps(payload, allow_nan=False).encode(),
                      headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout) as response:
            raw = response.read(limit + 1)
            status = response.status
    except HTTPError as exc:
        # Do not read or expose server error bodies, including reflected secrets.
        status = exc.code
        exc.close()
        return status, {}
    except (URLError, TimeoutError, OSError):
        raise LLMProviderError("LLM transport unavailable or timed out") from None
    if len(raw) > limit:
        raise LLMProviderError("LLM response exceeds size limit")
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise LLMProviderError("LLM response is not JSON") from None
    if not isinstance(data, dict):
        raise LLMProviderError("LLM response envelope must be an object")
    return status, data


def _check_schema(schema: dict[str, Any], depth: int = 0) -> None:
    """Explicit subset used by CATML specialists; unsupported constraints fail closed."""
    allowed = {"type", "properties", "required", "items", "enum", "additionalProperties", "description"}
    if depth > 20 or not isinstance(schema, dict) or set(schema) - allowed:
        raise ValueError("Unsupported response schema")
    if schema.get("type") not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ValueError("Schema requires a supported type")
    props = schema.get("properties", {})
    if not isinstance(props, dict) or any(not isinstance(key, str) for key in props):
        raise ValueError("Invalid schema properties")
    if schema["type"] != "object" and any(key in schema for key in ("properties", "required", "additionalProperties")):
        raise ValueError("Object constraints require object type")
    if "items" in schema and schema["type"] != "array":
        raise ValueError("Array constraints require array type")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        raise ValueError("Invalid schema required fields")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError("Invalid schema enum")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        raise ValueError("Only boolean additionalProperties is supported")
    for child in props.values():
        _check_schema(child, depth + 1)
    if "items" in schema:
        _check_schema(schema["items"], depth + 1)


def _validate(value: Any, schema: dict[str, Any], depth: int = 0) -> None:
    kinds = {"object": dict, "array": list, "string": str, "integer": int,
             "number": (int, float), "boolean": bool, "null": type(None)}
    kind = schema["type"]
    if depth > 40 or not isinstance(value, kinds[kind]) or (kind in {"integer", "number"} and isinstance(value, bool)):
        raise LLMProviderError("LLM output does not match response schema")
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        raise LLMProviderError("LLM output contains an invalid enum value")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(key not in value for key in schema.get("required", [])):
            raise LLMProviderError("LLM output is missing required fields")
        if schema.get("additionalProperties") is False and set(value) - set(props):
            raise LLMProviderError("LLM output contains unexpected fields")
        for key, child in value.items():
            if key in props:
                _validate(child, props[key], depth + 1)
    if isinstance(value, list) and "items" in schema:
        for child in value:
            _validate(child, schema["items"], depth + 1)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON value")
    return number


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON value")


class HTTPProvider:
    """Adapter implementing LLMProviderPort; execution/governance stays in CATML.

    Audit contains metadata only. Tokens are None when usage is unknown, including
    failures; LLMResponse retains the existing contract's integer defaults.
    Instances are session-scoped and intended for sequential specialist calls.
    """
    def __init__(self, config: LLMProviderConfig, *, transport: Callable[..., Any] | None = None) -> None:
        if config.provider == "fake":
            raise ValueError("Use create_llm_provider for fake mode")
        self.config = config
        self._transport = transport or _http_post
        self.audit: deque[dict[str, Any]] = deque(maxlen=256)

    def _redact(self, text: str) -> str:
        for secret in (self.config.api_key, *self.config.sensitive_values):
            if secret:
                text = text.replace(secret, "[REDACTED]")
        return re.sub(r"(?i)(api[_-]?key|password|authorization|token)(\s*[:=]\s*)([^\s,;]+)",
                      r"\1\2[REDACTED]", text)

    def generate(self, prompt: str, system_prompt: str | None = None,
                 response_schema: dict[str, Any] | None = None, **kwargs: Any) -> LLMResponse:
        if kwargs:
            raise ValueError("Configure generation limits through LLMProviderConfig")
        if not isinstance(prompt, str) or (system_prompt is not None and not isinstance(system_prompt, str)):
            raise ValueError("Prompts must be strings")
        if response_schema is not None:
            _check_schema(response_schema)
            if response_schema["type"] != "object":
                raise ValueError("Response schema root must be an object")
            system_prompt = (system_prompt or "") + "\nReturn only a JSON object matching this schema: " + json.dumps(response_schema, allow_nan=False)
        if len(prompt) + len(system_prompt or "") > self.config.max_input_chars:
            raise ValueError("LLM prompt exceeds configured input limit")
        prompt, system = self._redact(prompt), self._redact(system_prompt or "")
        url, headers, payload = self._request(prompt, system, response_schema)
        start = time.monotonic()
        data: dict[str, Any] = {}
        record: dict[str, Any] = {"provider": self.config.provider, "model": self.config.model,
                                  "attempts": 0, "status": "failed", "prompt_tokens": None,
                                  "completion_tokens": None, "total_tokens": None}
        try:
            for attempt in range(self.config.max_retries + 1):
                record["attempts"] += 1
                # Retry only an explicit transient HTTP response. Network failures
                # may have been billed already; never resend them automatically.
                status, data = self._transport(url, headers, payload, self.config.timeout_seconds,
                                               self.config.max_response_bytes)
                record["http_status"] = status
                if status == 200:
                    break
                if status not in {429, 500, 502, 503, 504} or attempt == self.config.max_retries:
                    raise LLMProviderError(f"LLM HTTP failure ({status})")
                time.sleep(min(0.25 * 2 ** attempt, 1.0))
            content, input_tokens, output_tokens = self._decode(data)
            record.update(prompt_tokens=input_tokens, completion_tokens=output_tokens,
                          total_tokens=None if input_tokens is None or output_tokens is None else input_tokens + output_tokens)
            parsed = None
            if response_schema is not None:
                try:
                    parsed = json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float)
                except (ValueError, TypeError, RecursionError):
                    raise LLMProviderError("LLM output is not a valid JSON object") from None
                _validate(parsed, response_schema)
            record["status"] = "ok"
            return LLMResponse(content=content, parsed=parsed, provider=self.config.provider,
                               model=self.config.model, prompt_tokens=input_tokens or 0,
                               completion_tokens=output_tokens or 0, total_tokens=record["total_tokens"] or 0,
                               latency_seconds=time.monotonic() - start)
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            raise LLMProviderError("Malformed LLM response envelope") from None
        finally:
            record["latency_seconds"] = time.monotonic() - start
            self.audit.append(record)

    def _request(self, prompt: str, system: str, schema: dict[str, Any] | None) -> tuple[str, dict[str, str], dict[str, Any]]:
        cfg = self.config
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        payload: dict[str, Any] = {"model": cfg.model}
        if cfg.provider == "openai":
            url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/") + "/responses"
            payload.update(input=messages, max_output_tokens=cfg.max_output_tokens, store=False)
            if schema is not None:
                payload["text"] = {"format": {"type": "json_object"}}
        elif cfg.provider == "anthropic":
            url = (cfg.base_url or "https://api.anthropic.com/v1").rstrip("/") + "/messages"
            headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"}
            payload.update(messages=[{"role": "user", "content": prompt}], max_tokens=cfg.max_output_tokens)
            if system:
                payload["system"] = system
        elif cfg.provider == "ollama":
            url = (cfg.base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/chat"
            payload.update(messages=messages, stream=False, options={"num_predict": cfg.max_output_tokens})
            if schema is not None:
                payload["format"] = schema
        else:
            url = cfg.base_url.rstrip("/") + "/chat/completions"
            payload.update(messages=messages, max_tokens=cfg.max_output_tokens, stream=False)
            if schema is not None:
                payload["response_format"] = {"type": "json_object"}
        return url, headers, payload

    def _decode(self, data: dict[str, Any]) -> tuple[str, int | None, int | None]:
        provider = self.config.provider
        usage = data.get("usage", {})
        if provider == "openai":
            if data.get("status") != "completed":
                raise LLMProviderError("LLM generation was incomplete")
            blocks = [part for item in data["output"] if item.get("type") == "message" for part in item.get("content", [])]
            if any(part.get("type") == "refusal" for part in blocks):
                raise LLMProviderError("LLM refused the request")
            content = "".join(part["text"] for part in blocks if part.get("type") == "output_text")
            first, second = usage.get("input_tokens"), usage.get("output_tokens")
        elif provider == "anthropic":
            if data.get("stop_reason") not in {"end_turn", "stop_sequence"}:
                raise LLMProviderError("LLM generation was incomplete")
            content = "".join(part["text"] for part in data["content"] if part.get("type") == "text")
            first = usage.get("input_tokens")
            if first is not None:
                first += usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
            second = usage.get("output_tokens")
        elif provider == "ollama":
            if not data.get("done") or data.get("done_reason") == "length":
                raise LLMProviderError("LLM generation was incomplete")
            content = data["message"]["content"]
            first, second = data.get("prompt_eval_count"), data.get("eval_count")
        else:
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise LLMProviderError("LLM generation was incomplete")
            content = choice["message"]["content"]
            first, second = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM returned no text")
        for tokens in (first, second):
            if tokens is not None and (type(tokens) is not int or tokens < 0):
                raise LLMProviderError("LLM reported invalid usage")
        return content, first, second


def create_llm_provider(config: LLMProviderConfig | None = None, *, transport: Callable[..., Any] | None = None) -> FakeLLMProvider | HTTPProvider:
    """Explicit composition boundary; no automatic selection or remote fallback."""
    config = config or LLMProviderConfig.from_env()
    if config.provider == "fake":
        return FakeLLMProvider()
    return HTTPProvider(config, transport=transport)
