# Interchangeable LLM providers — Persona A / A5

These optional infrastructure adapters implement the existing `LLMProviderPort`.
They generate text/proposals; CATML's application layer retains execution,
approvals and experimental verification. They require no new dependencies.
Importing or constructing a provider never sends a request.

## Choose a provider

Set `CATML_LLM_PROVIDER` and `CATML_LLM_MODEL`, then call the factory explicitly.
The default is `fake`, suitable for reproducible offline tests. Real models must
be specified explicitly: model availability and access vary by installation.

| Provider | Endpoint default | Credentials |
| --- | --- | --- |
| `openai` | `https://api.openai.com/v1/responses` | `OPENAI_API_KEY` |
| `anthropic` | `https://api.anthropic.com/v1/messages` | `ANTHROPIC_API_KEY` |
| `ollama` | `http://127.0.0.1:11434/api/chat` | None for local Ollama |
| `openai-compatible` | Explicit base URL + `/chat/completions` | Optional `CATML_LLM_API_KEY` |

For local testing, install/start Ollama and pull the model separately. Example:

```bash
ollama pull llama3.2
export CATML_LLM_PROVIDER=ollama
export CATML_LLM_MODEL=llama3.2
```

The example requires sufficient local resources. Change `CATML_LLM_MODEL` to any
model installed in your Ollama server. For Claude, use `anthropic`, the model ID
available in your account, and `ANTHROPIC_API_KEY`. For OpenAI, use `openai`, your
available model ID, and `OPENAI_API_KEY`. Set secrets through your environment;
do not commit them.

For a compatible local server (e.g. LM Studio/vLLM), select
`openai-compatible`, supply its model ID, and set `CATML_LLM_BASE_URL`, including
`/v1`, e.g. `http://127.0.0.1:1234/v1`. The server/model must support Chat
Completions and JSON object output. There is no silent downgrade if unsupported.
Other remote servers require HTTPS; HTTP is allowed only for loopback addresses.
There is no automatic fallback to another remote provider.

## Use the existing specialists

```python
from automl.infrastructure.llm import LLMProviderConfig, create_llm_provider
from automl.application.agents.contracts import ContextPayload
from automl.application.agents.specialists.planner import Planner

provider = create_llm_provider(LLMProviderConfig.from_env())
planner = Planner(llm_provider=provider)
context = ContextPayload(
    run_id="example-run", dataset_id="example-dataset",
    task_type="binary_classification", target_metric="roc_auc",
    dataset_profile={"n_rows": 1000, "n_columns": 8},
)
proposal = planner.analyze(context)
print(proposal.hypothesis)  # Proposal only; no training/execution here.
if hasattr(provider, "audit"):
    print(list(provider.audit))  # Metadata only; no prompts or response bodies.
```

For an existing workspace, obtain context with
`ContextBuilder().build(workspace, run_id)` instead. `Critic` and `FeatureAdvisor`
also accept `llm_provider=provider`. Inject these instances into the existing
`AgentSessionManager(..., planner=planner, critic=critic, feature_advisor=advisor)`
to compose a session. Commands still go through the existing executor/buses.

This A5 change does **not** activate remote LLMs automatically in the CLI or
Workbench. Their default specialists remain deterministic. Persona B owns the
CLI/composition, LangGraph/checkpointer and persistent session integration.

The existing specialists fall back to deterministic behavior when their LLM
call fails. Consult `provider.audit` to distinguish a real response from fallback.
For a direct diagnostic that propagates errors instead, call:

```python
response = provider.generate(
    "Summarize the purpose of tabular AutoML in one sentence.",
    system_prompt="Be precise and concise.",
)
print(response.content)
```

## Controls and limitations

- `CATML_LLM_TIMEOUT_SECONDS` defaults to 30 seconds per socket operation;
  `CATML_LLM_MAX_RETRIES` defaults to 1 (maximum 3). This is bounded socket I/O,
  not a hard wall-clock cancellation guarantee for a trickling server.
- Only HTTP 429/500/502/503/504 responses are retried, with bounded backoff.
  Transport timeouts are not retried because a request may have been processed.
  Invalid outputs, refusals, incomplete generation and authentication failures
  fail immediately. Redirects are refused so credentials are not forwarded.
- `CATML_LLM_MAX_OUTPUT_TOKENS` defaults to 1024. Input character and response
  byte limits are configurable via `LLMProviderConfig` in Python.
- Structured responses are validated locally against the explicit schema subset:
  type, properties, required, items, enum, boolean additionalProperties, description.
  Unsupported schema constraints fail before sending a request. Duplicate JSON
  fields, non-finite numbers, missing fields and incorrect types are rejected.
  OpenAI requests JSON mode (`store=False`), Ollama receives the schema, and
  Claude receives a JSON instruction. Claude is not guaranteed to obey the schema;
  every provider still receives the same local validation.
- Prompts redact the configured API key, explicit `sensitive_values`, and common
  plaintext secret assignments. This is **not** a general anonymizer. Use bounded
  `ContextBuilder` context, which excludes raw dataframe samples; review summaries
  before enabling a remote provider. Local-first ML does not imply remote LLM
  prompts stay on the machine. Provider data policies still apply.
- `audit` retains the latest 256 generation records in memory: provider/model,
  attempt count, last HTTP status, duration and final-response token counts.
  Prompts, responses, credentials and server error bodies are excluded. Unknown
  usage is `None`, including unsuccessful attempts; do not interpret it as free.
  The shared `LLMResponse` DTO retains its existing zero defaults for unknown usage.
  Audit is not a persisted ledger and final-response counts do not measure earlier
  unsuccessful attempts. Monetary costs are not estimated.
- Instances are scoped to sequential specialist calls within a session. There
  is no additional provider-level budget policy: application budget enforcement
  and durable LLM accounting must be integrated by B5 through agreed contracts.
  An LLM proposal never establishes measured model improvement.

## Validation and provider references

Tests use canned API envelopes and an actual local HTTP fixture; no credentials
or paid requests. End-to-end live-provider acceptance remains an explicit manual
step against the model chosen by the developer.

- [OpenAI Responses](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- [OpenAI JSON output](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Claude Messages](https://platform.claude.com/docs/en/api/messages/create)
- [Ollama chat](https://docs.ollama.com/api/chat)
