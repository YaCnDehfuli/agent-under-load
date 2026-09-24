"""The model is an interface, and an unconfigured run raises.

Three implementations:

`OpenAICompatModel`
    Chat Completions with native tool calling. One client for every provider
    in the study: OpenAI, Groq, DeepSeek, and local weights served by Ollama or
    vLLM on their /v1 endpoints. Reproducing the results without a paid API is
    a non-negotiable for this repo, and the local route goes through the same
    code as the hosted ones.

`AnthropicModel`
    The Anthropic Messages API. Structured output via a forced tool call, so
    the verdict arrives as a validated object rather than JSON scraped out of
    prose.

`ScriptedModel`
    Deterministic, offline, used by the test suite. Never scored, and never
    attacked. A stub cannot be prompt-injected in any meaningful sense, so a
    number produced against one would be fiction dressed as a measurement.

Models are named by key in `models.yml`, which holds each one's provider,
endpoint, generation parameters and price. There is no fallback chain: an
unknown key or a missing credential raises. The failure this guards against is
a results table quietly populated by something that was never a language model.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import yaml
from pydantic import BaseModel, ValidationError

#: Pinned low for repeatability. Sampling is still not a guarantee of identical
#: output across runs, which is why determinism is measured rather than assumed
#: (tests/test_determinism.py).
DEFAULT_TEMPERATURE = 0.0
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"

#: Name the model must call to submit its answer. Forcing the answer through a
#: tool is what makes the contract enforceable at the boundary.
SUBMIT_TOOL = "submit_verdict"


class ModelError(RuntimeError):
    pass


class ModelNotConfigured(ModelError):
    pass


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclasses.dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_call_id: str = ""
    #: On an assistant turn, the tool calls it made, replayed next to their results.
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class Usage:
    """Token counts as the provider reported them. None where it did not say."""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    #: What the provider says the call cost, where it says so (OpenRouter).
    reported_cost_usd: float | None = None


#: What a reply that neither called a tool nor answered becomes. Without it the
#: graph would resend the identical prompt until the turn budget ran out.
NO_TOOL_OR_ANSWER = "reply neither called a tool nor submitted a verdict"


@dataclasses.dataclass
class ModelReply:
    """Either the model wants tools, or it has answered."""

    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    answer: BaseModel | None = None
    raw_text: str = ""
    #: Set when the model produced something the contract rejected. Kept rather
    #: than retried silently, because a malformed answer under attack is itself
    #: a result.
    invalid: str = ""
    usage: Usage | None = None
    latency_s: float | None = None
    #: The model id the provider says served the request, which can differ from
    #: the one asked for when an alias resolves to a snapshot.
    served_model: str = ""
    #: The host behind a router such as OpenRouter; empty when there is none.
    served_by: str = ""
    finish_reason: str = ""
    attempts: int = 1

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Model(Protocol):
    name: str
    temperature: float | None

    @property
    def config(self) -> dict[str, Any]:
        """What produced the numbers: provider, endpoint host, model, parameters."""

    def respond(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        schema: type[BaseModel],
    ) -> ModelReply: ...


def _schema_of(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for the verdict, inlined so no $ref survives.

    Providers differ on how much of JSON Schema they accept in a tool
    definition, and a $defs indirection is the usual casualty.
    """
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = node["$ref"].rsplit("/", 1)[-1]
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return inline({**defs.get(target, {}), **merged})
            return {k: inline(v) for k, v in node.items()}
        if isinstance(node, list):
            return [inline(item) for item in node]
        return node

    return inline(raw)


def _validate(schema: type[BaseModel], arguments: dict) -> tuple[BaseModel | None, str]:
    try:
        return schema.model_validate(arguments), ""
    except ValidationError as exc:
        return None, f"contract rejected the verdict: {exc.errors(include_url=False)}"


# ---------------------------------------------------------------------------
# scripted, for tests
# ---------------------------------------------------------------------------


class ScriptedModel:
    """Replays a fixed script. Deterministic by construction.

    Used to pin the graph's control flow and the contract's enforcement without
    a network call. Not scored: see the module docstring.
    """

    def __init__(
        self,
        script: Sequence[ModelReply] | Callable[[Sequence[Message]], ModelReply],
        name: str = "scripted",
    ):
        self.name = name
        self.temperature = 0.0
        self._script = script
        self._position = 0
        #: Every prompt this model was shown, so a test can assert on what
        #: reached it — which is how the injection tests verify placement.
        self.seen: list[tuple[str, list[Message]]] = []

    @property
    def config(self) -> dict[str, Any]:
        return {"provider": "scripted", "host": "", "model": self.name,
                "temperature": self.temperature, "params": {}}

    def respond(self, *, system, messages, tools, schema) -> ModelReply:
        self.seen.append((system, list(messages)))
        if callable(self._script):
            return self._script(messages)
        if self._position >= len(self._script):
            raise ModelError(
                f"scripted model exhausted after {self._position} replies; "
                "the graph asked for more turns than the script provides"
            )
        reply = self._script[self._position]
        self._position += 1
        return reply


# ---------------------------------------------------------------------------
# hosted
# ---------------------------------------------------------------------------


class AnthropicModel:
    def __init__(
        self,
        name: str = DEFAULT_ANTHROPIC_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise ModelNotConfigured("anthropic is not installed") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ModelNotConfigured("ANTHROPIC_API_KEY is not set")
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=key)

    @property
    def config(self) -> dict[str, Any]:
        return {"provider": "anthropic", "host": "api.anthropic.com",
                "model": self.name, "temperature": self.temperature,
                "params": {"max_tokens": self.max_tokens}}

    def respond(self, *, system, messages, tools, schema) -> ModelReply:
        payload_tools = [
            {"name": t.name, "description": t.description,
             "input_schema": t.parameters}
            for t in tools
        ]
        payload_tools.append({
            "name": SUBMIT_TOOL,
            "description": "Submit the final answer. Call this exactly once, "
                           "when the evidence supports a conclusion.",
            "input_schema": _schema_of(schema),
        })

        started = time.monotonic()
        response = self._client.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            tools=payload_tools,
            messages=_to_anthropic(messages),
        )
        meta = {
            "latency_s": round(time.monotonic() - started, 3),
            "served_model": getattr(response, "model", "") or "",
            "finish_reason": getattr(response, "stop_reason", "") or "",
            "usage": Usage(
                input_tokens=response.usage.input_tokens,
                cached_input_tokens=getattr(response.usage,
                                            "cache_read_input_tokens", None),
                output_tokens=response.usage.output_tokens,
            ),
        }

        calls: list[ToolCall] = []
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(name=block.name,
                                      arguments=dict(block.input or {}),
                                      call_id=block.id))

        text = "\n".join(text_parts)
        submissions = [c for c in calls if c.name == SUBMIT_TOOL]
        if submissions:
            answer, error = _validate(schema, submissions[-1].arguments)
            return ModelReply(answer=answer, raw_text=text, invalid=error, **meta)
        if not calls:
            return ModelReply(raw_text=text, invalid=NO_TOOL_OR_ANSWER, **meta)
        return ModelReply(tool_calls=calls, raw_text=text, **meta)


def _to_anthropic(messages: Sequence[Message]) -> list[dict]:
    """Flatten the transcript into Anthropic's message shape.

    Tool results are folded into a user turn that names the call, which keeps
    the transcript readable in the audit log without depending on a particular
    provider's content-block layout. An assistant turn with no text is dropped:
    the folded results already name the calls it made.
    """
    out: list[dict] = []
    for message in messages:
        if message.role == "assistant" and not message.content:
            continue
        if message.role == "tool":
            out.append({
                "role": "user",
                "content": f"[result of {message.tool_call_id}]\n{message.content}",
            })
        else:
            out.append({"role": message.role, "content": message.content})
    return out


# ---------------------------------------------------------------------------
# OpenAI-compatible: OpenAI, Groq, DeepSeek, and Ollama / vLLM through /v1
# ---------------------------------------------------------------------------

#: Worth retrying: rate limits and transient server failures. Anything else in
#: the 4xx range is a malformed request, and retrying it would only spend money.
RETRYABLE = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 8
MAX_BACKOFF_S = 60.0
#: Extra tries when a routed host fails mid-generation behind a 200 response.
HOST_ERROR_RETRIES = 2


class OpenAICompatModel:
    """A Chat Completions endpoint with native tool calling.

    One client for every provider in the study. Provider differences that are
    only request fields (`max_completion_tokens` versus `max_tokens`,
    `reasoning_effort`, `service_tier`) live in the registry's `params` and are
    sent verbatim, so they are recorded exactly as they were sent.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        temperature: float | None = None,
        params: dict[str, Any] | None = None,
        provider: str = "openai-compat",
        timeout: float = 300.0,
        transport: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        import httpx

        self.name = name
        #: None means the field is not sent: some reasoning models reject it.
        self.temperature = temperature
        self.params = dict(params or {})
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout, headers=headers,
                                    transport=transport)

    @property
    def config(self) -> dict[str, Any]:
        from urllib.parse import urlparse

        return {"provider": self.provider, "host": urlparse(self.base_url).netloc,
                "model": self.name, "temperature": self.temperature,
                "params": self.params}

    def respond(self, *, system, messages, tools, schema) -> ModelReply:
        body: dict[str, Any] = {
            "model": self.name,
            "messages": [{"role": "system", "content": system},
                         *_to_openai(messages)],
            "tools": [_function(t.name, t.description, t.parameters) for t in tools]
                     + [_function(SUBMIT_TOOL,
                                  "Submit the final answer. Call this exactly once, "
                                  "when the evidence supports a conclusion.",
                                  _schema_of(schema))],
            **self.params,
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature

        data, latency, attempts = self._post(body)
        # a router can answer 200 while the host behind it failed mid-generation;
        # that is the host's fault, so it is retried and then raised, never
        # scored as a model that gave no answer
        for _ in range(HOST_ERROR_RETRIES):
            if not _host_failed(data):
                break
            self._sleep(_backoff(None, attempts))
            data, more_latency, more_attempts = self._post(body)
            latency, attempts = latency + more_latency, attempts + more_attempts
        if _host_failed(data):
            raise ModelError(f"{data.get('provider') or self.provider} failed "
                             f"mid-generation after {attempts} attempt(s): "
                             f"{_host_error(data)}")
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        meta = {
            "usage": _openai_usage(data.get("usage") or {}),
            "latency_s": latency,
            "served_model": data.get("model", ""),
            "served_by": data.get("provider") or "",
            "finish_reason": choice.get("finish_reason") or "",
            "attempts": attempts,
        }
        text = message.get("content") or ""

        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                return ModelReply(raw_text=text, **meta, invalid=(
                    f"arguments to {function.get('name')!r} were not valid JSON"))
            calls.append(ToolCall(name=function.get("name", ""),
                                  arguments=arguments if isinstance(arguments, dict)
                                  else {},
                                  call_id=raw.get("id", "")))

        submissions = [c for c in calls if c.name == SUBMIT_TOOL]
        if submissions:
            answer, error = _validate(schema, submissions[-1].arguments)
            return ModelReply(answer=answer, raw_text=text, invalid=error, **meta)
        if calls:
            # ids are synthesised only when the provider sent none, and from the
            # turn position, so a replayed transcript stays byte-identical
            turn = sum(1 for m in messages if m.role == "assistant")
            calls = [c if c.call_id else dataclasses.replace(c, call_id=f"call_{turn}_{i}")
                     for i, c in enumerate(calls)]
            return ModelReply(tool_calls=calls, raw_text=text, **meta)
        if meta["finish_reason"] == "length":
            return ModelReply(raw_text=text, invalid="output hit the token limit",
                              **meta)
        return ModelReply(raw_text=text, invalid=NO_TOOL_OR_ANSWER, **meta)

    def _post(self, body: dict) -> tuple[dict, float, int]:
        started = time.monotonic()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = self._client.post(f"{self.base_url}/chat/completions",
                                         json=body)
            if response.status_code == 200:
                return response.json(), round(time.monotonic() - started, 3), attempt
            if response.status_code not in RETRYABLE or attempt == MAX_ATTEMPTS:
                raise ModelError(
                    f"{self.provider} returned {response.status_code} after "
                    f"{attempt} attempt(s): {response.text[:300]}")
            self._sleep(_backoff(response, attempt))
        raise AssertionError("unreachable")  # pragma: no cover


def _backoff(response: Any, attempt: int) -> float:
    try:
        return min(float(response.headers["retry-after"]), MAX_BACKOFF_S)
    except (AttributeError, KeyError, ValueError):
        return min(2.0 ** (attempt - 1), MAX_BACKOFF_S)


def _host_failed(data: dict) -> bool:
    choice = (data.get("choices") or [{}])[0]
    return choice.get("finish_reason") == "error" or bool(choice.get("error"))


def _host_error(data: dict) -> str:
    error = (data.get("choices") or [{}])[0].get("error") or {}
    return str(error.get("message") if isinstance(error, dict) else error)[:300]


def _function(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": parameters}}


def _to_openai(messages: Sequence[Message]) -> list[dict]:
    out: list[dict] = []
    for message in messages:
        if message.role == "tool":
            out.append({"role": "tool", "tool_call_id": message.tool_call_id,
                        "content": message.content})
        elif message.role == "assistant" and message.tool_calls:
            out.append({"role": "assistant", "content": message.content or None,
                        "tool_calls": [
                            {"id": c.call_id, "type": "function",
                             "function": {"name": c.name,
                                          "arguments": json.dumps(c.arguments,
                                                                  sort_keys=True)}}
                            for c in message.tool_calls]})
        else:
            out.append({"role": message.role, "content": message.content})
    return out


def _openai_usage(usage: dict) -> Usage:
    prompt = usage.get("prompt_tokens_details") or {}
    completion = usage.get("completion_tokens_details") or {}
    cached = prompt.get("cached_tokens")
    if cached is None:
        cached = usage.get("prompt_cache_hit_tokens")  # DeepSeek's field
    return Usage(input_tokens=usage.get("prompt_tokens"),
                 cached_input_tokens=cached,
                 output_tokens=usage.get("completion_tokens"),
                 reasoning_tokens=completion.get("reasoning_tokens"),
                 reported_cost_usd=usage.get("cost"))  # OpenRouter's field


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

REGISTRY = Path(__file__).resolve().parent.parent / "models.yml"
ENV_FILE = REGISTRY.parent / ".env"


def load_env(path: Path | None = None) -> None:
    """Put the keys from a .env file into the environment.

    Accepts `KEY=value` and `KEY = value`, quoted or not, with # comments.
    A variable that is already set wins, so a key exported in the shell is
    never replaced by the file.
    """
    path = Path(path or ENV_FILE)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        key = key.removeprefix("export ").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def registry(path: Path | None = None) -> dict[str, dict]:
    return yaml.safe_load((path or REGISTRY).read_text()) or {}


def load_model(key: str, path: Path | None = None) -> Model:
    """Build the model a registry key names, or raise.

    There is deliberately no stub fallback: a scored run must involve a real
    model, and an unknown key or a missing credential stops the run.
    """
    entries = registry(path)
    if key not in entries:
        raise ModelNotConfigured(
            f"{key!r} is not in the model registry. Known: {', '.join(sorted(entries))}")
    entry = entries[key]
    api_key = ""
    if entry.get("api_key_env"):
        api_key = os.environ.get(entry["api_key_env"], "")
        if not api_key:
            raise ModelNotConfigured(f"{entry['api_key_env']} is not set")

    if entry["provider"] == "anthropic":
        return AnthropicModel(name=entry["model"],
                              temperature=entry.get("temperature", DEFAULT_TEMPERATURE),
                              api_key=api_key)
    return OpenAICompatModel(name=entry["model"], base_url=entry["base_url"],
                             api_key=api_key, temperature=entry.get("temperature"),
                             params=entry.get("params"), provider=entry["provider"],
                             timeout=entry.get("timeout_s", 300.0))
