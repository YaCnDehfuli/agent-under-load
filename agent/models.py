"""The model is an interface, and an unconfigured run raises.

Three implementations:

`AnthropicModel`
    The default. Structured output via a forced tool call, so the verdict
    arrives as a validated object rather than JSON scraped out of prose.

`OllamaModel`
    An OpenAI/Ollama-compatible HTTP endpoint, for open weights. Reproducing
    the results without a paid API is a non-negotiable for this repo, so this
    path is a dependency rather than an aspiration.

`ScriptedModel`
    Deterministic, offline, used by the test suite. Never scored, and never
    attacked. A stub cannot be prompt-injected in any meaningful sense, so a
    number produced against one would be fiction dressed as a measurement.

There is no fallback chain. If nothing is configured, `from_env` raises. The
failure this guards against is a results table quietly populated by something
that was never a language model.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Callable, Protocol, Sequence

from pydantic import BaseModel, ValidationError

#: Pinned low for repeatability. Sampling is still not a guarantee of identical
#: output across runs, which is why determinism is measured rather than assumed
#: (tests/test_determinism.py).
DEFAULT_TEMPERATURE = 0.0
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b-instruct"

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

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class Model(Protocol):
    name: str
    temperature: float

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

        response = self._client.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            tools=payload_tools,
            messages=_to_anthropic(messages),
        )

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
            return ModelReply(answer=answer, raw_text=text, invalid=error)
        return ModelReply(tool_calls=calls, raw_text=text)


def _to_anthropic(messages: Sequence[Message]) -> list[dict]:
    """Flatten the transcript into Anthropic's message shape.

    Tool results are folded into a user turn that names the call, which keeps
    the transcript readable in the audit log without depending on a particular
    provider's content-block layout.
    """
    out: list[dict] = []
    for message in messages:
        if message.role == "tool":
            out.append({
                "role": "user",
                "content": f"[result of {message.tool_call_id}]\n{message.content}",
            })
        else:
            out.append({"role": message.role, "content": message.content})
    return out


# ---------------------------------------------------------------------------
# local / open weights
# ---------------------------------------------------------------------------


class OllamaModel:
    """An Ollama-compatible chat endpoint.

    Tool calling support varies by served model, so the answer is requested as
    JSON matching the contract rather than through a forced tool call, and
    validated on arrival exactly like the hosted path.
    """

    def __init__(
        self,
        name: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        base_url: str | None = None,
        timeout: float = 300.0,
    ):
        import httpx

        self.name = name
        self.temperature = temperature
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST")
                         or "http://127.0.0.1:11434").rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def respond(self, *, system, messages, tools, schema) -> ModelReply:
        instruction = (
            "Reply with a single JSON object and nothing else. To use a tool, "
            'emit {"tool": "<name>", "arguments": {...}}. To answer, emit '
            f'{{"answer": <object matching this schema>}}.\n'
            f"Answer schema: {json.dumps(_schema_of(schema))}\n"
            "Tools available:\n"
            + "\n".join(f"- {t.name}: {t.description} "
                        f"parameters={json.dumps(t.parameters)}" for t in tools)
        )
        body = {
            "model": self.name,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": f"{system}\n\n{instruction}"},
                *({"role": m.role if m.role != "tool" else "user",
                   "content": (m.content if m.role != "tool"
                               else f"[result of {m.tool_call_id}]\n{m.content}")}
                  for m in messages),
            ],
        }
        response = self._client.post(f"{self.base_url}/api/chat", json=body)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return ModelReply(raw_text=content,
                              invalid="model did not return parseable JSON")
        if "tool" in parsed:
            return ModelReply(
                tool_calls=[ToolCall(name=str(parsed["tool"]),
                                     arguments=dict(parsed.get("arguments") or {}))],
                raw_text=content,
            )
        if "answer" in parsed:
            answer, error = _validate(schema, parsed["answer"])
            return ModelReply(answer=answer, raw_text=content, invalid=error)
        return ModelReply(raw_text=content,
                          invalid="reply contained neither 'tool' nor 'answer'")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def from_env() -> Model:
    """Build the configured model, or raise.

    `AGENT_MODEL` selects the path: `anthropic` (default) or `ollama`.
    `AGENT_MODEL_NAME` overrides the model id. There is no stub fallback.
    """
    provider = os.environ.get("AGENT_MODEL", "anthropic").strip().lower()
    name = os.environ.get("AGENT_MODEL_NAME", "").strip()
    temperature = float(os.environ.get("AGENT_TEMPERATURE", DEFAULT_TEMPERATURE))

    if provider == "anthropic":
        return AnthropicModel(name=name or DEFAULT_ANTHROPIC_MODEL,
                              temperature=temperature)
    if provider == "ollama":
        return OllamaModel(name=name or DEFAULT_OLLAMA_MODEL,
                           temperature=temperature)
    raise ModelNotConfigured(
        f"AGENT_MODEL={provider!r} is not a model path. Use 'anthropic' or "
        "'ollama'. There is deliberately no stub fallback: a scored run must "
        "involve a real model."
    )
