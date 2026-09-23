"""The OpenAI-compatible client, against a fake transport.

Every provider in the study goes through this one class, so what it sends and
what it makes of the reply are pinned here without a network call. The live
check is the smoke run against each provider.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent.contracts import TriageVerdict
from agent.models import (
    NO_TOOL_OR_ANSWER,
    SUBMIT_TOOL,
    Message,
    ModelError,
    ModelNotConfigured,
    OpenAICompatModel,
    ToolCall,
    ToolSpec,
    load_model,
)

TOOLS = [ToolSpec(name="lookup_rule", description="read the rule",
                  parameters={"type": "object", "properties": {}})]

VERDICT = {"verdict": "true_positive", "confidence": 0.9,
           "evidence": [{"event_index": 1, "field": "TargetImage",
                         "quote": "lsass.exe", "supports": "target"}],
           "rationale": "procdump opened lsass"}


def _completion(message: dict, finish_reason: str = "tool_calls",
                usage: dict | None = None, model: str = "served-id") -> dict:
    return {"model": model,
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage or {}}


def _tool_call(name: str, arguments, call_id: str = "call_a") -> dict:
    encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": encoded}}


def _model(responses, sent=None, sleeps=None, **kwargs) -> OpenAICompatModel:
    """A client whose transport replays `responses` and records each request."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if sent is not None:
            sent.append(json.loads(request.content))
        status, body, headers = queue.pop(0)
        return httpx.Response(status, json=body, headers=headers)

    return OpenAICompatModel(
        name="some-model", base_url="https://provider.test/v1", api_key="k",
        transport=httpx.MockTransport(handler),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
        **kwargs)


def _respond(model, messages=None):
    return model.respond(system="sys", tools=TOOLS, schema=TriageVerdict,
                         messages=messages or [Message(role="user", content="case")])


# -- replies ---------------------------------------------------------------


def test_a_tool_call_arrives_with_its_id_and_arguments():
    reply = _respond(_model([(200, _completion(
        {"content": "", "tool_calls": [_tool_call("lookup_rule", {"x": 1})]}), {})]))
    assert reply.tool_calls == [ToolCall(name="lookup_rule", arguments={"x": 1},
                                         call_id="call_a")]
    assert reply.served_model == "served-id"
    assert reply.finish_reason == "tool_calls"
    assert not reply.invalid


def test_usage_is_read_including_cached_and_reasoning_tokens():
    usage = {"prompt_tokens": 1500, "completion_tokens": 300,
             "prompt_tokens_details": {"cached_tokens": 1280},
             "completion_tokens_details": {"reasoning_tokens": 220}}
    reply = _respond(_model([(200, _completion(
        {"tool_calls": [_tool_call("lookup_rule", {})]}, usage=usage), {})]))
    assert reply.usage.input_tokens == 1500
    assert reply.usage.cached_input_tokens == 1280
    assert reply.usage.output_tokens == 300
    assert reply.usage.reasoning_tokens == 220
    assert reply.latency_s is not None


def test_deepseek_reports_cache_hits_under_its_own_field():
    usage = {"prompt_tokens": 900, "completion_tokens": 50,
             "prompt_cache_hit_tokens": 640, "prompt_cache_miss_tokens": 260}
    reply = _respond(_model([(200, _completion(
        {"tool_calls": [_tool_call("lookup_rule", {})]}, usage=usage), {})]))
    assert reply.usage.cached_input_tokens == 640
    assert reply.usage.reasoning_tokens is None


def test_a_valid_submission_becomes_the_answer():
    reply = _respond(_model([(200, _completion(
        {"tool_calls": [_tool_call(SUBMIT_TOOL, VERDICT)]}), {})]))
    assert isinstance(reply.answer, TriageVerdict)
    assert not reply.invalid


def test_a_submission_the_contract_refuses_is_kept_as_invalid():
    bad = dict(VERDICT, evidence=[])  # a decisive verdict must cite
    reply = _respond(_model([(200, _completion(
        {"tool_calls": [_tool_call(SUBMIT_TOOL, bad)]}), {})]))
    assert reply.answer is None
    assert reply.invalid.startswith("contract rejected the verdict")


def test_arguments_that_are_not_json_are_invalid_not_empty():
    reply = _respond(_model([(200, _completion(
        {"tool_calls": [_tool_call("lookup_rule", "{not json")]}), {})]))
    assert not reply.tool_calls
    assert "not valid JSON" in reply.invalid


def test_a_text_only_reply_is_invalid():
    reply = _respond(_model([(200, _completion(
        {"content": "I think it is benign."}, finish_reason="stop"), {})]))
    assert reply.invalid == NO_TOOL_OR_ANSWER
    assert reply.raw_text == "I think it is benign."


def test_running_out_of_output_tokens_is_reported_as_such():
    reply = _respond(_model([(200, _completion(
        {"content": "Let me think"}, finish_reason="length"), {})]))
    assert reply.invalid == "output hit the token limit"


def test_missing_call_ids_are_synthesised_from_the_turn_position():
    raw = {"type": "function", "function": {"name": "lookup_rule", "arguments": "{}"}}
    reply = _respond(_model([(200, _completion({"tool_calls": [raw]}), {})]))
    assert reply.tool_calls[0].call_id == "call_0_0"


# -- requests --------------------------------------------------------------


def test_the_transcript_is_sent_with_native_tool_turns():
    sent: list[dict] = []
    history = [
        Message(role="user", content="case"),
        Message(role="assistant", content="",
                tool_calls=[ToolCall(name="lookup_rule", arguments={"b": 2, "a": 1},
                                     call_id="call_a")]),
        Message(role="tool", content="rule text", tool_call_id="call_a"),
    ]
    _respond(_model([(200, _completion(
        {"tool_calls": [_tool_call(SUBMIT_TOOL, VERDICT)]}), {})], sent=sent), history)

    messages = sent[0]["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_a", "type": "function",
         "function": {"name": "lookup_rule", "arguments": '{"a": 1, "b": 2}'}}]}
    assert messages[3] == {"role": "tool", "tool_call_id": "call_a",
                           "content": "rule text"}
    names = [t["function"]["name"] for t in sent[0]["tools"]]
    assert names == ["lookup_rule", SUBMIT_TOOL]


def test_registry_params_are_sent_verbatim_and_temperature_only_when_set():
    sent: list[dict] = []
    params = {"reasoning_effort": "low", "max_completion_tokens": 2048}
    reply = _completion({"tool_calls": [_tool_call("lookup_rule", {})]})
    _respond(_model([(200, reply, {})], sent=sent, params=params))
    _respond(_model([(200, reply, {})], sent=sent, params=params, temperature=0.0))

    assert sent[0]["reasoning_effort"] == "low"
    assert sent[0]["max_completion_tokens"] == 2048
    assert "temperature" not in sent[0]
    assert sent[1]["temperature"] == 0.0


# -- failures --------------------------------------------------------------


def test_a_rate_limit_is_retried_after_the_advertised_wait():
    sleeps: list[float] = []
    reply = _respond(_model([
        (429, {"error": "slow down"}, {"retry-after": "7"}),
        (200, _completion({"tool_calls": [_tool_call("lookup_rule", {})]}), {}),
    ], sleeps=sleeps))
    assert sleeps == [7.0]
    assert reply.attempts == 2
    assert reply.tool_calls


def test_a_bad_request_is_not_retried():
    sleeps: list[float] = []
    with pytest.raises(ModelError, match="400"):
        _respond(_model([(400, {"error": "unknown parameter"}, {})], sleeps=sleeps))
    assert sleeps == []


def test_persistent_server_errors_raise_after_the_last_attempt():
    sleeps: list[float] = []
    with pytest.raises(ModelError, match="503 after 6 attempt"):
        _respond(_model([(503, {}, {})] * 6, sleeps=sleeps))
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


# -- registry --------------------------------------------------------------


def _registry(tmp_path, body: str):
    path = tmp_path / "models.yml"
    path.write_text(body)
    return path


def test_an_unknown_key_is_refused(tmp_path):
    path = _registry(tmp_path, "a: {provider: groq, base_url: x, model: m}\n")
    with pytest.raises(ModelNotConfigured, match="not in the model registry"):
        load_model("b", path)


def test_a_missing_credential_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    path = _registry(tmp_path, "a: {provider: groq, base_url: https://h.test/v1, "
                               "api_key_env: SOME_KEY, model: m}\n")
    with pytest.raises(ModelNotConfigured, match="SOME_KEY"):
        load_model("a", path)


def test_a_registry_entry_becomes_a_configured_client(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_KEY", "secret")
    path = _registry(tmp_path, """\
a:
  provider: groq
  base_url: https://api.provider.test/openai/v1
  api_key_env: SOME_KEY
  model: org/model-20b
  temperature: null
  params: {reasoning_effort: low}
""")
    model = load_model("a", path)
    assert model.config == {"provider": "groq", "host": "api.provider.test",
                            "model": "org/model-20b", "temperature": None,
                            "params": {"reasoning_effort": "low"}}
    assert "secret" not in json.dumps(model.config)


def test_the_committed_registry_parses_and_names_its_credentials():
    from agent.models import registry

    entries = registry()
    for key, entry in entries.items():
        assert entry["provider"] and entry["model"], key
        assert "price" in entry, key
        if entry["provider"] != "ollama":
            assert entry["api_key_env"], key
