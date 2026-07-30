"""Non-determinism has to be a known quantity before anything is measured.

Two different things are being pinned, and only one of them is about the model.

The graph's own behaviour is deterministic by construction: given the same case
and the same model replies, the node sequence, the tool calls, the prompts and
the result are identical. That is testable offline and is tested here. If it
were not true, a difference between two Phase D rows could be the harness rather
than the control.

The model's behaviour is not guaranteed. Temperature is pinned at 0, which
reduces variation without eliminating it, so the model-backed test measures
repeatability across identical runs and reports it rather than asserting it
away. It is opt-in, because it costs API calls.
"""

from __future__ import annotations

import os

import pytest

from agent.contracts import EvidenceCitation, TriageVerdict, Verdict
from agent.graph import ALL_CONTROLS, AgentConfig, Control, TriageGraph
from agent.models import DEFAULT_TEMPERATURE, ModelReply, ScriptedModel, ToolCall
from tests.support import FakeStore, synthetic_triage_case, write_rule


def _script():
    """A fresh script each time: ScriptedModel is stateful by design."""
    return [
        ModelReply(tool_calls=[ToolCall(name="lookup_rule", arguments={},
                                       call_id="c1")]),
        ModelReply(tool_calls=[ToolCall(name="query_events",
                                        arguments={"event_id": 10, "limit": 2},
                                        call_id="c2")]),
        ModelReply(answer=TriageVerdict(
            verdict=Verdict.TRUE_POSITIVE, confidence=0.9,
            evidence=[EvidenceCitation(event_index=1, field="TargetImage",
                                       quote="lsass.exe", supports="target")],
            rationale="procdump opened lsass with full access")),
    ]


def _run(tmp_path, controls=frozenset()):
    rule = write_rule(tmp_path)
    case = synthetic_triage_case(rule)
    model = ScriptedModel(_script())
    graph = TriageGraph(model, AgentConfig(controls=controls), store=FakeStore())
    result = graph.run(case)
    return result, graph, model


def test_the_same_case_produces_the_same_result(tmp_path):
    first, first_graph, _ = _run(tmp_path)
    second, second_graph, _ = _run(tmp_path)

    assert first.label == second.label
    assert first.tool_calls == second.tool_calls
    assert first.rejections == second.rejections
    assert ([e.kind for e in first_graph.audit]
            == [e.kind for e in second_graph.audit])


def test_the_prompt_is_byte_identical_across_runs(tmp_path):
    """An unstable prompt would make every comparison in this repo noisy."""
    _, _, first = _run(tmp_path)
    _, _, second = _run(tmp_path)

    assert len(first.seen) == len(second.seen)
    for (system_a, messages_a), (system_b, messages_b) in zip(first.seen,
                                                              second.seen):
        assert system_a == system_b
        assert [(m.role, m.content) for m in messages_a] \
            == [(m.role, m.content) for m in messages_b]


def test_the_node_sequence_is_stable(tmp_path):
    _, graph, _ = _run(tmp_path)
    kinds = [e.kind for e in graph.audit]
    assert kinds == ["run_started", "model_turn", "tool_call", "model_turn",
                     "tool_call", "model_turn", "run_finished"]


def test_turning_a_control_on_changes_the_prompt_and_nothing_else(tmp_path):
    """The Phase D comparison depends on exactly one thing moving."""
    plain, _, plain_model = _run(tmp_path)
    tagged, _, tagged_model = _run(tmp_path,
                                   frozenset({Control.PROVENANCE_TAGS}))

    assert plain.label == tagged.label
    assert plain.tool_calls == tagged.tool_calls
    plain_system = plain_model.seen[0][0]
    tagged_system = tagged_model.seen[0][0]
    assert tagged_system.startswith(plain_system)
    assert "adversary-writable" in tagged_system
    assert "adversary-writable" not in plain_system


def test_control_labels_are_stable_and_sorted():
    """Row labels in the results table must not depend on set iteration order."""
    assert AgentConfig().label == "undefended"
    assert AgentConfig(controls=ALL_CONTROLS).label == "all-controls"
    pair = frozenset({Control.ENFORCED_CITATION, Control.PROVENANCE_TAGS})
    assert AgentConfig(controls=pair).label == "enforced_citation+provenance_tags"


def test_temperature_is_pinned_low_by_default():
    assert DEFAULT_TEMPERATURE == 0.0


def test_the_run_records_what_produced_it(tmp_path):
    """A number without its model and temperature is not reproducible."""
    _, graph, _ = _run(tmp_path)
    started = graph.audit.of_kind("run_started")[0].payload
    assert started["model"] == "scripted"
    assert started["temperature"] == 0.0
    assert started["controls"] == []
    assert started["rule_id"] and started["capture_id"]


@pytest.mark.model
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"),
                    reason="needs a configured model")
def test_model_repeatability_is_measured_not_assumed(tmp_path):
    """Runs the same case three times and reports agreement.

    Deliberately not an equality assertion. Temperature 0 is not a determinism
    guarantee, and a flaky test that fails on provider variance would teach the
    reader to ignore it. The number belongs in the results, not in an assert.
    """
    from agent import corpus, models

    evaluation = corpus.evaluation_set()
    case = evaluation.triage[0]
    labels = []
    for _ in range(3):
        graph = TriageGraph(models.from_env(), AgentConfig())
        labels.append(graph.run(case).label)
    agreement = labels.count(max(set(labels), key=labels.count)) / len(labels)
    print(f"\nrepeatability on {case.case_id}: {labels} -> {agreement:.0%}")
    assert labels  # the measurement is the point
