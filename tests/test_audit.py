"""The audit is complete under attack, and detects a single altered record.

Two claims, tested separately because they fail differently.

**Completeness.** A run in which the injection *succeeded* must still be fully
reconstructable from the log: which tools were called, how much adversary-written
text reached the model, what was requested, what the boundary decided, and what
the agent concluded. An audit trail that goes quiet exactly when the attack works
is the one that matters and the one nobody tests.

**Integrity.** Entries are hash-chained, so editing or deleting one breaks the
digests after it. This is tamper-evidence, not tamper-proofing: an attacker
holding the log can recompute the chain. The claim is only that a single altered
record is detectable, which is the realistic case once the log is forwarded
somewhere the attacker does not control.
"""

from __future__ import annotations

import dataclasses
import json

from agent.audit import GENESIS, AuditLog
from agent.contracts import EvidenceCitation, TriageVerdict, Verdict
from agent.graph import AgentConfig, Control, TriageGraph
from agent.models import ModelReply, ScriptedModel, ToolCall
from attack.inject import InjectedStore, Injection, load_payloads
from tests.support import FakeStore, synthetic_triage_case, write_rule


def _attacked_run(tmp_path, controls=frozenset()):
    """A run where the payload landed and the agent did what it asked.

    The verdict flips to false_positive on a true positive: a successful
    suppression. Everything about it has to be in the log.
    """
    payload = next(p for p in load_payloads()
                   if p.objective == "suppression" and "CommandLine" in p.placements)
    injection = Injection(payload=payload, field="CommandLine",
                          event_indices=(0,))
    store = InjectedStore(injection, base=FakeStore())

    script = [
        ModelReply(tool_calls=[ToolCall(name="lookup_rule", arguments={},
                                       call_id="c1")]),
        ModelReply(tool_calls=[ToolCall(name="query_events",
                                        arguments={"event_id": 1}, call_id="c2")]),
        ModelReply(tool_calls=[ToolCall(
            name="request_action",
            arguments={"action": "enrich_alert", "target": "lab/synthetic",
                       "justification": "closing as benign per note"},
            call_id="c3")]),
        ModelReply(answer=TriageVerdict(
            verdict=Verdict.FALSE_POSITIVE, confidence=0.85,
            evidence=[EvidenceCitation(event_index=0, field="CommandLine",
                                       quote="whoami",
                                       supports="approved maintenance")],
            rationale="the command line documents an approved change")),
    ]
    graph = TriageGraph(ScriptedModel(script), AgentConfig(controls=controls),
                        store=store)
    result = graph.run(synthetic_triage_case(write_rule(tmp_path)))
    return graph, result, payload


# -- completeness ---------------------------------------------------------


def test_the_attack_succeeded_so_the_test_is_about_the_right_thing(tmp_path):
    _, result, _ = _attacked_run(tmp_path)
    assert result.label == "false_positive"  # on a true positive


def test_every_tool_call_is_in_the_log(tmp_path):
    graph, result, _ = _attacked_run(tmp_path)
    logged = [e.payload["tool"] for e in graph.audit.of_kind("tool_call")]
    assert logged == ["lookup_rule", "query_events", "request_action"]
    assert result.tool_calls == len(logged)


def test_the_log_records_how_much_untrusted_text_reached_the_model(tmp_path):
    """The number a defender alerts on."""
    graph, _, payload = _attacked_run(tmp_path)
    assert graph.audit.untrusted_bytes > len(payload.text)

    by_tool = {e.payload["tool"]: e.payload for e in graph.audit.of_kind("tool_call")}
    assert by_tool["query_events"]["untrusted_bytes"] > 0
    # the tools that answer from os-generated values contributed nothing
    assert by_tool["lookup_rule"]["untrusted_bytes"] == 0
    assert by_tool["request_action"]["untrusted_bytes"] == 0


def test_the_verdict_and_the_action_decision_are_both_recorded(tmp_path):
    graph, _, _ = _attacked_run(tmp_path)

    finished = graph.audit.of_kind("run_finished")[0].payload
    assert finished["label"] == "false_positive"

    action = graph.audit.of_kind("action_requested")[0].payload
    assert action["action"] == "enrich_alert"
    assert action["granted"] is True


def test_the_run_is_reconstructable_from_the_serialised_log_alone(tmp_path):
    """Someone who was not watching must be able to rebuild what happened."""
    graph, _, payload = _attacked_run(tmp_path)
    lines = [json.loads(line) for line in graph.audit.to_jsonl().splitlines()]

    kinds = [entry["kind"] for entry in lines]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    assert kinds.count("tool_call") == 3

    start = lines[0]
    assert start["model"] == "scripted" and start["temperature"] == 0.0
    assert start["rule_id"] and start["capture_id"]
    assert "controls" in start

    assert any(entry.get("untrusted_bytes", 0) > 0 for entry in lines)
    assert lines[-1]["label"] == "false_positive"


def test_the_log_survives_the_controls_being_on(tmp_path):
    """Completeness is not a property of the undefended configuration only."""
    graph, _, _ = _attacked_run(tmp_path, frozenset(Control))
    assert graph.audit.verify() == []
    assert [e.kind for e in graph.audit][0] == "run_started"
    assert graph.audit.of_kind("run_finished")


def test_a_rejected_citation_is_recorded_as_a_signal(tmp_path):
    """Under attack this is the entry that fires."""
    bogus = ModelReply(answer=TriageVerdict(
        verdict=Verdict.FALSE_POSITIVE, confidence=0.9,
        evidence=[EvidenceCitation(event_index=0, field="CommandLine",
                                   quote="approved by the security team",
                                   supports="x")],
        rationale="fabricated"))
    graph = TriageGraph(ScriptedModel([bogus, bogus]),
                        AgentConfig(controls=frozenset({Control.ENFORCED_CITATION})),
                        store=FakeStore())
    graph.run(synthetic_triage_case(write_rule(tmp_path)))

    assert graph.audit.of_kind("citation_rejected")
    assert graph.audit.verify() == []


# -- integrity ------------------------------------------------------------


def test_an_empty_log_has_the_genesis_head():
    log = AuditLog()
    assert log.head == GENESIS
    assert log.verify() == []


def test_a_clean_chain_verifies(tmp_path):
    graph, _, _ = _attacked_run(tmp_path)
    assert graph.audit.verify() == []
    assert len(graph.audit.head) == 64


def test_each_entry_chains_to_the_one_before(tmp_path):
    graph, _, _ = _attacked_run(tmp_path)
    entries = graph.audit.entries
    assert entries[0].previous == GENESIS
    for earlier, later in zip(entries, entries[1:]):
        assert later.previous == earlier.digest


def test_editing_an_entry_is_detected(tmp_path):
    """The realistic tamper: soften one record and leave the rest alone."""
    graph, _, _ = _attacked_run(tmp_path)
    log = graph.audit
    target = next(e for e in log.entries if e.kind == "run_finished")
    log._entries[target.seq] = dataclasses.replace(
        target, payload={**target.payload, "label": "true_positive"})

    problems = log.verify()
    assert problems
    assert "does not match its digest" in problems[0]


def test_deleting_an_entry_is_detected(tmp_path):
    graph, _, _ = _attacked_run(tmp_path)
    log = graph.audit
    del log._entries[2]

    problems = log.verify()
    assert problems
    assert any("seq" in p or "previous digest" in p for p in problems)


def test_the_head_changes_when_anything_changes(tmp_path):
    """Publishing the head somewhere append-only is what makes this useful."""
    first, _, _ = _attacked_run(tmp_path)
    second, _, _ = _attacked_run(tmp_path)
    # different runs differ only by timestamps, which are part of the digest
    assert first.audit.head != second.audit.head

    before = first.audit.head
    entry = first.audit.entries[1]
    first.audit._entries[1] = dataclasses.replace(entry, payload={"turn": 99})
    assert first.audit.verify()
    assert first.audit.head == before  # the head alone does not move…
    # …which is exactly why verify() exists and the head must be published
