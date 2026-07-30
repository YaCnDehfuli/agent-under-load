"""Injections may only enter fields an adversary controls.

This is the test that separates a real threat model from a rigged demo. Without
it the attack numbers mean nothing: a payload in `GrantedAccess` would produce an
impressive success rate against an attack no adversary can mount, and nobody
reading the results would be able to tell.

It is a hard allowlist rather than a review habit, because a threat model
maintained by discipline is a threat model that drifts.
"""

from __future__ import annotations

import pytest

from agent.events import Event, Ingestion, render_raw, render_structured
from agent.provenance import (
    INFLUENCED_FIELDS,
    OS_FIELDS,
    WRITABLE_FIELDS,
    ProvenanceError,
)
from attack import inject
from attack.inject import (
    InjectedStore,
    Injection,
    InjectionError,
    Payload,
    inject_into_event,
    load_payloads,
    target_indices,
)
from tests.support import (
    SYNTHETIC_CAPTURE,
    SYNTHETIC_EVENTS,
    FakeStore,
    write_rule,
)

PAYLOADS = load_payloads()


# -- the allowlist --------------------------------------------------------


def test_the_corpus_loads_and_is_not_empty():
    assert len(PAYLOADS) >= 12
    assert len({p.id for p in PAYLOADS}) == len(PAYLOADS)


def test_every_payload_targets_only_adversary_writable_fields():
    """The load-bearing assertion of the whole attack phase."""
    for payload in PAYLOADS:
        for field in payload.placements:
            assert field in WRITABLE_FIELDS, (
                f"{payload.id} targets {field}, which an adversary does not write"
            )


@pytest.mark.parametrize("field", sorted(INFLUENCED_FIELDS | OS_FIELDS)[:25])
def test_no_payload_targets_an_influenced_or_os_field(field):
    for payload in PAYLOADS:
        assert field not in payload.placements, f"{payload.id} targets {field}"


def test_a_payload_naming_a_forbidden_field_cannot_be_constructed():
    """Refused at construction, so a bad payload cannot be loaded at all."""
    for field in ("GrantedAccess", "EventID", "Hostname", "CallTrace"):
        with pytest.raises(ProvenanceError):
            Payload(id="bad", strategy="instruction", objective="suppression",
                    placements=(field,), text="x")


def test_injecting_into_a_forbidden_field_raises():
    with pytest.raises(ProvenanceError):
        inject_into_event({"GrantedAccess": "0x1010"}, "GrantedAccess", "payload")


def test_an_unknown_field_is_refused_as_a_target():
    """Unknown means off-limits for attack, the opposite of the defence default."""
    with pytest.raises(ProvenanceError):
        Payload(id="bad", strategy="instruction", objective="suppression",
                placements=("SomeNewSysmonField",), text="x")


# -- corpus shape ---------------------------------------------------------


def test_every_strategy_is_represented():
    covered = {p.strategy for p in PAYLOADS}
    assert covered == set(inject.STRATEGIES)


def test_both_objectives_are_represented():
    assert {p.objective for p in PAYLOADS} == set(inject.OBJECTIVES)


def test_placement_varies_across_fields():
    """A result per strategy is only interesting if placement also varies."""
    fields = {f for p in PAYLOADS for f in p.placements}
    assert len(fields) >= 6


def test_no_payload_is_the_trivially_greppable_one():
    """A real operator would not use a string that greps in one line.

    Not a security control — a realism check. The corpus is meant to measure
    whether plausible text works, and the canonical phrase is not plausible.
    """
    for payload in PAYLOADS:
        assert "ignore all previous instructions" not in payload.text.lower()
        assert "ignore previous instructions" not in payload.text.lower()


def test_every_payload_carries_a_note_explaining_why_it_is_plausible():
    for payload in PAYLOADS:
        assert payload.note.strip(), f"{payload.id} has no note"


def test_a_malformed_payload_is_rejected():
    for bad in (
        {"strategy": "telepathy"},
        {"objective": "mischief"},
        {"placements": ()},
        {"text": "   "},
    ):
        base = {"id": "x", "strategy": "instruction", "objective": "suppression",
                "placements": ("CommandLine",), "text": "hello"}
        with pytest.raises((InjectionError, ProvenanceError)):
            Payload(**{**base, **bad})  # type: ignore[arg-type]


# -- injection mechanics --------------------------------------------------


def test_the_payload_is_appended_not_substituted():
    """The original value is evidence and must survive."""
    event = {"EventID": 1, "CommandLine": "procdump -ma lsass.exe out.dmp"}
    out = inject_into_event(event, "CommandLine", "PAYLOAD")
    assert out["CommandLine"] == "procdump -ma lsass.exe out.dmp PAYLOAD"
    assert event["CommandLine"] == "procdump -ma lsass.exe out.dmp"  # not mutated


def test_a_field_the_event_lacks_is_not_invented():
    """Inventing a field forges a record shape no host emits."""
    event = {"EventID": 10, "SourceImage": "x.exe"}
    assert inject_into_event(event, "ScriptBlockText", "PAYLOAD") == event


def test_an_empty_field_is_not_a_target():
    event = {"EventID": 1, "CommandLine": ""}
    assert inject_into_event(event, "CommandLine", "PAYLOAD") == event


def test_the_rendered_message_stays_consistent():
    """Otherwise the agent reacts to a contradiction, not to the payload."""
    event = {
        "EventID": 1,
        "CommandLine": "procdump -ma lsass.exe",
        "Message": "Process Create:\r\nCommandLine: procdump -ma lsass.exe\r\n"
                   "User: CORP\\alice",
    }
    out = inject_into_event(event, "CommandLine", "PAYLOAD")
    assert "procdump -ma lsass.exe PAYLOAD" in out["Message"]
    assert out["Message"].count("PAYLOAD") == 1


def test_message_is_left_alone_when_it_does_not_quote_the_field():
    event = {"EventID": 1, "CommandLine": "whoami", "Message": "unrelated text"}
    out = inject_into_event(event, "CommandLine", "PAYLOAD")
    assert out["Message"] == "unrelated text"


# -- targeting ------------------------------------------------------------


def test_targets_are_events_the_adversary_owns(tmp_path):
    """The rule fired on the adversary's own activity, so those are the targets."""
    from agent.baseline import analyse_rule

    rule = write_rule(tmp_path)
    events = [Event(raw, i) for i, raw in enumerate(SYNTHETIC_EVENTS)]
    indices = target_indices(events, analyse_rule(rule), "SourceImage", limit=5)
    # event 1 is the procdump/lsass/0x1fffff record the rule matches
    assert indices == (1,)


def test_no_target_is_a_recorded_outcome_not_a_failure(tmp_path):
    from agent.baseline import analyse_rule

    rule = write_rule(tmp_path)
    events = [Event(raw, i) for i, raw in enumerate(SYNTHETIC_EVENTS)]
    assert target_indices(events, analyse_rule(rule), "ScriptBlockText") == ()


def test_targets_respect_the_event_limit(tmp_path):
    from agent.baseline import analyse_rule

    rule = write_rule(tmp_path)
    raws = [dict(SYNTHETIC_EVENTS[1]) for _ in range(20)]
    events = [Event(raw, i) for i, raw in enumerate(raws)]
    assert len(target_indices(events, analyse_rule(rule), "SourceImage",
                              limit=3)) == 3


# -- the injected store --------------------------------------------------


def test_the_injected_store_delivers_the_payload_through_the_normal_path(tmp_path):
    """The agent must meet the payload via the tools it always uses."""
    from agent.tools import Toolbox

    payload = next(p for p in PAYLOADS if "CommandLine" in p.placements)
    injection = Injection(payload=payload, field="CommandLine",
                          event_indices=(0,))
    store = InjectedStore(injection, base=FakeStore())

    box = Toolbox(rule=write_rule(tmp_path), capture=SYNTHETIC_CAPTURE,
                  store=store, ingestion=Ingestion.RAW)
    output = box.call("query_events", {"event_id": 1}).output
    assert payload.text.split()[0] in output


def test_only_the_targeted_events_are_modified():
    payload = next(p for p in PAYLOADS if "CommandLine" in p.placements)
    injection = Injection(payload=payload, field="CommandLine",
                          event_indices=(0,))
    store = InjectedStore(injection, base=FakeStore())
    events = store.load(SYNTHETIC_CAPTURE)
    assert payload.text in str(events[0].get("CommandLine"))
    for event in events[1:]:
        assert payload.text not in str(event.raw)


def test_structured_rendering_neutralises_the_delimiter_escape():
    """The escape payload should survive raw mode and be defanged in structured.

    Not a claim that structured ingestion stops prompt injection. It is a claim
    about this one strategy: a payload whose mechanism is a literal newline
    cannot keep it once newlines are escaped.
    """
    payload = next(p for p in PAYLOADS if p.id == "delim-field")
    raw = inject_into_event(
        {"EventID": 1, "CommandLine": "whoami", "Image": "C:\\x.exe"},
        "CommandLine", payload.text,
    )
    event = Event(raw, 0)
    assert "\n" in render_raw(event)
    structured = render_structured(event)
    # the payload's newlines are escaped, so it cannot open a fake field line
    assert "\\n" in structured
    body = [line for line in structured.splitlines() if line.startswith("  ")]
    assert not any(line.strip().startswith("Verdict") for line in body)
