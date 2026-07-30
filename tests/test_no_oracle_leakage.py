"""The inputs a predictor sees must not contain the answer.

The sibling's labels are a deterministic function of five observable
quantities, so leaking any of them turns both the agent and the baseline into
a reimplementation of the labeller. This is the test that keeps the benchmark
honest, so it checks the serialised form rather than the field names: a nested
dict or a reformatted number would slip past an attribute check.
"""

from __future__ import annotations

import json
import re

import pytest

from agent import corpus

#: Field names produced by the labeller. None may appear in a case's inputs.
ORACLE_KEYS = frozenset({
    "truth", "label", "class", "reason", "candidates", "matched",
    "field_present", "group_satisfied", "tool_specific", "in_scope",
    "detected", "tools_detected", "classes", "per_100k", "captures_hit",
})

#: Values that would give a four-way label away if they appeared as text.
ORACLE_VALUES = ("miss-logic", "miss-telemetry", "out-of-scope",
                 "true_positive", "false_positive")


def _keys(obj, into: set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            into.add(key)
            _keys(value, into)
    elif isinstance(obj, list):
        for item in obj:
            _keys(item, into)


def _assert_clean(inputs: dict, case_id: str) -> None:
    keys: set[str] = set()
    _keys(inputs, keys)
    leaked = keys & ORACLE_KEYS
    assert not leaked, f"{case_id}: oracle keys in inputs: {sorted(leaked)}"

    blob = json.dumps(inputs)
    # the case id is allowed to encode the split it came from; nothing else is
    body = blob.replace(json.dumps(inputs.get("case_id", "")), '""')
    for value in ORACLE_VALUES:
        assert value not in body, f"{case_id}: oracle value {value!r} in inputs"


def test_synthetic_triage_case_inputs_are_clean(synthetic_rule, synthetic_capture):
    case = corpus.TriageCase(
        case_id="tp:synthetic",
        rule=synthetic_rule,
        capture=synthetic_capture,
        fire_count=3,
        truth="true_positive",
    )
    _assert_clean(case.inputs(), case.case_id)


def test_synthetic_miss_case_inputs_are_clean(synthetic_rule, synthetic_capture):
    case = corpus.MissCase(
        case_id="miss:synthetic",
        rule=synthetic_rule,
        capture=synthetic_capture,
        truth="miss-logic",
    )
    _assert_clean(case.inputs(), case.case_id)


def test_truth_is_reachable_on_the_case_but_not_in_inputs(
    synthetic_rule, synthetic_capture
):
    """The scorer needs the label; the predictor must not have it."""
    case = corpus.MissCase(
        case_id="miss:synthetic",
        rule=synthetic_rule,
        capture=synthetic_capture,
        truth="out-of-scope",
    )
    assert case.truth == "out-of-scope"
    assert "out-of-scope" not in json.dumps(case.inputs())


@pytest.mark.corpus
def test_every_real_case_has_clean_inputs(evaluation):
    for case in (*evaluation.triage, *evaluation.miss):
        _assert_clean(case.inputs(), case.case_id)


def _shape(obj) -> object:
    """The structure of a case's inputs, with values replaced by their type.

    Presence and absence are part of the shape, so a field populated for one
    label and null for the other shows up as a difference.
    """
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return ["…"] if obj else []
    return type(obj).__name__


@pytest.mark.corpus
def test_the_two_triage_labels_have_identically_shaped_inputs(evaluation):
    """A field present for one label and absent for the other is a shortcut.

    This caught a real one: the sibling publishes an event count per campaign
    capture but not per benign capture, so `host_events` was an integer on every
    true positive and null on every false positive. That is a perfect classifier
    requiring no telemetry at all.
    """
    shapes: dict[str, set[str]] = {}
    for case in evaluation.triage:
        shapes.setdefault(case.truth, set()).add(json.dumps(_shape(case.inputs())))

    assert set(shapes) == {"true_positive", "false_positive"}
    for truth, seen in shapes.items():
        assert len(seen) == 1, f"{truth} inputs vary in shape: {seen}"
    assert shapes["true_positive"] == shapes["false_positive"]


@pytest.mark.corpus
def test_the_capture_handle_hides_which_corpus_a_case_came_from(evaluation):
    """Capture names announce the label: LSASS_campaign_01 versus tactic/name."""
    for case in evaluation.triage:
        handle = case.inputs()["capture"]["id"]
        assert handle.startswith("capture-")
        assert "LSASS" not in handle and "campaign" not in handle
        assert case.capture.id not in handle
    # and distinct captures stay distinguishable
    handles = {c.inputs()["capture"]["id"] for c in evaluation.triage}
    captures = {c.capture.id for c in evaluation.triage}
    assert len(handles) == len(captures)


@pytest.mark.corpus
def test_lab_identifiers_do_not_reach_the_agent(evaluation):
    """The domain suffix separated the two classes perfectly. It must not survive.

    All seven true-positive captures are hosts in pandalab.com; every
    false-positive capture is in theshire.local, mordor.local or shire.com. That
    lives in Hostname, an os-generated field, so provenance tagging cannot help.
    """
    from agent.events import Ingestion
    from agent.tools import CaptureStore, Toolbox

    lab = re.compile(r"pandalab|theshire|mordor|shire\.com", re.IGNORECASE)
    store = CaptureStore()
    seen_labels = set()
    for case in evaluation.triage:
        if case.truth in seen_labels:
            continue
        seen_labels.add(case.truth)
        box = Toolbox(case.rule, case.capture, store=store,
                      ingestion=Ingestion.STRUCTURED)
        output = box.call("query_events", {"limit": 20}).output
        assert not lab.search(output), f"{case.case_id}: lab identifier survived"


def test_well_known_principals_are_not_pseudonymised():
    """NT AUTHORITY\\SYSTEM names the OS, not the lab, and it is evidence."""
    from agent.pseudonymise import Pseudonymiser

    events = [{"Hostname": "MKT01.pandalab.com", "TargetUser": "NT AUTHORITY\\SYSTEM",
               "SourceUser": "PANDALAB\\pedro"}]
    table = Pseudonymiser(events)
    rewritten = table.event(events[0])
    assert rewritten["TargetUser"] == "NT AUTHORITY\\SYSTEM"
    assert rewritten["SourceUser"] == "CORP\\pedro"
    assert rewritten["Hostname"] == "HOST1.corp.example"
