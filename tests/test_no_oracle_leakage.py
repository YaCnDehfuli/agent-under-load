"""The inputs a predictor sees must not contain the answer.

The sibling's labels are a deterministic function of five observable
quantities, so leaking any of them turns both the agent and the baseline into
a reimplementation of the labeller. This is the test that keeps the benchmark
honest, so it checks the serialised form rather than the field names: a nested
dict or a reformatted number would slip past an attribute check.
"""

from __future__ import annotations

import json

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
