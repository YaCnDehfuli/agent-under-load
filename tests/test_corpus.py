"""The evaluation set must match what the sibling repo published.

These counts are the sibling's, not mine. If a rerun there changes them, these
tests fail and the results in docs/ are stale — which is the point.
"""

from __future__ import annotations

import pytest

from agent import corpus

EXPECTED_MISS = {
    "out-of-scope": 273,
    "miss-logic": 207,
    "miss-telemetry": 57,
    "detected": 44,
}
EXPECTED_TRIAGE = {"true_positive": 44, "false_positive": 36}


@pytest.mark.corpus
def test_miss_set_matches_published_labels(evaluation):
    assert evaluation.truth_distribution()["miss"] == EXPECTED_MISS
    assert len(evaluation.miss) == sum(EXPECTED_MISS.values()) == 581


@pytest.mark.corpus
def test_triage_set_is_symmetric_across_both_labels(evaluation):
    """Both sides come from the same rules and the same kind of archive.

    If one label were only ever backed by captures that are missing from disk,
    the task would be separable by data availability rather than by evidence.
    """
    assert evaluation.truth_distribution()["triage"] == EXPECTED_TRIAGE
    for case in evaluation.triage:
        assert case.capture.archive.exists(), case.case_id
    truths = {case.truth for case in evaluation.triage}
    assert truths == {"true_positive", "false_positive"}


@pytest.mark.corpus
def test_true_positives_come_from_attack_captures_only(evaluation):
    for case in evaluation.triage:
        assert case.capture.is_attack == (case.truth == "true_positive"), case.case_id


@pytest.mark.corpus
def test_every_case_resolves_a_readable_rule(evaluation):
    for case in (*evaluation.triage, *evaluation.miss):
        assert case.rule.path.exists(), case.case_id
        assert case.rule.parsed().get("detection"), case.case_id


@pytest.mark.corpus
def test_absences_are_explicit_and_reasoned(evaluation):
    """A missing input is an Absence with a reason, never a silent drop."""
    for absence in evaluation.absences:
        assert absence.kind and absence.ref and absence.reason


@pytest.mark.corpus
def test_pins_and_digests_verify():
    assert corpus.verify() == []


@pytest.mark.corpus
def test_events_stream_as_flattened_dicts(evaluation):
    capture = next(c.capture for c in evaluation.triage if not c.capture.is_attack)
    stream = corpus.events(capture)
    first = next(stream)
    assert isinstance(first, dict)
    assert "EventID" in first
