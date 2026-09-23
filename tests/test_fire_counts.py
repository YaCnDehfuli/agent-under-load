"""The match-count leak check, and the alert header that no longer carries a count."""

from __future__ import annotations

from agent.baseline import analyse_rule, count_matches
from agent.events import Event
from agent.graph import _case_prompt
from score.fire_counts import best_threshold, recount, report
from tests.support import (
    SYNTHETIC_EVENTS,
    FakeStore,
    synthetic_triage_case,
    write_rule,
)


def test_count_matches_counts_the_events_the_rule_matches(tmp_path):
    events = [Event(raw, i) for i, raw in enumerate(SYNTHETIC_EVENTS)]
    # event 1 is procdump opening lsass with 0x1fffff; event 2 targets svchost
    assert count_matches(analyse_rule(write_rule(tmp_path)), events) == 1


def test_recount_uses_each_case_s_own_capture(tmp_path):
    case = synthetic_triage_case(write_rule(tmp_path))
    assert recount([case], FakeStore()) == {case.case_id: 1}


def test_the_header_carries_no_match_count(tmp_path):
    assert "Events matched" not in _case_prompt(synthetic_triage_case(write_rule(tmp_path)))


def test_the_threshold_check_finds_a_count_that_splits_the_labels():
    counts = {"a": 1, "b": 2, "c": 40, "d": 50}
    truth = {"a": "true_positive", "b": "true_positive",
             "c": "false_positive", "d": "false_positive"}
    assert best_threshold(counts, truth) == (4, 40, "false_positive")
    assert "4/4 correct" in report(counts, truth)


def test_the_report_lists_alerts_the_recount_finds_no_match_for():
    counts = {"a": 0, "b": 3}
    truth = {"a": "false_positive", "b": "true_positive"}
    assert "a (false_positive)" in report(counts, truth)
