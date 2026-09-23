"""The alert's "events matched" line: counted per capture, the same way for both labels."""

from __future__ import annotations

import dataclasses
import json

import pytest

from agent import corpus
from agent.baseline import analyse_rule, count_matches
from agent.events import Event
from agent.graph import _case_prompt
from agent.models import ScriptedModel
from score.fire_counts import best_threshold, recount, report
from score.run import run_agent
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


def test_the_header_shows_the_recomputed_count(tmp_path):
    case = dataclasses.replace(synthetic_triage_case(write_rule(tmp_path)), fire_count=4)
    assert "Events matched by the rule: 4" in _case_prompt(case)


def test_the_header_leaves_the_line_out_when_there_is_no_count(tmp_path):
    case = dataclasses.replace(synthetic_triage_case(write_rule(tmp_path)), fire_count=None)
    assert "Events matched" not in _case_prompt(case)


def test_an_agent_run_refuses_cases_without_a_count(tmp_path):
    case = dataclasses.replace(synthetic_triage_case(write_rule(tmp_path)), fire_count=None)
    with pytest.raises(ValueError, match="score.fire_counts"):
        run_agent([case], ScriptedModel([]), "scripted", {"price": None},
                  run_dir=tmp_path / "run", store=FakeStore())


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


def test_the_counts_file_is_read_and_fingerprinted(tmp_path, monkeypatch):
    path = tmp_path / "fire-counts.json"
    monkeypatch.setattr(corpus, "FIRE_COUNTS", path)
    assert corpus.load_fire_counts() == {}
    assert corpus.fire_counts_digest() is None

    path.write_text(json.dumps({"counts": {"tp:r@c": 2}}))
    assert corpus.load_fire_counts() == {"tp:r@c": 2}
    assert len(corpus.fire_counts_digest()) == 64
