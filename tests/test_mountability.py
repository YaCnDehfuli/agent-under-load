"""The mountability count: which placements have anywhere to sit, with no model."""

from __future__ import annotations

from attack.inject import load_payloads
from attack.mountability import mountability
from tests.support import FakeStore, synthetic_triage_case, write_proc_creation_rule


def test_counts_are_mountable_over_attempted_per_field_and_strategy(tmp_path):
    case = synthetic_triage_case(write_proc_creation_rule(tmp_path))
    payloads = load_payloads()
    result = mountability([case], FakeStore(), payloads)
    assert result["attempts"] == result["placements"] == sum(
        len(p.placements) for p in payloads)
    assert 0 < result["mountable"] < result["attempts"]
    assert sum(m for m, _ in result["by_field"].values()) == result["mountable"]
    assert sum(n for _, n in result["by_strategy"].values()) == result["attempts"]
    # the synthetic capture has command lines but no script blocks
    assert result["by_field"]["CommandLine"][0] > 0
    assert result["by_field"]["ScriptBlockText"][0] == 0
