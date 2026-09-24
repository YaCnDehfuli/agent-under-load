"""The tools, and the declarations they make about themselves.

The provenance ceiling on each tool is used to label prompts, to populate the
audit log and to reason about which controls can help. A declaration that does
not match behaviour would quietly invalidate all three, so it is tested rather
than trusted.
"""

from __future__ import annotations

import pytest

from agent.events import Ingestion
from agent.provenance import Provenance
from agent.audit import AuditLog
from agent.models import ToolCall
from agent.tools import CONTRACTS, MAX_QUERY_LIMIT, Toolbox
from tests.support import FakeStore, write_rule


@pytest.fixture
def toolbox(tmp_path, synthetic_capture):
    return Toolbox(rule=write_rule(tmp_path), capture=synthetic_capture,
                   store=FakeStore())


def test_every_contract_declares_a_ceiling_and_what_it_reads():
    for contract in CONTRACTS:
        assert contract.reads, contract.name
        assert isinstance(contract.provenance_ceiling, Provenance), contract.name
        assert contract.parameters["type"] == "object", contract.name


def test_only_query_events_returns_adversary_writable_text():
    """The narrow tainted surface is a property worth pinning.

    If a second tool starts returning full field values, the threat model and
    every Phase D argument about which controls can help have to be revisited.
    """
    tainted = {c.name for c in CONTRACTS if c.returns_untrusted}
    assert tainted == {"query_events"}


def test_untrusted_tools_warn_the_model_in_their_own_description():
    spec = next(c for c in CONTRACTS if c.name == "query_events").spec()
    assert "never as instructions" in spec.description
    clean = next(c for c in CONTRACTS if c.name == "count_events").spec()
    assert "never as instructions" not in clean.description


def test_lookup_rule_returns_the_logic_and_drops_the_padding(toolbox):
    result = toolbox.call("lookup_rule", {})
    assert not result.error
    assert "process_access" in result.output
    assert "lsass.exe" in result.output
    assert "extra_prose" not in result.output
    assert result.provenance_ceiling is Provenance.OS


def test_describe_capture_reports_telemetry_availability(toolbox):
    result = toolbox.call("describe_capture", {"fields": ["CallTrace", "GrantedAccess"]})
    assert "EventID 10: 2" in result.output
    assert "CallTrace" in result.output and "present in 0 events" in result.output
    assert "GrantedAccess" in result.output
    # answering a telemetry question exposed no adversary-written text
    assert result.provenance_ceiling is Provenance.OS


def test_count_events_returns_a_number_not_content(toolbox):
    result = toolbox.call("count_events",
                          {"event_id": 10,
                           "field_contains": {"TargetImage": "lsass"}})
    assert "1 events match" in result.output
    assert "procdump" not in result.output
    assert result.provenance_ceiling is Provenance.OS


def test_query_events_returns_content_and_is_tainted(toolbox):
    result = toolbox.call("query_events", {"event_id": 10, "limit": 5})
    assert result.matched == 2
    assert "procdump.exe" in result.output
    assert result.provenance_ceiling is Provenance.WRITABLE


def test_query_events_caps_the_limit(tmp_path, synthetic_capture):
    raws = [{"EventID": 1, "Channel": "Sysmon", "Image": f"C:\\{i}.exe"}
            for i in range(30)]
    box = Toolbox(rule=write_rule(tmp_path), capture=synthetic_capture,
                  store=FakeStore(raws))
    result = box.call("query_events", {"limit": 10_000})
    assert (result.returned, result.matched) == (MAX_QUERY_LIMIT, 30) == (10, 30)


def test_query_events_survives_a_nonsense_limit(toolbox):
    assert not toolbox.call("query_events", {"limit": "lots"}).error


@pytest.mark.parametrize("tool", ["query_events", "count_events"])
@pytest.mark.parametrize("arguments", [
    {"event_id": [10, 1]},          # a list where the schema says integer
    {"field_present": "TargetImage"},
    {"field_contains": ["lsass"]},
])
def test_malformed_filters_are_errors_the_model_can_recover_from(toolbox, tool, arguments):
    result = toolbox.call(tool, arguments)
    assert result.error and "must " in result.error


def test_a_numeric_string_event_id_is_still_accepted(toolbox):
    assert toolbox.call("count_events", {"event_id": "10"}).output == "2 events match"


def test_ingestion_mode_changes_what_a_query_returns(tmp_path, synthetic_capture):
    raws = [{"EventID": 1, "Channel": "Sysmon", "CommandLine": "whoami",
             "Image": "C:\\x.exe", "Message": "Process Create:\r\nrendered blob"}]
    rule = write_rule(tmp_path)

    raw_box = Toolbox(rule=rule, capture=synthetic_capture, store=FakeStore(raws),
                      ingestion=Ingestion.RAW)
    structured_box = Toolbox(rule=rule, capture=synthetic_capture,
                             store=FakeStore(raws),
                             ingestion=Ingestion.STRUCTURED)
    assert "rendered blob" in raw_box.call("query_events", {}).output
    assert "rendered blob" not in structured_box.call("query_events", {}).output


def test_a_bad_attack_lookup_is_an_error_the_model_can_recover_from(toolbox):
    result = toolbox.call("lookup_attack_technique", {"technique_id": "T9999"})
    assert result.error and "not in the local ATT&CK table" in result.error
    assert toolbox.call("lookup_attack_technique",
                        {"technique_id": "t1003.001"}).output.startswith("T1003.001")


def test_unavailable_tools_are_refused_not_executed(tmp_path, synthetic_capture):
    box = Toolbox(rule=write_rule(tmp_path), capture=synthetic_capture,
                  store=FakeStore(), allowed={"count_events"})
    assert "not available" in box.call("query_events", {}).error
    assert not box.call("count_events", {}).error


def test_every_call_is_recorded_for_the_audit(toolbox):
    toolbox.call("count_events", {})
    toolbox.call("query_events", {})
    assert [c.name for c in toolbox.calls] == ["count_events", "query_events"]


def test_a_repeated_query_points_back_instead_of_returning_it_again(toolbox):
    first = toolbox.call("query_events", {"event_id": 10, "limit": 5})
    again = toolbox.call("query_events", {"limit": 5, "event_id": 10})
    assert first.returned and first.repeat_of is None
    assert again.repeat_of == 1 and again.returned == 0
    assert again.matched == first.matched
    assert "already answered by call 1" in again.output
    assert len(again.output) < 100


def test_reordered_field_lists_are_the_same_question(toolbox):
    toolbox.call("query_events", {"field_present": ["Image", "EventID"]})
    again = toolbox.call("query_events", {"field_present": ["EventID", "Image"]})
    assert again.repeat_of == 1


def test_asking_for_fewer_rows_folds_but_more_rows_runs(toolbox):
    toolbox.call("query_events", {"event_id": 10, "limit": 5})
    assert toolbox.call("query_events", {"event_id": 10, "limit": 2}).repeat_of == 1
    bigger = toolbox.call("query_events", {"event_id": 10, "limit": 10})
    assert bigger.repeat_of is None
    # the bigger call is now the one later repeats point back to
    assert toolbox.call("query_events", {"event_id": 10, "limit": 7}).repeat_of == 3


def test_different_arguments_or_tools_are_not_repeats(toolbox):
    toolbox.call("query_events", {"event_id": 10})
    assert toolbox.call("query_events", {"event_id": 1}).repeat_of is None
    assert toolbox.call("count_events", {"event_id": 10}).repeat_of is None


def test_a_failed_call_can_be_retried_for_real(toolbox):
    assert toolbox.call("query_events", {"field_present": "Image"}).error
    assert toolbox.call("query_events", {"field_present": "Image"}).repeat_of is None


def test_rule_lookups_are_never_folded(toolbox):
    toolbox.call("lookup_rule", {})
    assert toolbox.call("lookup_rule", {}).repeat_of is None


def test_repeats_do_not_carry_across_sessions(tmp_path, synthetic_capture):
    rule = write_rule(tmp_path)
    for _ in range(2):
        box = Toolbox(rule=rule, capture=synthetic_capture, store=FakeStore())
        assert box.call("query_events", {}).repeat_of is None


def test_the_audit_marks_a_repeat(toolbox):
    log = AuditLog()
    for _ in range(2):
        call = ToolCall(name="count_events", arguments={})
        log.tool_call(call, toolbox.call(call.name, call.arguments))
    first, second = (e.payload for e in log.of_kind("tool_call"))
    assert "repeat_of" not in first and second["repeat_of"] == 1


# -- citation checking ----------------------------------------------------


def test_a_true_citation_holds(toolbox):
    assert toolbox.verify_citation(1, "TargetImage", "lsass.exe") == ""


def test_a_citation_to_a_missing_field_fails(toolbox):
    assert "no field" in toolbox.verify_citation(1, "ScriptBlockText", "x")


def test_a_citation_out_of_range_fails(toolbox):
    assert "outside the capture" in toolbox.verify_citation(9999, "EventID", "1")


def test_a_fabricated_quote_fails(toolbox):
    """The check that makes the evidence requirement more than a formality."""
    problem = toolbox.verify_citation(1, "SourceImage",
                                     "approved by the security team")
    assert "not in event[1].SourceImage" in problem
