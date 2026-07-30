"""The tools, and the declarations they make about themselves.

The provenance ceiling on each tool is used to label prompts, to populate the
audit log and to reason about which controls can help. A declaration that does
not match behaviour would quietly invalidate all three, so it is tested rather
than trusted.
"""

from __future__ import annotations

import pytest

from agent import corpus
from agent.events import Ingestion
from agent.provenance import Provenance
from agent.tools import CONTRACTS, CaptureStore, Toolbox

SYNTHETIC = [
    {"EventID": 1, "Channel": "Sysmon", "UtcTime": "t0",
     "Image": "C:\\Windows\\System32\\cmd.exe", "CommandLine": "cmd /c whoami",
     "Hostname": "H"},
    {"EventID": 10, "Channel": "Sysmon", "UtcTime": "t1",
     "SourceImage": "C:\\tools\\procdump.exe",
     "TargetImage": "C:\\Windows\\system32\\lsass.exe",
     "GrantedAccess": "0x1fffff", "Hostname": "H"},
    {"EventID": 10, "Channel": "Sysmon", "UtcTime": "t2",
     "SourceImage": "C:\\Windows\\explorer.exe",
     "TargetImage": "C:\\Windows\\system32\\svchost.exe",
     "GrantedAccess": "0x1010", "Hostname": "H"},
]


class FakeStore(CaptureStore):
    """A store over a handful of hand-written events.

    The unit suite must pass with no corpus on disk, and these tests are about
    the tools rather than about the data.
    """

    def __init__(self, raws=None):
        super().__init__()
        from agent.events import Event
        self._events = [Event(raw, i) for i, raw in enumerate(raws or SYNTHETIC)]

    def load(self, capture):
        return self._events


@pytest.fixture
def toolbox(tmp_path, synthetic_capture):
    rule_path = tmp_path / "rule.yml"
    rule_path.write_text(
        "title: Suspicious LSASS Access\n"
        "id: abc\n"
        "status: test\n"
        "logsource:\n  category: process_access\n"
        "detection:\n"
        "  selection:\n"
        "    TargetImage|endswith: '\\lsass.exe'\n"
        "    GrantedAccess: '0x1fffff'\n"
        "  condition: selection\n"
        "level: high\n"
        "extra_prose: this key should not reach the prompt\n"
    )
    rule = corpus.RuleRef(id="abc", title="Suspicious LSASS Access", level="high",
                          path=rule_path, source="sigmahq", selected_by="tag")
    return Toolbox(rule=rule, capture=synthetic_capture, store=FakeStore())


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


def test_query_events_caps_the_limit(toolbox):
    result = toolbox.call("query_events", {"limit": 10_000})
    assert result.returned <= 20


def test_query_events_survives_a_nonsense_limit(toolbox):
    assert not toolbox.call("query_events", {"limit": "lots"}).error


def test_ingestion_mode_changes_what_a_query_returns(tmp_path, synthetic_capture):
    raws = [{"EventID": 1, "Channel": "Sysmon", "CommandLine": "whoami",
             "Image": "C:\\x.exe", "Message": "Process Create:\r\nrendered blob"}]
    rule_path = tmp_path / "r.yml"
    rule_path.write_text("title: t\ndetection:\n  selection:\n    a: b\n"
                         "  condition: selection\n")
    rule = corpus.RuleRef(id="r", title="t", level="low", path=rule_path,
                          source="sigmahq", selected_by="tag")

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
    rule_path = tmp_path / "r.yml"
    rule_path.write_text("title: t\ndetection:\n  selection:\n    a: b\n"
                         "  condition: selection\n")
    rule = corpus.RuleRef(id="r", title="t", level="low", path=rule_path,
                          source="sigmahq", selected_by="tag")
    box = Toolbox(rule=rule, capture=synthetic_capture, store=FakeStore(),
                  allowed={"count_events"})
    assert "not available" in box.call("query_events", {}).error
    assert not box.call("count_events", {}).error


def test_every_call_is_recorded_for_the_audit(toolbox):
    toolbox.call("count_events", {})
    toolbox.call("query_events", {})
    assert [c.name for c in toolbox.calls] == ["count_events", "query_events"]


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
