"""The evidence conditions: what each one gives the agent, and what it withholds."""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import pytest

from agent import corpus
from agent.baseline import ConstantBaseline, RulePriorBaseline
from agent.contracts import (
    EvidenceCitation,
    ForcedTriageVerdict,
    TriageVerdict,
    UncitedTriageVerdict,
    Verdict,
)
from agent.events import Event
from agent.graph import (
    SYSTEM_TRIAGE,
    AgentConfig,
    Condition,
    Control,
    TriageGraph,
    system_prompt,
)
from agent.models import ModelReply, ScriptedModel, ToolCall, _schema_of
from agent.pseudonymise import capture_handle
from agent.tools import CaptureStore
from score.pairing import donors
from score.run import run_agent
from score.ledger import RunDir
from tests.support import SYNTHETIC_EVENTS, FakeStore, synthetic_triage_case, write_rule

ANSWER = TriageVerdict(
    verdict=Verdict.TRUE_POSITIVE, confidence=0.9,
    evidence=[EvidenceCitation(event_index=0, field="CommandLine",
                               quote="whoami", supports="x")],
    rationale="x")


class Recording(ScriptedModel):
    """A scripted model that also keeps the tools and schema it was offered."""

    def __init__(self, script):
        super().__init__(script)
        self.tools: list[list[str]] = []
        self.schemas: list[type] = []

    def respond(self, *, system, messages, tools, schema):
        self.tools.append([t.name for t in tools])
        self.schemas.append(schema)
        return super().respond(system=system, messages=messages, tools=tools,
                               schema=schema)


class ByCapture(CaptureStore):
    """Different events per capture, so a test can tell which capture was read."""

    def __init__(self, events: dict[str, list[dict]]):
        super().__init__()
        self._events = {cid: [Event(raw, i) for i, raw in enumerate(raws)]
                        for cid, raws in events.items()}

    def load(self, capture):
        return self._events[capture.id]


def _capture(cid: str) -> corpus.CaptureRef:
    return corpus.CaptureRef(id=cid, group="atomic", archive=Path("nowhere.zip"))


def _cases(tmp_path, layout: dict[str, tuple[str, int]]):
    """{capture_id: (label, number of cases)} -> cases, rule ids distinct per case."""
    base = synthetic_triage_case(write_rule(tmp_path))
    out = []
    for cid, (label, n) in layout.items():
        for i in range(n):
            out.append(dataclasses.replace(
                base, case_id=f"{label[:2]}:{cid}:{i}", capture=_capture(cid),
                truth=label, rule=dataclasses.replace(base.rule, id=f"rule-{i}")))
    return out


LAYOUT = {"tp1": ("true_positive", 3), "tp2": ("true_positive", 2),
          "fp1": ("false_positive", 2), "fp2": ("false_positive", 2),
          "fp3": ("false_positive", 1)}


# -- tools and prompts ---------------------------------------------------------


@pytest.mark.parametrize("condition,tools", [
    (Condition.ALERT_ONLY, []),
    (Condition.RULE_ONLY, ["lookup_rule", "lookup_attack_technique"]),
])
def test_no_evidence_conditions_offer_only_their_tools(tmp_path, condition, tools):
    model = Recording([ModelReply(answer=UncitedTriageVerdict(
        verdict=Verdict.TRUE_POSITIVE, confidence=0.5, rationale="x"))])
    graph = TriageGraph(model, AgentConfig(condition=condition), store=FakeStore())
    result = graph.run(synthetic_triage_case(write_rule(tmp_path)))
    assert sorted(model.tools[0]) == sorted(tools)
    assert model.schemas[0] is UncitedTriageVerdict
    assert result.label == "true_positive"


def test_the_reference_run_keeps_every_tool_and_the_citation_rule(tmp_path):
    model = Recording([ModelReply(answer=ANSWER)])
    TriageGraph(model, AgentConfig(), store=FakeStore()).run(
        synthetic_triage_case(write_rule(tmp_path)))
    assert {"query_events", "count_events", "describe_capture"} <= set(model.tools[0])
    assert model.schemas[0] is TriageVerdict


def test_the_reference_prompt_is_unchanged():
    assert system_prompt("triage_verdict", AgentConfig()) == SYSTEM_TRIAGE
    assert "query the capture's events" in SYSTEM_TRIAGE


@pytest.mark.parametrize("condition", [Condition.ALERT_ONLY, Condition.RULE_ONLY])
def test_a_no_evidence_prompt_changes_only_the_investigation_sentences(condition):
    reference = system_prompt("triage_verdict", AgentConfig()).splitlines()
    variant = system_prompt("triage_verdict", AgentConfig(condition=condition)).splitlines()
    assert len(reference) == len(variant)
    changed = [i for i, (a, b) in enumerate(zip(reference, variant)) if a != b]
    assert len(changed) == 2  # what can be looked at, and what can be cited
    assert "query the capture's events" not in "\n".join(variant)


def test_controls_still_apply_on_top_of_a_condition():
    config = AgentConfig(condition=Condition.ALERT_ONLY,
                         controls=frozenset({Control.PROVENANCE_TAGS}))
    assert "adversary-writable" in system_prompt("triage_verdict", config)


def test_the_uncited_schema_is_only_the_citation_rule_relaxed():
    assert UncitedTriageVerdict(verdict="true_positive", confidence=0.5, rationale="x")
    with pytest.raises(ValueError):
        TriageVerdict(verdict="true_positive", confidence=0.5, rationale="x")
    with pytest.raises(ValueError):  # the rest of the contract still holds
        UncitedTriageVerdict(verdict="maybe", confidence=0.5, rationale="x")


# -- the forced-choice probe -------------------------------------------------------


def test_the_forced_probe_has_no_tools_and_no_inconclusive(tmp_path):
    model = Recording([ModelReply(answer=ForcedTriageVerdict(
        verdict=Verdict.FALSE_POSITIVE, confidence=0.6, rationale="x"))])
    graph = TriageGraph(model, AgentConfig(condition=Condition.ALERT_ONLY_FORCED),
                        store=FakeStore())
    result = graph.run(synthetic_triage_case(write_rule(tmp_path)))
    assert model.tools[0] == []
    assert model.schemas[0] is ForcedTriageVerdict
    assert result.label == "false_positive"


def test_the_forced_schema_offers_only_the_two_labels():
    offered = _schema_of(ForcedTriageVerdict)["properties"]["verdict"]["enum"]
    assert sorted(offered) == ["false_positive", "true_positive"]
    assert ForcedTriageVerdict(verdict="true_positive", confidence=0.5, rationale="x")
    with pytest.raises(ValueError):
        ForcedTriageVerdict(verdict="inconclusive", confidence=0.5, rationale="x")


def test_an_inconclusive_in_the_forced_probe_is_rejected_and_repaired(tmp_path):
    bad = ModelReply(invalid="verdict: input should be 'true_positive' or "
                             "'false_positive'")
    good = ModelReply(answer=ForcedTriageVerdict(
        verdict=Verdict.TRUE_POSITIVE, confidence=0.5, rationale="x"))
    graph = TriageGraph(ScriptedModel([bad, good]),
                        AgentConfig(condition=Condition.ALERT_ONLY_FORCED),
                        store=FakeStore())
    result = graph.run(synthetic_triage_case(write_rule(tmp_path)))
    assert result.label == "true_positive"
    assert result.rejections


def test_the_forced_prompt_differs_from_alert_only_in_one_line():
    plain = system_prompt("triage_verdict",
                          AgentConfig(condition=Condition.ALERT_ONLY)).splitlines()
    forced = system_prompt("triage_verdict",
                           AgentConfig(condition=Condition.ALERT_ONLY_FORCED)).splitlines()
    changed = [b for a, b in zip(plain, forced) if a != b]
    assert len(plain) == len(forced) and len(changed) == 1
    assert "Inconclusive is not available" in changed[0]


# -- mismatched evidence -----------------------------------------------------------


EVENTS = {
    "own": [{"EventID": 1, "CommandLine": "own capture", "Image": "a.exe"}],
    "donor": [{"EventID": 1, "CommandLine": "donor capture", "Image": "b.exe"}],
}


def _mismatched(tmp_path, script, controls=frozenset()):
    case = dataclasses.replace(synthetic_triage_case(write_rule(tmp_path)),
                               capture=_capture("own"))
    model = Recording(script)
    graph = TriageGraph(model, AgentConfig(condition=Condition.MISMATCH_CROSS,
                                           controls=controls),
                        store=ByCapture(EVENTS))
    return case, model, graph


def test_a_mismatched_run_reads_the_donor_s_events(tmp_path):
    case, model, graph = _mismatched(tmp_path, [
        ModelReply(tool_calls=[ToolCall(name="query_events", arguments={},
                                        call_id="c1")]),
        ModelReply(answer=ANSWER)])
    graph.run(case, donor=_capture("donor"))
    seen = "\n".join(m.content for m in model.seen[-1][1])
    assert "donor capture" in seen and "own capture" not in seen
    started = graph.audit.of_kind("run_started")[0].payload
    assert started["evidence_capture_id"] == "donor"
    assert started["capture_id"] == "own"


def test_the_alert_names_the_donor_s_handle_and_nothing_else_about_it(tmp_path):
    case, model, graph = _mismatched(tmp_path, [ModelReply(answer=ANSWER)])
    graph.run(case, donor=_capture("donor"))
    alert = model.seen[0][1][0].content
    assert capture_handle("donor") in alert
    assert capture_handle("own") not in alert
    for leaked in ("donor", "own", "true_positive", "false_positive"):
        assert leaked not in alert


def test_citations_are_checked_against_the_donor(tmp_path):
    donor_quote = TriageVerdict(verdict=Verdict.TRUE_POSITIVE, confidence=0.9,
                                evidence=[EvidenceCitation(
                                    event_index=0, field="CommandLine",
                                    quote="donor capture", supports="x")],
                                rationale="x")
    own_quote = TriageVerdict(verdict=Verdict.TRUE_POSITIVE, confidence=0.9,
                              evidence=[EvidenceCitation(
                                  event_index=0, field="CommandLine",
                                  quote="own capture", supports="x")],
                              rationale="x")
    citing = frozenset({Control.ENFORCED_CITATION})

    case, _, graph = _mismatched(tmp_path, [ModelReply(answer=donor_quote)], citing)
    assert graph.run(case, donor=_capture("donor")).label == "true_positive"

    case, _, graph = _mismatched(tmp_path, [ModelReply(answer=own_quote)] * 2, citing)
    assert graph.run(case, donor=_capture("donor")).label is None


def test_a_mismatch_condition_needs_a_donor_and_the_others_refuse_one(tmp_path):
    case = synthetic_triage_case(write_rule(tmp_path))
    with pytest.raises(ValueError, match="needs"):
        TriageGraph(ScriptedModel([]), AgentConfig(condition=Condition.MISMATCH_SAME),
                    store=FakeStore()).run(case)
    with pytest.raises(ValueError, match="takes no"):
        TriageGraph(ScriptedModel([]), AgentConfig(), store=FakeStore()).run(
            case, donor=_capture("x"))


# -- pairing -----------------------------------------------------------------------


@pytest.mark.parametrize("stratum", ["cross", "same"])
def test_pairing_keeps_the_stratum_and_never_pairs_a_capture_with_itself(tmp_path, stratum):
    cases = _cases(tmp_path, LAYOUT)
    pairs = donors(cases, stratum, seed=0)
    labels = {c.capture.id: c.truth for c in cases}
    for case in cases:
        donor, donor_label = pairs[case.case_id]
        assert donor.id != case.capture.id
        assert donor_label == labels[donor.id]
        assert (donor_label == case.truth) == (stratum == "same")


def test_pairing_is_seeded(tmp_path):
    cases = _cases(tmp_path, LAYOUT)
    first = {k: v[0].id for k, v in donors(cases, "cross", seed=3).items()}
    again = {k: v[0].id for k, v in donors(cases, "cross", seed=3).items()}
    other = {k: v[0].id for k, v in donors(cases, "cross", seed=4).items()}
    assert first == again
    assert first != other or len(set(first.values())) == 1


def test_pairing_spreads_the_donors_evenly(tmp_path):
    cases = _cases(tmp_path, LAYOUT)
    tp_cases = [c for c in cases if c.truth == "true_positive"]  # 5 cases, 3 FP donors
    uses = Counter(donors(cases, "cross", seed=0)[c.case_id][0].id for c in tp_cases)
    assert max(uses.values()) - min(uses.values()) <= 1


def test_a_capture_with_two_labels_is_refused(tmp_path):
    cases = _cases(tmp_path, {"x": ("true_positive", 1), "y": ("false_positive", 1)})
    cases.append(dataclasses.replace(cases[1], case_id="odd", capture=_capture("x")))
    with pytest.raises(ValueError, match="both labels"):
        donors(cases, "cross", seed=0)


def test_a_mismatch_run_records_its_donors(tmp_path):
    cases = _cases(tmp_path, LAYOUT)
    # every capture holds the procdump/lsass record the synthetic rule matches
    events = {cid: [SYNTHETIC_EVENTS[1]] for cid in LAYOUT}
    model = ScriptedModel(lambda _: ModelReply(answer=ANSWER))
    meta = run_agent(cases, model, "scripted", {"price": None},
                     run_dir=tmp_path / "run",
                     config=AgentConfig(condition=Condition.MISMATCH_CROSS),
                     store=ByCapture(events), seed=0)
    assert meta["condition"] == "mismatch-cross" and meta["seed"] == 0
    assert len(meta["pairing_sha256"]) == 64
    for record in RunDir(tmp_path / "run").records():
        assert record["donor"] and record["donor"] != record["capture"]
        assert record["donor_truth"] != record["truth"]
        assert record["rule_fires_on_donor"] is True


# -- no-evidence baselines -------------------------------------------------------------


def test_the_constant_baseline_says_one_thing(tmp_path):
    cases = _cases(tmp_path, LAYOUT)
    labels = {ConstantBaseline("false_positive").predict(c).label for c in cases}
    assert labels == {"false_positive"}


def test_the_rule_prior_never_learns_from_the_case_s_own_capture(tmp_path):
    base = synthetic_triage_case(write_rule(tmp_path))

    def case(cid, capture, label, rule):
        return dataclasses.replace(base, case_id=cid, capture=_capture(capture),
                                   truth=label,
                                   rule=dataclasses.replace(base.rule, id=rule))

    cases = [
        # rule A: two FP alerts on the case's own capture, one TP elsewhere
        case("a1", "own", "false_positive", "A"),
        case("a2", "own", "false_positive", "A"),
        case("a3", "other", "true_positive", "A"),
    ]
    prior = RulePriorBaseline(cases)
    assert prior.predict(cases[0]).label == "true_positive"


def test_the_rule_prior_falls_back_to_other_captures_when_the_rule_is_unseen(tmp_path):
    base = synthetic_triage_case(write_rule(tmp_path))
    cases = [dataclasses.replace(base, case_id=cid, capture=_capture(cap), truth=lab,
                                 rule=dataclasses.replace(base.rule, id=rule))
             for cid, cap, lab, rule in [("x", "c1", "true_positive", "only-here"),
                                         ("y", "c2", "false_positive", "B"),
                                         ("z", "c3", "false_positive", "C")]]
    assert RulePriorBaseline(cases).predict(cases[0]).label == "false_positive"


# -- the technique question --------------------------------------------------------


def test_the_technique_question_changes_only_what_a_true_positive_is():
    reference = system_prompt("triage_verdict", AgentConfig()).splitlines()
    variant = system_prompt("triage_verdict",
                            AgentConfig(condition=Condition.TECHNIQUE_QUESTION)).splitlines()
    changed = [b for a, b in zip(reference, variant) if a != b]
    assert len(reference) == len(variant) and len(changed) == 1
    assert "T1003.001" in changed[0] and "T1003.001" not in "\n".join(reference)


def test_the_technique_question_is_otherwise_a_reference_run(tmp_path):
    model = Recording([ModelReply(answer=ANSWER)])
    TriageGraph(model, AgentConfig(condition=Condition.TECHNIQUE_QUESTION),
                store=FakeStore()).run(synthetic_triage_case(write_rule(tmp_path)))
    assert {"query_events", "count_events", "describe_capture"} <= set(model.tools[0])
    assert model.schemas[0] is TriageVerdict


def test_the_log_records_what_an_accepted_verdict_cites(tmp_path):
    graph = TriageGraph(ScriptedModel([ModelReply(answer=ANSWER)]), AgentConfig(),
                        store=FakeStore())
    graph.run(synthetic_triage_case(write_rule(tmp_path)))
    (entry,) = graph.audit.of_kind("verdict_evidence")
    assert entry.payload["cited"] == [
        {"event_index": 0, "field": "CommandLine", "event_id": 1}]
