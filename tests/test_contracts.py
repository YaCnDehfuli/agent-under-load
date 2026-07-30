"""The verdict contract refuses what it should.

A verdict that cannot point at the event justifying it is not usable by an
analyst, who will go and look at the event anyway. The requirement is also a
defence: a model talked into "routine administrative activity" by text inside a
command line still has to name a record and quote a field, and the citation
control then checks that the quote is really there.

So these tests are about what the contract *rejects*. A contract that accepts
everything is decoration.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.contracts import (
    ActionRequest,
    EvidenceCitation,
    MissClass,
    MissVerdict,
    TriageResult,
    TriageVerdict,
    Verdict,
)

CITATION = EvidenceCitation(event_index=4, field="TargetImage",
                            quote="lsass.exe", supports="the target was LSASS")


def test_a_decisive_triage_verdict_must_cite():
    for verdict in (Verdict.TRUE_POSITIVE, Verdict.FALSE_POSITIVE):
        with pytest.raises(ValidationError, match="cites no event"):
            TriageVerdict(verdict=verdict, confidence=0.9, evidence=[],
                          rationale="because I say so")


def test_inconclusive_may_abstain_without_evidence():
    """Abstention is allowed, and the scorer counts it against recall.

    If abstaining were forbidden the model would be pushed into guessing; if it
    were free, always abstaining would be a perfect-precision strategy.
    """
    verdict = TriageVerdict(verdict=Verdict.INCONCLUSIVE, confidence=0.2,
                            evidence=[], rationale="not enough in the capture")
    assert verdict.verdict is Verdict.INCONCLUSIVE


def test_claims_about_specific_records_must_cite():
    for label in (MissClass.DETECTED, MissClass.MISS_TELEMETRY):
        with pytest.raises(ValidationError, match="cites no event"):
            MissVerdict(label=label, confidence=0.8, evidence=[], rationale="x")


def test_claims_about_the_whole_capture_need_no_citation():
    """miss-logic and out-of-scope are statements about an absence.

    Demanding a citation for "nothing in 50,000 events matched" would be
    demanding a citation for nothing in particular.
    """
    for label in (MissClass.MISS_LOGIC, MissClass.OUT_OF_SCOPE):
        assert MissVerdict(label=label, confidence=0.6, evidence=[],
                           rationale="x").label is label


def test_the_verdict_vocabulary_is_closed():
    with pytest.raises(ValidationError):
        TriageVerdict(verdict="probably_bad", confidence=0.5,  # type: ignore[arg-type]
                      evidence=[CITATION], rationale="x")
    with pytest.raises(ValidationError):
        MissVerdict(label="miss_logic", confidence=0.5,  # type: ignore[arg-type]
                    evidence=[], rationale="x")


def test_confidence_is_bounded():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValidationError):
            TriageVerdict(verdict=Verdict.TRUE_POSITIVE, confidence=bad,
                          evidence=[CITATION], rationale="x")


def test_rationale_is_required_and_bounded():
    with pytest.raises(ValidationError):
        TriageVerdict(verdict=Verdict.TRUE_POSITIVE, confidence=0.5,
                      evidence=[CITATION], rationale="")
    with pytest.raises(ValidationError):
        TriageVerdict(verdict=Verdict.TRUE_POSITIVE, confidence=0.5,
                      evidence=[CITATION], rationale="x" * 4001)


def test_extra_fields_are_refused():
    """A model inventing a field is a model whose output was not understood."""
    with pytest.raises(ValidationError):
        TriageVerdict(verdict=Verdict.TRUE_POSITIVE, confidence=0.5,
                      evidence=[CITATION], rationale="x",
                      severity_override="critical")  # type: ignore[call-arg]


def test_a_citation_must_be_specific_enough_to_check():
    with pytest.raises(ValidationError):
        EvidenceCitation(event_index=-1, field="Image", quote="x", supports="y")
    for missing in ({"field": ""}, {"quote": ""}, {"supports": ""}):
        with pytest.raises(ValidationError):
            EvidenceCitation(**{"event_index": 0, "field": "Image",
                                "quote": "x", "supports": "y", **missing})


def test_action_names_are_normalised():
    """So the authorization boundary compares like with like."""
    request = ActionRequest(action="  Isolate Host  ", target="MKT01",
                            justification="credential theft in progress")
    assert request.action == "isolate_host"


def test_reversibility_is_only_a_claim():
    request = ActionRequest(action="isolate_host", target="MKT01",
                            justification="x", claimed_reversible=True)
    assert request.claimed_reversible is True  # advisory; the boundary decides


def test_result_reports_whichever_label_the_task_produced():
    triage = TriageResult(case_id="c", verdict=TriageVerdict(
        verdict=Verdict.FALSE_POSITIVE, confidence=0.7, evidence=[CITATION],
        rationale="x"))
    assert triage.label == "false_positive"

    miss = TriageResult(case_id="c", miss_verdict=MissVerdict(
        label=MissClass.MISS_LOGIC, confidence=0.7, evidence=[], rationale="x"))
    assert miss.label == "miss-logic"

    assert TriageResult(case_id="c", rejections=["nope"]).label is None
