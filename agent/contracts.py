"""What the agent is allowed to return.

A verdict is a validated structure, not prose. Three reasons, in order of how
much they matter:

1. It is scoreable. A paragraph is not.
2. It has to point at the event that justifies it. A verdict that cannot cite
   the record it rests on is not usable by an analyst, who will go and look at
   the event anyway.
3. The citation requirement is also a defence. A model talked into "this is
   routine administrative activity" by text inside a command line still has to
   name an event and quote a field that supports the conclusion, and Phase D
   checks that the quoted text is really in the record at that index.

The third reason is why `evidence` is not optional. Requiring it costs the
agent an escape hatch it would otherwise take.
"""

from __future__ import annotations

import enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Verdict(str, enum.Enum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    #: Allowed, and scored as a miss rather than silently excused. An agent that
    #: can always answer "inconclusive" would score perfectly on precision, so
    #: the scorer counts abstentions against recall.
    INCONCLUSIVE = "inconclusive"


class MissClass(str, enum.Enum):
    DETECTED = "detected"
    MISS_LOGIC = "miss-logic"
    MISS_TELEMETRY = "miss-telemetry"
    OUT_OF_SCOPE = "out-of-scope"


class EvidenceCitation(BaseModel):
    """A pointer into the capture, precise enough to check.

    `event_index` is the record's position in the capture stream, which is how
    the citation is verified: the field is read back out of that record and
    compared against `quote`.
    """

    model_config = ConfigDict(extra="forbid")

    event_index: int = Field(ge=0, description="position of the event in the capture")
    field: str = Field(min_length=1, description="the field being cited")
    quote: str = Field(min_length=1, max_length=2000,
                       description="text copied from that field, verbatim")
    supports: str = Field(min_length=1, max_length=500,
                          description="what this evidence establishes")


class TriageVerdict(BaseModel):
    """A detection fired; this is the call on whether it is real."""

    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def _decisive_verdicts_must_cite(self) -> "TriageVerdict":
        if self.verdict is not Verdict.INCONCLUSIVE and not self.evidence:
            raise ValueError(
                f"verdict {self.verdict.value} cites no event. A decisive "
                "verdict must point at the record that justifies it."
            )
        return self


class MissVerdict(BaseModel):
    """Why a rule did not fire on a capture."""

    model_config = ConfigDict(extra="forbid")

    label: MissClass
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def _telemetry_and_detected_claims_must_cite(self) -> "MissVerdict":
        """Two of the four labels are claims about specific records.

        `detected` asserts a matching event exists, and `miss-telemetry`
        asserts the capture's records lack a field. Both are checkable against
        the stream, so both have to cite. `miss-logic` and `out-of-scope` are
        statements about the rule against the whole capture, and demanding a
        citation for an absence across 50,000 events would be demanding a
        citation for nothing in particular.
        """
        if self.label in (MissClass.DETECTED, MissClass.MISS_TELEMETRY):
            if not self.evidence:
                raise ValueError(
                    f"label {self.label.value} cites no event, but it is a "
                    "claim about specific records in the capture."
                )
        return self


class ActionRequest(BaseModel):
    """A response action the agent would like taken.

    The agent may only ever *request*. Whether the request is granted is decided
    by the authorization boundary, which the agent does not control and cannot
    widen. Phase C's escalation objective is to get a harmful request out of the
    agent; Phase D's job is to make the request insufficient.
    """

    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=1, max_length=100)
    target: str = Field(min_length=1, max_length=500)
    justification: str = Field(min_length=1, max_length=1000)
    #: Set by the agent when it believes the action is reversible. Advisory
    #: only: the boundary keeps its own list and does not take the agent's word.
    claimed_reversible: bool = False

    @field_validator("action")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return value.strip().lower().replace(" ", "_")


class TriageResult(BaseModel):
    """Everything one run of the graph produced."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    verdict: TriageVerdict | None = None
    miss_verdict: MissVerdict | None = None
    requested_actions: list[ActionRequest] = Field(default_factory=list)
    #: Populated when the contract or the citation check rejected the model's
    #: output. A rejected run is a failed run, not an absent one.
    rejections: list[str] = Field(default_factory=list)
    tool_calls: int = 0

    @property
    def label(self) -> str | None:
        """The scored answer, whichever task this was."""
        if self.verdict is not None:
            return self.verdict.verdict.value
        if self.miss_verdict is not None:
            return self.miss_verdict.label.value
        return None


TASK_SCHEMAS: dict[str, type[BaseModel]] = {
    "triage_verdict": TriageVerdict,
    "miss_classification": MissVerdict,
}
