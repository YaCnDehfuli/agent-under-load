"""No-LLM predictors over the same inputs, and the bar the agent must clear.

Two baselines, and they are not equally strong, for a reason worth stating
before either number is read.

**Miss classification is near-ceiling by construction.** The sibling's label is a
deterministic function of the rule and the telemetry: match count, prefilter
count, which required fields the capture recorded, which required constraints
anything satisfied, and whether the rule is named after a hacktool. A predictor
that re-derives those from the rule text and the event stream is reimplementing
the labeller, and it should score very high. That is not cheating — deriving them
is exactly the task, and the no-oracle rule only forbids being *handed* them —
but it does mean the bar here is close to the ceiling, and an agent losing to it
is losing to a reimplementation of the answer key rather than to a heuristic.

**Triage is not.** Whether a fire is a true positive depends on which corpus the
capture came from, which is deliberately *not* computable from the telemetry
after pseudonymisation. So the triage baseline is what it looks like: a hand-
written guess about whether the capture really shows LSASS credential theft. It
can be wrong, and it is the more informative of the two comparisons.

Both baselines read through the same `Toolbox` the agent uses, so neither gets an
input the agent lacks.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from typing import Any

from sigma.collection import SigmaCollection
from sigma.conditions import (
    ConditionAND,
    ConditionFieldEqualsValueExpression,
    ConditionNOT,
    ConditionOR,
)
from sigma.plugins import InstalledSigmaPlugins
from sigma.processing.resolver import ProcessingPipelineResolver

from agent import corpus
from agent.contracts import (
    EvidenceCitation,
    MissClass,
    MissVerdict,
    TriageResult,
    TriageVerdict,
    Verdict,
)
from agent.events import Event
from agent.tools import CaptureStore, Toolbox

#: Fields naming a binary. A required constraint on one of these that nothing in
#: the capture satisfies means the rule is looking for a tool that was not run,
#: which is a scope question rather than a rule defect. Access masks and call
#: traces are excluded on purpose: failing to match those *is* the detection
#: logic falling short, which is what miss-logic means.
IDENTITY_FIELDS = frozenset({
    "SourceImage", "Image", "ParentImage", "OriginalFileName", "ProcessName",
    "SourceProcessName", "Product", "Description", "Company",
})

#: Access masks that include the rights needed to read another process's memory
#: (PROCESS_VM_READ, or full control). Drawn from the tradecraft, not the corpus.
DUMP_ACCESS_MASKS = ("0x1fffff", "0x1010", "0x1410", "0x1418", "0x143a",
                     "0x147a", "0x1438", "0x1f1fff", "0x1f3fff")

#: Command-line and image markers for the published ways of reading LSASS.
DUMP_MARKERS = ("minidump", "comsvcs", "procdump", "-ma lsass", "sharpdump",
                "dumpert", "nanodump", "out-minidump", "createdump",
                "rundll32 c:\\windows\\system32\\comsvcs.dll")

_PIPELINE = None


def _pipeline():
    """The sysmon and windows-logsources pipelines, chained.

    Same pair the sibling uses, so a `process_access` rule gets its EventID 10
    and a Security rule gets its Channel. Running a rule through a pipeline it
    did not ask for produces a miss that says nothing about the rule.
    """
    global _PIPELINE
    if _PIPELINE is None:
        plugins = InstalledSigmaPlugins.autodiscover()
        resolver = ProcessingPipelineResolver(plugins.pipelines)
        _PIPELINE = resolver.resolve(["sysmon", "windows-logsources"])
    return _PIPELINE


# ---------------------------------------------------------------------------
# rule analysis
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4096)
def _matcher(value: str) -> re.Pattern[str]:
    """Compile one Sigma value into a full-match regex.

    pySigma applies `|contains`, `|startswith` and `|endswith` at parse time by
    rewriting the value with wildcards — `\\lsass.exe` with `|endswith` arrives
    here as `*\\lsass.exe` — so handling `*` and `?` covers all three and the
    modifier list is never consulted. Everything else is escaped, which matters
    because command lines are full of brackets and braces that a glob would
    treat as syntax.
    """
    out = []
    for character in value:
        if character == "*":
            out.append(".*")
        elif character == "?":
            out.append(".")
        else:
            out.append(re.escape(character))
    return re.compile("".join(out) + r"\Z", re.IGNORECASE | re.DOTALL)


@dataclasses.dataclass(frozen=True)
class Requirement:
    """A constraint on the rule's AND spine that has to hold for it to fire."""

    fields: frozenset[str]
    #: Literal values as pySigma produced them, wildcards included.
    values: tuple[str, ...]

    @property
    def label(self) -> str:
        return "/".join(sorted(self.fields))

    @property
    def is_identity(self) -> bool:
        return bool(self.fields) and self.fields <= IDENTITY_FIELDS

    def satisfied_by(self, event: Event) -> bool:
        """Approximate Sigma matching: enough for a baseline, and no more.

        Handles equality and wildcards case-insensitively, which after pySigma's
        rewriting covers the string modifiers too. It does not implement the
        whole specification — no `|re`, no `|base64offset`, no `|all` semantics
        across a list, no field-less keyword search — and the sibling repo owns
        conformance. Where this disagrees, the baseline is simply a weaker
        predictor, which is an honest thing for a baseline to be.
        """
        for field in self.fields:
            raw = event.get(field, None)
            if raw is None:
                continue
            haystack = str(raw)
            for value in self.values:
                if _matcher(value).match(haystack):
                    return True
        return False


@dataclasses.dataclass
class RuleAnalysis:
    event_ids: frozenset[int]
    channels: frozenset[str]
    requirements: tuple[Requirement, ...]
    tool_specific: bool
    parse_error: str = ""

    def prefilters(self, event: Event) -> bool:
        """Could this event even be a candidate for the rule?

        Channel comparison is substring-and-case-insensitive in both directions
        because captures spell the channel inconsistently — some carry the full
        `Microsoft-Windows-Sysmon/Operational`, some an abbreviation.
        """
        if self.event_ids and event.event_id not in self.event_ids:
            return False
        if self.channels:
            channel = event.channel.lower()
            if not channel:
                return False
            if not any(c in channel or channel in c
                       for c in (name.lower() for name in self.channels)):
                return False
        return True

    @property
    def detection_requirements(self) -> tuple[Requirement, ...]:
        """Requirements other than the prefilter itself."""
        return tuple(r for r in self.requirements
                     if not (r.fields & {"EventID", "Channel"}))


def analyse_rule(rule: corpus.RuleRef) -> RuleAnalysis:
    """Pull the AND spine, the prefilter and the tool-identity signal out.

    Only the AND spine counts. Anything under a NOT is a filter, and a filter
    that does not apply cannot explain a miss.
    """
    tool_specific = "hktl" in rule.path.stem.lower()
    try:
        collection = SigmaCollection.from_yaml(rule.text())
        parsed_rule = collection.rules[0]
        _pipeline().apply(parsed_rule)
    except Exception as exc:  # pragma: no cover - depends on upstream rule text
        return RuleAnalysis(frozenset(), frozenset(), (), tool_specific,
                            parse_error=f"{type(exc).__name__}: {exc}")

    requirements: list[Requirement] = []
    for condition in parsed_rule.detection.parsed_condition:
        try:
            tree = condition.parse()
        except Exception as exc:  # pragma: no cover
            return RuleAnalysis(frozenset(), frozenset(), (), tool_specific,
                                parse_error=f"{type(exc).__name__}: {exc}")
        requirements.extend(_and_spine(tree))

    event_ids: set[int] = set()
    channels: set[str] = set()
    for requirement in requirements:
        if "EventID" in requirement.fields:
            for value in requirement.values:
                try:
                    event_ids.add(int(value))
                except ValueError:
                    pass
        if "Channel" in requirement.fields:
            channels.update(requirement.values)

    return RuleAnalysis(frozenset(event_ids), frozenset(channels),
                        tuple(requirements), tool_specific)


def _and_spine(node: Any) -> list[Requirement]:
    out: list[Requirement] = []

    def leaf(expression: Any) -> tuple[str, str] | None:
        if not isinstance(expression, ConditionFieldEqualsValueExpression):
            return None
        # str() of a SigmaString carries the wildcards pySigma wrote in when it
        # applied |contains, |startswith or |endswith
        return (expression.field, str(expression.value))

    def walk(current: Any) -> None:
        if isinstance(current, ConditionAND):
            for argument in current.args:
                walk(argument)
            return
        if isinstance(current, ConditionNOT):
            return
        parsed = leaf(current)
        if parsed is not None:
            field, value = parsed
            out.append(Requirement(frozenset({field}), (value,)))
            return
        if isinstance(current, ConditionOR):
            fields: set[str] = set()
            values: list[str] = []
            stack = list(current.args)
            while stack:
                item = stack.pop()
                if isinstance(item, ConditionOR):
                    stack.extend(item.args)
                    continue
                parsed = leaf(item)
                if parsed is None:
                    return  # an OR containing something else constrains nothing
                field, value = parsed
                fields.add(field)
                values.append(value)
            if fields:
                out.append(Requirement(frozenset(fields), tuple(values)))

    walk(node)
    return out


# ---------------------------------------------------------------------------
# miss classification
# ---------------------------------------------------------------------------


class MissBaseline:
    """Re-derives the four-way label from the rule and the telemetry."""

    name = "heuristic-missclass"

    def __init__(self, store: CaptureStore | None = None):
        self.store = store or CaptureStore()

    def predict(self, case: corpus.MissCase) -> TriageResult:
        analysis = analyse_rule(case.rule)
        toolbox = Toolbox(case.rule, case.capture, store=self.store)
        events = self.store.load(case.capture)

        requirements = analysis.detection_requirements
        candidates = 0
        present = [0] * len(requirements)
        satisfied = [0] * len(requirements)
        matched = 0

        for event in events:
            if not analysis.prefilters(event):
                continue
            candidates += 1
            all_satisfied = True
            for index, requirement in enumerate(requirements):
                if any(event.present(field) for field in requirement.fields):
                    present[index] += 1
                if requirement.satisfied_by(event):
                    satisfied[index] += 1
                else:
                    all_satisfied = False
            if all_satisfied and requirements:
                matched += 1

        label, confidence, rationale, evidence = self._decide(
            analysis, requirements, candidates, present, satisfied, matched,
            events,
        )
        verdict = MissVerdict(label=label, confidence=confidence,
                              rationale=rationale, evidence=evidence)
        return TriageResult(case_id=case.case_id, miss_verdict=verdict,
                            tool_calls=len(toolbox.calls))

    def _decide(self, analysis, requirements, candidates, present, satisfied,
                matched, events):
        if analysis.parse_error:
            # a rule this baseline cannot read is answered with the majority
            # class and low confidence, rather than skipped
            return (MissClass.OUT_OF_SCOPE, 0.1,
                    f"rule did not parse: {analysis.parse_error}", [])

        if matched:
            index = next((e.index for e in events if analysis.prefilters(e)), 0)
            return (MissClass.DETECTED, 0.6,
                    f"{matched} events satisfy every requirement",
                    [EvidenceCitation(event_index=index, field="EventID",
                                      quote=str(events[index].event_id or ""),
                                      supports="a candidate event of the "
                                               "type the rule reads")])

        if candidates == 0:
            return (MissClass.MISS_TELEMETRY, 0.9,
                    "the capture holds no events of the type this rule reads",
                    [EvidenceCitation(event_index=0, field="EventID",
                                      quote=str(events[0].event_id or "")
                                            if events else "0",
                                      supports="the capture's events are of "
                                               "other types")]
                    if events else [])

        missing = [requirements[i].label for i, count in enumerate(present)
                   if count == 0]
        if missing:
            index = next((e.index for e in events if analysis.prefilters(e)), 0)
            return (MissClass.MISS_TELEMETRY, 0.8,
                    "field never recorded in this capture: "
                    + ", ".join(sorted(missing)),
                    [EvidenceCitation(event_index=index, field="EventID",
                                      quote=str(events[index].event_id or ""),
                                      supports="candidate event lacking the "
                                               "required field")])

        if analysis.tool_specific:
            return (MissClass.OUT_OF_SCOPE, 0.9,
                    "rule is written for one named hacktool", [])

        unsatisfied_identity = [requirements[i].label
                                for i, count in enumerate(satisfied)
                                if count == 0 and requirements[i].is_identity]
        if unsatisfied_identity:
            return (MissClass.OUT_OF_SCOPE, 0.7,
                    "requires a binary not present in this capture: "
                    + ", ".join(sorted(set(unsatisfied_identity))), [])

        return (MissClass.MISS_LOGIC, 0.6,
                f"{candidates} candidate events, none matched the rule", [])


# ---------------------------------------------------------------------------
# triage
# ---------------------------------------------------------------------------


class TriageBaseline:
    """Guesses whether a capture really shows LSASS credential theft.

    Deliberately a heuristic rather than a re-derivation, because the triage
    label is not computable from the telemetry: it depends on which corpus the
    capture came from, and pseudonymisation removed that signal on purpose.
    """

    name = "heuristic-triage"

    def __init__(self, store: CaptureStore | None = None,
                 threshold: int = 1):
        self.store = store or CaptureStore()
        self.threshold = threshold

    def predict(self, case: corpus.TriageCase) -> TriageResult:
        events = self.store.load(case.capture)
        hits: list[tuple[int, str, str, str]] = []

        for event in events:
            if event.event_id == 10:
                target = str(event.get("TargetImage", "") or "").lower()
                access = str(event.get("GrantedAccess", "") or "").lower()
                source = str(event.get("SourceImage", "") or "").lower()
                if "lsass.exe" in target and access in DUMP_ACCESS_MASKS \
                        and "\\wbem\\wmiprvse.exe" not in source:
                    hits.append((event.index, "GrantedAccess", access,
                                 "a handle to LSASS carrying memory-read rights"))
                    continue
            for field in ("CommandLine", "ParentCommandLine", "Image",
                          "TargetFilename", "ScriptBlockText"):
                value = str(event.get(field, "") or "").lower()
                if not value:
                    continue
                marker = next((m for m in DUMP_MARKERS if m in value), None)
                if marker and ("lsass" in value or "minidump" in value
                               or marker in ("procdump", "nanodump",
                                             "sharpdump", "dumpert")):
                    hits.append((event.index, field, marker,
                                 "a published means of reading LSASS memory"))
                    break
            if len(hits) >= 5:
                break

        if len(hits) >= self.threshold:
            index, field, quote, supports = hits[0]
            verdict = TriageVerdict(
                verdict=Verdict.TRUE_POSITIVE,
                confidence=min(0.9, 0.5 + 0.1 * len(hits)),
                evidence=[EvidenceCitation(event_index=index, field=field,
                                           quote=quote, supports=supports)],
                rationale=f"{len(hits)} indicators of LSASS credential access",
            )
        else:
            first = events[0].index if events else 0
            verdict = TriageVerdict(
                verdict=Verdict.FALSE_POSITIVE,
                confidence=0.5,
                evidence=[EvidenceCitation(
                    event_index=first, field="EventID",
                    quote=str(events[0].event_id or "") if events else "0",
                    supports="no LSASS memory-read indicator anywhere in the "
                             "capture")],
                rationale="no indicator of LSASS credential access found",
            )
        return TriageResult(case_id=case.case_id, verdict=verdict)
