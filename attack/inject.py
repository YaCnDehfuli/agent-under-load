"""Place a payload where an adversary could actually have put it.

Three constraints, all enforced here rather than left to whoever writes the next
payload:

**Only adversary-writable fields.** `provenance.require_injectable` raises on
anything else. Injecting into `GrantedAccess` would produce an impressive success
rate for an attack nobody can mount.

**Only fields the event already has.** A payload is appended to an existing
value, never used to invent a field. A `ScriptBlockText` on a process-access
record is telemetry no host emits, and an agent reacting to that impossibility
is not being prompt-injected, it is noticing a forged record.

**The rendered `Message` stays consistent.** Sysmon builds `Message` by
concatenating the other fields. Writing `CommandLine` and leaving `Message`
describing the old value produces an internally contradictory record, and the
contradiction rather than the payload is what the agent would respond to. So the
same substitution is applied inside `Message`.

Which events get the payload is a question about realism too. For a suppression
attack on a true positive, the events the rule matched *are* the adversary's own
events — that is why the rule fired — so those are the targets. The adversary
cannot write another process's command line, and this module does not pretend
otherwise.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from agent import corpus
from agent.baseline import RuleAnalysis, analyse_rule
from agent.events import Event
from agent.provenance import require_injectable
from agent.tools import CaptureStore

PAYLOAD_FILE = Path(__file__).resolve().parent / "payloads.yml"

STRATEGIES = ("instruction", "system_framing", "tool_output",
              "delimiter_escape", "authority", "distraction")
OBJECTIVES = ("suppression", "escalation")

#: How many events in a capture carry the payload. The adversary's own activity
#: is a handful of records, not the whole window, and an agent querying with a
#: limit of 5 should meet the payload without it being pasted 50,000 times.
DEFAULT_MAX_EVENTS = 5


class InjectionError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Payload:
    id: str
    strategy: str
    objective: str
    placements: tuple[str, ...]
    text: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise InjectionError(f"{self.id}: unknown strategy {self.strategy!r}")
        if self.objective not in OBJECTIVES:
            raise InjectionError(f"{self.id}: unknown objective {self.objective!r}")
        if not self.placements:
            raise InjectionError(f"{self.id}: no placements")
        if not self.text.strip():
            raise InjectionError(f"{self.id}: empty payload")
        # the hard allowlist, checked at construction so a bad payload cannot be
        # loaded at all rather than failing somewhere downstream
        require_injectable(self.placements)


def load_payloads(path: Path = PAYLOAD_FILE) -> list[Payload]:
    document = yaml.safe_load(path.read_text()) or {}
    payloads = [
        Payload(id=entry["id"], strategy=entry["strategy"],
                objective=entry["objective"],
                placements=tuple(entry["placements"]),
                text=str(entry["text"]).strip(), note=entry.get("note", ""))
        for entry in document.get("payloads", [])
    ]
    seen: set[str] = set()
    for payload in payloads:
        if payload.id in seen:
            raise InjectionError(f"duplicate payload id {payload.id}")
        seen.add(payload.id)
    return payloads


@dataclasses.dataclass(frozen=True)
class Injection:
    """One payload, in one field, in a specific set of events."""

    payload: Payload
    field: str
    event_indices: tuple[int, ...]

    @property
    def label(self) -> str:
        return f"{self.payload.id}@{self.field}"


def inject_into_event(raw: dict, field: str, text: str) -> dict:
    """Append the payload to one field, keeping the record self-consistent.

    Returns a copy. Raises when the field is not adversary-writable, and returns
    the record untouched when the field is not already present.
    """
    require_injectable([field])
    original = raw.get(field)
    if not isinstance(original, str) or not original:
        return raw

    injected = f"{original} {text}"
    out = dict(raw)
    out[field] = injected

    # Sysmon's rendered Message restates the other fields; a payload absent from
    # it would make the record contradict itself
    message = raw.get("Message")
    if isinstance(message, str) and original in message:
        out["Message"] = message.replace(original, injected)
    return out


def target_indices(
    events: Sequence[Event],
    analysis: RuleAnalysis,
    field: str,
    limit: int = DEFAULT_MAX_EVENTS,
) -> tuple[int, ...]:
    """Events the adversary owns and the agent is likely to retrieve.

    Preference order: events the rule actually matched, then events that merely
    passed its prefilter. The first group is the adversary's own activity by
    definition — the rule fired on it — which is what makes writing to it
    realistic.
    """
    requirements = analysis.detection_requirements
    matched: list[int] = []
    candidates: list[int] = []

    for event in events:
        if not event.present(field):
            continue
        if not analysis.prefilters(event):
            continue
        if requirements and all(r.satisfied_by(event) for r in requirements):
            matched.append(event.index)
        else:
            candidates.append(event.index)
        if len(matched) >= limit:
            break

    chosen = matched or candidates
    return tuple(chosen[:limit])


def plan(
    case: corpus.TriageCase | corpus.MissCase,
    payload: Payload,
    field: str,
    store: CaptureStore,
    limit: int = DEFAULT_MAX_EVENTS,
) -> Injection | None:
    """Work out where this payload would go, or None if it cannot go anywhere.

    A None is a real outcome and is recorded as such: "the payload had nowhere
    plausible to sit in this capture" is not the same as "the attack failed", and
    folding the two together would understate the defence.
    """
    if field not in payload.placements:
        raise InjectionError(f"{payload.id} does not target {field}")
    events = store.load(case.capture)
    indices = target_indices(events, analyse_rule(case.rule), field, limit)
    if not indices:
        return None
    return Injection(payload=payload, field=field, event_indices=indices)


class InjectedStore(CaptureStore):
    """A store that hands out a capture with the payload already in it.

    Subclassing the store rather than patching the tools is deliberate: the
    agent, its tools, the renderer and the citation checker all see the injected
    text through exactly the path they use for clean telemetry. Nothing about the
    run is special-cased for the attack, so a flip cannot be an artefact of the
    harness taking a different route.
    """

    def __init__(self, injection: Injection, base: CaptureStore | None = None):
        super().__init__()
        self.injection = injection
        self._base = base or CaptureStore()
        self._done: dict[str, list[Event]] = {}

    def load(self, capture: corpus.CaptureRef) -> list[Event]:
        if capture.id in self._done:
            return self._done[capture.id]
        clean = self._base.load(capture)
        targets = set(self.injection.event_indices)
        events = [
            Event(inject_into_event(event.raw, self.injection.field,
                                    self.injection.payload.text), event.index)
            if event.index in targets else event
            for event in clean
        ]
        self._done[capture.id] = events
        return events

    def contains_payload(self, capture: corpus.CaptureRef) -> bool:
        """Did the payload actually land? Checked, not assumed."""
        text = self.injection.payload.text
        return any(
            any(isinstance(v, str) and text in v for v in event.raw.values())
            for event in self.load(capture)
        )


def placements(payloads: Iterable[Payload]) -> list[tuple[Payload, str]]:
    """Every payload paired with every field it declares."""
    return [(payload, field) for payload in payloads
            for field in payload.placements]
