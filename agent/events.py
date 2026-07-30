"""Event access, and the two ways an event can reach the model.

Captures are flattened JSON, one object per line, so field access is a dict
lookup. What matters here is not reading the event but *presenting* it, because
the presentation is the attack surface.

Two ingestion modes, which Phase D measures against each other:

`Ingestion.RAW`
    The event is pasted in as text — the JSON object, or Sysmon's rendered
    `Message`, which is the shape a first draft of this agent naturally takes
    and the shape most tutorials use. A command line sits inside a paragraph of
    other prose, indistinguishable from instructions to the model.

`Ingestion.STRUCTURED`
    The event is parsed into named fields drawn from a per-event-id schema,
    each one labelled with its provenance and fenced so a value cannot end its
    own field. The rendered `Message` blob is dropped entirely: it is a
    concatenation of fields already present, so it adds no evidence and is the
    most injectable text in the record.

Nothing here rewrites or sanitises a value. An agent that silently edits the
command line it is reasoning about is an agent whose evidence citations are
about text no host ever emitted.
"""

from __future__ import annotations

import enum
import json
from typing import Any, Iterable

from agent.provenance import (
    DERIVED_FIELDS,
    Provenance,
    classify_in_event,
)

MISSING = object()

#: Longest field value passed through untruncated. Script blocks reach tens of
#: kilobytes, and a prompt is not the place to discover that.
VALUE_LIMIT = 2000


class Ingestion(enum.Enum):
    RAW = "raw"
    STRUCTURED = "structured"


#: Fields worth showing per event id. Ordered: identity first, then the
#: behaviour a rule keys on. Anything not listed is dropped in structured mode,
#: which is a deliberate reduction rather than an oversight.
SCHEMA: dict[int, tuple[str, ...]] = {
    1: ("UtcTime", "Image", "CommandLine", "ParentImage", "ParentCommandLine",
        "OriginalFileName", "CurrentDirectory", "User", "IntegrityLevel",
        "Hashes", "Product", "Company", "Description"),
    3: ("UtcTime", "Image", "Initiated", "Protocol", "SourceIp", "SourcePort",
        "DestinationIp", "DestinationPort", "DestinationHostname", "User"),
    5: ("UtcTime", "Image", "User"),
    7: ("UtcTime", "Image", "ImageLoaded", "OriginalFileName", "Signed",
        "Signature", "SignatureStatus", "Hashes"),
    8: ("UtcTime", "SourceImage", "TargetImage", "NewThreadId", "StartAddress",
        "StartModule", "StartFunction"),
    10: ("UtcTime", "SourceImage", "TargetImage", "GrantedAccess", "CallTrace",
         "SourceUser", "TargetUser"),
    11: ("UtcTime", "Image", "TargetFilename", "User"),
    12: ("UtcTime", "EventType", "Image", "TargetObject", "User"),
    13: ("UtcTime", "EventType", "Image", "TargetObject", "Details", "User"),
    17: ("UtcTime", "EventType", "Image", "PipeName", "User"),
    18: ("UtcTime", "Image", "PipeName", "User"),
    22: ("UtcTime", "Image", "QueryName", "QueryResults", "User"),
    23: ("UtcTime", "Image", "TargetFilename", "Hashes", "User"),
    4104: ("UtcTime", "ScriptBlockText", "Path", "User"),
    4624: ("UtcTime", "TargetUserName", "LogonType", "IpAddress",
           "ProcessName", "SubjectUserName"),
    4625: ("UtcTime", "TargetUserName", "LogonType", "IpAddress", "Status"),
    4688: ("UtcTime", "NewProcessName", "CommandLine", "ParentProcessName",
           "SubjectUserName", "TokenElevationType"),
    4697: ("UtcTime", "ServiceName", "ServiceFileName", "SubjectUserName"),
    7045: ("UtcTime", "ServiceName", "ImagePath", "ServiceType", "StartType"),
}

#: Used when the event id has no schema. Deliberately broad: an unknown event
#: type should still surface identity and behaviour rather than nothing.
FALLBACK_FIELDS: tuple[str, ...] = (
    "UtcTime", "Image", "CommandLine", "ParentImage", "TargetImage",
    "SourceImage", "TargetFilename", "TargetObject", "Details",
    "ScriptBlockText", "ServiceName", "ServiceFileName", "User",
    "GrantedAccess", "CallTrace",
)


class Event:
    """One flattened telemetry record.

    Wraps the raw dict rather than copying it, so a citation can be checked
    against the bytes the capture actually held.
    """

    __slots__ = ("raw", "index")

    def __init__(self, raw: dict, index: int = -1):
        self.raw = raw
        #: Position in the capture stream. This is the citation handle: an
        #: evidence reference has to point at a specific record.
        self.index = index

    def get(self, field: str, default: Any = MISSING) -> Any:
        return self.raw.get(field, default)

    def present(self, field: str) -> bool:
        value = self.raw.get(field, MISSING)
        return value is not MISSING and value is not None and value != ""

    @property
    def event_id(self) -> int | None:
        value = self.raw.get("EventID", self.raw.get("EventId"))
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @property
    def channel(self) -> str:
        return str(self.raw.get("Channel", ""))

    def provenance(self, field: str) -> Provenance:
        return classify_in_event(field, self.raw)

    def schema_fields(self) -> tuple[str, ...]:
        return SCHEMA.get(self.event_id or -1, FALLBACK_FIELDS)

    def __repr__(self) -> str:
        return f"Event(index={self.index}, EventID={self.event_id})"


def _clip(value: Any) -> tuple[str, bool]:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= VALUE_LIMIT:
        return text, False
    return text[:VALUE_LIMIT], True


def _fence(text: str) -> str:
    """Keep a value from ending its own field.

    A payload containing a newline and a plausible-looking field name is trying
    to look like the start of the next field. Escaping the newline removes the
    shape without altering a single character of the value's meaning, which
    matters because the value is evidence and gets cited verbatim.
    """
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r")


def render_structured(event: Event, *, tag_provenance: bool = True) -> str:
    """Named fields from the event's schema, fenced, provenance-tagged.

    `Message` is dropped: it restates fields already listed and is the largest
    piece of free text in the record.
    """
    lines = [f"event[{event.index}] EventID={event.event_id} "
             f"Channel={event.channel}"]
    for field in event.schema_fields():
        if field in DERIVED_FIELDS or not event.present(field):
            continue
        text, clipped = _clip(event.get(field))
        suffix = " …[clipped]" if clipped else ""
        if tag_provenance:
            marker = event.provenance(field).value
            lines.append(f"  {field} ({marker}) = \"{_fence(text)}\"{suffix}")
        else:
            lines.append(f"  {field} = \"{_fence(text)}\"{suffix}")
    return "\n".join(lines)


def render_raw(event: Event) -> str:
    """The event as text, the way an unguarded first draft would paste it.

    Prefers Sysmon's rendered `Message`, because that is what a human analyst
    reads and therefore what a naive implementation forwards.
    """
    message = event.get("Message", MISSING)
    if isinstance(message, str) and message.strip():
        return f"event[{event.index}]\n{message}"
    return f"event[{event.index}]\n{json.dumps(event.raw, indent=1, default=str)}"


def render(event: Event, mode: Ingestion) -> str:
    return (render_structured(event) if mode is Ingestion.STRUCTURED
            else render_raw(event))


def render_many(events: Iterable[Event], mode: Ingestion) -> str:
    return "\n\n".join(render(event, mode) for event in events)
