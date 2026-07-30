"""Who wrote this field: the adversary, the OS, or something in between.

The agent reads command lines, filenames, script blocks and service names. An
adversary *writes* those. That is the whole reason this repo exists, so the
question "could an attacker have chosen this text?" is a type here rather than
a comment.

Three classes, because the middle one changes what an attack can do:

`WRITABLE`
    Free text the adversary chooses outright. A command line can contain an
    English sentence, so it can contain an instruction. This is the injection
    surface, and the only class an injection is allowed into.

`INFLUENCED`
    The value reflects adversary behaviour but is drawn from a space the OS
    controls. `GrantedAccess` is a hex access mask: an adversary decides whether
    it is 0x1010 or 0x1fffff, and can decide nothing else about it. It cannot
    carry a sentence, so it cannot carry a payload.

`OS`
    The adversary cannot touch it without already owning the logging pipeline,
    at which point telemetry integrity is gone and prompt injection is the least
    of the problem. Event ids, channels, GUIDs, timestamps, host names.

Two defaults, pointing opposite ways, and the asymmetry is deliberate:

- Classifying an unknown field for *defence* returns `WRITABLE`. An unrecognised
  field is untrusted until someone establishes otherwise.
- Choosing an unknown field as an *injection target* is refused. The attack
  corpus may only touch fields explicitly established as adversary-written,
  because an attack on a field no adversary controls inflates the success rate
  with something nobody can mount.

So the defence over-distrusts and the attack under-reaches. Both errors are the
safe direction for the claim being made.
"""

from __future__ import annotations

import enum
from typing import Iterable


class Provenance(enum.Enum):
    """Ordered by how much an adversary can put in the field."""

    OS = "os-generated"
    INFLUENCED = "adversary-influenced"
    WRITABLE = "adversary-writable"

    @property
    def rank(self) -> int:
        return {"os-generated": 0, "adversary-influenced": 1,
                "adversary-writable": 2}[self.value]

    @property
    def is_untrusted(self) -> bool:
        return self is not Provenance.OS


#: Free text an adversary chooses. Command lines, paths, script bodies, service
#: names, registry values and the PE metadata of a binary they compiled.
WRITABLE_FIELDS = frozenset({
    # process creation and access
    "CommandLine", "ParentCommandLine", "ProcessCommandLine",
    "Image", "ParentImage", "SourceImage", "TargetImage", "NewProcessName",
    "ParentProcessName", "OriginalFileName", "CurrentDirectory",
    # modules and files
    "ImageLoaded", "TargetFilename", "Device", "PipeName",
    # PE metadata: attacker-supplied when the attacker built the binary
    "Product", "Description", "Company", "FileVersion", "Signature",
    # scripts
    "ScriptBlockText", "Payload", "CommandName", "CommandPath",
    # services and scheduled tasks
    "ServiceName", "ServiceFileName", "StartFunction", "StartModule",
    "TaskName",
    # registry
    "TargetObject", "Details", "NewName",
    # network and dns
    "DestinationHostname", "DestinationPortName", "QueryName", "QueryResults",
    "SourceHostname",
    # wmi
    "Operation", "EventNamespace", "Name", "Query", "Consumer", "Filter",
    "Destination",
})

#: Adversary behaviour is visible in the value, but the value's vocabulary is
#: the OS's. None of these can hold a sentence.
INFLUENCED_FIELDS = frozenset({
    "GrantedAccess", "CallTrace", "IntegrityLevel", "Hashes", "Hash",
    "User", "SourceUser", "TargetUser", "ParentUser", "SubjectUserName",
    "TargetUserName", "LogonId", "SubjectLogonId", "TerminalSessionId",
    "EventType", "Protocol", "Initiated", "SourcePort", "DestinationPort",
    "DestinationIp", "SourceIp", "DestinationIsIpv6", "SourceIsIpv6",
    "TokenElevationType", "MandatoryLabel", "IsExecutable", "Archived",
    "Signed", "SignatureStatus", "AccessMask", "AccessList", "ObjectType",
})

#: The adversary cannot write these without owning the logging pipeline.
OS_FIELDS = frozenset({
    "EventID", "EventId", "Channel", "Provider", "ProviderGuid", "SourceName",
    "Task", "Level", "Keywords", "Opcode", "Version", "RecordNumber",
    "EventRecordID", "@timestamp", "UtcTime", "TimeCreated", "SystemTime",
    "Hostname", "Computer", "ComputerName", "ProcessGuid", "ProcessId",
    "ThreadId", "ExecutionProcessID", "ExecutionThreadID",
    "SourceProcessGUID", "SourceProcessId", "SourceThreadId",
    "TargetProcessGUID", "TargetProcessId", "ParentProcessGuid",
    "ParentProcessId", "SequenceNumber", "Tags", "host", "@version",
    # Sysmon's RuleName comes from the defender's own configuration
    "RuleName",
})

#: Fields the host renders by concatenating other fields. Their provenance is
#: the strongest provenance of anything inside them, which in practice means a
#: rendered Message is adversary-writable whenever the event has a command line.
DERIVED_FIELDS = frozenset({"Message", "message", "RenderedDescription", "param1"})


class ProvenanceError(ValueError):
    pass


def classify(field: str) -> Provenance:
    """Provenance of one field name, defaulting to WRITABLE when unknown.

    The default is the untrusting one on purpose. A field this module has never
    heard of is not thereby safe, and a taxonomy that silently trusts new
    telemetry is a taxonomy that degrades every time a Sysmon schema changes.
    """
    if field in OS_FIELDS:
        return Provenance.OS
    if field in INFLUENCED_FIELDS:
        return Provenance.INFLUENCED
    if field in DERIVED_FIELDS:
        # a rendered message inherits from its parts; without the event in hand
        # the safe answer is the strongest class it could contain
        return Provenance.WRITABLE
    return Provenance.WRITABLE


def classify_in_event(field: str, event: dict) -> Provenance:
    """Provenance of a field given the event it sits in.

    Only differs from `classify` for derived fields, where the answer depends on
    what the host actually concatenated.
    """
    if field not in DERIVED_FIELDS:
        return classify(field)
    others = [classify(name) for name in event if name not in DERIVED_FIELDS]
    if not others:
        return Provenance.WRITABLE
    return max(others, key=lambda provenance: provenance.rank)


def is_injectable(field: str) -> bool:
    """May the attack corpus write a payload into this field?

    Only fields explicitly established as adversary-written. Unknown fields are
    refused, which is the opposite default to `classify`: the defence treats
    unknown as untrusted, the attack treats unknown as off-limits.
    """
    return field in WRITABLE_FIELDS


def require_injectable(fields: Iterable[str]) -> None:
    """Raise unless every field is an allowed injection target."""
    refused = sorted({f for f in fields if not is_injectable(f)})
    if refused:
        raise ProvenanceError(
            "refusing to inject into fields an adversary does not write: "
            + ", ".join(refused)
            + ". Add to WRITABLE_FIELDS only with a reason a real adversary "
              "could choose the value."
        )


def tag_event(event: dict) -> dict[str, Provenance]:
    """Provenance for every field present in one event."""
    return {field: classify_in_event(field, event) for field in event}


def untrusted_fields(event: dict) -> list[str]:
    """Fields in this event an adversary could have written, worst first."""
    tagged = tag_event(event)
    return sorted(
        (f for f, p in tagged.items() if p is Provenance.WRITABLE),
        key=lambda f: (f in DERIVED_FIELDS, f),
    )
