"""Test doubles, so the unit suite needs no corpus and no API key."""

from __future__ import annotations

from pathlib import Path

from agent import corpus
from agent.events import Event
from agent.tools import CaptureStore

#: A handful of hand-written Sysmon-shaped events. Enough to exercise the tools
#: and the citation checker without 50,000 records.
SYNTHETIC_EVENTS: list[dict] = [
    {"EventID": 1, "Channel": "Microsoft-Windows-Sysmon/Operational", "UtcTime": "t0",
     "Image": "C:\\Windows\\System32\\cmd.exe", "CommandLine": "cmd /c whoami",
     "Hostname": "H"},
    {"EventID": 10, "Channel": "Microsoft-Windows-Sysmon/Operational", "UtcTime": "t1",
     "SourceImage": "C:\\tools\\procdump.exe",
     "TargetImage": "C:\\Windows\\system32\\lsass.exe",
     "GrantedAccess": "0x1fffff", "Hostname": "H"},
    {"EventID": 10, "Channel": "Microsoft-Windows-Sysmon/Operational", "UtcTime": "t2",
     "SourceImage": "C:\\Windows\\explorer.exe",
     "TargetImage": "C:\\Windows\\system32\\svchost.exe",
     "GrantedAccess": "0x1010", "Hostname": "H"},
]

RULE_YAML = """\
title: Suspicious LSASS Access
id: 6f7a3d5e-1c2b-4a9e-8d31-0f5b7c9a2e44
status: test
logsource:
  product: windows
  category: process_access
detection:
  selection:
    TargetImage|endswith: '\\lsass.exe'
    GrantedAccess: '0x1fffff'
  condition: selection
level: high
extra_prose: this key should not reach the prompt
"""


class FakeStore(CaptureStore):
    """A CaptureStore over in-memory events, ignoring the capture argument."""

    def __init__(self, raws: list[dict] | None = None):
        super().__init__()
        self._events = [Event(raw, index) for index, raw
                        in enumerate(raws if raws is not None else SYNTHETIC_EVENTS)]

    def load(self, capture):  # noqa: D102 - signature matches the parent
        return self._events


def write_rule(directory: Path, body: str = RULE_YAML,
               name: str = "rule.yml") -> corpus.RuleRef:
    path = directory / name
    path.write_text(body)
    return corpus.RuleRef(id="6f7a3d5e-1c2b-4a9e-8d31-0f5b7c9a2e44", title="Suspicious LSASS Access", level="high",
                          path=path, source="sigmahq", selected_by="tag")


SYNTHETIC_CAPTURE = corpus.CaptureRef(
    id="lab/synthetic", group="atomic", archive=Path("nowhere.zip"), events=3,
)


def synthetic_triage_case(rule: corpus.RuleRef) -> corpus.TriageCase:
    return corpus.TriageCase(
        case_id="tp:synthetic",
        rule=rule,
        capture=SYNTHETIC_CAPTURE,
        fire_count=1,
        truth="true_positive",
    )


def synthetic_miss_case(rule: corpus.RuleRef) -> corpus.MissCase:
    return corpus.MissCase(
        case_id="miss:synthetic",
        rule=rule,
        capture=SYNTHETIC_CAPTURE,
        truth="miss-logic",
    )
