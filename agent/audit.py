"""An append-only record of what the agent did.

Agent telemetry is telemetry. The point of this module is that a run can be
reconstructed afterwards by someone who was not watching it: which tools were
called, what came back, how much of what came back was text an adversary wrote,
and what the agent concluded.

Entries are append-only and sequenced. Tamper-evidence is added later, as its
own control with its own test; right now the guarantee is completeness and
ordering, not integrity.

Fields worth alerting on, for the detection-engineering side of this:

- `untrusted_bytes` — how much adversary-written text reached the model. A
  sudden jump is worth a look.
- `citation_rejected` — the agent tried to justify a verdict with evidence that
  was not in the capture. Under attack this is the signal that fires.
- `verdict_rejected` — the model returned something the contract refused.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Iterator

from agent.provenance import Provenance


@dataclasses.dataclass(frozen=True)
class AuditEntry:
    seq: int
    kind: str
    at: float
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "at": round(self.at, 6),
                **self.payload}


class AuditLog:
    """One run's trail. Append-only by construction: there is no mutator."""

    def __init__(self, clock=time.time):
        self._entries: list[AuditEntry] = []
        self._clock = clock

    # -- writing ----------------------------------------------------------

    def _append(self, kind: str, **payload: Any) -> AuditEntry:
        entry = AuditEntry(seq=len(self._entries), kind=kind,
                           at=self._clock(), payload=payload)
        self._entries.append(entry)
        return entry

    def started(self, **payload: Any) -> None:
        self._append("run_started", **payload)

    def model_turn(self, turn: int, reply: Any) -> None:
        self._append(
            "model_turn",
            turn=turn,
            requested_tools=[c.name for c in getattr(reply, "tool_calls", [])],
            answered=getattr(reply, "answer", None) is not None,
            invalid=getattr(reply, "invalid", ""),
            text_bytes=len(getattr(reply, "raw_text", "") or ""),
        )

    def tool_call(self, call: Any, result: Any) -> None:
        ceiling: Provenance = result.provenance_ceiling
        body = result.error or result.output
        self._append(
            "tool_call",
            tool=call.name,
            arguments=_safe(call.arguments),
            provenance_ceiling=ceiling.value,
            # the number a defender would alert on: how much adversary-written
            # text this call put in front of the model
            untrusted_bytes=len(body) if ceiling is Provenance.WRITABLE else 0,
            output_bytes=len(body),
            matched=result.matched,
            returned=result.returned,
            error=result.error,
        )

    def citation_rejected(self, problems: list[str]) -> None:
        self._append("citation_rejected", problems=list(problems))

    def verdict_rejected(self, reasons: list[str]) -> None:
        self._append("verdict_rejected", reasons=list(reasons))

    def action_requested(self, request: Any, decision: Any = None) -> None:
        self._append(
            "action_requested",
            action=request.action,
            target=request.target,
            granted=None if decision is None else bool(decision.granted),
            reason="" if decision is None else decision.reason,
        )

    def finished(self, result: Any) -> None:
        self._append(
            "run_finished",
            case_id=result.case_id,
            label=result.label,
            rejections=list(result.rejections),
            tool_calls=result.tool_calls,
        )

    # -- reading ----------------------------------------------------------

    def __iter__(self) -> Iterator[AuditEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def of_kind(self, kind: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.kind == kind]

    @property
    def untrusted_bytes(self) -> int:
        """Total adversary-written text that reached the model this run."""
        return sum(e.payload.get("untrusted_bytes", 0) for e in self._entries)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e.to_json(), default=str) for e in self._entries)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(self.to_jsonl() + "\n")


def _safe(value: Any, limit: int = 500) -> Any:
    """Arguments go in the log verbatim, but bounded.

    Tool arguments can contain adversary text (a `field_contains` filter the
    model built out of something it read), so they are recorded — that is the
    trail — but not without a length ceiling.
    """
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= limit else text[:limit] + "…"
