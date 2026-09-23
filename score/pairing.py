"""Which capture's events each case reads in the mismatched conditions.

The label is constant per capture in this corpus (every alert on an LSASS
campaign capture is a true positive, every alert on the others a false
positive), so pairing happens at capture level:

- cross: the donor capture has the other label. If the verdict follows the
  evidence it should flip.
- same: the donor capture has the same label but is a different capture. The
  evidence is wrong but points the same way, so the verdict should hold; if it
  flips anyway, the agent is reacting to the swap rather than to the content.

Donors come from a seeded shuffle and are handed out least-used first, so every
capture on the donor side is used about equally and a rerun with the same seed
gives the same pairing.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter

from agent import corpus

STRATA = ("cross", "same")
OTHER = {"true_positive": "false_positive", "false_positive": "true_positive"}


def labels_by_capture(cases) -> dict[str, str]:
    labels: dict[str, str] = {}
    for case in cases:
        seen = labels.setdefault(case.capture.id, case.truth)
        if seen != case.truth:
            raise ValueError(f"capture {case.capture.id} carries both labels; "
                             "capture-level pairing assumes one")
    return labels


def donors(cases, stratum: str, seed: int) -> dict[str, tuple[corpus.CaptureRef, str]]:
    """case_id -> (donor capture, donor's label)."""
    if stratum not in STRATA:
        raise ValueError(f"stratum must be one of {STRATA}, not {stratum!r}")
    labels = labels_by_capture(cases)
    captures = {c.capture.id: c.capture for c in cases}

    # nosec B311: a seeded shuffle for a reproducible pairing, not a secret
    rng = random.Random(f"{seed}:{stratum}")  # nosec B311
    pools: dict[str, list[corpus.CaptureRef]] = {}
    for label in OTHER:
        pool = sorted((captures[i] for i, lab in labels.items() if lab == label),
                      key=lambda c: c.id)
        rng.shuffle(pool)
        pools[label] = pool

    used: Counter[str] = Counter()
    out: dict[str, tuple[corpus.CaptureRef, str]] = {}
    for case in sorted(cases, key=lambda c: c.case_id):
        label = case.truth if stratum == "same" else OTHER[case.truth]
        pool = [c for c in pools[label] if c.id != case.capture.id]
        if not pool:
            raise ValueError(f"no {stratum}-label donor for {case.case_id}")
        # least used first; the shuffled order breaks ties, so it stays seeded
        donor = min(pool, key=lambda c: used[c.id])
        used[donor.id] += 1
        out[case.case_id] = (donor, label)
    return out


def digest(pairing: dict[str, tuple[corpus.CaptureRef, str]]) -> str:
    flat = {case_id: donor.id for case_id, (donor, _) in pairing.items()}
    return hashlib.sha256(json.dumps(flat, sort_keys=True).encode()).hexdigest()
