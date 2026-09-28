"""How much of the injection surface can actually be attacked, with no model.

    python -m attack.mountability [--out PATH]

Every payload placement against every true-positive case: a placement is
mountable when the capture has an event the adversary owns that carries the
field (see attack.inject.plan). Writes benchmark/attack-mountability.json,
which docs/results-attack.md quotes; counts are [mountable, attempted].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent import corpus
from agent.tools import CaptureStore
from attack.inject import load_payloads, placements, plan

OUT = Path(__file__).resolve().parent.parent / "benchmark" / "attack-mountability.json"


def mountability(cases, store: CaptureStore, payloads=None) -> dict:
    pairs = placements(payloads if payloads is not None else load_payloads())
    by_field: dict[str, list[int]] = {}
    by_strategy: dict[str, list[int]] = {}
    mountable = 0
    for case in cases:
        for payload, field in pairs:
            ok = plan(case, payload, field, store) is not None
            mountable += ok
            for table, key in ((by_field, field), (by_strategy, payload.strategy)):
                counts = table.setdefault(key, [0, 0])
                counts[0] += ok
                counts[1] += 1
    return {"cases": len(cases), "placements": len(pairs),
            "attempts": len(cases) * len(pairs), "mountable": mountable,
            "by_field": by_field, "by_strategy": by_strategy}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    cases = sorted((c for c in corpus.evaluation_set().triage
                    if c.truth == "true_positive"),
                   key=lambda c: (c.capture.id, c.case_id))
    result = mountability(cases, CaptureStore())
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"{result['mountable']} of {result['attempts']} placements can be mounted "
          f"({result['mountable'] / result['attempts']:.1%})")
    print(f"written: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
