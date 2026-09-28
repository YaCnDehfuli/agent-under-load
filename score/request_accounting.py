"""Count provider requests in the primary four-model triage matrix.

The run ledger can contain duplicate case rows after resume. Audit trajectories
are keyed by case and repeat, so this reads each final trajectory once. Each
successful model turn records the number of HTTP attempts, including retries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = (
    "gpt-6-luna",
    "deepseek-v4-pro",
    "gpt-oss-20b",
    "gpt-oss-120b",
)
CONDITIONS = (
    "reference",
    "alert-only",
    "alert-only-forced",
    "rule-only",
    "mismatch-cross",
    "mismatch-same",
)
TRAJECTORIES_PER_CELL = 80 * 3


def account(runs: Path) -> dict:
    cells = []
    for model in MODELS:
        for condition in CONDITIONS:
            directory = runs / f"triage-{model}-{condition}-undefended"
            traces = sorted((directory / "trajectories").glob("*.jsonl"))
            if len(traces) != TRAJECTORIES_PER_CELL:
                raise ValueError(f"{directory}: expected 240 audit traces, found {len(traces)}")

            turns = requests = retried_turns = 0
            for trace in traces:
                for line in trace.read_text().splitlines():
                    event = json.loads(line)
                    if event.get("kind") != "model_turn":
                        continue
                    attempts = event.get("attempts")
                    if not isinstance(attempts, int) or attempts < 1:
                        raise ValueError(f"{trace}: model turn has no valid attempt count")
                    turns += 1
                    requests += attempts
                    retried_turns += attempts > 1

            latest = {}
            for line in (directory / "cases.jsonl").read_text().splitlines():
                row = json.loads(line)
                latest[(row["case_id"], row["repeat"])] = row
            if len(latest) != TRAJECTORIES_PER_CELL:
                raise ValueError(f"{directory}: expected 240 unique case repeats")
            if any(row["outcome"] == "error" for row in latest.values()):
                raise ValueError(f"{directory}: provider errors make the request total incomplete")
            if sum(row.get("turns") or 0 for row in latest.values()) != turns:
                raise ValueError(f"{directory}: case turn total differs from audit traces")

            cells.append({
                "model": model,
                "condition": condition,
                "trajectories": len(traces),
                "model_turns": turns,
                "provider_requests": requests,
                "turns_with_retries": retried_turns,
            })

    return {
        "scope": "Primary triage matrix: four models, 80 cases, six evidence conditions, three repeats; undefended configuration",
        "method": "Sum attempts on model_turn events in each final audit trajectory. Attempts include HTTP retries and host-error retries. Resumed duplicate case rows are excluded. Smoke, wording-check and control runs are excluded.",
        "totals": {
            "trajectories": sum(row["trajectories"] for row in cells),
            "model_turns": sum(row["model_turns"] for row in cells),
            "provider_requests": sum(row["provider_requests"] for row in cells),
            "turns_with_retries": sum(row["turns_with_retries"] for row in cells),
        },
        "cells": cells,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(account(args.runs), indent=2) + "\n"
    if args.out:
        args.out.write_text(result)
    else:
        print(result, end="")
