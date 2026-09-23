"""Where an agent run keeps its records, and what it has spent.

    runs/<run>/run.json          what produced the run, totals, one report per repeat
    runs/<run>/cases.jsonl       one line per trajectory, appended as it finishes
    runs/<run>/trajectories/     the full audit log of each trajectory

Lines are appended as trajectories finish, so a run that dies halfway — a rate
limit, a spending cap, a laptop lid — resumes from where it stopped instead of
paying for the finished part twice. A trajectory that failed for infrastructure
reasons is recorded as `error` and tried again on resume; one where the model
gave no usable answer is `unanswered`, counts against the model, and is not.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

ERROR = "error"
PER_MILLION = 1_000_000


class BudgetExceeded(RuntimeError):
    pass


def cost_usd(usage: dict[str, int | None], price: dict | None) -> float | None:
    """What a trajectory cost, or None when the price or the usage is unknown.

    Reasoning tokens are already counted inside the output tokens on the APIs
    used here, so they are not added again. A provider that does not report
    cached tokens is billed as if nothing was cached, which over-estimates.
    """
    if not price or any(price.get(k) is None for k in ("input", "output")):
        return None
    total_in, out = usage.get("input_tokens"), usage.get("output_tokens")
    if total_in is None or out is None:
        return None
    cached = usage.get("cached_input_tokens") or 0
    cached_rate = price.get("cached_input")
    if cached_rate is None:
        cached_rate = price["input"]
    return ((total_in - cached) * price["input"] + cached * cached_rate
            + out * price["output"]) / PER_MILLION


def check_budget(budget: float | None, spent: float, mean: float | None,
                 pending: int) -> float | None:
    """The projected cost of what is left. Raises when it would pass the cap."""
    if mean is None:
        return None
    remaining = mean * pending
    if budget is not None and spent + remaining > budget:
        raise BudgetExceeded(
            f"projected ${spent + remaining:.2f} ({pending} trajectories at "
            f"${mean:.4f}, ${spent:.2f} already spent) is over the ${budget:.2f} cap")
    return remaining


class RunDir:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.cases_path = self.path / "cases.jsonl"
        self.meta_path = self.path / "run.json"
        self.trajectories = self.path / "trajectories"

    def create(self) -> None:
        self.trajectories.mkdir(parents=True, exist_ok=True)

    def records(self) -> Iterator[dict]:
        if not self.cases_path.exists():
            return
        with open(self.cases_path) as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def latest(self) -> dict[tuple[str, int], dict]:
        """The last record for each (case, repeat): a rerun supersedes an error."""
        return {(r["case_id"], r["repeat"]): r for r in self.records()}

    def completed(self) -> set[tuple[str, int]]:
        return {key for key, r in self.latest().items() if r["outcome"] != ERROR}

    def spent(self) -> float:
        return sum(r["cost_usd"] or 0.0 for r in self.records())

    def mean_cost(self) -> float | None:
        costs = [r["cost_usd"] for r in self.records()
                 if r["outcome"] != ERROR and r["cost_usd"] is not None]
        return sum(costs) / len(costs) if costs else None

    def append(self, record: dict) -> None:
        with open(self.cases_path, "a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def trajectory_path(self, case_id: str, repeat: int) -> Path:
        return self.trajectories / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', case_id)}-r{repeat}.jsonl"

    def read_meta(self) -> dict[str, Any] | None:
        return json.loads(self.meta_path.read_text()) if self.meta_path.exists() else None

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=1, default=str))
