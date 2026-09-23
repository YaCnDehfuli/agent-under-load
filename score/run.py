"""Score a predictor against the corpus and write the artefact.

    python -m score.run --predictor baseline --task miss
    python -m score.run --predictor agent --model gpt-oss-20b --task triage \
        --repeats 3 --budget-usd 5

A baseline run writes one JSON artefact under `runs/`. An agent run writes a
run directory (see `score/ledger.py`) that it appends to as trajectories finish,
so it can be stopped and resumed. Docs quote artefacts; nothing in `docs/` is
typed by hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from agent import corpus
from agent.audit import AuditLog
from agent.baseline import MissBaseline, TriageBaseline
from agent.graph import ALL_CONTROLS, AgentConfig, Control, TriageGraph, system_prompt
from agent.tools import CaptureStore
from score.ledger import (
    ERROR,
    BudgetExceeded,
    RunDir,
    check_budget,
    cost_usd,
    model_mean_cost,
    model_spend,
)
from score.metrics import Prediction, Report, score

RUNS = Path(__file__).resolve().parent.parent / "runs"

TRIAGE_CLASSES = ("true_positive", "false_positive")
MISS_CLASSES = corpus.MISS_CLASSES


def _controls(spec: str) -> frozenset[Control]:
    spec = (spec or "").strip().lower()
    if spec in ("", "none", "undefended"):
        return frozenset()
    if spec == "all":
        return ALL_CONTROLS
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(Control(part))
        except ValueError:
            raise SystemExit(
                f"unknown control {part!r}. Known: "
                + ", ".join(c.value for c in Control)
            )
    return frozenset(out)


def _sorted_by_capture(cases):
    """Group cases by capture so the LRU store loads each archive once.

    Ordering is by capture id, which is stable, so a partial run with --limit
    covers the same cases every time.
    """
    return sorted(cases, key=lambda c: (c.capture.id, c.case_id))


def run(
    task: str,
    predictor: str,
    controls: frozenset[Control] = frozenset(),
    limit: int | None = None,
) -> tuple[Report, list[dict]]:
    cases = _cases(task, limit)

    store = CaptureStore()
    classes = TRIAGE_CLASSES if task == "triage" else MISS_CLASSES

    if predictor == "baseline":
        engine = (TriageBaseline(store=store) if task == "triage"
                  else MissBaseline(store=store))
        name = engine.name
        predict = engine.predict
    else:
        raise SystemExit(f"unknown predictor {predictor!r}: agent runs go through run_agent")

    if controls:
        name = f"{name} {AgentConfig(controls=controls).label}"

    predictions: list[Prediction] = []
    records: list[dict] = []
    started = time.time()

    for index, case in enumerate(cases, start=1):
        try:
            result = predict(case)
            label, rejections = result.label, result.rejections
            tool_calls = result.tool_calls
        except Exception as exc:  # a failed case is recorded, never dropped
            label, rejections, tool_calls = None, [f"{type(exc).__name__}: {exc}"], 0
        predictions.append(Prediction(case_id=case.case_id, truth=case.truth,
                                      predicted=label))
        records.append({"case_id": case.case_id, "truth": case.truth,
                        "predicted": label, "rejections": rejections,
                        "tool_calls": tool_calls,
                        "rule_id": case.rule.id, "capture": case.capture.id})
        if index % 25 == 0 or index == len(cases):
            print(f"  {index}/{len(cases)} cases", file=sys.stderr)

    report = score(predictions, classes, task=task, predictor=name)
    artefact = {
        "task": task,
        "predictor": name,
        "model": "",
        "temperature": None,
        "controls": sorted(c.value for c in controls),
        "cases": len(cases),
        "limited": bool(limit),
        "seconds": round(time.time() - started, 1),
        "report": report.to_json(),
        "predictions": records,
    }
    return report, artefact


def _cases(task: str, limit: int | None):
    evaluation = corpus.evaluation_set()
    cases = _sorted_by_capture(evaluation.triage if task == "triage"
                               else evaluation.miss)
    return cases[:limit] if limit else cases


# ---------------------------------------------------------------------------
# agent runs
# ---------------------------------------------------------------------------

TASK_NAMES = {"triage": "triage_verdict", "miss": "miss_classification"}

#: A run directory is refused on resume when any of these differ from what it
#: was started with: mixing two configurations inside one run would make its
#: numbers describe neither.
IDENTITY = ("task", "model_key", "model_entry", "controls", "system_prompt_sha256")


def _git_revision() -> dict:
    try:
        sha = corpus._run(["git", "rev-parse", "HEAD"], cwd=RUNS.parent)
        dirty = corpus._run(["git", "status", "--porcelain", "--untracked-files=no"],
                            cwd=RUNS.parent)
    except (corpus.CorpusError, OSError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": bool(dirty)}


def _manifest_digest() -> str | None:
    try:
        return corpus.manifest_digest()
    except corpus.CorpusError:
        return None


def _sum(turns: list[dict], field: str, *, all_or_nothing: bool) -> int | None:
    values = [t.get(field) for t in turns]
    known = [v for v in values if v is not None]
    if not known or (all_or_nothing and len(known) != len(values)):
        return None
    return sum(known)


def _usage(log: AuditLog) -> dict:
    turns = [e.payload for e in log.of_kind("model_turn")]
    return {
        # a turn with unreported input or output makes the total unknown rather
        # than silently low; cached and reasoning counts are optional extras
        "input_tokens": _sum(turns, "input_tokens", all_or_nothing=True),
        "cached_input_tokens": _sum(turns, "cached_input_tokens", all_or_nothing=False),
        "output_tokens": _sum(turns, "output_tokens", all_or_nothing=True),
        "reasoning_tokens": _sum(turns, "reasoning_tokens", all_or_nothing=False),
        "reported_cost_usd": _sum(turns, "reported_cost_usd", all_or_nothing=True),
        "turns": len(turns),
        "latency_s": round(sum(t.get("latency_s") or 0.0 for t in turns), 3),
        "served_models": sorted({t["served_model"] for t in turns if t.get("served_model")}),
        "served_by": sorted({t["served_by"] for t in turns if t.get("served_by")}),
    }


def run_agent(
    cases,
    model,
    model_key: str,
    model_entry: dict,
    run_dir: Path,
    task: str = "triage",
    config: AgentConfig | None = None,
    repeats: int = 1,
    budget_usd: float | None = None,
    model_cap_usd: float | None = None,
    store: CaptureStore | None = None,
) -> dict:
    """Run the agent over cases × repeats into a run directory, resuming if it exists.

    Two caps: `budget_usd` for this run directory, and `model_cap_usd` for
    everything this model has cost across the run directories next to it.
    """
    config = config or AgentConfig()
    price = model_entry.get("price")
    capped = budget_usd is not None or model_cap_usd is not None
    if capped and cost_usd({"input_tokens": 0, "output_tokens": 0}, price) is None:
        raise ValueError(f"{model_key} has no price in models.yml, so a spending "
                         "cap cannot be enforced")

    system = system_prompt(TASK_NAMES[task], config)
    identity = {
        "task": task,
        "model_key": model_key,
        "model_entry": model_entry,
        "controls": sorted(c.value for c in config.controls),
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
    }
    rundir = RunDir(run_dir)
    meta = rundir.read_meta()
    if meta is not None:
        changed = [k for k in IDENTITY if meta.get(k) != identity[k]]
        if changed:
            raise ValueError(f"{run_dir} was started with a different "
                             f"{', '.join(changed)}; use a new --run-dir")
    else:
        meta = {**identity, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "model_config": model.config, "git": _git_revision(),
                "corpus_manifest_sha256": _manifest_digest()}
    rundir.create()
    # written now, not only at the end, so an interrupted first run still
    # refuses to be resumed under a different configuration
    rundir.write_meta(meta)

    done = rundir.completed()
    pending = [(case, repeat) for repeat in range(1, repeats + 1) for case in cases
               if (case.case_id, repeat) not in done]
    # what this model cost in other run directories; constant for this run
    elsewhere = model_spend(rundir.path.parent, model_key) - rundir.spent()

    def mean() -> float | None:
        # this run's own mean once it has one; until then the model's mean from
        # its other runs, so a fresh run directory can still be projected
        own = rundir.mean_cost()
        return own if own is not None else model_mean_cost(rundir.path.parent, model_key)

    projected = check_budget(budget_usd, rundir.spent(), mean(), len(pending))
    check_budget(model_cap_usd, elsewhere + rundir.spent(), mean(), len(pending))
    print(f"  {len(pending)} trajectories to run, {len(done)} already done; "
          + (f"projected ${projected:.2f} more" if projected is not None
             else "no cost measured yet"), file=sys.stderr)

    graph = TriageGraph(model, config, store=store or CaptureStore())
    budget_stop = False
    for index, (case, repeat) in enumerate(pending, start=1):
        next_cost, spent = mean(), rundir.spent()
        over = [f"the ${cap:.2f} {name} cap" for name, cap, total in (
                    ("run", budget_usd, spent),
                    (f"{model_key} model", model_cap_usd, elsewhere + spent))
                if cap is not None and next_cost is not None and total + next_cost > cap]
        if over:
            budget_stop = True
            print(f"  stopping before trajectory {index}: the next one would pass "
                  + " and ".join(over), file=sys.stderr)
            break

        log = AuditLog()
        label, rejections, tool_calls, error = None, [], 0, ""
        try:
            result = graph.run(case, audit=log)
            label, rejections, tool_calls = result.label, result.rejections, result.tool_calls
            outcome = "answered" if label is not None else "unanswered"
        except Exception as exc:  # recorded and retried on resume, never dropped
            outcome, error = ERROR, f"{type(exc).__name__}: {exc}"

        usage = _usage(log)
        path = rundir.trajectory_path(case.case_id, repeat)
        path.write_text(log.to_jsonl() + "\n")
        rundir.append({
            "case_id": case.case_id, "repeat": repeat, "truth": case.truth,
            "predicted": label, "outcome": outcome, "rejections": rejections,
            "error": error, "tool_calls": tool_calls, **usage,
            # what the provider billed when it says, otherwise tokens × price table
            "cost_usd": (usage["reported_cost_usd"]
                         if usage["reported_cost_usd"] is not None
                         else cost_usd(usage, price)),
            "audit_head": log.head, "trajectory": path.name,
            "rule_id": case.rule.id, "capture": case.capture.id,
        })
        if index % 10 == 0 or index == len(pending):
            print(f"  {index}/{len(pending)} trajectories", file=sys.stderr)

    latest = list(rundir.latest().values())
    classes = TRIAGE_CLASSES if task == "triage" else MISS_CLASSES
    reports = {}
    for repeat in range(1, repeats + 1):
        scored = [r for r in latest if r["repeat"] == repeat and r["outcome"] != ERROR]
        if scored:
            reports[str(repeat)] = score(
                [Prediction(r["case_id"], r["truth"], r["predicted"]) for r in scored],
                classes, task=task, predictor=f"agent[{model_key}]").to_json()

    spent = rundir.spent()
    completed = rundir.completed()
    remaining = sum((case.case_id, repeat) not in completed
                    for case in cases for repeat in range(1, repeats + 1))
    per_trajectory = mean()
    meta.update({
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "repeats": max(repeats, meta.get("repeats", 0)),
        "totals": {
            "trajectories": len(latest),
            **{o: sum(r["outcome"] == o for r in latest)
               for o in ("answered", "unanswered", ERROR)},
            **{f: sum(r[f] or 0 for r in latest)
               for f in ("input_tokens", "cached_input_tokens",
                         "output_tokens", "reasoning_tokens")},
        },
        "budget": {
            "budget_usd": budget_usd,
            "actual_cost_usd": round(spent, 6),
            "projected_remaining_cost_usd": (round(per_trajectory * remaining, 6)
                                             if per_trajectory is not None else None),
            "budget_stop": budget_stop,
            "model_cap_usd": model_cap_usd,
            "model_spent_usd": round(elsewhere + spent, 6),
            "pricing_source": (price or {}).get("source"),
        },
        "reports": reports,
        "errors": dict(Counter(r["error"] for r in latest if r["outcome"] == ERROR)),
    })
    rundir.write_meta(meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("triage", "miss"), required=True)
    parser.add_argument("--predictor", choices=("baseline", "agent"),
                        default="baseline")
    parser.add_argument("--model", default="",
                        help="model key in models.yml, required for --predictor agent")
    parser.add_argument("--controls", default="",
                        help="none | all | comma-separated control names")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N cases, for a smoke run")
    parser.add_argument("--repeats", type=int, default=1,
                        help="agent only: runs per case")
    parser.add_argument("--budget-usd", type=float, default=None,
                        help="agent only: stop before spending more than this")
    parser.add_argument("--run-dir", default="",
                        help="agent only: run directory, resumed if it exists")
    parser.add_argument("--out", default="", help="baseline artefact path")
    parser.add_argument("--env-file", default="",
                        help="agent only: file with API keys (default: .env in the repo)")
    args = parser.parse_args(argv)
    controls = _controls(args.controls)

    if args.predictor == "agent":
        from agent import models
        models.load_env(Path(args.env_file) if args.env_file else None)
        if not args.model:
            raise SystemExit("--predictor agent needs --model: a key in models.yml")
        model = models.load_model(args.model)
        label = AgentConfig(controls=controls).label
        run_dir = (Path(args.run_dir) if args.run_dir
                   else RUNS / f"{args.task}-{args.model}-{label}")
        try:
            meta = run_agent(_cases(args.task, args.limit), model, args.model,
                             models.registry()[args.model], run_dir, task=args.task,
                             config=AgentConfig(controls=controls),
                             repeats=args.repeats, budget_usd=args.budget_usd,
                             model_cap_usd=models.registry()[args.model].get("budget_usd"))
        except (ValueError, BudgetExceeded) as exc:
            raise SystemExit(str(exc))
        print(json.dumps({k: meta[k] for k in ("totals", "budget")}, indent=1))
        for message, count in list(meta["errors"].items())[:3]:
            print(f"  error x{count}: {message}", file=sys.stderr)
        print(f"\nrun directory: {run_dir}", file=sys.stderr)
        return 0

    report, artefact = run(args.task, args.predictor, controls, args.limit)

    print(report.to_markdown())

    RUNS.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.task}-{args.predictor}"
            + (f"-{artefact['controls'] and '+'.join(artefact['controls'])}"
               if artefact["controls"] else ""))
    path = Path(args.out) if args.out else RUNS / f"{stem}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artefact, indent=1, default=str))
    print(f"\nartefact: {path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
