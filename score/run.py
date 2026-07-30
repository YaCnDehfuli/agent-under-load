"""Score a predictor against the corpus and write the artefact.

    python -m score.run --predictor baseline --task miss
    python -m score.run --predictor agent --task triage --controls all

Every run writes a JSON artefact under `runs/` carrying the predictor, the model
and temperature where there is one, the control set, and the per-case
predictions. Docs quote artefacts; nothing in `docs/` is typed by hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from agent import corpus
from agent.baseline import MissBaseline, TriageBaseline
from agent.graph import ALL_CONTROLS, AgentConfig, Control, TriageGraph
from agent.tools import CaptureStore
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
    evaluation = corpus.evaluation_set()
    cases = evaluation.triage if task == "triage" else evaluation.miss
    cases = _sorted_by_capture(cases)
    if limit:
        cases = cases[:limit]

    store = CaptureStore()
    classes = TRIAGE_CLASSES if task == "triage" else MISS_CLASSES

    if predictor == "baseline":
        engine = (TriageBaseline(store=store) if task == "triage"
                  else MissBaseline(store=store))
        name = engine.name
        predict = engine.predict
        model_name, temperature = "", None
    elif predictor == "agent":
        from agent import models
        model = models.from_env()
        graph = TriageGraph(model, AgentConfig(controls=controls), store=store)
        name = f"agent[{model.name}]"
        predict = graph.run
        model_name, temperature = model.name, model.temperature
    else:
        raise SystemExit(f"unknown predictor {predictor!r}: use baseline or agent")

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
        "model": model_name,
        "temperature": temperature,
        "controls": sorted(c.value for c in controls),
        "cases": len(cases),
        "limited": bool(limit),
        "seconds": round(time.time() - started, 1),
        "report": report.to_json(),
        "predictions": records,
    }
    return report, artefact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("triage", "miss"), required=True)
    parser.add_argument("--predictor", choices=("baseline", "agent"),
                        default="baseline")
    parser.add_argument("--controls", default="",
                        help="none | all | comma-separated control names")
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N cases, for a smoke run")
    parser.add_argument("--out", default="", help="artefact path")
    args = parser.parse_args(argv)

    report, artefact = run(args.task, args.predictor, _controls(args.controls),
                           args.limit)

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
