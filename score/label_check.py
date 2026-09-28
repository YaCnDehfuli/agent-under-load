"""The label and question checks fixed in docs/prereg-labels-and-question.md.

    python -m score.label_check --model gpt-6-luna

Needs benchmark/label-review.yml filled in, and a technique-question run next
to the model's reference run. Makes no model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from score.analysis import CaseRuns, condition_cases, load, macro_f1, paired
from score.ledger import ERROR, RunDir
from score.run import RUNS

REVIEW = RUNS.parent / "benchmark" / "label-review.yml"
TP, FP = "true_positive", "false_positive"
ANSWERS = ("yes", "no", "unsure")
#: Events that record a handle being opened on a process: Sysmon ProcessAccess,
#: and the Security log's handle request and object access.
ACCESS_EVENTS = {10, 4656, 4663}


def read_review(path: Path = REVIEW) -> dict[str, str]:
    """capture -> "clear" or "ambiguous". Refuses a review that isn't finished."""
    captures = (yaml.safe_load(path.read_text()) or {}).get("captures") or []
    out, missing = {}, []
    for entry in captures:
        answer = str(entry.get("credential_theft") or "").strip().lower()
        if answer not in ANSWERS:
            missing.append(entry["capture"])
            continue
        out[entry["capture"]] = "clear" if answer == "no" else "ambiguous"
    if missing:
        raise ValueError(f"the label review isn't finished: credential_theft is "
                         f"not yes/no/unsure for {', '.join(missing)}")
    return out


def _rate(cases: list[CaseRuns]) -> dict:
    right = sum(c.majority == c.truth for c in cases)
    return {"cases": len(cases), "right": right,
            "rate": round(right / len(cases), 4) if cases else None}


def by_label(cases: dict[str, CaseRuns], review: dict[str, str]) -> dict:
    runs = list(cases.values())
    return {
        "tp_recall": _rate([c for c in runs if c.truth == TP]),
        "fp_specificity_clear": _rate([c for c in runs if c.truth == FP
                                       and review.get(c.capture) == "clear"]),
        "fp_specificity_ambiguous": _rate([c for c in runs if c.truth == FP
                                           and review.get(c.capture) == "ambiguous"]),
    }


def citations(run_dir: Path) -> dict:
    """Do decisive verdicts cite a process-access event, or only other events?"""
    rundir = RunDir(run_dir)
    counts: dict[str, dict[str, int]] = {}
    for record in rundir.latest().values():
        if record["outcome"] == ERROR or record["predicted"] not in (TP, FP):
            continue
        path = rundir.trajectories / record["trajectory"]
        cited = [c for line in path.read_text().splitlines() if line.strip()
                 for entry in [json.loads(line)]
                 if entry.get("kind") == "verdict_evidence"
                 for c in entry.get("cited", [])]
        kind = ("access event" if any(c.get("event_id") in ACCESS_EVENTS for c in cited)
                else "other events only" if cited else "nothing logged")
        key = f"{record['truth']} called {record['predicted']}"
        counts.setdefault(key, {}).setdefault(kind, 0)
        counts[key][kind] += 1
    return counts


def analyse(model_key: str, runs_root: Path = RUNS, review_path: Path = REVIEW) -> dict:
    review = read_review(review_path)
    cases, _ = load(runs_root, model_key)
    for needed in ("reference", "technique-question"):
        if needed not in cases:
            raise ValueError(f"{model_key} has no {needed} run under {runs_root}")
    reference, technique = cases["reference"], cases["technique-question"]
    clear = {c for c, v in review.items() if v == "clear"}

    def without_ambiguous(runs: dict[str, CaseRuns]) -> list[CaseRuns]:
        return [c for c in runs.values() if c.truth == TP or c.capture in clear]

    return {
        "model_key": model_key,
        "review": {"clear": sorted(clear),
                   "ambiguous": sorted(c for c, v in review.items() if v == "ambiguous")},
        "primary": paired(list(technique.values()), reference),
        "reference": by_label(reference, review),
        "technique_question": by_label(technique, review),
        "rescored_without_ambiguous": {
            condition: {"all": round(macro_f1(list(runs.values())), 4),
                        "without_ambiguous": round(macro_f1(without_ambiguous(runs)), 4)}
            for condition, runs in cases.items()},
        "citations": citations(condition_cases(runs_root, model_key,
                                               "technique-question")[0]),
    }


def to_markdown(result: dict) -> str:
    p = result["primary"]
    ci = p.get("difference_ci")
    lines = [
        f"**{result['model_key']}**. Review: {len(result['review']['clear'])} clear, "
        f"{len(result['review']['ambiguous'])} ambiguous false-positive captures"
        + (f" ({', '.join(result['review']['ambiguous'])})"
           if result["review"]["ambiguous"] else "") + ".",
        "",
        f"Technique question minus reference, macro-F1: {p['difference']:+.2f}"
        + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
        + f", McNemar p = {p['mcnemar_p']:.3f}. The wording "
        + ("**matters**" if ci and (ci[0] > 0 or ci[1] < 0) else "**isn't shown to matter**")
        + " by the rule written down in advance.",
        "",
        "| | reference | technique question |",
        "|---|---|---|",
    ]
    for key, name in (("tp_recall", "true-positive recall"),
                      ("fp_specificity_clear", "false-positive specificity, clear"),
                      ("fp_specificity_ambiguous", "false-positive specificity, ambiguous")):
        cells = [result[side][key] for side in ("reference", "technique_question")]
        lines.append(f"| {name} | " + " | ".join(
            "—" if c["rate"] is None else f"{c['rate']:.0%} ({c['right']}/{c['cases']})"
            for c in cells) + " |")
    lines += ["", "| condition | macro-F1 | without ambiguous captures |", "|---|---|---|"]
    for condition, row in result["rescored_without_ambiguous"].items():
        lines.append(f"| {condition} | {row['all']:.2f} | {row['without_ambiguous']:.2f} |")
    lines += ["", "What decisive verdicts in the technique-question run cite:", "",
              "| verdict | access event | other events only | nothing logged |",
              "|---|---|---|---|"]
    for key, row in sorted(result["citations"].items()):
        lines.append(f"| {key} | {row.get('access event', 0)} "
                     f"| {row.get('other events only', 0)} | {row.get('nothing logged', 0)} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", default=str(RUNS))
    parser.add_argument("--review", default=str(REVIEW))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    try:
        result = analyse(args.model, Path(args.runs), Path(args.review))
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print(f"written: {args.out}", file=sys.stderr)
    print(to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
