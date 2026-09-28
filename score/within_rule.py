"""The within-rule analysis fixed in docs/prereg-within-rule.md.

    python -m score.within_rule --model gpt-6-luna

For rules the published detector fires on for both labels, the alert text is
the same whichever capture it came from, so a verdict that differs between
those captures can only have come from the telemetry. This reads the existing
reference and alert-only-forced runs; it makes no model calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agent import corpus
from score.analysis import (
    BENCHMARK,
    CaseRuns,
    by_capture,
    cluster_bootstrap,
    condition_cases,
    macro_f1,
    mcnemar,
    strata,
)
from score.ledger import ERROR, RunDir
from score.run import RUNS

TP, FP = "true_positive", "false_positive"

#: The subset as written down before the analysis ran; derived again from the
#: detector's results every time, and the run refuses if the two disagree.
PREREGISTERED = {
    "250ae82f-736e-4844-a68b-0b5e8cc887da": (8, 4),
    "4a1b6da0-d94f-4fc3-98fc-2d9cb9e5ee76": (13, 3),
    "4b447e9d-1c82-47f6-9a01-a1bb0a22d684": (12, 7),
    "5ef9853e-4d0e-4a70-846f-a9ca37d876da": (8, 4),
    "678dfc63-fefb-47a5-a04c-26bcf8cc9f65": (6, 2),
    "962fe167-e48d-4fd6-9974-11e5b9a5d6d1": (3, 1),
}


def detector_fires(results: dict) -> dict[str, set[str]]:
    """rule id -> every capture the published detector fired on, either label."""
    fires: dict[str, set[str]] = {}
    for capture, per_capture in results["per_capture"].items():
        for rule, info in per_capture["rules"].items():
            if info["class"] == "detected":
                fires.setdefault(rule, set()).add(capture)
    for rule, info in results["false_positives"]["per_rule"].items():
        fires.setdefault(rule, set()).update(info["captures_hit"] or ())
    return fires


def mixed_rules(results: dict) -> set[str]:
    """Rules that fire on at least one capture of each label."""
    on_tp = {rule for per_capture in results["per_capture"].values()
             for rule, info in per_capture["rules"].items()
             if info["class"] == "detected"}
    on_fp = {rule for rule, info in results["false_positives"]["per_rule"].items()
             if info["captures_hit"]}
    return on_tp & on_fp


def _rule(case_id: str) -> str:
    return case_id.split(":", 1)[1].split("@", 1)[0]


def check_subset(cases: dict[str, CaseRuns], rules: set[str],
                 expected: dict = PREREGISTERED) -> None:
    found = {}
    for case in cases.values():
        if _rule(case.case_id) in rules:
            n, tp = found.get(_rule(case.case_id), (0, 0))
            found[_rule(case.case_id)] = (n + 1, tp + (case.truth == TP))
    if found != expected:
        raise ValueError(f"the subset in the runs {found} is not the one written "
                         f"down in docs/prereg-within-rule.md {expected}")


def primary(reference: dict[str, CaseRuns], forced: dict[str, CaseRuns],
            subset: list[str], seed: int = 0) -> dict:
    """Reference minus forced guess, macro-F1, with a capture-resampled interval."""
    ref = [reference[c] for c in subset]
    probe = {c: forced[c] for c in subset}
    groups = by_capture(ref)

    def difference(sample: list[str]) -> float:
        mine = [c for capture in sample for c in groups[capture]]
        return macro_f1(mine) - macro_f1([probe[c.case_id] for c in mine])

    only_ref = sum(r.majority == r.truth and probe[r.case_id].majority != r.truth
                   for r in ref)
    only_probe = sum(r.majority != r.truth and probe[r.case_id].majority == r.truth
                     for r in ref)
    interval = cluster_bootstrap(strata(groups), difference, seed=seed)
    return {
        "cases": len(ref),
        "reference_macro_f1": round(macro_f1(ref), 4),
        "forced_macro_f1": round(macro_f1(list(probe.values())), 4),
        "difference": round(macro_f1(ref) - macro_f1(list(probe.values())), 4),
        "difference_ci": [round(v, 4) for v in interval] if interval else None,
        "evidence_sensitive": bool(interval and interval[0] > 0),
        "only_reference_right": only_ref,
        "only_forced_right": only_probe,
        "mcnemar_p": round(mcnemar(only_ref, only_probe), 4),
    }


def per_rule(reference: dict[str, CaseRuns], subset: list[str]) -> dict[str, dict]:
    out = {}
    for rule in sorted({_rule(c) for c in subset}):
        cases = [reference[c] for c in subset if _rule(c) == rule]
        tps = [c for c in cases if c.truth == TP]
        fps = [c for c in cases if c.truth == FP]
        right = {c.case_id: c.majority == c.truth for c in cases}
        pairs = [(t, f) for t in tps for f in fps]
        out[rule] = {
            "cases": len(cases),
            "tp_recall": _share([right[c.case_id] for c in tps]),
            "fp_specificity": _share([right[c.case_id] for c in fps]),
            "both_right_pairs": _share([right[t.case_id] and right[f.case_id]
                                        for t, f in pairs]),
            "pairs": len(pairs),
        }
    return out


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def baseline_predictions(name: str, benchmark: Path = BENCHMARK) -> dict[str, CaseRuns]:
    data = json.loads((benchmark / f"triage-{name}.json").read_text())
    return {p["case_id"]: CaseRuns(p["case_id"], p["truth"], p["capture"],
                                   answers=[p["predicted"] or "unanswered"])
            for p in data["predictions"]}


def mismatch_by_donor(run_dir: Path, fires: dict[str, set[str]]) -> dict:
    """Exploratory: the swap conditions split by whether the rule fires on the donor."""
    per_case: dict[str, dict] = {}
    for record in RunDir(run_dir).latest().values():
        if record["outcome"] == ERROR:
            continue
        entry = per_case.setdefault(record["case_id"], {
            "runs": CaseRuns(record["case_id"], record["truth"], record["capture"]),
            "donor_truth": record.get("donor_truth"),
            "fires": record.get("donor") in fires.get(_rule(record["case_id"]), set()),
        })
        entry["runs"].answers.append(record["predicted"] or "unanswered")
    out = {}
    for label in (TP, FP):
        for fired in (True, False):
            group = [e for e in per_case.values()
                     if e["runs"].truth == label and e["fires"] is fired]
            if not group:
                continue
            out[f"{label}, rule {'fires' if fired else 'does not fire'} on donor"] = {
                "cases": len(group),
                "held": _share([e["runs"].majority == e["runs"].truth for e in group]),
                "followed_donor_label": _share([e["runs"].majority == e["donor_truth"]
                                                for e in group]),
                "unanswered": _share([e["runs"].majority is None for e in group]),
            }
    return out


def analyse(model_key: str, runs_root: Path = RUNS, results: dict | None = None,
            benchmark: Path = BENCHMARK, expected: dict = PREREGISTERED) -> dict:
    results = results or corpus.load_results()
    rules = mixed_rules(results)
    found = condition_cases(runs_root, model_key, "reference")
    forced = condition_cases(runs_root, model_key, "alert-only-forced")
    if found is None or forced is None:
        raise ValueError(f"{model_key} needs both a reference and an "
                         "alert-only-forced run under runs/")
    reference = found[1]
    check_subset(reference, rules, expected)
    subset = sorted(c for c in reference if _rule(c) in rules)

    out = {
        "model_key": model_key,
        "subset": {"rules": sorted(rules), "cases": len(subset),
                   "true_positives": sum(reference[c].truth == TP for c in subset)},
        "primary": primary(reference, forced[1], subset),
        "per_rule": per_rule(reference, subset),
        "leave_one_rule_out": {
            rule: (primary(reference, forced[1], rest)["difference_ci"] if rest else None)
            for rule in sorted(rules)
            for rest in [[c for c in subset if _rule(c) != rule]]},
        "subset_abstained": _share([reference[c].majority == "inconclusive"
                                    for c in subset]),
        "subset_unanswered": _share([reference[c].majority is None for c in subset]),
        "baselines_on_subset": {},
    }
    for name in ("baseline", "rule-prior"):
        preds = baseline_predictions(name, benchmark)
        out["baselines_on_subset"][name] = round(macro_f1([preds[c] for c in subset]), 4)

    heuristic = baseline_predictions("baseline", benchmark)
    both = [c for c in reference if c in heuristic]
    only_model = sum(reference[c].majority == reference[c].truth
                     and heuristic[c].majority != heuristic[c].truth for c in both)
    only_heuristic = sum(reference[c].majority != reference[c].truth
                         and heuristic[c].majority == heuristic[c].truth for c in both)
    out["against_heuristic_all_cases"] = {
        "cases": len(both), "model_macro_f1": round(macro_f1([reference[c] for c in both]), 4),
        "heuristic_macro_f1": round(macro_f1([heuristic[c] for c in both]), 4),
        "only_model_right": only_model, "only_heuristic_right": only_heuristic,
        "mcnemar_p": round(mcnemar(only_model, only_heuristic), 4),
    }

    fires = detector_fires(results)
    out["exploratory_mismatch_by_donor"] = {}
    for condition in ("mismatch-cross", "mismatch-same"):
        swapped = condition_cases(runs_root, model_key, condition)
        if swapped is not None:
            out["exploratory_mismatch_by_donor"][condition] = mismatch_by_donor(
                swapped[0], fires)
    return out


def to_markdown(result: dict) -> str:
    p = result["primary"]
    ci = p["difference_ci"]
    lines = [
        f"**{result['model_key']}**, rules that fire on both labels: "
        f"{result['subset']['cases']} cases ({result['subset']['true_positives']} "
        "true positives).",
        "",
        f"Reference macro-F1 {p['reference_macro_f1']:.2f} against its forced guess "
        f"from the alert alone {p['forced_macro_f1']:.2f}: difference "
        f"{p['difference']:+.2f}"
        + (f" [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "")
        + f", McNemar p = {p['mcnemar_p']:.3f}. Evidence-sensitive by the "
        f"preregistered rule: **{'yes' if p['evidence_sensitive'] else 'not detectable'}**.",
        "",
        "Baselines on the same cases: "
        + ", ".join(f"{k} {v:.2f}" for k, v in result["baselines_on_subset"].items())
        + f". Abstained {result['subset_abstained']:.0%}, unanswered "
        f"{result['subset_unanswered']:.0%}.",
        "",
        "| rule | cases | TP recall | FP specificity | both right, per TP×FP pair |",
        "|---|---|---|---|---|",
    ]
    for rule, row in result["per_rule"].items():
        lines.append(f"| `{rule[:8]}` | {row['cases']} | {_pct(row['tp_recall'])} "
                     f"| {_pct(row['fp_specificity'])} "
                     f"| {_pct(row['both_right_pairs'])} of {row['pairs']} |")
    lines += ["", "Leaving one rule out, the interval for the difference: "
              + "; ".join(f"`{r[:8]}` {_ci(v)}"
                          for r, v in result["leave_one_rule_out"].items())]
    h = result["against_heuristic_all_cases"]
    lines += ["", f"All {h['cases']} cases, against the heuristic: "
              f"{h['model_macro_f1']:.2f} vs {h['heuristic_macro_f1']:.2f}, "
              f"{h['only_model_right']} right only for the model, "
              f"{h['only_heuristic_right']} only for the heuristic, McNemar p = "
              f"{h['mcnemar_p']:.3f}."]
    if result["exploratory_mismatch_by_donor"]:
        lines += ["", "Exploratory, the swaps by whether the published detector fires "
                      "on the donor:", "",
                  "| condition | group | cases | held | followed donor | unanswered |",
                  "|---|---|---|---|---|---|"]
        for condition, groups in result["exploratory_mismatch_by_donor"].items():
            for name, row in groups.items():
                lines.append(f"| {condition} | {name} | {row['cases']} "
                             f"| {_pct(row['held'])} | {_pct(row['followed_donor_label'])} "
                             f"| {_pct(row['unanswered'])} |")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _ci(interval: list[float] | None) -> str:
    return "—" if not interval else f"[{interval[0]:+.2f}, {interval[1]:+.2f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", default=str(RUNS))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    try:
        result = analyse(args.model, Path(args.runs))
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print(f"written: {args.out}", file=sys.stderr)
    print(to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
