"""Turn one model's run directories into the study's numbers.

    python -m score.analysis --model gpt-oss-120b

Reads every run directory under runs/ that belongs to the model and carries no
defences, one per condition, and prints the tables docs/results-agent.md quotes.
Nothing there is typed by hand.

Three choices shape every number:

- A case's answer is the majority over its repeats. Without a majority (two
  repeats that disagree, or three different answers) it counts as unanswered:
  a model that can't make up its mind hasn't answered.
- Confidence intervals resample captures, not cases. The label is constant per
  capture and cases on one capture share their evidence, so 80 cases are
  nowhere near 80 independent observations; the 24 captures are. True- and
  false-positive captures are resampled separately, keeping 7 and 17: a
  resample with no false-positive capture has no macro-F1 to speak of.
- Conditions are compared to the reference on the same cases, so the paired
  tests (McNemar, and a paired bootstrap of the difference) use each case as
  its own control.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from agent.contracts import Verdict
from score.ledger import ERROR, RunDir
from score.metrics import Prediction, score
from score.run import HARNESS_VERSION, IDENTITY, RUNS

CLASSES = (Verdict.TRUE_POSITIVE.value, Verdict.FALSE_POSITIVE.value)
INCONCLUSIVE = Verdict.INCONCLUSIVE.value
BENCHMARK = RUNS.parent / "benchmark"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0
TURN_LIMIT = "turn limit"

#: Table order. Anything else found on disk goes after these.
ORDER = ("reference", "alert-only", "alert-only-forced", "rule-only",
         "mismatch-cross", "mismatch-same")
BASELINES = {
    "constant-fp": "triage-constant-fp.json",
    "constant-tp": "triage-constant-tp.json",
    "rule-prior": "triage-rule-prior.json",
    "heuristic": "triage-baseline.json",
}


@dataclasses.dataclass
class CaseRuns:
    """Every scored trajectory of one case under one condition."""

    case_id: str
    truth: str
    capture: str
    answers: list[str] = dataclasses.field(default_factory=list)
    donor_truth: str | None = None
    rule_fires_on_donor: bool | None = None

    @property
    def majority(self) -> str | None:
        """The answer more than half the repeats gave, else None (unanswered)."""
        if not self.answers:
            return None
        answer, count = Counter(self.answers).most_common(1)[0]
        if count * 2 <= len(self.answers) or answer == "unanswered":
            return None
        return answer

    @property
    def agreed(self) -> bool:
        return len(set(self.answers)) == 1


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load(runs_root: Path, model_key: str) -> tuple[dict[str, dict[str, CaseRuns]],
                                                   dict[str, list[dict]]]:
    """condition -> case_id -> CaseRuns, and condition -> every record incl. errors."""
    found: dict[str, tuple[Path, dict]] = {}
    for meta_path in sorted(Path(runs_root).glob("*/run.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("model_key") != model_key or meta.get("controls"):
            continue
        if meta.get("harness_version") != HARNESS_VERSION:
            # an earlier harness showed the model something different; its
            # spending still counts against the model's cap, its answers don't
            print(f"skipped {meta_path.parent.name}: harness version "
                  f"{meta.get('harness_version')}, not {HARNESS_VERSION}", file=sys.stderr)
            continue
        condition = meta.get("condition", "reference")
        if condition in found:
            earlier, other = found[condition]
            changed = [k for k in IDENTITY if meta.get(k) != other.get(k)]
            why = (f"different {', '.join(changed)}" if changed
                   else "the same configuration")
            raise ValueError(f"two {condition} runs for {model_key} with {why}: "
                             f"{earlier.parent} and {meta_path.parent}; keep one")
        found[condition] = (meta_path, meta)

    cases: dict[str, dict[str, CaseRuns]] = {}
    records: dict[str, list[dict]] = {}
    for condition, (meta_path, _) in found.items():
        rundir = RunDir(meta_path.parent)
        latest = rundir.latest()
        records[condition] = list(latest.values())
        per_case: dict[str, CaseRuns] = {}
        for (case_id, _), record in sorted(latest.items()):
            if record["outcome"] == ERROR:
                continue
            runs = per_case.setdefault(case_id, CaseRuns(
                case_id=case_id, truth=record["truth"], capture=record["capture"],
                donor_truth=record.get("donor_truth"),
                rule_fires_on_donor=record.get("rule_fires_on_donor")))
            runs.answers.append(record["predicted"] or "unanswered")
        cases[condition] = per_case
    return cases, records


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def macro_f1(cases: list[CaseRuns]) -> float:
    predictions = [Prediction(c.case_id, c.truth, c.majority) for c in cases]
    return score(predictions, CLASSES, "triage", "analysis").macro_f1


def by_capture(cases: list[CaseRuns]) -> dict[str, list[CaseRuns]]:
    out: dict[str, list[CaseRuns]] = {}
    for case in cases:
        out.setdefault(case.capture, []).append(case)
    return out


def strata(groups: dict[str, list[CaseRuns]]) -> list[list[str]]:
    """Captures split by label, the unit and the strata of the bootstrap."""
    out: dict[str, list[str]] = {}
    for capture, cases in sorted(groups.items()):
        out.setdefault(cases[0].truth, []).append(capture)
    return [out[label] for label in sorted(out)]


def cluster_bootstrap(strata: list[list[str]],
                      statistic: Callable[[list[str]], float | None],
                      samples: int = BOOTSTRAP_SAMPLES,
                      seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:
    """95% percentile interval of `statistic` over resampled capture lists.

    Each stratum is resampled with replacement at its own size.
    """
    # nosec B311: a seeded resample for a reproducible interval, not a secret
    rng = random.Random(seed)  # nosec B311
    values = []
    for _ in range(samples):
        value = statistic([rng.choice(stratum) for stratum in strata
                           for _ in stratum])
        if value is not None and not math.isnan(value):
            values.append(value)
    if not values:
        return None
    values.sort()
    return (values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))])


def mcnemar(only_first_right: int, only_second_right: int) -> float:
    """Exact two-sided McNemar p-value: a binomial test on the discordant pairs."""
    n = only_first_right + only_second_right
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_first_right,
                                                   only_second_right) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _correct(case: CaseRuns) -> bool:
    return case.majority == case.truth


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------


def condition_row(condition: str, cases: dict[str, CaseRuns], records: list[dict],
                  reference: dict[str, CaseRuns] | None) -> dict:
    runs = list(cases.values())
    groups = by_capture(runs)
    captures = strata(groups)
    scored = [r for r in records if r["outcome"] != ERROR]
    turns = [r["turns"] for r in scored if r.get("turns") is not None]
    costs = [r["cost_usd"] for r in scored if r.get("cost_usd") is not None]

    def f1_of(sample: list[str]) -> float:
        return macro_f1([c for capture in sample for c in groups[capture]])

    row = {
        "condition": condition,
        "cases": len(runs),
        "captures": len(groups),
        "trajectories": len(scored),
        "errors": len(records) - len(scored),
        "macro_f1": round(macro_f1(runs), 4),
        "macro_f1_ci": _round(cluster_bootstrap(captures, f1_of)),
        "abstained": _share(scored, lambda r: r["predicted"] == INCONCLUSIVE),
        "unanswered": _share(scored, lambda r: r["predicted"] is None),
        "hit_turn_limit": _share(scored, lambda r: any(
            TURN_LIMIT in str(reason) for reason in r.get("rejections") or [])),
        "repeats_agree": _share(runs, lambda c: c.agreed),
        "mean_turns": round(sum(turns) / len(turns), 2) if turns else None,
        "mean_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "total_cost_usd": round(sum(costs), 6),
    }
    if reference is not None and condition != "reference":
        row.update(paired(runs, reference))
    return row


def paired(runs: list[CaseRuns], reference: dict[str, CaseRuns]) -> dict:
    """This condition against the reference, on the cases both have."""
    both = [c for c in runs if c.case_id in reference]
    if not both:
        return {}
    ref = {c.case_id: reference[c.case_id] for c in both}
    groups = by_capture(both)
    captures = strata(groups)

    def pick(sample: list[str]) -> tuple[list[CaseRuns], list[CaseRuns]]:
        chosen = [c for capture in sample for c in groups[capture]]
        return chosen, [ref[c.case_id] for c in chosen]

    def difference(sample: list[str]) -> float:
        mine, theirs = pick(sample)
        return macro_f1(mine) - macro_f1(theirs)

    def retention(sample: list[str]) -> float | None:
        mine, theirs = pick(sample)
        base = macro_f1(theirs)
        return macro_f1(mine) / base if base else None

    ref_f1 = macro_f1(list(ref.values()))
    only_ref = sum(_correct(ref[c.case_id]) and not _correct(c) for c in both)
    only_this = sum(_correct(c) and not _correct(ref[c.case_id]) for c in both)
    return {
        "paired_cases": len(both),
        "difference": round(macro_f1(both) - ref_f1, 4),
        "difference_ci": _round(cluster_bootstrap(captures, difference)),
        "retention": round(macro_f1(both) / ref_f1, 4) if ref_f1 else None,
        "retention_ci": _round(cluster_bootstrap(captures, retention)),
        "only_reference_right": only_ref,
        "only_this_right": only_this,
        "mcnemar_p": round(mcnemar(only_ref, only_this), 4),
    }


def mismatch_row(condition: str, cases: dict[str, CaseRuns]) -> dict:
    """Did the verdict follow the swapped-in evidence or the alert?

    In the cross-label cell the two point opposite ways. In the same-label cell
    they coincide, so the one number there is how often the verdict held.
    """
    runs = list(cases.values())

    def rates(subset: list[CaseRuns]) -> dict:
        out = {
            "cases": len(subset),
            "follows_evidence": _share(subset, lambda c: c.majority == c.donor_truth),
            "follows_alert": _share(subset, lambda c: c.majority == c.truth),
            "abstained": _share(subset, lambda c: c.majority == INCONCLUSIVE),
            "unanswered": _share(subset, lambda c: c.majority is None),
        }
        if condition == "mismatch-same":
            out["held"] = out.pop("follows_alert")
            del out["follows_evidence"]
        return out

    row = {"condition": condition, **rates(runs), "by_rule_fires_on_donor": {}}
    for fires in (True, False, None):
        subset = [c for c in runs if c.rule_fires_on_donor is fires]
        if subset:
            row["by_rule_fires_on_donor"][str(fires).lower()] = rates(subset)
    return row


def probe_row(cases: dict[str, CaseRuns], benchmark: Path) -> dict | None:
    """The forced guess next to rule-prior: is the model's prior the rule's history?"""
    path = benchmark / BASELINES["rule-prior"]
    if not path.exists():
        return None
    prior = {p["case_id"]: p["predicted"]
             for p in json.loads(path.read_text())["predictions"]}
    runs = [c for c in cases.values() if c.case_id in prior]
    return {
        "cases": len(runs),
        "macro_f1": round(macro_f1(runs), 4),
        "rule_prior_macro_f1_same_cases": round(macro_f1([
            dataclasses.replace(c, answers=[prior[c.case_id]]) for c in runs]), 4),
        "agrees_with_rule_prior": _share(runs, lambda c: c.majority == prior[c.case_id]),
    }


def baselines(benchmark: Path) -> dict[str, float]:
    out = {}
    for name, filename in BASELINES.items():
        path = benchmark / filename
        if path.exists():
            out[name] = json.loads(path.read_text())["report"]["macro_f1"]
    # guessing each label at its base rate gives each class an expected F1
    # equal to its prevalence, so the expected macro-F1 is exactly 0.5
    out["class prior (random)"] = 0.5
    return out


def analyse(runs_root: Path, model_key: str, benchmark: Path = BENCHMARK) -> dict:
    cases, records = load(runs_root, model_key)
    if not cases:
        raise ValueError(f"no undefended run directories for {model_key} under {runs_root}")
    order = [c for c in ORDER if c in cases] + sorted(set(cases) - set(ORDER))
    reference = cases.get("reference")
    result = {
        "model_key": model_key,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
                      "unit": "capture"},
        "conditions": [condition_row(c, cases[c], records[c], reference) for c in order],
        "mismatch": [mismatch_row(c, cases[c]) for c in order if c.startswith("mismatch")],
        "baselines": baselines(benchmark),
    }
    if "alert-only-forced" in cases:
        result["forced_probe"] = probe_row(cases["alert-only-forced"], benchmark)
    return result


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def to_markdown(result: dict) -> str:
    lines = [
        f"**{result['model_key']}**: majority over repeats; 95% intervals from "
        f"{result['bootstrap']['samples']} resamples of captures.",
        "",
        "| condition | cases | macro-F1 [95% CI] | retention [95% CI] | McNemar p "
        "| abstained | unanswered | turn limit | repeats agree | turns | $/traj |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["conditions"]:
        retention = (f"{row['retention']:.2f} {_ci(row.get('retention_ci'))}"
                     if row.get("retention") is not None else "—")
        lines.append(
            f"| {row['condition']} | {row['cases']} "
            f"| {row['macro_f1']:.2f} {_ci(row['macro_f1_ci'])} | {retention} "
            f"| {_num(row.get('mcnemar_p'), 3)} | {_pct(row['abstained'])} "
            f"| {_pct(row['unanswered'])} | {_pct(row['hit_turn_limit'])} "
            f"| {_pct(row['repeats_agree'])} | {_num(row['mean_turns'], 1)} "
            f"| {_num(row['mean_cost_usd'], 4)} |")
    if result["mismatch"]:
        lines += ["", "| mismatch | cases | follows evidence | follows alert / held "
                      "| abstained | unanswered |", "|---|---|---|---|---|---|"]
        for row in result["mismatch"]:
            lines.append(
                f"| {row['condition']} | {row['cases']} "
                f"| {_pct(row.get('follows_evidence'))} "
                f"| {_pct(row.get('follows_alert', row.get('held')))} "
                f"| {_pct(row['abstained'])} | {_pct(row['unanswered'])} |")
        lines.append("")
        lines.append("_Split by whether the alert's rule matches anything in the "
                     "donor capture is in the JSON; that matcher is approximate._")
    probe = result.get("forced_probe")
    if probe:
        lines += ["", f"Forced guess from the alert alone: macro-F1 "
                      f"{probe['macro_f1']:.2f} on {probe['cases']} cases, against "
                      f"rule-prior {probe['rule_prior_macro_f1_same_cases']:.2f} on "
                      f"the same cases; agrees with rule-prior on "
                      f"{_pct(probe['agrees_with_rule_prior'])}."]
    lines += ["", "| baseline (reads no model) | macro-F1 |", "|---|---|"]
    lines += [f"| {name} | {value:.2f} |" for name, value in result["baselines"].items()]
    return "\n".join(lines)


def _share(items: list, predicate: Callable) -> float | None:
    return round(sum(1 for i in items if predicate(i)) / len(items), 4) if items else None


def _round(interval: tuple[float, float] | None) -> list[float] | None:
    return [round(v, 4) for v in interval] if interval else None


def _ci(interval: list[float] | None) -> str:
    return f"[{interval[0]:.2f}, {interval[1]:.2f}]" if interval else ""


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="model key in models.yml")
    parser.add_argument("--runs", default=str(RUNS), help="where the run directories are")
    parser.add_argument("--out", default="", help="write the numbers as JSON here too")
    args = parser.parse_args(argv)
    try:
        result = analyse(Path(args.runs), args.model)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
        print(f"written: {args.out}", file=sys.stderr)
    print(to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
