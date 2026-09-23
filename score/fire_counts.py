"""Recount how many events each triage alert matched, per capture.

    python -m score.fire_counts

Writes benchmark/fire-counts.json, which the alert header reads its "events
matched" line from. Both labels are counted the same way, by this repo's
matcher, on the capture the case is about. The sibling's results can't be used
for this: they give a true positive's count per capture but a false positive's
only as the rule's total over the whole benign corpus, and that difference on
its own separated the labels.

Prints the check that matters before any agent sees the numbers: how the counts
are distributed per label, and how well a single threshold on them alone would
do.
"""

from __future__ import annotations

import json
import statistics
import sys

from agent import corpus
from agent.baseline import analyse_rule, count_matches
from agent.tools import CaptureStore

MATCHER = ("agent.baseline.count_matches: this repo's approximate Sigma matcher "
           "(prefilter plus the AND spine of the detection, NOT branches ignored)")


def recount(cases, store) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in sorted(cases, key=lambda c: (c.capture.id, c.case_id)):
        counts[case.case_id] = count_matches(analyse_rule(case.rule),
                                             store.load(case.capture))
    return counts


def best_threshold(counts: dict[str, int], truth: dict[str, str]) -> tuple[int, int, str]:
    """Most cases a single cut on the count gets right, the cut, and which side is FP."""
    best = (0, 0, "")
    for cut in sorted(set(counts.values())) + [max(counts.values()) + 1]:
        for high in ("false_positive", "true_positive"):
            right = sum((counts[c] >= cut) == (truth[c] == high) for c in counts)
            best = max(best, (right, cut, high))
    return best


def report(counts: dict[str, int], truth: dict[str, str]) -> str:
    lines = []
    for label in ("true_positive", "false_positive"):
        values = sorted(n for c, n in counts.items() if truth[c] == label)
        lines.append(f"{label}: n={len(values)} median={statistics.median(values)} "
                     f"values={values}")
    right, cut, high = best_threshold(counts, truth)
    lines.append(f"best single threshold: {right}/{len(counts)} correct "
                 f"(count >= {cut} -> {high})")
    zero = sorted(c for c, n in counts.items() if n == 0)
    lines.append(f"cases the recount finds no match for: {len(zero)}")
    lines.extend(f"  {c} ({truth[c]})" for c in zero)
    return "\n".join(lines)


def main() -> int:
    cases = corpus.evaluation_set().triage
    counts = recount(cases, CaptureStore())
    truth = {c.case_id: c.truth for c in cases}
    corpus.FIRE_COUNTS.write_text(json.dumps({
        "matcher": MATCHER,
        "corpus_manifest_sha256": corpus.manifest_digest(),
        "counts": counts,
        "no_match": sorted(c for c, n in counts.items() if n == 0),
    }, indent=1, sort_keys=True) + "\n")
    print(report(counts, truth))
    print(f"\nwritten: {corpus.FIRE_COUNTS}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
