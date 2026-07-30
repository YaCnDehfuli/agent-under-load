"""Per-class metrics, and no way to ask for bare accuracy.

The miss-classification set is 47% `out-of-scope`. A predictor that answers
`out-of-scope` every time scores 47% accuracy and knows nothing, so accuracy is
not merely discouraged here — there is no field for it and no method that
returns it. `tests/test_metrics.py` drives a degenerate predictor through the
scorer and asserts the report makes it look bad, because a scorer that flatters
a constant answer is broken.

Abstention and rejection are one thing, `UNANSWERED`, and it is a predicted
class that can never be correct. So abstaining costs recall while leaving
precision untouched, which is the honest treatment: an agent that answers only
when sure should be visibly cautious rather than invisibly perfect.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Iterable, Sequence

#: What a rejected verdict, an exhausted turn budget or an explicit abstention
#: all become. Never equal to a truth label, so it can never score as correct.
UNANSWERED = "unanswered"


@dataclasses.dataclass(frozen=True)
class Prediction:
    case_id: str
    truth: str
    predicted: str | None

    @property
    def answer(self) -> str:
        return self.predicted if self.predicted is not None else UNANSWERED

    @property
    def correct(self) -> bool:
        return self.predicted is not None and self.predicted == self.truth


@dataclasses.dataclass(frozen=True)
class ClassMetrics:
    label: str
    support: int          # how many cases truly are this class
    predicted: int        # how many times the predictor said this class
    true_positives: int

    @property
    def precision(self) -> float:
        return self.true_positives / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.support if self.support else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


@dataclasses.dataclass(frozen=True)
class Report:
    """A scored run. Carries no accuracy field, deliberately."""

    task: str
    predictor: str
    classes: tuple[str, ...]
    per_class: tuple[ClassMetrics, ...]
    confusion: dict[str, dict[str, int]]
    total: int
    unanswered: int
    #: Set when the run is not a fair comparison, e.g. it covered a subset.
    note: str = ""

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1 over the true classes.

        Unweighted on purpose: weighting by support would let the majority class
        carry the score, which is the thing this module exists to prevent.
        """
        if not self.per_class:
            return 0.0
        return sum(m.f1 for m in self.per_class) / len(self.per_class)

    @property
    def macro_recall(self) -> float:
        if not self.per_class:
            return 0.0
        return sum(m.recall for m in self.per_class) / len(self.per_class)

    @property
    def worst_class(self) -> ClassMetrics | None:
        return min(self.per_class, key=lambda m: m.f1, default=None)

    def to_json(self) -> dict:
        return {
            "task": self.task,
            "predictor": self.predictor,
            "total": self.total,
            "unanswered": self.unanswered,
            "macro_f1": round(self.macro_f1, 4),
            "macro_recall": round(self.macro_recall, 4),
            "per_class": [
                {"label": m.label, "support": m.support, "predicted": m.predicted,
                 "precision": round(m.precision, 4), "recall": round(m.recall, 4),
                 "f1": round(m.f1, 4)}
                for m in self.per_class
            ],
            "confusion": self.confusion,
            "note": self.note,
        }

    def to_markdown(self) -> str:
        lines = [
            f"**{self.predictor}** on `{self.task}` — {self.total} cases, "
            f"{self.unanswered} unanswered",
            "",
            "| class | support | predicted | precision | recall | F1 |",
            "|---|---|---|---|---|---|",
        ]
        for metrics in self.per_class:
            lines.append(
                f"| `{metrics.label}` | {metrics.support} | {metrics.predicted} "
                f"| {metrics.precision:.2f} | {metrics.recall:.2f} "
                f"| {metrics.f1:.2f} |"
            )
        lines += [
            f"| **macro** | {self.total} | | | {self.macro_recall:.2f} "
            f"| **{self.macro_f1:.2f}** |",
            "",
            "Confusion, rows are truth:",
            "",
            "| truth \\ predicted | " + " | ".join(
                f"`{c}`" for c in (*self.classes, UNANSWERED)) + " |",
            "|---" * (len(self.classes) + 2) + "|",
        ]
        for truth in self.classes:
            row = self.confusion.get(truth, {})
            cells = " | ".join(str(row.get(p, 0))
                               for p in (*self.classes, UNANSWERED))
            lines.append(f"| `{truth}` | {cells} |")
        if self.note:
            lines += ["", f"_{self.note}_"]
        return "\n".join(lines)


def score(
    predictions: Iterable[Prediction],
    classes: Sequence[str],
    task: str,
    predictor: str,
    note: str = "",
) -> Report:
    predictions = list(predictions)
    truths = Counter(p.truth for p in predictions)
    answers = Counter(p.answer for p in predictions)
    hits = Counter(p.truth for p in predictions if p.correct)

    confusion: dict[str, dict[str, int]] = {t: {} for t in classes}
    for prediction in predictions:
        row = confusion.setdefault(prediction.truth, {})
        row[prediction.answer] = row.get(prediction.answer, 0) + 1

    per_class = tuple(
        ClassMetrics(label=label, support=truths.get(label, 0),
                     predicted=answers.get(label, 0),
                     true_positives=hits.get(label, 0))
        for label in classes
    )
    return Report(
        task=task, predictor=predictor, classes=tuple(classes),
        per_class=per_class, confusion=confusion, total=len(predictions),
        unanswered=answers.get(UNANSWERED, 0), note=note,
    )


def compare(baseline: Report, candidate: Report) -> str:
    """The one line that decides whether the agent was worth building."""
    delta = candidate.macro_f1 - baseline.macro_f1
    verdict = "beats" if delta > 0 else ("ties" if abs(delta) < 1e-9 else "loses to")
    return (f"{candidate.predictor} macro-F1 {candidate.macro_f1:.3f} {verdict} "
            f"{baseline.predictor} {baseline.macro_f1:.3f} "
            f"(delta {delta:+.3f})")
