"""Pin the scorer, including against a predictor that knows nothing.

A scorer that flatters a constant answer is broken, and on a set that is 47% one
class it is easy to build one by accident. So the central test here drives a
degenerate predictor through the real scorer and asserts the report makes it look
bad.
"""

from __future__ import annotations

import pytest

from agent import corpus
from score.metrics import UNANSWERED, Prediction, compare, score

MISS = corpus.MISS_CLASSES

#: The real distribution, so the degenerate predictor is tested against the
#: imbalance it would exploit.
SUPPORT = {"out-of-scope": 273, "miss-logic": 207, "miss-telemetry": 57,
           "detected": 44}


def _cases():
    out = []
    for label, count in SUPPORT.items():
        out.extend([label] * count)
    return out


def _report(predict, predictor="test"):
    return score(
        [Prediction(case_id=f"c{i}", truth=truth, predicted=predict(truth))
         for i, truth in enumerate(_cases())],
        MISS, task="miss_classification", predictor=predictor,
    )


def test_a_constant_predictor_scores_badly():
    """Answers the 47% majority class every time. Accuracy would say 47%."""
    report = _report(lambda truth: "out-of-scope", "always-out-of-scope")

    assert report.macro_f1 < 0.25
    assert report.macro_recall == pytest.approx(0.25)
    # three of four classes are never recovered at all
    zeros = [m.label for m in report.per_class if m.recall == 0.0]
    assert sorted(zeros) == ["detected", "miss-logic", "miss-telemetry"]


def test_the_report_has_no_accuracy_anywhere():
    """Not discouraged — absent. There is no field and no method."""
    report = _report(lambda truth: "out-of-scope")
    assert not hasattr(report, "accuracy")
    assert "accuracy" not in report.to_json()
    assert "accuracy" not in report.to_markdown().lower()


def test_a_perfect_predictor_scores_one():
    report = _report(lambda truth: truth, "oracle")
    assert report.macro_f1 == pytest.approx(1.0)
    assert report.unanswered == 0


def test_macro_f1_is_unweighted_so_the_majority_class_cannot_carry_it():
    """Right on the big class, wrong on the small ones, must not look good."""
    def predict(truth):
        return truth if truth == "out-of-scope" else "out-of-scope"

    report = _report(predict, "majority-only")
    out_of_scope = next(m for m in report.per_class if m.label == "out-of-scope")
    assert out_of_scope.recall == pytest.approx(1.0)
    assert report.macro_f1 < 0.25


def test_abstaining_costs_recall_and_leaves_precision_alone():
    """Otherwise "always inconclusive" is a perfect-precision strategy."""
    def predict(truth):
        return truth if truth == "detected" else None

    report = _report(predict, "answers-only-when-sure")
    detected = next(m for m in report.per_class if m.label == "detected")
    assert detected.precision == pytest.approx(1.0)
    assert detected.recall == pytest.approx(1.0)

    for label in ("out-of-scope", "miss-logic", "miss-telemetry"):
        metrics = next(m for m in report.per_class if m.label == label)
        assert metrics.recall == 0.0
        assert metrics.precision == 0.0
    assert report.unanswered == 581 - SUPPORT["detected"]
    assert report.macro_f1 == pytest.approx(0.25)


def test_unanswered_can_never_score_as_correct():
    report = score([Prediction("c", "miss-logic", None)], MISS, "t", "p")
    assert report.unanswered == 1
    assert report.macro_f1 == 0.0
    assert report.confusion["miss-logic"][UNANSWERED] == 1


def test_confusion_rows_account_for_every_case():
    report = _report(lambda truth: "miss-logic")
    for label, support in SUPPORT.items():
        assert sum(report.confusion[label].values()) == support
    assert report.total == 581


def test_a_class_never_predicted_has_zero_precision_not_a_crash():
    report = _report(lambda truth: "miss-logic")
    detected = next(m for m in report.per_class if m.label == "detected")
    assert detected.predicted == 0
    assert detected.precision == 0.0
    assert detected.f1 == 0.0


def test_worst_class_is_surfaced():
    report = _report(lambda truth: "out-of-scope")
    assert report.worst_class is not None
    assert report.worst_class.f1 == 0.0


def test_compare_says_plainly_which_way_it_went():
    weak = _report(lambda truth: "out-of-scope", "baseline")
    strong = _report(lambda truth: truth, "agent")
    assert "beats" in compare(weak, strong)
    assert "loses to" in compare(strong, weak)
    assert "ties" in compare(weak, _report(lambda t: "out-of-scope", "other"))


def test_markdown_reports_support_per_class():
    """A recall figure without its support is not interpretable."""
    table = _report(lambda truth: truth).to_markdown()
    for label, support in SUPPORT.items():
        assert f"`{label}` | {support}" in table
