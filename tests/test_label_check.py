"""The label review and the technique-question comparison, on synthetic runs."""

from __future__ import annotations

import json

import pytest
import yaml

from score import label_check
from score.run import HARNESS_VERSION

TP, FP = "true_positive", "false_positive"
CASES = [("tp:r@tp1", "tp1", TP), ("tp:r@tp2", "tp2", TP),
         ("fp:r@clear1", "clear1", FP), ("fp:r@clear2", "clear2", FP),
         ("fp:r@pth", "pth", FP)]


def _review(tmp_path, answers):
    path = tmp_path / "review.yml"
    path.write_text(yaml.safe_dump({"captures": [
        {"capture": c, "credential_theft": a} for c, a in answers.items()]}))
    return path


REVIEW = {"clear1": "no", "clear2": "no", "pth": "unsure"}


def _run(root, name, condition, answer, cited=None):
    path = root / name
    (path / "trajectories").mkdir(parents=True)
    (path / "run.json").write_text(json.dumps({
        "model_key": "m", "controls": [], "condition": condition, "task": "triage",
        "harness_version": HARNESS_VERSION}))
    with open(path / "cases.jsonl", "w") as fh:
        for case_id, capture, truth in CASES:
            trajectory = f"{capture}.jsonl"
            (path / "trajectories" / trajectory).write_text(json.dumps(
                {"kind": "verdict_evidence", "cited": cited or []}) + "\n")
            fh.write(json.dumps({
                "case_id": case_id, "repeat": 1, "truth": truth, "capture": capture,
                "predicted": answer(truth, capture), "outcome": "answered",
                "trajectory": trajectory}) + "\n")


def test_an_unfinished_review_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pth"):
        label_check.read_review(_review(tmp_path, {"clear1": "no", "pth": None}))


def test_unsure_counts_as_ambiguous(tmp_path):
    assert label_check.read_review(_review(tmp_path, REVIEW)) == {
        "clear1": "clear", "clear2": "clear", "pth": "ambiguous"}


def test_the_question_comparison_and_the_split_by_review(tmp_path):
    # the reference calls everything true positive; the technique question
    # gets the clear false positives right but still calls pth a true positive
    _run(tmp_path, "ref", "reference", lambda t, c: TP)
    _run(tmp_path, "tech", "technique-question",
         lambda t, c: FP if c.startswith("clear") else TP,
         cited=[{"event_index": 3, "field": "GrantedAccess", "event_id": 10}])
    result = label_check.analyse("m", tmp_path, _review(tmp_path, REVIEW))

    assert result["review"]["ambiguous"] == ["pth"]
    assert result["primary"]["difference"] > 0
    assert result["reference"]["fp_specificity_clear"]["rate"] == 0.0
    assert result["technique_question"]["fp_specificity_clear"]["rate"] == 1.0
    assert result["technique_question"]["fp_specificity_ambiguous"]["rate"] == 0.0
    assert result["technique_question"]["tp_recall"]["rate"] == 1.0
    rescored = result["rescored_without_ambiguous"]["technique-question"]
    assert rescored["without_ambiguous"] == 1.0 > rescored["all"]
    assert result["citations"]["true_positive called true_positive"] == {"access event": 2}
    text = label_check.to_markdown(result)
    assert "false-positive specificity, clear | 0% (0/2) | 100% (2/2)" in text


def test_a_missing_technique_question_run_is_refused(tmp_path):
    _run(tmp_path, "ref", "reference", lambda t, c: TP)
    with pytest.raises(ValueError, match="technique-question"):
        label_check.analyse("m", tmp_path, _review(tmp_path, REVIEW))


def test_the_committed_sheet_is_the_seventeen_false_positive_captures():
    captures = yaml.safe_load(label_check.REVIEW.read_text())["captures"]
    assert len(captures) == 17 and all(c["description"] for c in captures)
