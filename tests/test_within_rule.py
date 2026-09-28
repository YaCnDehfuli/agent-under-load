"""The preregistered within-rule analysis, on synthetic runs and the real subset."""

from __future__ import annotations

import json

import pytest

from score import within_rule
from score.run import HARNESS_VERSION

TP, FP = "true_positive", "false_positive"

# rule "r1" fires on tp1, tp2, fp1, fp2; rule "r2" only on true positives
RESULTS = {
    "per_capture": {
        "tp1": {"rules": {"r1": {"class": "detected"}, "r2": {"class": "detected"}}},
        "tp2": {"rules": {"r1": {"class": "detected"}}},
    },
    "false_positives": {"per_rule": {
        "r1": {"captures_hit": ["fp1", "fp2"]}, "r2": {"captures_hit": []}}},
}
CASES = [("tp:r1@tp1", "tp1", TP), ("tp:r1@tp2", "tp2", TP),
         ("fp:r1@fp1", "fp1", FP), ("fp:r1@fp2", "fp2", FP),
         ("tp:r2@tp1", "tp1", TP)]
EXPECTED = {"r1": (4, 2)}


def _write(root, name, condition, answers, extra=None):
    path = root / name
    path.mkdir(parents=True)
    (path / "run.json").write_text(json.dumps({
        "model_key": "m", "controls": [], "condition": condition, "task": "triage",
        "harness_version": HARNESS_VERSION}))
    with open(path / "cases.jsonl", "w") as fh:
        for case_id, capture, truth in CASES:
            for repeat, answer in enumerate(answers(case_id, truth), start=1):
                fh.write(json.dumps({
                    "case_id": case_id, "repeat": repeat, "truth": truth,
                    "capture": capture, "predicted": answer, "outcome": "answered",
                    **((extra or {}).get(case_id, {}))}) + "\n")


def _baselines(root):
    for name in ("baseline", "rule-prior"):
        (root / f"triage-{name}.json").write_text(json.dumps({"predictions": [
            {"case_id": c, "truth": t, "capture": cap, "predicted": TP}
            for c, cap, t in CASES]}))


def _analyse(tmp_path):
    _baselines(tmp_path)
    return within_rule.analyse("m", tmp_path, RESULTS, benchmark=tmp_path,
                               expected=EXPECTED)


def test_a_model_that_reads_the_capture_beats_its_guess_from_the_alert(tmp_path):
    _write(tmp_path, "ref", "reference", lambda c, t: [t, t, t])
    _write(tmp_path, "forced", "alert-only-forced", lambda c, t: [TP, TP, TP])
    result = _analyse(tmp_path)
    p = result["primary"]
    assert result["subset"]["cases"] == 4  # r2 fires on one label only
    assert p["reference_macro_f1"] == 1.0 and p["difference"] > 0
    assert (p["only_reference_right"], p["only_forced_right"]) == (2, 0)
    assert result["per_rule"]["r1"]["both_right_pairs"] == 1.0
    assert result["per_rule"]["r1"]["pairs"] == 4


def test_the_same_answer_for_every_capture_is_not_evidence_use(tmp_path):
    _write(tmp_path, "ref", "reference", lambda c, t: [TP, TP, TP])
    _write(tmp_path, "forced", "alert-only-forced", lambda c, t: [TP, TP, TP])
    p = _analyse(tmp_path)["primary"]
    assert p["difference"] == 0 and not p["evidence_sensitive"]


def test_a_subset_other_than_the_one_written_down_is_refused(tmp_path):
    _write(tmp_path, "ref", "reference", lambda c, t: [t])
    _write(tmp_path, "forced", "alert-only-forced", lambda c, t: [TP])
    _baselines(tmp_path)
    with pytest.raises(ValueError, match="prereg"):
        within_rule.analyse("m", tmp_path, RESULTS, benchmark=tmp_path,
                            expected={"r1": (5, 2)})


def test_swaps_are_split_by_whether_the_rule_fires_on_the_donor(tmp_path):
    _write(tmp_path, "ref", "reference", lambda c, t: [t])
    _write(tmp_path, "forced", "alert-only-forced", lambda c, t: [TP])
    donors = {"tp:r1@tp1": {"donor": "fp1", "donor_truth": FP},     # fires
              "tp:r1@tp2": {"donor": "fp9", "donor_truth": FP},     # does not
              "fp:r1@fp1": {"donor": "tp2", "donor_truth": TP},
              "fp:r1@fp2": {"donor": "tp2", "donor_truth": TP},
              "tp:r2@tp1": {"donor": "fp1", "donor_truth": FP}}
    _write(tmp_path, "cross", "mismatch-cross", lambda c, t: [FP], extra=donors)
    split = _analyse(tmp_path)["exploratory_mismatch_by_donor"]["mismatch-cross"]
    fires = split["true_positive, rule fires on donor"]
    assert fires["cases"] == 1 and fires["followed_donor_label"] == 1.0
    assert split["true_positive, rule does not fire on donor"]["cases"] == 2


def test_the_real_subset_is_the_one_written_down():
    """Against the sibling's published results and the committed baselines."""
    from agent import corpus
    try:
        results = corpus.load_results()
    except corpus.CorpusError:
        pytest.skip("detection-under-load is not checked out")
    rules = within_rule.mixed_rules(results)
    heuristic = within_rule.baseline_predictions("baseline")
    subset = {c: r for c, r in heuristic.items() if within_rule._rule(c) in rules}
    within_rule.check_subset(subset, rules)
    prior = within_rule.baseline_predictions("rule-prior")
    assert round(within_rule.macro_f1(list(subset.values())), 3) == 0.495
    assert round(within_rule.macro_f1([prior[c] for c in subset]), 3) == 0.408
