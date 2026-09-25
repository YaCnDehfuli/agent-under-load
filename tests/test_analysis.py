"""The analysis step: majority votes, capture-level intervals, paired tests."""

from __future__ import annotations

import json
import math

import pytest

from score.analysis import (
    CaseRuns,
    analyse,
    cluster_bootstrap,
    load,
    macro_f1,
    mcnemar,
    to_markdown,
)
from score.run import HARNESS_VERSION

TP, FP = "true_positive", "false_positive"


def _write_run(root, name, condition, rows, model_key="m", **meta):
    """rows: (case_id, capture, truth, [answers per repeat], extra fields)."""
    path = root / name
    path.mkdir(parents=True)
    (path / "run.json").write_text(json.dumps(
        {"model_key": model_key, "controls": [], "condition": condition,
         "task": "triage", "harness_version": HARNESS_VERSION, **meta}))
    with open(path / "cases.jsonl", "w") as fh:
        for case_id, capture, truth, answers, *extra in rows:
            for repeat, answer in enumerate(answers, start=1):
                record = {"case_id": case_id, "repeat": repeat, "truth": truth,
                          "capture": capture, "predicted": answer,
                          "outcome": "answered" if answer else "unanswered",
                          "rejections": [], "turns": 3, "cost_usd": 0.001,
                          **(extra[0] if extra else {})}
                if answer == "error":
                    record.update(predicted=None, outcome="error")
                fh.write(json.dumps(record) + "\n")
    return path


def _reference_rows():
    # 2 TP captures, 2 FP captures, 2 cases each; everything right
    return [(f"{cap}:{i}", cap, truth, [truth])
            for cap, truth in (("tp1", TP), ("tp2", TP), ("fp1", FP), ("fp2", FP))
            for i in range(2)]


# -- the vote ----------------------------------------------------------------


@pytest.mark.parametrize("answers,majority", [
    ([TP], TP),
    ([TP, TP, FP], TP),
    ([TP, FP], None),                      # a tie is no answer
    ([TP, FP, "inconclusive"], None),
    (["unanswered", "unanswered", TP], None),
    (["inconclusive", "inconclusive", TP], "inconclusive"),
])
def test_the_majority_vote(answers, majority):
    case = CaseRuns("c", TP, "cap", answers=answers)
    assert case.majority == majority


def test_errored_trajectories_are_left_out_of_the_vote(tmp_path):
    _write_run(tmp_path, "r", "reference", [("c", "cap", TP, [TP, "error", TP])])
    cases, records = load(tmp_path, "m")
    assert cases["reference"]["c"].answers == [TP, TP]
    assert len(records["reference"]) == 3


# -- statistics ------------------------------------------------------------------


def test_mcnemar_matches_a_hand_computed_binomial():
    # 1 vs 7 discordant: 2 * (C(8,0) + C(8,1)) / 2^8 = 18/256
    assert math.isclose(mcnemar(1, 7), 18 / 256)
    assert mcnemar(7, 1) == mcnemar(1, 7)
    assert mcnemar(0, 0) == 1.0
    assert mcnemar(4, 4) == 1.0


def test_the_bootstrap_resamples_captures_and_is_seeded():
    groups = [["a", "b", "c", "d"]]
    values = {"a": 0.0, "b": 1.0, "c": 1.0, "d": 1.0}

    def mean(sample):
        return sum(values[g] for g in sample) / len(sample)

    first = cluster_bootstrap(groups, mean, samples=500, seed=1)
    assert first == cluster_bootstrap(groups, mean, samples=500, seed=1)
    assert first[0] < 0.75 < first[1]


def test_each_label_keeps_its_share_of_captures():
    seen = []
    cluster_bootstrap([["fp1", "fp2", "fp3"], ["tp1"]],
                      lambda s: seen.append(s) or 0.0, samples=200)
    assert all(sum(c.startswith("tp") for c in s) == 1 and len(s) == 4 for s in seen)


def test_one_capture_gives_a_zero_width_interval():
    cases = [CaseRuns(f"c{i}", TP, "only", answers=[TP if i % 2 else FP])
             for i in range(6)]
    interval = cluster_bootstrap([["only"]], lambda s: macro_f1(cases), samples=50)
    assert interval[0] == interval[1]


# -- the tables ----------------------------------------------------------------------


def test_a_perfect_reference_and_a_blind_condition(tmp_path):
    _write_run(tmp_path, "ref", "reference", _reference_rows())
    blind = [(cid, cap, truth, ["inconclusive"]) for cid, cap, truth, _ in _reference_rows()]
    _write_run(tmp_path, "blind", "alert-only", blind)
    result = analyse(tmp_path, "m", benchmark=tmp_path)
    ref, alert = result["conditions"]
    assert ref["condition"] == "reference" and ref["macro_f1"] == 1.0
    assert ref["macro_f1_ci"] == [1.0, 1.0]
    assert alert["macro_f1"] == 0.0 and alert["abstained"] == 1.0
    assert alert["retention"] == 0.0
    assert (alert["only_reference_right"], alert["only_this_right"]) == (8, 0)
    assert math.isclose(alert["mcnemar_p"], round(2 / 2 ** 8, 4))


def test_mismatch_rates_split_evidence_from_alert(tmp_path):
    _write_run(tmp_path, "ref", "reference", _reference_rows())
    rows = [
        ("tp1:0", "tp1", TP, [FP], {"donor_truth": FP, "rule_fires_on_donor": True}),
        ("tp1:1", "tp1", TP, [TP], {"donor_truth": FP, "rule_fires_on_donor": False}),
        ("fp1:0", "fp1", FP, [TP], {"donor_truth": TP, "rule_fires_on_donor": True}),
        ("fp1:1", "fp1", FP, [None], {"donor_truth": TP, "rule_fires_on_donor": None}),
    ]
    _write_run(tmp_path, "cross", "mismatch-cross", rows)
    same = [(cid, cap, truth, [truth], {"donor_truth": truth})
            for cid, cap, truth, *_ in rows]
    _write_run(tmp_path, "same", "mismatch-same", same)

    result = analyse(tmp_path, "m", benchmark=tmp_path)
    cross, same_row = result["mismatch"]
    assert (cross["follows_evidence"], cross["follows_alert"],
            cross["unanswered"]) == (0.5, 0.25, 0.25)
    assert cross["by_rule_fires_on_donor"]["true"]["follows_evidence"] == 1.0
    assert same_row["held"] == 1.0 and "follows_evidence" not in same_row


def test_the_forced_probe_is_set_against_rule_prior(tmp_path):
    _write_run(tmp_path, "forced", "alert-only-forced",
               [(cid, cap, truth, [TP]) for cid, cap, truth, _ in _reference_rows()])
    prior = [{"case_id": cid, "predicted": TP if cap == "tp1" else FP}
             for cid, cap, _, _ in _reference_rows()]
    (tmp_path / "triage-rule-prior.json").write_text(json.dumps(
        {"predictions": prior, "report": {"macro_f1": 0.7}}))
    probe = analyse(tmp_path, "m", benchmark=tmp_path)["forced_probe"]
    assert probe["cases"] == 8
    # the model guessed TP everywhere; rule-prior says TP only on tp1's 2 cases
    assert probe["agrees_with_rule_prior"] == 0.25
    assert "rule-prior" in analyse(tmp_path, "m", benchmark=tmp_path)["baselines"]


def test_two_runs_of_one_condition_are_refused(tmp_path):
    _write_run(tmp_path, "a", "reference", _reference_rows())
    _write_run(tmp_path, "b", "reference", _reference_rows(), system_prompt_sha256="x")
    with pytest.raises(ValueError, match="system_prompt_sha256"):
        load(tmp_path, "m")


def test_other_models_and_defended_runs_are_ignored(tmp_path):
    _write_run(tmp_path, "ref", "reference", _reference_rows())
    _write_run(tmp_path, "other", "reference", _reference_rows(), model_key="n")
    path = _write_run(tmp_path, "defended", "reference", _reference_rows())
    meta = json.loads((path / "run.json").read_text())
    (path / "run.json").write_text(json.dumps({**meta, "controls": ["provenance_tags"]}))
    cases, _ = load(tmp_path, "m")
    assert list(cases) == ["reference"]


def test_the_markdown_carries_every_condition(tmp_path):
    _write_run(tmp_path, "ref", "reference", _reference_rows())
    _write_run(tmp_path, "rule", "rule-only",
               [(cid, cap, truth, [TP]) for cid, cap, truth, _ in _reference_rows()])
    text = to_markdown(analyse(tmp_path, "m", benchmark=tmp_path))
    assert "| reference |" in text and "| rule-only |" in text
    assert "class prior (random)" in text


def test_runs_from_an_earlier_harness_are_skipped_not_refused(tmp_path, capsys):
    _write_run(tmp_path, "ref", "reference", _reference_rows())
    old = _write_run(tmp_path, "smoke-old", "reference", _reference_rows())
    meta = json.loads((old / "run.json").read_text())
    del meta["harness_version"], meta["condition"]
    (old / "run.json").write_text(json.dumps(meta))
    cases, _ = load(tmp_path, "m")
    assert list(cases) == ["reference"]
    assert "skipped smoke-old" in capsys.readouterr().err


def test_a_model_that_always_says_tp_gains_no_evidence_shift(tmp_path):
    # it "follows the evidence" on every FP case (TP donor) without reading it
    rows = _reference_rows()
    _write_run(tmp_path, "ref", "reference",
               [(cid, cap, truth, [TP]) for cid, cap, truth, _ in rows])
    other = {TP: FP, FP: TP}
    _write_run(tmp_path, "cross", "mismatch-cross",
               [(cid, cap, truth, [TP], {"donor_truth": other[truth]})
                for cid, cap, truth, _ in rows])
    cross = analyse(tmp_path, "m", benchmark=tmp_path)["mismatch"][0]
    fp_half = cross["by_truth"][FP]
    assert fp_half["follows_evidence"] == 1.0
    assert fp_half["donor_label_in_reference"] == 1.0
    assert fp_half["shift_toward_evidence"] == 0.0
    assert cross["by_truth"][TP]["shift_toward_evidence"] == 0.0


def test_unanswered_cases_are_separated_from_wrong_ones(tmp_path):
    rows = _reference_rows()
    # half the cases unanswered, the other half all right
    _write_run(tmp_path, "ref", "reference",
               [(cid, cap, truth, [truth if cid.endswith(":0") else None])
                for cid, cap, truth, _ in rows])
    ref = analyse(tmp_path, "m", benchmark=tmp_path)["conditions"][0]
    assert ref["decided"] == 0.5
    assert ref["macro_f1_decided"] == 1.0
    assert ref["macro_f1"] < 1.0
