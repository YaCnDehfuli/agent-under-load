"""Run directories: what they record, how they resume, and when they stop spending.

Driven by the scripted model over synthetic cases, so none of this needs a
network or a corpus.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter

import pytest

from agent.contracts import EvidenceCitation, TriageVerdict, Verdict
from agent.models import ModelError, ModelReply, ScriptedModel, ToolCall, Usage
from score import run as score_run
from score.ledger import BudgetExceeded, RunDir, check_budget, cost_usd, model_spend
from score.run import balanced_sample, run_agent
from tests.support import FakeStore, synthetic_triage_case, write_rule

PRICE = {"input": 1.0, "cached_input": 0.1, "output": 4.0, "source": "test"}
ENTRY = {"provider": "scripted", "model": "scripted", "price": PRICE}
UNPRICED = {"provider": "scripted", "model": "scripted",
            "price": {"input": None, "cached_input": None, "output": None}}

VERDICT = TriageVerdict(
    verdict=Verdict.TRUE_POSITIVE, confidence=0.9,
    evidence=[EvidenceCitation(event_index=1, field="TargetImage",
                               quote="lsass.exe", supports="target")],
    rationale="procdump opened lsass")


class CountingModel(ScriptedModel):
    """Looks up the rule, then answers. Each turn costs 1,000 input and 100 output tokens."""

    def __init__(self, answer=VERDICT, fail_on: set[str] = frozenset(),
                 interrupt_on: str = ""):
        super().__init__(self._reply)
        self.answer = answer
        self.fail_on = fail_on
        self.interrupt_on = interrupt_on
        self.calls = 0

    def _reply(self, messages):
        self.calls += 1
        if self.interrupt_on and self.interrupt_on in messages[0].content:
            raise KeyboardInterrupt
        if any(case_id in messages[0].content for case_id in self.fail_on):
            raise ModelError("provider returned 503 after 6 attempt(s)")
        usage = Usage(input_tokens=1000, cached_input_tokens=0, output_tokens=100)
        if not any(m.role == "tool" for m in messages):
            return ModelReply(tool_calls=[ToolCall(name="lookup_rule", arguments={},
                                                   call_id="c1")], usage=usage)
        if self.answer is None:
            return ModelReply(invalid="contract rejected the verdict", usage=usage)
        return ModelReply(answer=self.answer, usage=usage)


def _cases(tmp_path, n=3):
    base = synthetic_triage_case(write_rule(tmp_path))
    # the case prompt carries the rule id, so a distinct id per case lets the
    # model fail on one of them by name
    return [dataclasses.replace(base, case_id=f"tp:case{i}",
                                rule=dataclasses.replace(base.rule, id=f"rule-case{i}"))
            for i in range(n)]


def _run(tmp_path, model, cases, **kwargs):
    kwargs.setdefault("model_entry", ENTRY)
    return run_agent(cases, model, "scripted", run_dir=tmp_path / "run",
                     store=FakeStore(), **kwargs)


# -- what a run records -----------------------------------------------------


def test_a_run_writes_its_records_and_trajectories(tmp_path):
    cases = _cases(tmp_path)
    meta = _run(tmp_path, CountingModel(), cases, repeats=2)
    rundir = RunDir(tmp_path / "run")

    records = list(rundir.records())
    assert len(records) == 6
    assert {(r["case_id"], r["repeat"]) for r in records} == \
        {(c.case_id, n) for c in cases for n in (1, 2)}
    assert all(r["outcome"] == "answered" and r["predicted"] == "true_positive"
               for r in records)

    first = records[0]
    assert (first["turns"], first["input_tokens"], first["output_tokens"]) == (2, 2000, 200)
    assert first["cost_usd"] == pytest.approx((2000 * 1.0 + 200 * 4.0) / 1e6)

    # the trajectory on disk is the audit log whose head the record carries
    lines = (rundir.trajectories / first["trajectory"]).read_text().splitlines()
    assert json.loads(lines[-1])["digest"] == first["audit_head"]

    assert meta["totals"]["answered"] == 6
    assert set(meta["reports"]) == {"1", "2"}
    assert meta["reports"]["1"]["total"] == 3


def test_the_run_records_what_produced_it(tmp_path, monkeypatch):
    monkeypatch.setattr(score_run, "_manifest_digest", lambda: "a" * 64)
    meta = _run(tmp_path, CountingModel(), _cases(tmp_path, 1))
    assert meta["corpus_manifest_sha256"] == "a" * 64
    assert set(meta["git"]) == {"sha", "dirty"}
    assert len(meta["system_prompt_sha256"]) == 64
    assert meta["model_entry"] == ENTRY
    assert meta["model_config"]["provider"] == "scripted"
    assert meta["budget"]["pricing_source"] == "test"


# -- resuming -----------------------------------------------------------------


def test_a_resumed_run_does_not_repeat_finished_work(tmp_path):
    cases = _cases(tmp_path)
    _run(tmp_path, CountingModel(), cases)
    again = CountingModel()
    _run(tmp_path, again, cases)
    assert again.calls == 0
    assert len(list(RunDir(tmp_path / "run").records())) == 3


def test_an_interrupted_run_resumes_without_duplicates(tmp_path):
    cases = _cases(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, CountingModel(interrupt_on="rule-case2"), cases)
    assert [r["case_id"] for r in RunDir(tmp_path / "run").records()] == \
        ["tp:case0", "tp:case1"]

    with pytest.raises(ValueError, match="model_entry"):
        _run(tmp_path, CountingModel(), cases, model_entry=dict(ENTRY, model="other"))

    meta = _run(tmp_path, CountingModel(), cases)
    records = list(RunDir(tmp_path / "run").records())
    assert [r["case_id"] for r in records] == ["tp:case0", "tp:case1", "tp:case2"]
    assert meta["totals"]["answered"] == 3


def test_more_repeats_extend_a_finished_run(tmp_path):
    cases = _cases(tmp_path)
    _run(tmp_path, CountingModel(), cases)
    more = CountingModel()
    meta = _run(tmp_path, more, cases, repeats=2)
    assert more.calls == 3 * 2  # three new trajectories, two turns each
    assert meta["totals"]["trajectories"] == 6


def test_an_infrastructure_error_is_recorded_and_retried(tmp_path):
    cases = _cases(tmp_path)
    meta = _run(tmp_path, CountingModel(fail_on={"rule-case1"}), cases)
    assert meta["totals"]["error"] == 1
    assert meta["reports"]["1"]["total"] == 2  # errors are not scored

    errored = [r for r in RunDir(tmp_path / "run").records() if r["outcome"] == "error"]
    assert errored[0]["case_id"] == "tp:case1"
    assert "503" in errored[0]["error"]

    retry = CountingModel()
    meta = _run(tmp_path, retry, cases)
    assert retry.calls == 2  # only the errored case ran again
    assert meta["totals"]["error"] == 0
    assert meta["totals"]["answered"] == 3


def test_a_model_that_never_answers_is_unanswered_and_not_retried(tmp_path):
    cases = _cases(tmp_path, 1)
    meta = _run(tmp_path, CountingModel(answer=None), cases)
    assert meta["totals"]["unanswered"] == 1
    assert meta["reports"]["1"]["unanswered"] == 1

    again = CountingModel()
    _run(tmp_path, again, cases)
    assert again.calls == 0


def test_a_run_directory_is_not_reused_for_another_configuration(tmp_path):
    cases = _cases(tmp_path, 1)
    _run(tmp_path, CountingModel(), cases)
    other = dict(ENTRY, model="another-model")
    with pytest.raises(ValueError, match="model_entry"):
        _run(tmp_path, CountingModel(), cases, model_entry=other)


# -- cost and the cap ------------------------------------------------------------


def test_cost_counts_cached_input_at_the_cached_rate():
    usage = {"input_tokens": 10_000, "cached_input_tokens": 8_000, "output_tokens": 500}
    assert cost_usd(usage, PRICE) == pytest.approx(
        (2_000 * 1.0 + 8_000 * 0.1 + 500 * 4.0) / 1e6)


def test_unreported_cached_tokens_are_billed_as_uncached():
    usage = {"input_tokens": 10_000, "cached_input_tokens": None, "output_tokens": 500}
    assert cost_usd(usage, PRICE) == pytest.approx((10_000 * 1.0 + 500 * 4.0) / 1e6)


def test_cost_is_unknown_without_a_price_or_without_usage():
    usage = {"input_tokens": 10, "output_tokens": 5}
    assert cost_usd(usage, UNPRICED["price"]) is None
    assert cost_usd({"input_tokens": None, "output_tokens": 5}, PRICE) is None


def test_a_cap_needs_prices(tmp_path):
    with pytest.raises(ValueError, match="no price"):
        _run(tmp_path, CountingModel(), _cases(tmp_path, 1),
             model_entry=UNPRICED, budget_usd=1.0)


def test_the_cap_stops_the_run_before_it_is_passed(tmp_path):
    # each trajectory costs 0.0028; the cap allows two and not a third
    cases = _cases(tmp_path, 1)
    first = _run(tmp_path, CountingModel(), cases, budget_usd=0.006)
    assert first["totals"]["trajectories"] == 1

    more = _cases(tmp_path, 4)
    with pytest.raises(BudgetExceeded):
        _run(tmp_path, CountingModel(), more, budget_usd=0.006)


def test_the_cap_stops_mid_run_when_no_projection_was_possible(tmp_path):
    # nothing measured yet, so the run starts; it stops once the measured mean
    # says the next trajectory would pass the cap
    meta = _run(tmp_path, CountingModel(), _cases(tmp_path, 5), budget_usd=0.006)
    assert meta["totals"]["trajectories"] == 2
    assert meta["budget"]["budget_stop"] is True
    assert meta["budget"]["actual_cost_usd"] <= 0.006
    assert meta["budget"]["projected_remaining_cost_usd"] == pytest.approx(3 * 0.0028)


def test_the_projection_is_the_measured_mean_times_what_is_left():
    assert check_budget(None, 0.0, None, 10) is None
    assert check_budget(5.0, 1.0, 0.25, 8) == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        check_budget(5.0, 1.0, 0.25, 20)


# -- the per-model cap --------------------------------------------------------------


def _run_in(tmp_path, name, model, cases, key="scripted", **kwargs):
    return run_agent(cases, model, key, ENTRY, run_dir=tmp_path / name,
                     store=FakeStore(), **kwargs)


def test_spending_in_other_run_directories_counts_against_the_model_cap(tmp_path):
    # 0.0028 a trajectory; two already spent elsewhere leave room for one more
    _run_in(tmp_path, "smoke", CountingModel(), _cases(tmp_path, 2))
    assert model_spend(tmp_path, "scripted") == pytest.approx(2 * 0.0028)

    meta = _run_in(tmp_path, "matrix", CountingModel(), _cases(tmp_path, 1),
                   model_cap_usd=0.0085)
    assert meta["budget"]["model_spent_usd"] == pytest.approx(3 * 0.0028)

    with pytest.raises(BudgetExceeded):
        _run_in(tmp_path, "more", CountingModel(), _cases(tmp_path, 3),
                model_cap_usd=0.0085)


def test_another_models_spending_does_not_count(tmp_path):
    _run_in(tmp_path, "other", CountingModel(), _cases(tmp_path, 3), key="other")
    meta = _run_in(tmp_path, "mine", CountingModel(), _cases(tmp_path, 1),
                   model_cap_usd=0.003)
    assert meta["totals"]["trajectories"] == 1
    assert meta["budget"]["model_spent_usd"] == pytest.approx(0.0028)


def test_a_model_cap_needs_prices(tmp_path):
    with pytest.raises(ValueError, match="no price"):
        run_agent(_cases(tmp_path, 1), CountingModel(), "scripted", UNPRICED,
                  run_dir=tmp_path / "run", store=FakeStore(), model_cap_usd=1.0)


# -- the committed registry --------------------------------------------------------


def test_every_paid_model_has_a_price_and_a_cap():
    from agent.models import registry

    for key, entry in registry().items():
        if entry["provider"] in ("ollama", "anthropic"):
            continue
        price = entry["price"]
        assert price["input"] and price["output"] and price["source"], key
        assert entry["budget_usd"] > 0, key


def test_registry_entries_survive_the_trip_through_run_json():
    """Resume compares the entry with the copy in run.json, so it must round-trip."""
    from agent.models import registry

    for key, entry in registry().items():
        assert json.loads(json.dumps(entry, default=str)) == entry, key


# -- provider-reported cost and errors ----------------------------------------------


class ReportingModel(CountingModel):
    """Like CountingModel, but each turn arrives with the cost the provider billed."""

    def _reply(self, messages):
        reply = super()._reply(messages)
        reply.usage = Usage(input_tokens=1000, cached_input_tokens=0, output_tokens=100,
                            reported_cost_usd=0.0001)
        reply.served_by = "SomeHost"
        return reply


def test_a_cost_the_provider_reports_is_used_over_the_price_table(tmp_path):
    _run(tmp_path, ReportingModel(), _cases(tmp_path, 1))
    record = next(RunDir(tmp_path / "run").records())
    assert record["cost_usd"] == pytest.approx(0.0002)  # two turns, not 0.0028
    assert record["served_by"] == ["SomeHost"]


def test_run_json_lists_the_errors_it_saw(tmp_path):
    meta = _run(tmp_path, CountingModel(fail_on={"rule-case0", "rule-case1"}),
                _cases(tmp_path))
    assert meta["errors"] == {"ModelError: provider returned 503 after 6 attempt(s)": 2}


# -- parallel workers ------------------------------------------------------------


def test_workers_produce_one_record_per_trajectory(tmp_path):
    cases = _cases(tmp_path, 6)
    meta = _run(tmp_path, CountingModel(), cases, repeats=2, workers=3)
    records = list(RunDir(tmp_path / "run").records())
    assert len(records) == 12
    assert {(r["case_id"], r["repeat"]) for r in records} == \
        {(c.case_id, n) for c in cases for n in (1, 2)}
    assert meta["totals"]["answered"] == 12
    for r in records:
        lines = (tmp_path / "run" / "trajectories" / r["trajectory"]).read_text().splitlines()
        assert json.loads(lines[-1])["digest"] == r["audit_head"]


def test_the_cap_counts_trajectories_still_running(tmp_path):
    # 0.0028 each. Nothing is measured at the start, so two go out unguarded;
    # after that a third fits under 0.0085, but a fourth started alongside it
    # would not. Ignoring the one in flight would have spent 0.0112.
    meta = _run(tmp_path, CountingModel(), _cases(tmp_path, 5),
                model_cap_usd=0.0085, workers=2)
    assert meta["totals"]["trajectories"] == 3
    assert meta["budget"]["model_spent_usd"] == pytest.approx(3 * 0.0028)
    assert meta["budget"]["budget_stop"] is True


# -- a balanced smoke sample ------------------------------------------------


def _labelled_cases(tmp_path):
    base = synthetic_triage_case(write_rule(tmp_path))
    layout = {"tp1": ("true_positive", 4), "tp2": ("true_positive", 3),
              "fp1": ("false_positive", 2), "fp2": ("false_positive", 2),
              "fp3": ("false_positive", 2)}
    cases = []
    for cid, (label, n) in layout.items():
        capture = dataclasses.replace(base.capture, id=cid)
        cases += [dataclasses.replace(base, case_id=f"{cid}:{i}", capture=capture,
                                      truth=label) for i in range(n)]
    return cases


def test_a_sample_alternates_labels_and_spreads_over_captures(tmp_path):
    picked = balanced_sample(_labelled_cases(tmp_path), 6, seed=0)
    labels = Counter(c.truth for c in picked)
    assert labels == {"true_positive": 3, "false_positive": 3}
    fp_captures = {c.capture.id for c in picked if c.truth == "false_positive"}
    assert fp_captures == {"fp1", "fp2", "fp3"}  # one each before any repeats
    assert {c.capture.id for c in picked if c.truth == "true_positive"} == {"tp1", "tp2"}


def test_a_sample_is_seeded(tmp_path):
    cases = _labelled_cases(tmp_path)
    ids = lambda seed: [c.case_id for c in balanced_sample(cases, 5, seed)]
    assert ids(0) == ids(0)
    assert len({tuple(ids(s)) for s in range(6)}) > 1


def test_a_sample_larger_than_the_set_takes_everything(tmp_path):
    cases = _labelled_cases(tmp_path)
    assert len(balanced_sample(cases, 100)) == len(cases)
