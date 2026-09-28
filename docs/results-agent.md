# Triage results: the agents, and the baseline they're measured against

The baselines come from `python -m score.run --task triage --predictor baseline`
(and `--task miss`), the agents from `python -m score.analysis --model <key>` and
`python -m score.within_rule --model gpt-6-luna`. Every figure below is read out
of the files those write in `benchmark/`.

## What the agents did

Four models ran the full matrix: 80 cases, six conditions, three repeats each,
about $20 in total. A case's answer is the majority over its repeats. Numbers
come from `benchmark/analysis-<model>.json`, written by `score.analysis`.

| | reference macro-F1 [95% CI] | its guess from the alert alone | answered | repeats agree |
|---|---|---|---|---|
| gpt-6-luna | 0.73 [0.60, 0.86] | 0.37 | 100% | 81% |
| deepseek-v4-pro | 0.59 [0.46, 0.70] | 0.52 | 96% | 89% |
| gpt-oss-20b | 0.57 [0.44, 0.71] | 0.54 | 76% | 38% |
| gpt-oss-120b | 0.46 [0.34, 0.58] | 0.57 | 84% | 69% |

Only Luna gets clearly more out of the investigation than out of the alert
text. The other three do about as well guessing from the alert as they do with
tools, and the gpt-oss models often run out of turns. Luna's 0.73 against the
heuristic's 0.60 looks like a win but isn't one yet: on the same 80 cases it is
right on 16 the heuristic misses and wrong on 10 the heuristic gets, McNemar
p = 0.33.

Given nothing but the alert, every model mostly answers `inconclusive`. That's
reasonable, and it's why there is a forced variant that asks for the guess
anyway.

### The evidence swaps measured something else than intended

The swap conditions hand the agent another capture's events under the same
alert. Every model did badly on them, but most of the donors don't contain
anything the alert's rule fires on: the published detector fires on the donor
in 19 of 80 cross-label swaps and 25 of 80 same-label swaps. An agent that looks
for what the rule describes, doesn't find it, and calls the alert unsupported is
doing the analyst's job. Splitting Luna's swaps by that
(`benchmark/within-rule-gpt-6-luna.json`, exploratory):

- same label, rule fires on the donor: verdict held in 16 of 16; rule doesn't
  fire: held in 2 of 28
- cross label, rule fires on the donor: followed the donor's label in 3 of 4
  true-positive cases and 12 of 15 false-positive ones

So when the evidence supports the alert, Luna's verdict mostly follows what the
capture actually is, including calling a false-positive alert a true positive
once real attack telemetry sits under it. The earlier reading, that no model
would ever flip a false positive to a true positive, came from the donors that
didn't support the alert. The swap results are kept as they are but read as
sensitivity to whether the alert is supported, not as accuracy.

### Within one rule, does the verdict follow the capture?

That was the question the swaps were after, and it can be asked of the
reference runs directly. Six rules fire on captures of both labels; for them the
alert text is identical whichever capture it came from, so a verdict that
differs between their captures can only come from the telemetry. The analysis
was written down before it ran (`docs/prereg-within-rule.md`).

On those 50 cases Luna scores 0.72 with the telemetry and 0.32 guessing from the
alert, a difference of +0.40 [+0.27, +0.52], and the interval stays above zero
with any one rule left out. By the rule set in advance, Luna uses the evidence.
The heuristic gets 0.49 and rule-prior 0.41 on the same cases. It is uneven
across rules: all 16 true/false-positive pairs right for one rule, none of 2 for
another, and 33–80% for the rest.

What that does and doesn't say: on this corpus, Luna's verdict depends on the
capture's telemetry when the alert is held fixed. It says nothing about other
techniques or environments, the base is 7 attack captures from two labs, and
Luna was picked after seeing the results.

### Were the misses the labels, or the question?

Luna still called 16 of the 36 false positives true positives, and it wasn't
obvious those were mistakes: the labels mean "not T1003.001 credential
dumping", while the prompt asks whether "the activity the rule describes"
happened. Two checks, written down first (`docs/prereg-labels-and-question.md`).

The labels held up. Going through all 17 false-positive captures from the
dataset's own descriptions (`benchmark/label-review.yml`), none is credential
theft from LSASS, and only over-pass-the-hash touches LSASS at all: it writes a
key in rather than reading one out. The injection captures that looked
suspicious inject into notepad.

The question mattered, a little. The same reference run with only the
definition of a true positive changed to "credential theft from LSASS memory,
ATT&CK T1003.001" scored 0.84 against 0.73, a difference of +0.10 [+0.01,
+0.23] by the preregistered interval, though the per-case McNemar test doesn't
reach significance (p = 0.12). The gain is on the false positives (specificity
56% to 72%) without losing true positives (91% to 93%). So part of what looked
like Luna misjudging benign captures was Luna answering a looser question than
the one the labels ask. The 0.84 is not a result for Luna: the wording was
chosen after seeing the misses, and the reference run stays the number to quote.

This run also logged what each verdict cites. Almost all of them, right or
wrong, cite the process-access event on LSASS itself (Sysmon EventID 10); only
26 of 232 decisive verdicts rest on other events alone. Luna reads the access
the rule is about, the handle and its rights, much more than the surrounding
activity, which fits where its remaining false positives are.

Running it: keys live in a `.env` file at the repo root (ignored by git), which
agent runs read on start-up; `--env-file` points elsewhere, and a key already
exported in the shell takes precedence. Each run writes a
directory under `runs/` and resumes it if run again. Spending is capped twice:
`--budget-usd` for the run, and the model's `budget_usd` in `models.yml` across
every run of that model; with `--workers N` the cap also counts trajectories
still in flight. DeepSeek runs go outside its peak hours (01:00-04:00
and 06:00-10:00 UTC on weekdays), when its rates double.

`python -m score.analysis --model <key>` turns a model's run directories, one
per condition, into the tables quoted here: macro-F1 per condition with 95%
intervals from resampling captures (7 true-positive and 17 false-positive
captures, resampled separately), retention and exact McNemar tests against the
reference on the same cases, abstention and turn-limit rates, and for the
mismatch conditions whether the verdict followed the swapped-in evidence or the
alert. `--sample N` gives a smoke run both labels; `--limit` doesn't.

## Triage verdict — the informative comparison

80 cases: 44 true positives (a rule fired on a recorded LSASS intrusion), 36
false positives (the same rules firing on captures benign for T1003.001).

| class | support | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| `true_positive` | 44 | 70 | 0.63 | 1.00 | 0.77 |
| `false_positive` | 36 | 10 | 1.00 | 0.28 | 0.43 |
| **macro** | 80 | | | 0.64 | **0.60** |

| truth \ predicted | `true_positive` | `false_positive` | `unanswered` |
|---|---|---|---|
| `true_positive` | 44 | 0 | 0 |
| `false_positive` | 26 | 10 | 0 |

**Baseline macro-F1: 0.60.** This is the bar that matters, and it is beatable.

The failure is one-directional and worth reading closely: the heuristic finds
every true positive and calls 26 of 36 false positives true as well. It looks for
a handle to LSASS carrying memory-read rights, or a published dumping tool in a
command line. In the benign corpus it keeps finding them.

That is not a bug in the heuristic. It is a fact about the corpus, and it is the
reason this task is hard. The benign captures are *atomic attack simulations* —
Empire, Covenant, PurpleSharp, mimikatz exercising other techniques. Processes
open handles to LSASS in them constantly. Distinguishing "a tool read LSASS
memory to steal credentials" from "a tool touched LSASS while doing something
else" needs more than the presence of a handle, which is precisely the judgement
an analyst is paid for and the thing an agent might plausibly add.

Some examples the baseline gets wrong, all `false_positive` called
`true_positive`, all on `credential_access/empire_over_pth_patch_lsass`: rules
`06d71506`, `0f920ebe`, `4a1b6da0`, `4b447e9d`. That capture patches LSASS for
pass-the-hash rather than dumping it for credentials — a genuinely fine
distinction, and one the rules themselves do not draw.

### Baselines that read no evidence

The evidence conditions (`docs/decisions.md`) need floors that never open the
capture. These are they:

| predictor | what it uses | macro-F1 | artefact |
|---|---|---|---|
| `constant-fp` | nothing | 0.31 | `benchmark/triage-constant-fp.json` |
| `constant-tp` | nothing | 0.35 | `benchmark/triage-constant-tp.json` |
| `rule-prior` | the rule's label on the *other* captures | 0.56 | `benchmark/triage-rule-prior.json` |
| heuristic | the capture's events | 0.60 | `benchmark/triage-baseline.json` |

`rule-prior` predicts the majority label the same rule carries on every other
capture, never the case's own, and gets 45 of 80. The rule alone fits about 61
of 80 when the case is allowed to vote for itself, so most of that apparent
signal is memorising a capture, not something that transfers. An agent in the
alert-only condition that beats 0.56 is using something beyond rule identity:
its prior knowledge of what the rule's title means.

## Miss classification — a near-ceiling bar, by construction

581 rule/capture pairs.

| class | support | predicted | precision | recall | F1 |
|---|---|---|---|---|---|
| `out-of-scope` | 273 | 273 | 1.00 | 1.00 | 1.00 |
| `miss-logic` | 207 | 215 | 0.93 | 0.97 | 0.95 |
| `miss-telemetry` | 57 | 54 | 1.00 | 0.95 | 0.97 |
| `detected` | 44 | 39 | 0.85 | 0.75 | 0.80 |
| **macro** | 581 | | | 0.92 | **0.93** |

| truth \ predicted | `out-of-scope` | `miss-logic` | `miss-telemetry` | `detected` | `unanswered` |
|---|---|---|---|---|---|
| `out-of-scope` | 273 | 0 | 0 | 0 | 0 |
| `miss-logic` | 0 | 201 | 0 | 6 | 0 |
| `miss-telemetry` | 0 | 3 | 54 | 0 | 0 |
| `detected` | 0 | 11 | 0 | 33 | 0 |

**Baseline macro-F1: 0.93**, and this number should be read with suspicion
rather than admiration.

The label here is a deterministic function of the rule and the telemetry. The
baseline re-derives that function with pySigma and the same two pipelines the
sibling repo chains, so it is reimplementing the answer key. Scoring 0.93 is
what a faithful reimplementation *should* do. `out-of-scope` at 1.00/1.00 is not
a discovery; it is `hktl` in a filename plus an identity-field check, which is
exactly how the label is defined.

All 20 errors sit in the `detected` / `miss-logic` boundary — 11 `detected`
called `miss-logic`, 6 the other way — and every one of them is a gap between
this repo's approximate Sigma matching and a conforming evaluator. Examples:
rule `4a1b6da0` on `LSASS_campaign_01` (truly `detected`, called `miss-logic`),
rule `962fe167` on the same capture (the reverse). The three
`miss-telemetry` → `miss-logic` errors are the same cause.

So the honest framing for this task: an agent beating 0.93 would be remarkable,
and an agent losing to it has lost to a reimplementation of the label, not to a
heuristic. **The triage number is the one to judge the agent by.** This task
earns its place for statistical heft and because it is a task with a known-correct
answer, not because it is a fair fight.

## Cost

| run | cases | wall clock |
|---|---|---|
| triage baseline | 80 | 71 s |
| miss baseline | 581 | 132 s |

Both single-threaded on one core, dominated by decompressing and scanning
captures. Cases are ordered by capture so each archive is parsed once.

## Limitations

- **Luna was chosen after the results.** The within-rule analysis was written
  down before it ran, but the model it ran on was picked because it did best.
- **The prompt's question is looser than the labels.** It asks whether the
  rule's activity happened; the labels mean T1003.001 credential dumping.
  Naming the technique moved Luna by about 10 points; the headline numbers
  keep the original wording.
- **Residual confound: account names.** Host names and domains are pseudonymised
  (see `agent/pseudonymise.py`), which removed a perfect separator between the
  two triage classes. Account names are not. `pedro.gustavo` and `stevie.marie`
  belong to the campaign lab and `pgustavo` to the benign one, so a model that
  had memorised these public datasets could still tell them apart. The
  correlation is partial, unlike the domain suffix which was total.
- **Approximate Sigma matching.** The baseline implements equality, `contains`,
  `startswith`, `endswith` and wildcards, not the full specification. The 20
  miss-classification errors are all attributable to it. The sibling repo owns
  conformance and cross-checks itself against Zircolite; this does not.
- **The triage set is small.** 80 cases, and the false-positive side comes from
  17 captures. A macro-F1 difference of a few points is not resolvable here.
- **Two labs, one technique.** Every true positive is T1003.001. The agent is not
  shown to generalise to other techniques, because nothing here tests that.
- **The alert carries no match count.** The count available from the sibling
  leaked the label, and a per-capture recount with this repo's matcher found
  nothing on 23 of 80 alerts that fired; see `docs/decisions.md`.
