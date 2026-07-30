# Measured result — the baseline, and the bar it sets

Produced by:

```
python -m score.run --task triage --predictor baseline
python -m score.run --task miss    --predictor baseline
```

Artefacts: `benchmark/triage-baseline.json`, `benchmark/miss-baseline.json`.
Every figure below is read out of those files.

## The state of this document

The baseline numbers are real and reproducible with no API key. **The agent
column is not populated, because no model credential was available in the
environment this was built in.** The harness is complete and the command is:

```
AGENT_MODEL=anthropic ANTHROPIC_API_KEY=... python -m score.run --task triage --predictor agent
AGENT_MODEL=ollama                          python -m score.run --task miss    --predictor agent
```

An unconfigured run raises rather than falling back to a stub, so there is no way
for this table to fill itself with something that was never a language model.
Nothing here is an estimate of what the agent would score.

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

- **The agent is unmeasured.** Everything above is the bar, not the result.
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
- **Fire counts on the false-positive side are per-rule, not per-pair.** The
  sibling publishes a rule's total fires across the benign corpus rather than
  per capture, so an FP case's `fire_count` is that total. It is an input a real
  alert would carry more precisely.
