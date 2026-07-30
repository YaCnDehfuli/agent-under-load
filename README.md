# agent-under-load

An LLM agent that triages security detections, scored against exact ground
truth — and then attacked through the one channel a real adversary controls: the
telemetry it reads.

Ground truth is borrowed from the sibling repo
[`chain-under-load`](https://github.com/YaCnDehfuli/chain-under-load), which runs
83 detection rules against recorded Windows telemetry and labels every
rule/capture pair with a deterministic classifier.

## Results

**Read this section first, including the part that says what is missing.**

### The bar, measured

Reproducible with no API key.

| task | cases | no-LLM baseline (macro-F1) |
|---|---|---|
| **triage verdict** — a rule fired; is it a true positive? | 80 (44 TP / 36 FP) | **0.60** |
| **miss classification** — a rule did not fire; why? | 581 rule/capture pairs | **0.93** |

The two bars are not equally meaningful. Triage at 0.60 is the informative one:
the heuristic finds every true positive and calls 26 of 36 false positives true as
well, because the benign corpus is atomic attack simulations in which processes
open handles to LSASS constantly. Miss classification at 0.93 should be read with
suspicion — that label is a deterministic function of the rule and the telemetry,
and the baseline re-derives it, so a faithful reimplementation *should* score
high. Details and the wrong cases: [`docs/results-agent.md`](docs/results-agent.md).

### The attack surface, measured

Also model-free. 44 true-positive cases × 43 payload/field placements = 1,892
possible attempts. **248 are actually mountable — 13.1%.**

The rest fail one realism constraint: a payload may only be appended to a field
the event *already carries*. So an adversary does not choose where the payload
goes — only the fields carried by the events the firing rule matched, because
those are the events the agent retrieves and the ones the adversary owns. For a
`process_access` rule keyed to LSASS handles, that means image paths, with no
room for a paragraph. Registry `Details` is 0% mountable across all 396 attempts,
because none of these rules' candidate events are registry writes.

That result points the opposite way from the usual framing: the channel is real
and it is narrower than "the agent reads attacker text" suggests.
[`docs/results-attack.md`](docs/results-attack.md).

### What is not measured

**No model credential was available in the environment this repo was built in, so
the agent has never been run against a language model.** There is no
agent-versus-baseline number, no attack success rate, and no before/after control
table. Those cells are empty in the docs and say so.

The harness is complete and verified end to end against the real corpus with a
scripted model. To populate the tables:

```bash
export ANTHROPIC_API_KEY=...                    # or AGENT_MODEL=ollama
python -m score.run   --task triage --predictor agent
python -m attack.runner --objective suppression --controls none
python -m attack.runner --objective suppression --controls all
```

An unconfigured run raises rather than falling back to a stub, so no table here
can fill itself with something that was never a language model.

## What is here

```
agent/      corpus, provenance, events, contracts, tools, graph, baseline,
            authz, audit, pseudonymise
attack/     payloads.yml, inject, runner
score/      metrics, run
docs/       decisions, architecture, threat-model, results-*
benchmark/  committed run artefacts
```

- [`docs/decisions.md`](docs/decisions.md) — why the repo is shaped this way
- [`docs/architecture.md`](docs/architecture.md) — the loop, the tools, the trust boundary
- [`docs/threat-model.md`](docs/threat-model.md) — which fields an adversary writes, and what they aim for
- [`docs/results-controls.md`](docs/results-controls.md) — the control design, and how the numbers must be written up

## Running it

```bash
pip install -r requirements.txt
git clone https://github.com/YaCnDehfuli/chain-under-load ../chain-under-load
python -m agent.corpus --fetch      # ~30 MB, sparse, pinned; verifies digests
python -m agent.corpus --list
python -m pytest tests -q
```

The unit suite passes on a clean checkout with no corpus and no API key —
corpus- and model-backed tests skip themselves, and CI asserts that they do.

Pins are not duplicated here: `agent/corpus.py` reads the sibling's
`manifest.yml`, because the ground truth is only meaningful against the bytes the
sibling actually scored. The fetch is sparse and takes only what this repo scores
— the seven campaign archives plus the seventeen benign captures a rule fired on,
about 30 MB against the sibling's ~1.5 GB.

## Three things that were wrong, and got fixed

The methodology work is most of the value here, so the failures are on the record
rather than smoothed over.

**The triage set was confounded by the lab it came from.** Every true-positive
capture is a host in `pandalab.com`; every false-positive capture is in
`theshire.local`, `mordor.local` or `shire.com`. The domain suffix separated the
labels perfectly, and it lives in `Hostname` — an os-generated field, so no amount
of provenance tagging keeps it from the model. A predictor that noticed it would
have scored 100% while knowing nothing. Host names and domains are now
pseudonymised per capture; built-in principals like `NT AUTHORITY\SYSTEM` are
left alone because they name the OS rather than the lab.

**Two smaller leaks beside it.** The capture id announced its corpus
(`LSASS_campaign_01` versus `tactic/name`), and the sibling publishes an event
count per campaign capture but not per benign one — so `host_events` was an
integer on every true positive and null on every false positive. A test now
asserts the case inputs have an identical *shape* under both labels, which is
what caught the second one.

**A dead code path in rule matching.** `Requirement` carried a modifier tuple
read from pySigma's `parent.modifiers`, which is always `None`: pySigma applies
`|contains`, `|startswith` and `|endswith` at parse time by rewriting values with
wildcards. The modifier branches never ran. Behaviour was unchanged — both
baselines re-run identical — but matching now compiles a full-match regex instead
of using `fnmatch`, which matters because command lines are full of brackets a
glob reads as syntax.

## Pipeline

CI runs the unit suite (twice: once normally, once with the corpus path removed),
Bandit and Semgrep for SAST, pip-audit for dependencies, gitleaks for secrets,
and Trivy for the filesystem. It also re-derives the committed baseline numbers
and fails if they drift from what is in `benchmark/`.

Findings and what was done about them:

| finding | response |
|---|---|
| **Bandit B603** — `subprocess` call with a non-literal argument in `agent/corpus.py`. `fetch()` hands the manifest's `url` field to `git clone`, and the manifest is data from another repository. | Fixed rather than annotated. `_checked_url` requires `https` and a host in an allowlist, and the clone uses `--` to stop a leading dash being read as an option. This blocks `ext::` (git's transport that executes a shell command), `file://`, `ssh://` and host-prefix lookalikes. Six of those shapes are now test cases. The `# nosec` that remains sits next to that validation. |
| **Bandit B404** — `import subprocess`. | Informational; suppressed with a reason at the import. |
| **pip-audit PYSEC-2026-2447 / CVE-2025-69872** — `diskcache <= 5.6.3` deserialises cache entries with pickle, so write access to the cache directory yields code execution. Arrives transitively via pySigma. | No fixed version is published, so it cannot be pinned away. Ignored by ID with the reasoning in the workflow: exploitation needs local write access to the cache directory, which here is a developer's machine or an ephemeral runner. Every other advisory still fails the build, so this exception cannot silently absorb a new one. |

ZAP DAST is **not** in the pipeline. There is no HTTP service in this repo to
point it at — the agent is a library and a CLI — and adding a web wrapper purely
to have something to scan would be theatre.

## The honest claim

I built an agent, measured a baseline for it against exact ground truth, built a
realistic injection channel with the field allowlist enforced by a test, measured
how much of that channel is actually reachable, and built and tested the controls.

I have not measured the agent, and I have not measured an attack against a
language model. Those need a model credential and the commands are above.

Never "production." Never "frontier."

## Limitations

- **The agent is unmeasured.** Everything reported is the bar and the surface.
- **One technique.** Every true positive is T1003.001. Nothing here tests
  generalisation.
- **Residual confound: account names.** Host names and domains are
  pseudonymised; account names are not. `pedro.gustavo` belongs to one lab and
  `pgustavo` to the other, so a model that had memorised these public datasets
  could still tell them apart. Partial, unlike the domain suffix, which was total.
- **Approximate Sigma matching.** The baseline handles equality and wildcards, not
  `|re`, `|base64offset`, `|all` list semantics or field-less keyword search. All
  20 of its miss-classification errors are attributable to this. The sibling repo
  owns conformance and cross-checks itself against Zircolite; this does not.
- **Small sets.** 80 triage cases and 248 mountable attempts. Differences of a few
  points will not be resolvable.
- **The payload corpus is mine.** 18 payloads written by one person is not an
  adversary. A future 0% residual would mean the corpus is too weak, not that the
  agent is safe.
- **Model dependence.** Any agent or attack number, when measured, will be a
  statement about one model on one date and will not transfer.
- **A lab agent, not a deployed one.** No queue, no analyst feedback, no drift, no
  cost ceiling, and response actions are stubs behind a boundary.

## The set

- [`chain-under-load`](https://github.com/YaCnDehfuli/chain-under-load) — do
  published detection rules catch a technique executed seven different ways?
  Supplies the ground truth used here.
- **this repo** — can an agent do the triage, and can it be attacked through the
  telemetry?
