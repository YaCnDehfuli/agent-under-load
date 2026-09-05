# Agent Under Load

Harness that scores a detection-triage agent and a no-LLM baseline, and measures which injection placements recorded telemetry can carry.

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![CI](https://github.com/YaCnDehfuli/agent-under-load/actions/workflows/ci.yml/badge.svg)](https://github.com/YaCnDehfuli/agent-under-load/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Release](https://img.shields.io/github/v/release/YaCnDehfuli/agent-under-load)](https://github.com/YaCnDehfuli/agent-under-load/releases)

## Results

No-LLM baseline on 80 triage cases (44 true positive / 36 false positive): macro-F1 **0.60**. Bare accuracy is not reported.

Miss classification on 581 rule/capture pairs: macro-F1 **0.93**. That label is a deterministic function of the rule and the telemetry; the baseline re-derives it, so a faithful reimplementation should score high.

Mountable injection surface: **248 of 1892** placements (**13.1%**).

**No language-model run has been measured.**

![Agent Under Load full plan](docs/assets/agent-under-load-full-plan.svg)

**Research artifact.** v0.1.0 is a harness. This repository does not report a model-security result.

## Quickstart

```bash
pip install -r requirements.txt
git clone https://github.com/YaCnDehfuli/detection-under-load ../detection-under-load
python -m agent.corpus --fetch
python -m agent.corpus --list
python -m pytest tests -q
```

`python -m agent.corpus --fetch` pulls about 30 MB, sparse and pinned, and verifies digests.

Ground truth is borrowed from the sibling repo
[`detection-under-load`](https://github.com/YaCnDehfuli/detection-under-load),
which runs detection rules against recorded Windows telemetry and labels every
rule/capture pair with a deterministic classifier.

## The bar, measured

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

**The agent has not been run against a language model.** The harness is
complete and the commands below populate the tables; nothing above claims a
result that a model produced, because none has.

```bash
export ANTHROPIC_API_KEY=...                # or AGENT_MODEL=ollama
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

## Pipeline

CI runs the unit suite (twice: once normally, once with the corpus path removed),
Bandit and Semgrep for SAST, pip-audit for dependencies, gitleaks for secrets,
and Trivy for the filesystem. It also re-derives the committed baseline numbers
and fails if they drift from what is in `benchmark/`.

Which of those have actually been run, since a configured scanner is not a clean
scanner:

| scanner | run here? | outcome |
|---|---|---|
| Bandit | yes | 2 findings, both addressed below; now clean |
| pip-audit | yes | 1 finding, accepted with reasoning below |
| Semgrep | **no** | the environment this was built in cannot reach `semgrep.dev` to fetch the rule packs, so it is configured in CI but unverified locally |
| gitleaks | **no** | GitHub Actions only |
| Trivy | **no** | GitHub Actions only |

## Limitations

No language-model run has been measured, so this repository is not a
model-security result. The miss-classification macro-F1 of 0.93 is a near-ceiling
score by construction: the labels are a deterministic function of the rule and
the telemetry, and the baseline re-derives them. The no-LLM triage bar (macro-F1
0.60 on 80 cases) is the informative comparison. v0.1.0 is a harness.
