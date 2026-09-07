# Agent Under Load

Evaluation harness for detection-triage agents, built around deterministic
ground truth, a no-LLM baseline, and telemetry-constrained prompt-injection
tests.

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![CI](https://github.com/YaCnDehfuli/agent-under-load/actions/workflows/ci.yml/badge.svg)](https://github.com/YaCnDehfuli/agent-under-load/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Release](https://img.shields.io/github/v/release/YaCnDehfuli/agent-under-load)](https://github.com/YaCnDehfuli/agent-under-load/releases)

## Results

On 80 triage cases (44 true positive / 36 false positive), the committed no-LLM
baseline reaches macro-F1 **0.60**. Bare accuracy is not reported.

On 581 rule/capture pairs, the same baseline reaches macro-F1 **0.93** for miss
classification. That label is a deterministic function of the rule and the
telemetry; the baseline re-derives it, so a faithful implementation should
score high.

Of 1,892 possible payload placements, **248 are mountable (13.1%)** under the
recorded telemetry's actual field constraints.

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

The fetch command pulls about 30 MB, uses a pinned sparse checkout, and verifies
the downloaded files against committed digests.

Ground truth is borrowed from the sibling repo
[`detection-under-load`](https://github.com/YaCnDehfuli/detection-under-load),
which runs detection rules against recorded Windows telemetry and labels every
rule/capture pair with a deterministic classifier.

## Evaluation baselines

Both baselines are reproducible without an API key.

| task | cases | no-LLM baseline (macro-F1) |
|---|---|---|
| **triage verdict** — a rule fired; is it a true positive? | 80 (44 TP / 36 FP) | **0.60** |
| **miss classification** — a rule did not fire; why? | 581 rule/capture pairs | **0.93** |

The two scores serve different purposes. Triage at 0.60 is the informative one:
the heuristic finds every true positive and calls 26 of 36 false positives true as
well, because the benign corpus is atomic attack simulations in which processes
open handles to LSASS constantly. Miss classification at 0.93 is primarily an
implementation check: the label is a deterministic function of the rule and the
telemetry, and the baseline re-derives it, so a faithful implementation *should*
score high. See [`docs/results-agent.md`](docs/results-agent.md) for the full
results and error analysis.

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

The result narrows the threat model: the channel exists, but it is more
constrained than "the agent reads attacker text" suggests. See
[`docs/results-attack.md`](docs/results-attack.md) for the placement results.

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

## Repository map

```
agent/      corpus, provenance, events, contracts, tools, graph, baseline,
            authz, audit, pseudonymise
attack/     payloads.yml, inject, runner
score/      metrics, run
docs/       decisions, architecture, threat-model, results-*
benchmark/  committed run artefacts
```

## Verification

The CI workflow is configured to run the unit suite twice — once normally and
once with the corpus path removed — plus Bandit and Semgrep for SAST, pip-audit
for dependencies, gitleaks for secrets, and Trivy for the filesystem. It also
re-derives the committed baseline numbers and fails if they drift from
`benchmark/`.

Local execution is recorded separately from CI configuration; a configured
workflow alone is not evidence of a clean scan:

| scanner | local verification | CI configuration | local outcome |
|---|---|---|---|
| Bandit | run locally | configured | 2 findings addressed; current local run clean |
| pip-audit | run locally | configured | 1 finding accepted; rationale in [`docs/decisions.md`](docs/decisions.md#fix-what-a-scanner-finds-do-not-annotate-it-away) |
| Semgrep | not run; rule packs were unreachable from the build environment | configured | unverified locally |
| gitleaks | not run locally | configured | unverified locally |
| Trivy | not run locally | configured | unverified locally |

## Limitations

No language-model run has been measured, so this repository is not a
model-security result. The miss-classification macro-F1 of 0.93 is a near-ceiling
score by construction: the labels are a deterministic function of the rule and
the telemetry, and the baseline re-derives them. The no-LLM triage bar (macro-F1
0.60 on 80 cases) is the informative comparison. v0.1.0 is a harness.

## Related work in this portfolio

Memory forensics → detection engineering → evaluation of AI in security operations.

| Repository | What it establishes |
| --- | --- |
| [VolMemLyzer3](https://github.com/YaCnDehfuli/VolMemLyzer3-CLI_forensic_tool) | Volatility 3 orchestration and feature extraction; 2.4× parallel speedup on a pinned 10-plugin set |
| [VADViT](https://github.com/YaCnDehfuli/VADViT) | Published ViT classification of process memory — 99.2% binary accuracy, 92% macro-F1 |
| [MalGraph](https://github.com/YaCnDehfuli/MalGraph) | Why memory-time recovery matters: UPX packing leaves 5.4% of functions statically recoverable |
| [MemTriage](https://github.com/YaCnDehfuli/MemTriage) | The analyst workspace that consumes both |
| [detection-under-load](https://github.com/YaCnDehfuli/detection-under-load) | Published Sigma coverage for T1003.001 collapses under operator-controlled renaming |
| **agent-under-load** | Whether an LLM agent can triage those detections, measured against deterministic ground truth |
