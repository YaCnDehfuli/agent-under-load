# Decisions

Choices that shaped this repo, with the reasoning I had at the time. Newest
last.

## What the agent must beat, and why this task has ground truth

Most agent projects cannot say whether the agent is any good, because they have
no ground truth. They show a transcript that reads well and stop. Most AI
red-team projects have the mirror problem: they attack a target built to be
attacked, so the attack succeeding says nothing.

This repo avoids both by borrowing a labelled corpus. The sibling repo
[`detection-under-load`](https://github.com/YaCnDehfuli/detection-under-load) runs 83
detection rules against recorded Windows telemetry and labels every
rule/capture pair with a deterministic classifier. That gives two scored tasks
with exact answers:

| Task | Ground truth | Size |
|---|---|---|
| **Triage verdict** — a rule fired on a capture; is it a true positive? | A fire in an attack capture is a TP, a fire in a capture benign for that technique is an FP | 44 TP fires, 36 FP fires |
| **Miss classification** — a rule did not fire; why? | The four-way label from the sibling's `classify.py` | 581 rule/capture pairs |

Both are deployment-shaped questions. The first is the one an analyst is paid
to answer, and the one the attack in Phase C targets. The second is where the
statistical heft is.

## A no-LLM baseline is built first, and published either way

Before the agent is scored, a heuristic with no model in it is scored on the
same cases. If the agent does not beat it, that is the finding and it gets
published. An unbeaten baseline reported honestly is worth more than a hidden
one, and a project that cannot fail cannot demonstrate anything.

## Per-class metrics only. Bare accuracy is banned in the scorer's output

The miss-classification set is heavily imbalanced:

| class | count | share |
|---|---|---|
| out-of-scope | 273 | 47.0% |
| miss-logic | 207 | 35.6% |
| miss-telemetry | 57 | 9.8% |
| detected | 44 | 7.6% |

A predictor that answers `out-of-scope` every time scores 47% accuracy and
knows nothing. So the scorer refuses to emit a bare accuracy figure at all —
it reports precision, recall, F1 and support per class, plus a confusion
matrix. This is enforced by a test with a deliberately degenerate predictor,
not by discipline.

## The agent and the baseline see identical inputs, and never the oracle

This is the decision that makes the numbers mean anything, and it took some
thought.

The sibling's labels are not human judgement. They are a deterministic function
of five observable quantities: how many events matched, how many events passed
the rule's channel and event-id prefilter, whether each required field was
present, whether each required constraint was satisfiable, and whether the rule
is named after one hacktool. Anything that sees those five things can
reimplement `classify.py` and score close to 100% while understanding nothing.

That has two consequences.

A baseline handed those diagnostics is not a baseline, it is the oracle wearing
a hat. And an agent handed them is not doing triage, it is doing arithmetic.

So the case a predictor receives contains the rule, the capture identity, and
access to the capture's events through tools. It does not contain the label,
the classifier's `reason` string, the candidate count, or the per-group
presence and satisfaction counts. Both the agent and the baseline have to
derive what they need from the rule text and the telemetry, which is the actual
task. `tests/test_no_oracle_leakage.py` asserts that no serialised case
contains any oracle-derived field.

The cost is honest: the baseline is weaker than it could be, and so is the
agent. The alternative is a benchmark that measures whether I remembered to
pass a field.

## Untrusted text is a first-class type, not a warning in a docstring

The agent reads command lines, filenames, script blocks, service names and
registry values. An adversary *writes* those. Every field the agent can see
therefore carries a provenance tag, and the tag survives all the way into the
prompt.

Three classes, not two, because the middle one matters:

- **adversary-writable** — free text the adversary chooses outright:
  `CommandLine`, `Image`, `TargetFilename`, `ScriptBlockText`, `ServiceName`,
  registry `Details`, PE `Product` / `Description` / `Company`.
- **adversary-influenced** — the value reflects adversary behaviour but is
  drawn from a space the OS controls, so it cannot carry a sentence:
  `GrantedAccess`, `CallTrace`, `IntegrityLevel`.
- **os-generated** — the adversary cannot touch it without already owning the
  logging pipeline: `EventID`, `Channel`, `ProviderGuid`, `ProcessGuid`,
  timestamps, `Hostname`.

Injections may only be placed in the first class. That is enforced by
`tests/test_injection_targets.py`, because a threat model maintained by
discipline is a threat model that drifts. Injecting into `GrantedAccess` would
inflate attack success with an attack no adversary can actually mount.

Sysmon's rendered `Message` field is a concatenation of the other fields, so it
inherits the strongest provenance of any field it contains. An injector that
writes `CommandLine` and leaves `Message` describing the old value has produced
telemetry no host would emit, and the inconsistency, not the payload, is what
the agent would be reacting to.

## LangGraph, for auditability rather than ergonomics

A `while` loop around a chat completion would run this agent. LangGraph is here
because the graph makes the state explicit and the transitions enumerable: what
the agent knew at each step, which tool it called, and what came back is
recoverable from the checkpoint rather than reconstructed from logs written by
hand.

That matters because Phase E claims the audit is complete under attack. A claim
like that is only testable if the loop has a state object to inspect.

The cost is a dependency with its own release cadence in the middle of the hot
path, and a framework whose abstractions I have to understand to debug. Taken
knowingly.

## The model is an interface, and an unconfigured run fails loudly

Three implementations: a hosted Anthropic path (the default), a local
open-weights path over an Ollama-compatible HTTP API (so the results
reproduce without a paid API), and a scripted deterministic model used by the
test suite.

The scripted model exists for tests only and is never scored. A stub cannot be
prompt-injected in any meaningful sense, so a number produced against it would
be fiction. An unconfigured run therefore raises rather than falling back to a
stub, because the failure mode that matters here is a results table quietly
populated by something that was never a language model.

## Numbers appear only when a run produced them

Every table in `docs/` is either populated from a committed run artefact or
marked as not yet measured, with the command that would populate it. There is
no third state. The baseline numbers were produced without a model and are
real; the agent and attack numbers require a configured model.

## The triage set was confounded by the lab it came from, and had to be fixed

Found while building the baseline, and worth recording as a failure rather than
a feature.

The two triage classes come from two different labs. Every true-positive capture
is a host in `pandalab.com`; every false-positive capture is in `theshire.local`,
`mordor.local`, `shire.com` or a bare workstation name. The domain suffix
separates the labels perfectly, and it lives in `Hostname` — an os-generated
field, so no amount of provenance tagging keeps it away from the model.

Left alone, a predictor that noticed the suffix would have scored 100% on triage
while knowing nothing about credential theft, and the number would have looked
like a result. Two smaller versions of the same problem were sitting next to it:
the capture id itself (`LSASS_campaign_01` versus
`credential_access/empire_over_pth_patch_lsass`), and an event count the sibling
publishes per campaign but not per benign capture, so the field was an integer
on every true positive and null on every false positive.

The fixes:

- Host short names, DNS domains and NetBIOS domain names are pseudonymised per
  capture, at the single point where a capture becomes readable, so rendering,
  filtering and citation checking all see the same text.
- Capture ids become opaque handles.
- The event count is no longer in the case inputs at all. `describe_capture`
  reports it for either side on request.

What is deliberately preserved: whether two events name the *same* host. That is
real evidence in an investigation, and destroying it would damage the task
instead of de-confounding it. Built-in Windows principals are also left alone —
`NT AUTHORITY\SYSTEM` names the operating system, not the lab, and "SYSTEM
opened a handle to LSASS" reads very differently from "CORP\SYSTEM did".

Three tests now guard this: inputs must have an identical *shape* under both
labels (which is what caught the null event count), capture handles must not
contain the corpus name, and no lab identifier may survive into tool output.

The residual is stated rather than assumed away. Account names are not
pseudonymised: `pedro.gustavo` belongs to one lab and `pgustavo` to another, so
a model that had memorised these public datasets could still tell them apart.
The correlation is partial, fixing it means rewriting fields analysts legitimately
reason over, and it is recorded as a limitation in the results instead.

The general lesson, which applies to any borrowed corpus: when the positive and
negative classes come from different sources, something in the data identifies
the source, and it will be found by whatever you point at it.

## Only inject into fields the event already carries

The injector appends to an existing value and never creates a field. A
`ScriptBlockText` on a Sysmon process-access record, or a registry `Details` on a
process-creation record, is telemetry no Windows host emits. An agent that
discounted such a record would be noticing a forgery rather than resisting an
injection, and counting that as a defence success would be measuring the wrong
thing.

The cost is large and turned out to be the most interesting model-free result in
the repo: only 13.1% of the 1,892 payload/case placements can be mounted at all.
The adversary does not choose where the payload goes. They can only write into
fields carried by the events the firing rule matched, because those are the events
the agent retrieves and the ones the adversary owns. For a `process_access` rule
keyed to LSASS handles, the writable candidates are image paths — so the payload
has to be shaped like a path, and prose has nowhere to sit. That is why
`system_framing` is the least mountable strategy at 5.2%.

Reporting an aggregate success rate over all 1,892 placements would have averaged
in 396 registry attempts that could never happen.

## A results table with empty cells, rather than no table

No model credential was available in the environment this was built in, so the
agent has never run against a language model. The attack and ablation tables are
committed with every cell marked not measured, and the command that would fill
them sits above each one.

The alternative — omitting the tables until there are numbers — is worse in a
specific way: it makes the measurement design unreviewable, and it makes a
partial repo easier to mistake for a finished one. An empty cell that names its
own command is honest. A missing section is ambiguous.

The related rule, decided now rather than when the numbers exist: a residual of
zero is a bug in the attack corpus, not a triumph. Eighteen payloads written by
one person is not an adversary, and 0% against them would mean the corpus is too
weak to measure that configuration.

## Fix what a scanner finds, do not annotate it away

Bandit flagged the `subprocess` call in `agent/corpus.py`: `fetch()` passes the
manifest's `url` field to `git clone`, and the manifest is data from another
repository. The easy response is `# nosec` and a sentence about trusting the
sibling.

The actual response was to validate the URL — https only, host allowlist, `--`
before the URL so a leading dash cannot be read as an option — which blocks the
shapes that turn a manifest into code execution: `ext::`, which is git's
transport for running a shell command, plus `file://`, `ssh://` and host-prefix
lookalikes. Six of them are test cases now. The suppression that remains sits
next to the validation it depends on.

One finding is accepted rather than fixed, with the reasoning recorded in the
workflow: `diskcache <= 5.6.3` deserialises with pickle (CVE-2025-69872), arrives
transitively through pySigma, and has no published fix, so it cannot be pinned
away. It is ignored by ID so every other advisory still fails the build.

## The authorization boundary matched substrings, and one character was enough

Recorded as a defect rather than a refinement, because the module it sits in is
the one this repo points at when it claims defence in depth.

`_in_scope` tested containment in both directions:

```python
return any(needle in scope.lower() or scope.lower() in needle
           for scope in capability.targets)
```

Against a session scoped to `lab/synthetic`, the target `s` was granted — `s`
occurs in the scope string. So did `l`, `/`, `lab`, `synthetic`, and
`lab/synthetic and PRODUCTION-DC01`. Every adversarial target in what is now
`tests/test_authz.py` passed. The docstring called this "loose on purpose", and
the looseness was real, but it was not bounded by anything.

The interesting part is why the obvious fix is wrong. `Capability.for_case` was
minting `targets={case.capture.id}` — the raw corpus id, `LSASS_campaign_01`.
The agent never sees that string: it is shown `capture-<sha256[:8]}`, because
the capture id was pseudonymised to stop it announcing the label (see the
confound entry above). There is no substring relationship between a digest and
the id it came from. So the loose match was failing in both directions at once —
letting arbitrary fragments through, while never actually admitting a legitimate
request, since the agent could not produce the only string in the scope set.

Replacing the substring test with equality and stopping there would have
refused *everything*. The Phase D `capability_scope` row would then have shown a
control with a perfect record — not because scope was enforced, but because no
request could satisfy it, which is a measurement artefact dressed as a defence.
It is the same failure the empty-tables rule elsewhere in this file exists to
prevent: a number that looks like a result and is an artefact of the harness.

So the fix is two-sided, and both halves are load-bearing:

- the capability is minted over identifiers the agent can actually name — the
  capture handle, plus host pseudonyms the toolbox reads out of the capture's
  own substitution table when the session asks to act on one;
- matching is exact after casefolding and whitespace collapse, and nothing else
  is normalised, because every additional normalisation widens what counts as
  the same identifier and this function decides an authorization question.

Assets still come from the capture and never from the request, which was always
the mechanism. What changed is that the scope set and the agent's vocabulary are
now the same namespace, so the boundary can be strict without being vacuous.
