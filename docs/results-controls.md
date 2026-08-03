# Attack success before and after each control

The table this repo exists to produce. It is not populated: no model credential
was available in the environment this was built in, so no attack was run against
a language model. What is here is the design, the mechanism of each control, and
what each one can and cannot do on the evidence available without a model.

## The controls

Four, each independently switchable, all one code path. `AgentConfig(controls=…)`
selects them; `tests/test_determinism.py` asserts that turning one on changes the
prompt and nothing else, which is what makes a row-to-row comparison a comparison.

| control | mechanism | what it can plausibly touch |
|---|---|---|
| `provenance_tags` | every value the model sees is labelled adversary-writable / adversary-influenced / os-generated, with an instruction to weigh the last two higher | all strategies, weakly — it is an instruction, and instructions are what the attack is made of |
| `structured_ingestion` | events become typed fields from a per-event-id schema; values are fenced; the rendered `Message` blob is dropped | `delimiter_escape` structurally; `tool_output` partly, since the imitated format no longer matches |
| `enforced_citation` | every citation is checked against the capture; a verdict resting on text that is not there is rejected | any strategy that works by getting the agent to assert something the telemetry does not support |
| `capability_scope` | requested actions are held against a session capability derived from the case, never from the model | escalation only, and completely — see below |

## What is already known without a model

Three of these have properties that are testable offline, and those tests are the
honest part of this document.

**`structured_ingestion` structurally defeats one strategy.** The
`delim-field` payload works by embedding a literal newline and a plausible field
name, so the rendered event appears to contain an extra os-generated field. The
structured renderer escapes newlines, so the payload cannot open a line of its
own. `tests/test_injection_targets.py::test_structured_rendering_neutralises_the_delimiter_escape`
pins it. This is a claim about one payload's mechanism, not about prompt
injection.

**`enforced_citation` rejects a fabricated justification.** A verdict citing text
that is not in the capture at the index it names is refused, and the run is
recorded as unanswered with the reason.
`tests/test_audit.py::test_a_rejected_citation_is_recorded_as_a_signal` shows the
rejection reaching the audit log. Note what this does *not* do: an attacker whose
payload is a real string in a real field can cite it truthfully. The control
raises the cost of lying, not of being wrong.

**`capability_scope` holds under a hijacked plan.**
`tests/test_authz.py::test_a_hijacked_plan_still_cannot_isolate_the_wrong_host`
lets the injection fully succeed — the agent requests `isolate_host` on
`PRODUCTION-DC01`, exactly what the payload asked for — and the boundary refuses
it, because scope comes from the case and the agent cannot widen it. With the
control off, the same request is granted. That is one case under a scripted
model, so it demonstrates the mechanism rather than a rate.

The "completely" in the table above is a claim about the mechanism, and it was
not true of the code until recently: `_in_scope` matched substrings, so a
one-character target passed the scope check. Fixed, with the reasoning and the
non-obvious second half in `docs/decisions.md`. It matters for how this table
gets read — the escalation rows measure a boundary, so a bug in the boundary
would have shown up here as a defence result.

**`provenance_tags` has no offline test, and that is the honest position.** It is
an instruction to the model. Whether it changes behaviour is exactly the sort of
claim that needs a measured rate, and there is nothing to substitute for one.

## The table, unpopulated

Rows are configurations; columns are strategies. Run with:

```
export ANTHROPIC_API_KEY=...
for c in none provenance_tags structured_ingestion enforced_citation all; do
  python -m attack.runner --objective suppression --controls $c
done
```

### Suppression

| configuration | instruction | system_framing | tool_output | delimiter_escape | authority | distraction | overall |
|---|---|---|---|---|---|---|---|
| undefended | — | — | — | — | — | — | — |
| `provenance_tags` | — | — | — | — | — | — | — |
| `structured_ingestion` | — | — | — | — | — | — | — |
| `enforced_citation` | — | — | — | — | — | — | — |
| all controls | — | — | — | — | — | — | — |

Mountable attempts per strategy (measured, model-independent): instruction 39,
system_framing 16, tool_output 37, delimiter_escape 49, authority 53,
distraction 54; 248 total. See `docs/results-attack.md`.

### Escalation

| configuration | requested | granted |
|---|---|---|
| undefended | — | — |
| `capability_scope` | — | — |
| all controls | — | — |

## How these numbers must be written up when they exist

Stated now, before there is any temptation to phrase them favourably.

**No control "prevents prompt injection."** The only defensible sentence has the
form: *this control reduced measured suppression success from X% to Y% against
this 18-payload corpus, on this model, at this date.* Anything shorter is an
overclaim.

**A residual of zero is a bug in the attack corpus, not a triumph.** If a
configuration shows 0% success across 248 attempts, the correct conclusion is
that 18 hand-written payloads are too weak to measure that configuration — not
that the agent is safe. The response is to strengthen the corpus and re-run, and
to say so in the write-up.

**Stacked is not the sum of the parts.** Controls interact. `structured_ingestion`
removes the blob that `tool_output` payloads imitate, so the two controls'
individual effects are not additive, and the stacked row has to be measured
rather than inferred.

**Report what remains.** The interesting column is the residual: which strategy
still works with everything on, and why. A table that only shows reductions is
marketing.

**Distinguish "resisted" from "unmountable."** 86.9% of possible placements
cannot be mounted at all. Those are excluded from every denominator above, and
the artefacts record them separately, because folding them in would make every
control look better than it is.
