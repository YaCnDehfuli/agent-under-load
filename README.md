# agent-under-load

An LLM agent that triages security detections, scored against exact ground
truth — and then attacked through the one channel a real adversary controls:
the telemetry it reads.

Ground truth comes from the sibling repo
[`chain-under-load`](https://github.com/YaCnDehfuli/chain-under-load), which
executes SigmaHQ rules against recorded Windows telemetry and labels every
rule/capture pair deterministically.

Results, method and limitations are written up as they are measured. This
README is filled in last, on purpose: it reports numbers rather than
intentions.

- `docs/decisions.md` — why the repo is shaped the way it is
- `docs/architecture.md` — the loop, the tools, and where untrusted text enters
- `docs/threat-model.md` — which fields an adversary writes, and what they aim for

## Status

Under construction. Nothing in this README is a claim about production
readiness, and no number appears here that was not produced by a committed run.
