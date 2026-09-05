# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-05

First public GitHub release of the triage-agent evaluation harness (not a finished stable product).

### Added

- Detection-under-load corpus integration (sparse, pinned, digest-verified).
- Triage graph, tool contracts, authorization boundary, tamper-evident audit.
- Labelled evaluation set; no-LLM baseline predictor.
- Scorer with per-class metrics and no accuracy shortcut.
- Injection corpus, attack runner, and mountability measurement.
- Committed artefacts: `benchmark/triage-baseline.json`, `miss-baseline.json`, `attack-mountability.json`.
- CI: unit tests (with and without corpus path), Bandit, pip-audit, Semgrep, gitleaks, Trivy, baseline drift check.
- Architecture, threat-model, decisions, and results write-ups; MIT license.

### Changed

- Sibling clone uses the current detection-under-load repository name.
- README states which scanners have actually run locally vs only in Actions.

### Fixed

- Actions scoped to identifiers the agent can name.
- Actions pinned to commit SHAs.
- Attribution trailers from the editor toolchain disabled in this repo.
- Truncated “what is not measured” paragraph restored.
- `scored-run` clones `detection-under-load` into the workspace so `hashFiles` can key the corpus cache.

### Security

- `CHAIN_UNDER_LOAD` fallback removed.
- Lab identifiers pseudonymised out of the triage set (account-name residual confound remains; see `docs/results-agent.md`).

[0.1.0]: https://github.com/YaCnDehfuli/agent-under-load/releases/tag/v0.1.0
