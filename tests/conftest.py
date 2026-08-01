"""Shared fixtures, and the switch that keeps the unit suite corpus-free.

Tests marked `corpus` are skipped unless the sibling repo and the fetched
archives are both present, so a clean checkout runs green with no data and no
API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import corpus


def _corpus_ready() -> tuple[bool, str]:
    if not (corpus.DETECTION_REPO / "benchmark" / "results.json").exists():
        return (
            False,
            f"sibling repo not at {corpus.DETECTION_REPO} "
            "(set DETECTION_UNDER_LOAD)",
        )
    if not corpus.SECURITY_DATASETS.exists():
        return False, "corpus not fetched (python -m agent.corpus --fetch)"
    return True, ""


def pytest_collection_modifyitems(config, items):
    ready, reason = _corpus_ready()
    if ready:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "corpus" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def evaluation():
    return corpus.evaluation_set()


@pytest.fixture
def synthetic_rule() -> corpus.RuleRef:
    return corpus.RuleRef(
        id="00000000-0000-0000-0000-000000000000",
        title="Synthetic Rule",
        level="high",
        path=Path("rules/synthetic_rule.yml"),
        source="sigmahq",
        selected_by="tag",
    )


@pytest.fixture
def synthetic_capture() -> corpus.CaptureRef:
    return corpus.CaptureRef(
        id="credential_access/synthetic_capture",
        group="atomic",
        archive=Path("nowhere.zip"),
        events=1234,
    )
