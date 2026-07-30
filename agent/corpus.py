"""Turn the sibling repo's published results into typed evaluation cases.

`chain-under-load` runs 83 detection rules against recorded Windows telemetry
and writes `benchmark/results.json`: for every rule/capture pair, whether the
rule fired and — when it did not — a deterministic four-way explanation. That
file is the ground truth here. This module reads it and emits cases.

Two rules govern the shape of everything below.

Ground truth is kept out of the inputs. A case carries a `truth` field and an
`inputs` view, and nothing that derives from the labeller ever reaches the
second. The sibling's `reason` string literally states the answer, and its
`candidates` count decides `miss-telemetry` on its own, so neither is exposed.
`tests/test_no_oracle_leakage.py` enforces it.

Missing labels are explicit absences. A rule whose file cannot be resolved, or
a capture whose archive is not on disk, produces an `Absence` with a reason —
never a case with a guessed field.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, Literal

import yaml

from agent.pseudonymise import capture_handle

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_ROOT = REPO_ROOT / "corpus"
SECURITY_DATASETS = CORPUS_ROOT / "security-datasets"
SIGMA = CORPUS_ROOT / "sigma"
EXTRACT_ROOT = CORPUS_ROOT / "extracted"

#: Where the sibling repo is checked out. It is a data dependency, not a code
#: one: this repo reads its manifest and results and never imports from it.
CHAIN_REPO = Path(
    os.environ.get("CHAIN_UNDER_LOAD", REPO_ROOT.parent / "chain-under-load")
)

TriageTruth = Literal["true_positive", "false_positive"]
MissTruth = Literal["detected", "miss-logic", "miss-telemetry", "out-of-scope"]

MISS_CLASSES: tuple[MissTruth, ...] = (
    "out-of-scope",
    "miss-logic",
    "miss-telemetry",
    "detected",
)


class CorpusError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RuleRef:
    """A detection rule, resolved to a file on disk."""

    id: str
    title: str
    level: str
    path: Path
    source: Literal["sigmahq", "chain-under-load"]
    selected_by: str

    def text(self) -> str:
        return self.path.read_text(errors="replace")

    def parsed(self) -> dict:
        return yaml.safe_load(self.text()) or {}


@dataclasses.dataclass(frozen=True)
class CaptureRef:
    """One recorded telemetry window."""

    id: str
    group: Literal["lsass_campaign", "atomic"]
    archive: Path
    tool: str | None = None
    #: Total events in the window. Corpus size, not a hint about any rule.
    events: int | None = None

    @property
    def is_attack(self) -> bool:
        """True for the seven LSASS captures, which are recorded intrusions.

        Atomic captures reach this corpus only when the sibling established
        they carry no label for T1003.001 or a sibling technique, so a fire in
        one is a false positive by construction.
        """
        return self.group == "lsass_campaign"


@dataclasses.dataclass(frozen=True)
class Absence:
    """Something the corpus could not supply, and why."""

    kind: str
    ref: str
    reason: str


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TriageCase:
    """A detection fired. Is it a true positive?

    `fire_count` is the number of events the rule matched. It stays in the
    inputs because a real alert carries it, and it is present on both sides of
    the label, so it cannot act as a shortcut for one class.
    """

    case_id: str
    rule: RuleRef
    capture: CaptureRef
    fire_count: int
    truth: TriageTruth

    def inputs(self) -> dict:
        """Exactly what a predictor may see. No oracle-derived field."""
        return {
            "case_id": self.case_id,
            "task": "triage_verdict",
            "rule": {
                "id": self.rule.id,
                "title": self.rule.title,
                "level": self.rule.level,
                "source": self.rule.source,
            },
            # an opaque handle: "LSASS_campaign_01" versus
            # "credential_access/empire_over_pth_patch_lsass" announces the
            # label before a single event is read. The event count is left out
            # for the same reason: the sibling publishes it per campaign and not
            # per benign capture, so present-versus-absent would track the
            # label. describe_capture reports it for either side.
            "capture": {"id": capture_handle(self.capture.id)},
            "fire_count": self.fire_count,
        }


@dataclasses.dataclass(frozen=True)
class MissCase:
    """A rule ran against a capture. Which of the four states is it in?"""

    case_id: str
    rule: RuleRef
    capture: CaptureRef
    truth: MissTruth

    def inputs(self) -> dict:
        return {
            "case_id": self.case_id,
            "task": "miss_classification",
            "rule": {
                "id": self.rule.id,
                "title": self.rule.title,
                "level": self.rule.level,
                "source": self.rule.source,
                # the filename is an input an analyst genuinely has, and
                # SigmaHQ's hktl_ prefix is a real signal in it
                "filename": self.rule.path.name,
            },
            "capture": {"id": capture_handle(self.capture.id)},
        }


@dataclasses.dataclass(frozen=True)
class EvaluationSet:
    triage: list[TriageCase]
    miss: list[MissCase]
    absences: list[Absence]

    def truth_distribution(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {"triage": {}, "miss": {}}
        for case in self.triage:
            out["triage"][case.truth] = out["triage"].get(case.truth, 0) + 1
        for case in self.miss:
            out["miss"][case.truth] = out["miss"].get(case.truth, 0) + 1
        return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _chain_file(name: str) -> Path:
    path = CHAIN_REPO / name
    if not path.exists():
        raise CorpusError(
            f"{path} not found. Point CHAIN_UNDER_LOAD at a checkout of "
            f"https://github.com/YaCnDehfuli/chain-under-load"
        )
    return path


def load_results() -> dict:
    with open(_chain_file("benchmark/results.json")) as fh:
        return json.load(fh)


def load_manifest() -> dict:
    with open(_chain_file("benchmark/manifest.yml")) as fh:
        return yaml.safe_load(fh)


def _resolve_rule_path(file_field: str) -> tuple[Path, str]:
    """Map a `file` value from results.json onto a path here.

    The sibling records SigmaHQ rules as `corpus/sigma/...`, relative to its own
    corpus cache, and its own rules as `rules/...`, relative to its repo root.
    """
    if file_field.startswith("corpus/"):
        return CORPUS_ROOT / file_field[len("corpus/") :], "sigmahq"
    return CHAIN_REPO / file_field, "chain-under-load"


def rules(results: dict | None = None) -> tuple[dict[str, RuleRef], list[Absence]]:
    results = results or load_results()
    found: dict[str, RuleRef] = {}
    absences: list[Absence] = []
    for entry in results["summary"]["rules"]:
        path, source = _resolve_rule_path(entry["file"])
        if not path.exists():
            absences.append(Absence("rule", entry["id"], f"{path} not on disk"))
            continue
        found[entry["id"]] = RuleRef(
            id=entry["id"],
            title=entry["title"],
            level=entry["level"],
            path=path,
            source=source,  # type: ignore[arg-type]
            selected_by=entry.get("selected_by", ""),
        )
    return found, absences


def _atomic_archive(capture_id: str) -> Path | None:
    """`tactic/name` -> the capture archive under datasets/atomic/windows."""
    tactic, _, name = capture_id.partition("/")
    host = SECURITY_DATASETS / "datasets/atomic/windows" / tactic / "host"
    if not host.exists():
        return None
    for suffix in (".zip", ".gz", ".json"):
        candidate = host / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    # some captures carry a recording timestamp in the filename
    matches = sorted(p for p in host.glob(f"{name}.*")
                     if p.suffix in (".zip", ".gz", ".json"))
    return matches[0] if matches else None


def campaign_captures(
    results: dict | None = None, manifest: dict | None = None
) -> tuple[dict[str, CaptureRef], list[Absence]]:
    results = results or load_results()
    manifest = manifest or load_manifest()
    found: dict[str, CaptureRef] = {}
    absences: list[Absence] = []
    for entry in manifest["lsass_campaigns"]:
        archive = SECURITY_DATASETS / entry["archive"]
        if not archive.exists():
            absences.append(
                Absence("capture", entry["id"],
                        "archive not fetched (python -m agent.corpus --fetch)")
            )
            continue
        found[entry["id"]] = CaptureRef(
            id=entry["id"],
            group="lsass_campaign",
            archive=archive,
            tool=entry["tool"],
            events=results["per_capture"].get(entry["id"], {}).get("events"),
        )
    return found, absences


def benign_captures(
    results: dict | None = None,
) -> tuple[dict[str, CaptureRef], list[Absence]]:
    """Only the captures a rule actually fired on.

    Cases exist where a detection fired, so restricting the fetch and the
    inventory to captures with at least one fire is the task's definition
    rather than selection on the outcome: a capture with no fire produces no
    triage case for any rule.
    """
    results = results or load_results()
    wanted: set[str] = set()
    for info in results["false_positives"]["per_rule"].values():
        wanted.update(info["captures_hit"])

    found: dict[str, CaptureRef] = {}
    absences: list[Absence] = []
    for capture_id in sorted(wanted):
        archive = _atomic_archive(capture_id)
        if archive is None:
            absences.append(
                Absence("capture", capture_id,
                        "archive not fetched (python -m agent.corpus --fetch)")
            )
            continue
        found[capture_id] = CaptureRef(id=capture_id, group="atomic",
                                       archive=archive)
    return found, absences


def evaluation_set(results: dict | None = None) -> EvaluationSet:
    """Build both scored sets, with an explicit absence for anything missing."""
    results = results or load_results()
    rule_index, absences = rules(results)
    campaigns, campaign_absences = campaign_captures(results)
    benign, benign_absences = benign_captures(results)
    absences = [*absences, *campaign_absences, *benign_absences]

    triage: list[TriageCase] = []
    miss: list[MissCase] = []

    # true positives: a rule that fired on a recorded intrusion
    for capture_id, per_capture in results["per_capture"].items():
        capture = campaigns.get(capture_id)
        for rule_id, info in per_capture["rules"].items():
            rule = rule_index.get(rule_id)
            if rule is None:
                continue
            if capture is None:
                absences.append(
                    Absence("triage_case", f"{rule_id}@{capture_id}",
                            "capture archive missing")
                )
                continue
            if info["class"] == "detected":
                triage.append(TriageCase(
                    case_id=f"tp:{rule_id}@{capture_id}",
                    rule=rule, capture=capture,
                    fire_count=info["matched"], truth="true_positive",
                ))
            miss.append(MissCase(
                case_id=f"miss:{rule_id}@{capture_id}",
                rule=rule, capture=capture, truth=info["class"],
            ))

    # false positives: the same rules firing on captures benign for the technique
    for rule_id, info in results["false_positives"]["per_rule"].items():
        rule = rule_index.get(rule_id)
        if rule is None or not info["captures_hit"]:
            continue
        for capture_id in info["captures_hit"]:
            capture = benign.get(capture_id)
            if capture is None:
                absences.append(
                    Absence("triage_case", f"{rule_id}@{capture_id}",
                            "capture archive missing")
                )
                continue
            triage.append(TriageCase(
                case_id=f"fp:{rule_id}@{capture_id}",
                rule=rule, capture=capture,
                # per-capture fire counts are not published per pair, so the
                # alert carries the rule's total across the benign corpus
                fire_count=info["fires"], truth="false_positive",
            ))

    return EvaluationSet(triage=triage, miss=miss, absences=absences)


# ---------------------------------------------------------------------------
# reading events
# ---------------------------------------------------------------------------


def _extract(capture: CaptureRef) -> Path:
    if capture.archive.suffix == ".json":
        return capture.archive

    dest = EXTRACT_ROOT / capture.group / capture.id.replace("/", "__")
    if dest.exists():
        found = sorted(dest.glob("*.json"))
        if found:
            return found[0]
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if capture.archive.suffix == ".zip":
        with zipfile.ZipFile(capture.archive) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            if not names:
                raise CorpusError(f"{capture.id}: no json in archive")
            for name in names:
                with zf.open(name) as src, open(dest / Path(name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(capture.archive) as tf:
            names = [n for n in tf.getnames() if n.endswith(".json")]
            if not names:
                raise CorpusError(f"{capture.id}: no json in archive")
            for name in names:
                src = tf.extractfile(name)
                if src is None:
                    continue
                with open(dest / Path(name).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    found = sorted(dest.glob("*.json"))
    if not found:
        raise CorpusError(f"{capture.id}: extraction produced nothing")
    return found[0]


def events(capture: CaptureRef) -> Iterator[dict]:
    """Stream a capture as flattened event dicts, one json object per line."""
    path = _extract(capture)
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CorpusError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _required_dataset_paths(results: dict, manifest: dict) -> list[str]:
    """The archives this repo scores, and nothing else.

    The sibling fetches every Windows host capture because it measures a false
    positive rate over all of them. Cases here exist only where a rule fired,
    so the seven campaigns plus the captures with at least one fire is the whole
    requirement — about 30 MB against roughly 1.5 GB.
    """
    paths = [entry["archive"] for entry in manifest["lsass_campaigns"]]
    hit: set[str] = set()
    for info in results["false_positives"]["per_rule"].values():
        hit.update(info["captures_hit"])
    for capture_id in sorted(hit):
        tactic, _, name = capture_id.partition("/")
        paths.append(f"datasets/atomic/windows/{tactic}/host/{name}.*")
    return paths


def fetch(force: bool = False) -> None:
    """Sparse-fetch the pinned sources, taking the pins from the sibling repo.

    The pins are not duplicated here on purpose. A second copy of a commit hash
    is a second thing to keep in step, and the ground truth is only meaningful
    against the bytes the sibling actually scored.
    """
    results = load_results()
    manifest = load_manifest()
    CORPUS_ROOT.mkdir(parents=True, exist_ok=True)

    specs = [
        ("security-datasets", SECURITY_DATASETS,
         manifest["sources"]["security_datasets"],
         _required_dataset_paths(results, manifest), False),
        ("sigma", SIGMA, manifest["sources"]["sigmahq"],
         manifest["sources"]["sigmahq"]["sparse"], True),
    ]

    for name, dest, spec, sparse, cone in specs:
        if force and dest.exists():
            shutil.rmtree(dest)
        if dest.exists():
            head = _run(["git", "rev-parse", "HEAD"], cwd=dest)
            if head == spec["commit"]:
                print(f"{name}: already at {head[:12]}")
                continue
            print(f"{name}: at {head[:12]}, want {spec['commit'][:12]}, refetching")
            shutil.rmtree(dest)

        print(f"{name}: cloning {spec['url']}")
        _run(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
              spec["url"], str(dest)])
        _run(["git", "fetch", "--quiet", "origin", spec["commit"]], cwd=dest)
        _run(["git", "sparse-checkout", "init",
              "--cone" if cone else "--no-cone"], cwd=dest)
        _run(["git", "sparse-checkout", "set", *sparse], cwd=dest)
        _run(["git", "checkout", "--quiet", "--force", spec["commit"]], cwd=dest)
        print(f"{name}: at {spec['commit'][:12]}")


def verify() -> list[str]:
    """Check the pins and the campaign digests. Returns problems."""
    manifest = load_manifest()
    problems: list[str] = []

    for name, dest in (("security-datasets", SECURITY_DATASETS), ("sigma", SIGMA)):
        key = "security_datasets" if name == "security-datasets" else "sigmahq"
        spec = manifest["sources"][key]
        if not dest.exists():
            problems.append(f"{name}: not fetched")
            continue
        head = _run(["git", "rev-parse", "HEAD"], cwd=dest)
        if head != spec["commit"]:
            problems.append(f"{name}: at {head}, manifest pins {spec['commit']}")

    for entry in manifest["lsass_campaigns"]:
        archive = SECURITY_DATASETS / entry["archive"]
        if not archive.exists():
            problems.append(f"{entry['id']}: archive missing")
            continue
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            problems.append(f"{entry['id']}: sha256 {digest} != {entry['sha256']}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build the evaluation set")
    parser.add_argument("--fetch", action="store_true", help="sparse-fetch pins")
    parser.add_argument("--force", action="store_true", help="refetch from scratch")
    parser.add_argument("--verify", action="store_true", help="check pins and digests")
    parser.add_argument("--list", action="store_true", help="print the inventory")
    args = parser.parse_args(argv)

    if args.fetch:
        fetch(force=args.force)
    if args.verify or args.fetch:
        problems = verify()
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        if problems:
            return 1
        print("corpus verified")
    if args.list:
        evaluation = evaluation_set()
        distribution = evaluation.truth_distribution()
        print(f"triage cases: {len(evaluation.triage)}")
        for label, count in sorted(distribution["triage"].items()):
            print(f"    {label:18s} {count}")
        print(f"miss cases:   {len(evaluation.miss)}")
        for label in MISS_CLASSES:
            count = distribution["miss"].get(label, 0)
            share = 100.0 * count / max(1, len(evaluation.miss))
            print(f"    {label:18s} {count:4d}  {share:5.1f}%")
        if evaluation.absences:
            print(f"absences:     {len(evaluation.absences)}")
            for absence in evaluation.absences[:10]:
                print(f"    {absence.kind:12s} {absence.ref}: {absence.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
