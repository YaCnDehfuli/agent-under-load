"""Remove the lab's fingerprints from a capture before the agent reads it.

This module exists because of a confound found while building the baseline, and
it is worth stating plainly: the two triage classes came from two different labs.

Every true-positive capture is a host in `pandalab.com`. Every false-positive
capture is a host in `theshire.local`, `mordor.local`, `shire.com` or a bare
workstation name. The domain suffix therefore separates the labels perfectly,
and it sits in `Hostname` — an os-generated field, so no amount of provenance
tagging keeps it away from the model. A predictor that noticed it would score
100% on triage while knowing nothing about credential theft, and the number
would have looked like a result.

So identifiers are pseudonymised per capture before anything reads them:

- host short names become `HOST1`, `HOST2`, … in order of first appearance
- DNS domains become `corp.example`
- NetBIOS domain names become `CORP`

What survives on purpose: whether two events name the *same* host. That is real
evidence in a lateral-movement investigation and removing it would damage the
task rather than de-confound it. The mapping is stable within a capture and
independent across captures, so "same host" is preserved and "which lab" is not.

What does not survive: nothing else is touched. Command lines, images and script
blocks are the evidence and are passed through byte for byte, because they are
also the attack surface and rewriting them would put this module inside the
measurement it is supposed to protect.

### Limits, stated rather than implied

Account names are not pseudonymised. `pedro.gustavo` and `stevie.marie` belong
to one lab and `pgustavo` to another, so a model that had memorised these public
datasets could still tell them apart. Fixing that means rewriting user fields
that rules and analysts both legitimately reason about, and the correlation is
partial rather than perfect. It is recorded as a residual confound in
`docs/results-agent.md` instead of being quietly assumed away.

Pseudonymisation is applied after the sibling repo produced its labels, so it
cannot change any ground truth. It could in principle change whether a rule
would match — a rule keyed to a UNC path containing a host name — which is why
the substitution is confined to host and domain tokens and why the affected
field count is reported by `Pseudonymiser.stats`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Iterable

#: Fields naming the host that produced the event.
HOST_FIELDS = ("Hostname", "Computer", "ComputerName", "host")

#: Fields carrying `DOMAIN\user`, from which the NetBIOS domain is taken.
PRINCIPAL_FIELDS = ("User", "SourceUser", "TargetUser", "ParentUser",
                    "SubjectUserName", "TargetUserName", "SubjectDomainName",
                    "TargetDomainName")

DOMAIN_REPLACEMENT = "corp.example"
NETBIOS_REPLACEMENT = "CORP"

#: Matches a host pseudonym this module hands out. The authorization boundary
#: uses it to tell a host name apart from the other replacements in the same
#: table, since only hosts are assets a session can be scoped to.
HOST_PSEUDONYM = re.compile(r"HOST\d+")

#: Built-in Windows principals. These appear in the `DOMAIN\user` position but
#: name the operating system, not the lab, and they are load-bearing evidence:
#: "NT AUTHORITY\SYSTEM opened a handle to LSASS" reads very differently from
#: "CORP\SYSTEM did". Rewriting them would damage triage to no privacy end.
WELL_KNOWN_PRINCIPALS = frozenset({
    "nt authority", "nt service", "builtin", "iis apppool", "window manager",
    "font driver host", "local service", "network service", "system",
    "everyone", "creator owner", "nt virtual machine", "applicationpackage",
})


@dataclasses.dataclass
class PseudonymStats:
    hosts: int = 0
    domains: int = 0
    fields_rewritten: int = 0
    events: int = 0


class Pseudonymiser:
    """A substitution table derived from one capture, applied to that capture.

    Built by scanning the stream once for host and domain tokens, then applied
    as a single compiled alternation so a 50,000-event capture costs one pass.
    """

    def __init__(self, events: Iterable[dict], sample: int = 5000):
        self._mapping: dict[str, str] = {}
        self._pattern: re.Pattern[str] | None = None
        self.stats = PseudonymStats()
        self._build(events, sample)

    # -- construction -----------------------------------------------------

    def _build(self, events: Iterable[dict], sample: int) -> None:
        hosts: list[str] = []
        domains: list[str] = []
        netbios: list[str] = []

        for index, raw in enumerate(events):
            if index >= sample:
                break
            for field in HOST_FIELDS:
                value = raw.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                short, _, domain = value.partition(".")
                if short.strip().lower() in WELL_KNOWN_PRINCIPALS:
                    continue
                _add(hosts, short)
                if domain:
                    _add(domains, domain)
            for field in PRINCIPAL_FIELDS:
                value = raw.get(field)
                if isinstance(value, str) and "\\" in value:
                    name = value.split("\\", 1)[0]
                    if name.strip().lower() not in WELL_KNOWN_PRINCIPALS:
                        _add(netbios, name)

        # a NetBIOS name that is really the first label of a DNS domain
        # (PANDALAB for pandalab.com) maps to the same pseudonym
        domain_labels = {d.split(".")[0].lower() for d in domains}

        for position, host in enumerate(hosts, start=1):
            self._mapping[host.lower()] = f"HOST{position}"
        for domain in domains:
            self._mapping[domain.lower()] = DOMAIN_REPLACEMENT
        for name in netbios:
            if name.lower() in self._mapping:
                continue
            self._mapping[name.lower()] = (
                NETBIOS_REPLACEMENT if name.lower() in domain_labels
                else NETBIOS_REPLACEMENT
            )

        self.stats.hosts = len(hosts)
        self.stats.domains = len(domains)

        if not self._mapping:
            return
        # longest first, so pandalab.com is consumed before pandalab
        tokens = sorted(self._mapping, key=len, reverse=True)
        self._pattern = re.compile(
            r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in tokens)
            + r")(?![A-Za-z0-9])",
            re.IGNORECASE,
        )

    # -- application ------------------------------------------------------

    def text(self, value: str) -> str:
        if self._pattern is None:
            return value
        return self._pattern.sub(
            lambda m: self._mapping[m.group(1).lower()], value
        )

    def event(self, raw: dict) -> dict:
        """A rewritten copy. The original is never mutated."""
        if self._pattern is None:
            return raw
        out = dict(raw)
        rewritten = 0
        for field, value in raw.items():
            if not isinstance(value, str) or not value:
                continue
            replaced = self.text(value)
            if replaced != value:
                out[field] = replaced
                rewritten += 1
        self.stats.fields_rewritten += rewritten
        self.stats.events += 1
        return out

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._mapping)


def _add(into: list[str], value: str) -> None:
    value = value.strip()
    if value and value.lower() not in {v.lower() for v in into}:
        into.append(value)


def capture_handle(capture_id: str) -> str:
    """An opaque, stable handle for a capture.

    `LSASS_campaign_01` versus `credential_access/empire_over_pth_patch_lsass`
    announces the label before the agent reads a single event: one name is a
    compound intrusion recording and the other is an atomic test. The handle
    keeps captures distinguishable and citable without saying which corpus a
    case came from.
    """
    digest = hashlib.sha256(capture_id.encode()).hexdigest()[:8]
    return f"capture-{digest}"
