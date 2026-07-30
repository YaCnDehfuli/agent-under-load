"""The provenance taxonomy, and its two opposite defaults.

If these break, the threat model has drifted and every attack number in the repo
is measuring something other than what it claims.
"""

from __future__ import annotations

import pytest

from agent import provenance
from agent.provenance import Provenance


@pytest.mark.parametrize("field", [
    "CommandLine", "ParentCommandLine", "Image", "TargetFilename",
    "ScriptBlockText", "ServiceName", "Details", "Product", "TargetObject",
])
def test_free_text_fields_are_writable(field):
    assert provenance.classify(field) is Provenance.WRITABLE
    assert provenance.is_injectable(field)


@pytest.mark.parametrize("field", ["GrantedAccess", "CallTrace",
                                   "IntegrityLevel", "Hashes", "User"])
def test_constrained_fields_are_influenced_and_never_injectable(field):
    """An access mask cannot hold a sentence, so it cannot hold a payload.

    Injecting here would inflate attack success with an attack nobody can
    mount, which is the specific dishonesty this class exists to prevent.
    """
    assert provenance.classify(field) is Provenance.INFLUENCED
    assert not provenance.is_injectable(field)


@pytest.mark.parametrize("field", ["EventID", "Channel", "ProviderGuid",
                                   "UtcTime", "Hostname", "ProcessGuid",
                                   "SourceProcessId", "RuleName"])
def test_os_fields_are_os_and_never_injectable(field):
    assert provenance.classify(field) is Provenance.OS
    assert not provenance.is_injectable(field)
    assert not provenance.classify(field).is_untrusted


def test_unknown_field_is_untrusted_for_defence():
    """Defence over-distrusts: an unrecognised field is not thereby safe."""
    assert provenance.classify("SomeFieldSysmonAddedLastTuesday") is Provenance.WRITABLE


def test_unknown_field_is_refused_as_an_injection_target():
    """Attack under-reaches: unknown means off-limits, not fair game."""
    assert not provenance.is_injectable("SomeFieldSysmonAddedLastTuesday")
    with pytest.raises(provenance.ProvenanceError):
        provenance.require_injectable(["CommandLine", "GrantedAccess"])


def test_require_injectable_names_every_refused_field():
    with pytest.raises(provenance.ProvenanceError) as caught:
        provenance.require_injectable(["GrantedAccess", "EventID", "CommandLine"])
    message = str(caught.value)
    assert "GrantedAccess" in message and "EventID" in message
    assert "CommandLine" not in message.split(".")[0]


def test_rendered_message_inherits_the_strongest_provenance_present():
    """Sysmon's Message concatenates other fields, so it inherits from them."""
    with_command_line = {"EventID": 1, "Message": "...", "CommandLine": "whoami"}
    assert provenance.classify_in_event("Message", with_command_line) is \
        Provenance.WRITABLE

    os_only = {"EventID": 10, "Channel": "x", "Message": "...",
               "GrantedAccess": "0x1010"}
    assert provenance.classify_in_event("Message", os_only) is Provenance.INFLUENCED

    nothing_else = {"Message": "..."}
    assert provenance.classify_in_event("Message", nothing_else) is Provenance.WRITABLE


def test_the_three_classes_do_not_overlap():
    assert not (provenance.WRITABLE_FIELDS & provenance.INFLUENCED_FIELDS)
    assert not (provenance.WRITABLE_FIELDS & provenance.OS_FIELDS)
    assert not (provenance.INFLUENCED_FIELDS & provenance.OS_FIELDS)


def test_untrusted_fields_lists_what_an_adversary_wrote():
    event = {"EventID": 1, "GrantedAccess": "0x1010", "CommandLine": "whoami",
             "Image": "C:\\x.exe", "Hostname": "H"}
    assert provenance.untrusted_fields(event) == ["CommandLine", "Image"]
