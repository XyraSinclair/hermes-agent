from autosteward.parser import parse_auto_steward_directive
from autosteward.state import AutoStewardConfig


def _cfg(**overrides):
    base = AutoStewardConfig(
        enabled=True,
        opt_in_required=True,
        opt_in_token="/as",
        default_hops_when_armed=3,
        hard_cap_hops=8,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_parse_suffix_arms_with_default_hops():
    parsed = parse_auto_steward_directive("ship it /as", _cfg())
    assert parsed.armed is True
    assert parsed.raw_directive == "/as"
    assert parsed.requested_hops is None
    assert parsed.effective_hops == 3
    assert parsed.sanitized_message == "ship it"


def test_parse_bare_directive_arms_with_default_hops():
    parsed = parse_auto_steward_directive("/as", _cfg())
    assert parsed.armed is True
    assert parsed.raw_directive == "/as"
    assert parsed.requested_hops is None
    assert parsed.effective_hops == 3
    assert parsed.sanitized_message == ""


def test_parse_suffix_with_numeric_override():
    parsed = parse_auto_steward_directive("keep going /as7", _cfg())
    assert parsed.armed is True
    assert parsed.raw_directive == "/as7"
    assert parsed.requested_hops == 7
    assert parsed.effective_hops == 7
    assert parsed.sanitized_message == "keep going"


def test_parse_bare_numeric_override():
    parsed = parse_auto_steward_directive("/as7", _cfg())
    assert parsed.armed is True
    assert parsed.raw_directive == "/as7"
    assert parsed.requested_hops == 7
    assert parsed.effective_hops == 7
    assert parsed.sanitized_message == ""


def test_parse_clamps_numeric_override_to_hard_cap():
    parsed = parse_auto_steward_directive("keep going /as99", _cfg(hard_cap_hops=5))
    assert parsed.armed is True
    assert parsed.requested_hops == 99
    assert parsed.effective_hops == 5
    assert parsed.warnings


def test_parse_ignores_non_suffix_occurrence():
    parsed = parse_auto_steward_directive("mention /as in docs but do not arm", _cfg())
    assert parsed.armed is False
    assert parsed.sanitized_message == "mention /as in docs but do not arm"


def test_parse_ignores_inline_code_suffix_like_text():
    parsed = parse_auto_steward_directive("document `/as7` exactly", _cfg())
    assert parsed.armed is False
    assert parsed.sanitized_message == "document `/as7` exactly"


def test_parse_rejects_invalid_zero_suffix():
    parsed = parse_auto_steward_directive("keep going /as0", _cfg())
    assert parsed.armed is False
    assert parsed.sanitized_message == "keep going /as0"
    assert parsed.warnings


def test_parse_arms_without_suffix_when_opt_in_not_required():
    parsed = parse_auto_steward_directive("keep going normally", _cfg(opt_in_required=False))
    assert parsed.armed is True
    assert parsed.raw_directive is None
    assert parsed.effective_hops == 3
    assert parsed.sanitized_message == "keep going normally"


def test_parse_explicit_bare_override_even_when_opt_in_is_disabled():
    parsed = parse_auto_steward_directive(
        "/as5",
        _cfg(opt_in_required=False, opt_in_token=""),
    )
    assert parsed.armed is True
    assert parsed.raw_directive == "/as5"
    assert parsed.requested_hops == 5
    assert parsed.effective_hops == 5
    assert parsed.sanitized_message == ""
