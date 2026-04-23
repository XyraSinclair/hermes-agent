import asyncio
from types import SimpleNamespace

from cli import HermesCLI
from gateway.run import GatewayRunner
from agent.xyra_summary import gateway_summary_help_lines, parse_summary_directive, summary_help_entries


def test_parse_summary_directive_strips_trailing_suffix():
    parsed = parse_summary_directive("summarize the last output /sum4xyra")
    assert parsed.armed is True
    assert parsed.raw_directive == "/sum4xyra"
    assert parsed.sanitized_message == "summarize the last output"


def test_parse_summary_directive_does_not_trigger_on_inline_prose():
    parsed = parse_summary_directive("please literally mention /sum4xyra in the docs")
    assert parsed.armed is False
    assert parsed.sanitized_message == "please literally mention /sum4xyra in the docs"


def test_parse_summary_directive_bare_token_is_armed_with_empty_message():
    parsed = parse_summary_directive("/sum4xyra")
    assert parsed.armed is True
    assert parsed.sanitized_message == ""


def test_parse_summary_directive_can_be_always_on_without_suffix():
    parsed = parse_summary_directive("catch me up", token="/sum4xyra", opt_in_required=False)
    assert parsed.armed is True
    assert parsed.raw_directive is None
    assert parsed.sanitized_message == "catch me up"


def test_summary_help_entries_describe_default_on_mode():
    entries = summary_help_entries(enabled=True, opt_in_required=False)
    assert any(label == "/sum4xyra" for label, _ in entries)
    assert any(label == "default final responses" for label, _ in entries)


def test_gateway_summary_help_lines_render_token_and_default_mode():
    lines = gateway_summary_help_lines(enabled=True, opt_in_required=False)
    joined = "\n".join(lines)
    assert "`/sum4xyra`" in joined
    assert "default final responses" in joined


def test_gateway_summary_help_lines_hidden_when_disabled():
    assert gateway_summary_help_lines(enabled=False) == []


def test_cli_and_gateway_share_xyra_summary_parser_behavior():
    cli = HermesCLI.__new__(HermesCLI)
    runner = GatewayRunner.__new__(GatewayRunner)
    cli._xyra_summary_opt_in_token = "/sum4xyra"
    runner._xyra_summary_opt_in_token = "/sum4xyra"

    cli_parsed = cli._parse_xyra_summary_directive("catch me up /sum4xyra")
    gateway_parsed = runner._parse_xyra_summary_directive("catch me up /sum4xyra")

    assert cli_parsed.armed is True
    assert gateway_parsed.armed is True
    assert cli_parsed.sanitized_message == "catch me up"
    assert gateway_parsed.sanitized_message == "catch me up"


def test_cli_build_xyra_summary_block_formats_summary(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._xyra_summary_enabled = True
    cli._xyra_summary_armed = True
    cli._xyra_summary_two_pass = True
    cli._xyra_summary_heading = "Xyra summary"
    cli._xyra_summary_max_context_messages = 8
    cli._xyra_summary_max_chars_per_message = 1200
    cli.conversation_history = []

    monkeypatch.setattr(
        "cli.summarize_for_xyra_sync",
        lambda **kwargs: "Bottom line\n- this is the important part",
    )

    block = cli._build_xyra_summary_block(
        "summarize the last output",
        "Shipped the tranche.",
        SimpleNamespace(kind=SimpleNamespace(value="stop")),
    )

    assert "Xyra summary" in block
    assert "this is the important part" in block


def test_cli_build_direct_xyra_summary_uses_last_assistant_message(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._xyra_summary_two_pass = True
    cli._xyra_summary_max_context_messages = 8
    cli._xyra_summary_max_chars_per_message = 1200
    cli.conversation_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "old answer"},
        {"role": "assistant", "content": "latest answer"},
    ]

    captured = {}

    def _fake_summary(**kwargs):
        captured.update(kwargs)
        return "Bottom line\n- latest answer summary"

    monkeypatch.setattr("cli.summarize_for_xyra_sync", _fake_summary)

    out = cli._build_direct_xyra_summary()
    assert "latest answer summary" in out
    assert captured["assistant_response"] == "latest answer"


def test_cli_build_xyra_summary_block_skips_intermediate_followthrough(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._xyra_summary_enabled = True
    cli._xyra_summary_armed = True
    cli._xyra_summary_two_pass = True
    cli._xyra_summary_heading = "Xyra summary"
    cli._xyra_summary_max_context_messages = 8
    cli._xyra_summary_max_chars_per_message = 1200
    cli.conversation_history = []

    monkeypatch.setattr(
        "cli.summarize_for_xyra_sync",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("summary should not run")),
    )

    block = cli._build_xyra_summary_block(
        "summarize the last output",
        "Still working.",
        SimpleNamespace(kind=SimpleNamespace(value="continue")),
    )

    assert block == ""


def test_cli_build_xyra_summary_block_skips_failed_results(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli._xyra_summary_enabled = True
    cli._xyra_summary_armed = True
    cli._xyra_summary_two_pass = True
    cli._xyra_summary_heading = "Xyra summary"
    cli._xyra_summary_max_context_messages = 8
    cli._xyra_summary_max_chars_per_message = 1200
    cli.conversation_history = []

    monkeypatch.setattr(
        "cli.summarize_for_xyra_sync",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("summary should not run")),
    )

    block = cli._build_xyra_summary_block(
        "summarize the last output",
        "Error: provider failed",
        SimpleNamespace(kind=SimpleNamespace(value="stop")),
        {"failed": True},
    )

    assert block == ""


def test_gateway_build_xyra_summary_block_formats_summary(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._xyra_summary_enabled = True
    runner._xyra_summary_armed = {"sess": True}
    runner._xyra_summary_two_pass = True
    runner._xyra_summary_heading = "Xyra summary"
    runner._xyra_summary_max_context_messages = 8
    runner._xyra_summary_max_chars_per_message = 1200
    runner._xyra_summary_directives = {}

    async def _fake_summary(**kwargs):
        return "Bottom line\n- gateway summary"

    monkeypatch.setattr("gateway.run.summarize_for_xyra_async", _fake_summary)

    block = asyncio.run(
        runner._build_xyra_summary_block(
            "catch me up",
            "Gateway response",
            [],
            SimpleNamespace(kind=SimpleNamespace(value="stop")),
            session_key="sess",
        )
    )

    assert "Xyra summary" in block
    assert "gateway summary" in block


def test_gateway_build_direct_xyra_summary_uses_last_assistant_message(monkeypatch):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._xyra_summary_two_pass = True
    runner._xyra_summary_max_context_messages = 8
    runner._xyra_summary_max_chars_per_message = 1200

    captured = {}

    async def _fake_summary(**kwargs):
        captured.update(kwargs)
        return "Bottom line\n- gateway direct summary"

    monkeypatch.setattr("gateway.run.summarize_for_xyra_async", _fake_summary)

    out = asyncio.run(
        runner._build_direct_xyra_summary(
            [
                {"role": "assistant", "content": "older"},
                {"role": "assistant", "content": "newest"},
            ]
        )
    )
    assert "gateway direct summary" in out
    assert captured["assistant_response"] == "newest"
