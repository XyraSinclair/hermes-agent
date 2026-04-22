import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cli import HermesCLI
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class TestAutoStewardSharedParity:
    def test_cli_and_gateway_share_followthrough_prompt(self):
        cli = HermesCLI.__new__(HermesCLI)
        runner = GatewayRunner.__new__(GatewayRunner)
        assert cli._build_auto_steward_prompt() == runner._build_auto_steward_prompt()

    def test_cli_and_gateway_share_suffix_parser_behavior(self):
        cli = HermesCLI.__new__(HermesCLI)
        runner = GatewayRunner.__new__(GatewayRunner)
        cli._auto_steward_opt_in_token = "/as"
        runner._auto_steward_opt_in_token = "/as"

        cli_parsed = cli._parse_auto_steward_directive("fix the bug /as7")
        gateway_parsed = runner._parse_auto_steward_directive("fix the bug /as7")

        assert cli_parsed.armed is True
        assert gateway_parsed.armed is True
        assert cli_parsed.effective_hops == 7
        assert gateway_parsed.effective_hops == 7
        assert cli_parsed.sanitized_message == "fix the bug"
        assert gateway_parsed.sanitized_message == "fix the bug"

    def test_cli_and_gateway_share_bare_directive_behavior(self):
        cli = HermesCLI.__new__(HermesCLI)
        runner = GatewayRunner.__new__(GatewayRunner)
        cli._auto_steward_opt_in_token = "/as"
        runner._auto_steward_opt_in_token = "/as"

        cli_parsed = cli._parse_auto_steward_directive("/as5")
        gateway_parsed = runner._parse_auto_steward_directive("/as5")

        assert cli_parsed.armed is True
        assert gateway_parsed.armed is True
        assert cli_parsed.effective_hops == 5
        assert gateway_parsed.effective_hops == 5
        assert cli._coerce_auto_steward_message(cli_parsed).startswith("Continue from the existing context")
        assert runner._coerce_auto_steward_message(gateway_parsed).startswith("Continue from the existing context")

    def test_cli_pending_directive_survives_coerced_message(self):
        cli = HermesCLI.__new__(HermesCLI)
        cli._auto_steward_enabled = True
        cli._auto_steward_notice = True
        cli._auto_steward_opt_in_token = "/as"
        cli._auto_steward_opt_in_required = True
        cli._auto_steward_default_hops = 5
        cli._auto_steward_hard_cap_hops = 8
        cli._auto_steward_max_hops = 8
        parsed = cli._parse_auto_steward_directive("/as5")
        cli._auto_steward_pending_directive = parsed
        pending = cli._auto_steward_pending_directive
        assert pending.armed is True
        coerced = cli._coerce_auto_steward_message(parsed)
        reparsed = cli._parse_auto_steward_directive(coerced)
        assert reparsed.armed is False
        assert pending.effective_hops == 5

    def test_gateway_pending_directive_payload_survives_coerced_message(self):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._auto_steward_enabled = True
        runner._auto_steward_notice = True
        runner._auto_steward_opt_in_token = "/as"
        runner._auto_steward_opt_in_required = True
        runner._auto_steward_default_hops = 5
        runner._auto_steward_hard_cap_hops = 8
        runner._auto_steward_max_hops = 8
        directive = runner._parse_auto_steward_directive("/as5")
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")
        event = MessageEvent(
            text=runner._coerce_auto_steward_message(directive),
            message_type=MessageType.TEXT,
            source=source,
            raw_message={"auto_steward_directive": directive.to_payload()},
        )
        payload = event.raw_message["auto_steward_directive"]
        reconstructed = type(directive).from_payload(payload)
        assert reconstructed.armed is True
        assert reconstructed.effective_hops == 5

    def test_gateway_handler_consumes_pending_bare_directive_payload_before_reparse(self, monkeypatch):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._auto_steward_enabled = True
        runner._auto_steward_notice = True
        runner._auto_steward_opt_in_token = "/as"
        runner._auto_steward_opt_in_required = True
        runner._auto_steward_default_hops = 3
        runner._auto_steward_hard_cap_hops = 8
        runner._auto_steward_max_hops = 8
        runner._auto_steward_depths = {}
        runner._auto_steward_armed = {}
        runner._auto_steward_effective_hops = {}
        runner._auto_steward_directives = {}
        runner._auto_steward_episodes = {}
        runner._auto_steward_last_decisions = {}
        runner.hooks = SimpleNamespace(emit=AsyncMock())
        runner.adapters = {}
        runner.session_store = SimpleNamespace(
            get_or_create_session=lambda _source: SimpleNamespace(
                session_key="sess-1",
                session_id="session-1",
                created_at=1,
                updated_at=2,
                was_auto_reset=False,
                last_prompt_tokens=0,
            ),
            load_transcript=lambda _session_id: [],
            has_any_sessions=lambda: True,
        )
        runner.config = {}
        runner._set_session_env = lambda context: None
        runner._format_session_info = lambda: ""
        runner._run_agent = AsyncMock(return_value={"final_response": "done", "messages": [], "api_calls": 0})
        runner._should_drop_unarmed_auto_steward_message = MagicMock(return_value=False)
        monkeypatch.setattr("gateway.run.build_session_context", lambda source, config, session_entry: {})
        monkeypatch.setattr("gateway.run.build_session_context_prompt", lambda context, redact_pii=False: "")

        source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1")
        directive = runner._parse_auto_steward_directive("/as5")
        event = MessageEvent(
            text=runner._coerce_auto_steward_message(directive),
            message_type=MessageType.TEXT,
            source=source,
            raw_message={"auto_steward_directive": directive.to_payload()},
        )

        result = asyncio.run(runner._handle_message_with_agent(event, source, "quick-key", 0))

        assert result == (
            "Auto-steward armed for 5 hops, but there’s no active task in this chat to continue.\n\n"
            "Send your actual request with /as5 appended, e.g. “audit this repo and fix the failing tests /as5”\n\n"
            "Or just give the task now, and I’ll execute it."
        )
        assert runner._auto_steward_directives["sess-1"].raw_directive == "/as5"
        assert runner._auto_steward_armed["sess-1"] is True
        assert runner._auto_steward_effective_hops["sess-1"] == 5
        assert "auto_steward_directive" not in event.raw_message
        runner._run_agent.assert_not_awaited()
