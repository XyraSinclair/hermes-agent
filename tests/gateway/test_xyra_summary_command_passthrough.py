from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = [
        {"role": "assistant", "content": "latest assistant output"}
    ]
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._auto_steward_armed = {}
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._draining = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_gateway_bare_sum4xyra_is_not_treated_as_unknown_command(monkeypatch):
    import gateway.run as gateway_run

    runner = _make_runner()
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/sum4xyra should not leak to the agent path as unknown command")
    )
    runner._build_direct_xyra_summary = AsyncMock(return_value="Bottom line\n- summary")

    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"}
    )

    result = await runner._handle_message(_make_event("/sum4xyra"))

    assert result == "Bottom line\n- summary"
    runner._build_direct_xyra_summary.assert_awaited()
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_help_mentions_xyra_summary_when_enabled():
    runner = _make_runner()
    runner._xyra_summary_enabled = True
    runner._xyra_summary_opt_in_required = False
    runner._xyra_summary_opt_in_token = "/sum4xyra"

    result = await runner._handle_help_command(_make_event("/help"))

    assert "Xyra Summary" in result
    assert "`/sum4xyra`" in result
    assert "default final responses" in result


@pytest.mark.asyncio
async def test_gateway_commands_mentions_xyra_summary_when_enabled():
    runner = _make_runner()
    runner._xyra_summary_enabled = True
    runner._xyra_summary_opt_in_required = False
    runner._xyra_summary_opt_in_token = "/sum4xyra"

    result = await runner._handle_commands_command(_make_event("/commands"))

    assert "Xyra Summary" in result
    assert "`/sum4xyra`" in result