from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from autosteward.reviewer import review_decision_async, review_decision_sync, review_trigger_for
from autosteward.state import AutoStewardConfig, Decision, DecisionKind, EpisodeState, Frontier, ParsedDirective


def _cfg(**overrides):
    cfg = AutoStewardConfig(
        enabled=True,
        opt_in_required=True,
        opt_in_token="/as",
        default_hops_when_armed=5,
        hard_cap_hops=8,
        review_enabled=True,
        review_on_user_input_boundary=True,
        review_provider="anthropic",
        review_model="claude-opus-4.6",
        review_timeout=30,
        review_min_confidence=0.75,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _episode():
    return EpisodeState(
        episode_id="ep-1",
        session_id="sess-1",
        directive=ParsedDirective(
            armed=True,
            raw_directive="/as",
            requested_hops=None,
            effective_hops=5,
            sanitized_message="fix the issue",
            warnings=[],
        ),
        armed=True,
        effective_max_hops=5,
        hops_used=1,
        review_calls=0,
        low_progress_streak=0,
        current_frontier="IMPLEMENT_NEXT_SUBTASK",
        recent_frontiers=["IMPLEMENT_NEXT_SUBTASK"],
        recent_command_signatures=[],
        recent_error_signatures=[],
        cumulative_tokens=0,
        policy_version="test",
        progress_ema=0.5,
        recent_response_fingerprints=[],
        recent_response_texts=["Want me to deploy this now?"],
    )


def _response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_review_trigger_for_required_user_input_boundary():
    decision = Decision(kind=DecisionKind.STOP, reason_codes=["REQUIRED_USER_INPUT"], frontier=Frontier("VALIDATE_EDIT", 0.8))
    trigger = review_trigger_for(_cfg(), decision)
    assert trigger is not None
    assert trigger.requires_user_input_boundary is True


def test_review_decision_sync_can_override_required_user_input_stop():
    decision = Decision(kind=DecisionKind.STOP, reason_codes=["REQUIRED_USER_INPUT"], frontier=Frontier("VALIDATE_EDIT", 0.8))
    with patch("autosteward.reviewer.call_llm", return_value=_response('{"automate_now": true, "kind": "continue", "confidence": 0.93, "reason": "The assistant is only asking a politeness question; validating the edit is still the obvious bounded next step.", "frontier_kind": null, "allowed_scope": "Run one bounded validation step.", "forbidden_actions": ["deploy", "destructive changes"]}')):
        with patch("autosteward.reviewer.append_episode_log") as append_log:
            outcome = review_decision_sync(
                episode=_episode(),
                response="Done with the patch. Want me to deploy this now?",
                decision=decision,
                cfg=_cfg(),
            )
    assert outcome.approved is True
    assert outcome.replacement.kind == DecisionKind.CONTINUE
    assert "SMART_REVIEW_APPROVED" in outcome.replacement.reason_codes
    assert outcome.replacement.metadata["decision_source"] == "SMART_REVIEW_APPROVED"
    assert outcome.replacement.metadata["review_prior_kind"] == "stop"
    append_log.assert_called_once()
    logged = append_log.call_args.args[0]
    assert logged["event_type"] == "smart_review"
    assert logged["decision_source"] == "SMART_REVIEW_APPROVED"
    assert logged["final_decision"] == "continue"


def test_review_decision_sync_fails_closed_on_error():
    decision = Decision(kind=DecisionKind.REVIEW, reason_codes=["AMBIGUOUS_STATE"], frontier=Frontier("INSPECT_ERROR_SITE", 0.7))
    with patch("autosteward.reviewer.call_llm", side_effect=RuntimeError("provider unavailable")):
        with patch("autosteward.reviewer.append_episode_log") as append_log:
            outcome = review_decision_sync(
                episode=_episode(),
                response="I might continue, but the state is ambiguous.",
                decision=decision,
                cfg=_cfg(),
            )
    assert outcome.approved is True
    assert outcome.replacement.kind == DecisionKind.STOP
    assert "SMART_REVIEW_FAILED_CLOSED" in outcome.replacement.reason_codes
    assert outcome.replacement.metadata["decision_source"] == "SMART_REVIEW_FAILED_CLOSED"
    append_log.assert_called_once()
    logged = append_log.call_args.args[0]
    assert logged["decision_source"] == "SMART_REVIEW_FAILED_CLOSED"
    assert logged["final_decision"] == "stop"


@pytest.mark.asyncio
async def test_review_decision_async_blocks_low_confidence_override():
    decision = Decision(kind=DecisionKind.REVIEW, reason_codes=["AMBIGUOUS_STATE"], frontier=Frontier("RUN_TARGETED_TEST", 0.6))
    with patch("autosteward.reviewer.async_call_llm", new=AsyncMock(return_value=_response('{"automate_now": true, "kind": "continue", "confidence": 0.40, "reason": "Low confidence.", "frontier_kind": null, "allowed_scope": "", "forbidden_actions": []}'))):
        with patch("autosteward.reviewer.append_episode_log") as append_log:
            outcome = await review_decision_async(
                episode=_episode(),
                response="This might need a user choice, but maybe not.",
                decision=decision,
                cfg=_cfg(),
            )
    assert outcome.approved is True
    assert outcome.replacement.kind == DecisionKind.STOP
    assert "SMART_REVIEW_LOW_CONFIDENCE" in outcome.replacement.reason_codes
    assert outcome.replacement.metadata["decision_source"] == "SMART_REVIEW_BLOCKED"
    append_log.assert_called_once()
    logged = append_log.call_args.args[0]
    assert logged["decision_source"] == "SMART_REVIEW_BLOCKED"
    assert logged["final_decision"] == "stop"
