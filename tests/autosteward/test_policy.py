from autosteward.policy import decide_next_action
from autosteward.state import (
    AutoStewardConfig,
    DecisionKind,
    EpisodeState,
    Frontier,
    HeuristicScores,
    HopEvidence,
    ParsedDirective,
)


def _cfg(**overrides):
    cfg = AutoStewardConfig(
        enabled=True,
        opt_in_required=True,
        opt_in_token="/as",
        default_hops_when_armed=3,
        hard_cap_hops=8,
        continue_threshold=0.45,
        done_threshold=0.8,
        redirect_margin=0.2,
        low_progress_patience=2,
        review_start_hop=3,
        review_band=0.1,
        max_reviews_per_episode=2,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _episode(**overrides):
    ep = EpisodeState(
        episode_id="ep-1",
        session_id="sess-1",
        directive=ParsedDirective(
            armed=True,
            raw_directive="/as",
            requested_hops=None,
            effective_hops=3,
            sanitized_message="fix it",
            warnings=[],
        ),
        armed=True,
        effective_max_hops=3,
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
    )
    for key, value in overrides.items():
        setattr(ep, key, value)
    return ep


def _hop(**overrides):
    hop = HopEvidence(
        hop_index=1,
        assistant_summary="patched code and should validate next",
        commands=[],
        files_read=[],
        files_written=[],
        tests=[],
        error_signatures=[],
        interactive_terminal_wait=False,
        required_user_input=False,
        high_risk_action_requires_approval=False,
        token_usage=None,
    )
    for key, value in overrides.items():
        setattr(hop, key, value)
    return hop


def _scores(**overrides):
    scores = HeuristicScores(
        progress_ema=0.7,
        novelty_score=0.6,
        repetition_score=0.1,
        executability_score=0.85,
        confidence_proxy=0.8,
        blocked_score=0.0,
        done_confidence=0.1,
        low_progress_streak=0,
    )
    for key, value in overrides.items():
        setattr(scores, key, value)
    return scores


def test_policy_stops_when_done_evidence_is_strong():
    decision = decide_next_action(
        _episode(),
        _hop(),
        _scores(done_confidence=0.95),
        current_frontier=Frontier("VALIDATE_EDIT", 0.5),
        alternative_frontiers=[],
        cfg=_cfg(),
    )
    assert decision.kind == DecisionKind.STOP
    assert "DONE_EVIDENCE_STRONG" in decision.reason_codes


def test_policy_redirects_when_loop_detected_and_better_frontier_exists():
    decision = decide_next_action(
        _episode(progress_ema=0.1),
        _hop(error_signatures=["ImportError@foo.py"]),
        _scores(progress_ema=0.1, novelty_score=0.1, repetition_score=0.92, executability_score=0.6, confidence_proxy=0.25),
        current_frontier=Frontier("RUN_TARGETED_TEST", 0.45),
        alternative_frontiers=[Frontier("INSPECT_ERROR_SITE", 0.85)],
        cfg=_cfg(),
    )
    assert decision.kind == DecisionKind.REDIRECT
    assert decision.frontier.kind == "INSPECT_ERROR_SITE"
    assert "LOOP_DETECTED_BETTER_FRONTIER" in decision.reason_codes


def test_policy_continues_when_marginal_utility_is_positive():
    decision = decide_next_action(
        _episode(),
        _hop(files_written=["src/foo.py"]),
        _scores(),
        current_frontier=Frontier("VALIDATE_EDIT", 0.9),
        alternative_frontiers=[Frontier("IMPLEMENT_NEXT_SUBTASK", 0.5)],
        cfg=_cfg(),
    )
    assert decision.kind == DecisionKind.CONTINUE
    assert decision.frontier.kind == "VALIDATE_EDIT"


def test_policy_requests_review_only_for_ambiguous_state_after_threshold():
    decision = decide_next_action(
        _episode(hops_used=3, effective_max_hops=7),
        _hop(),
        _scores(progress_ema=0.2, novelty_score=0.2, repetition_score=0.55, executability_score=0.6, confidence_proxy=0.35),
        current_frontier=Frontier("RUN_TARGETED_TEST", 0.48),
        alternative_frontiers=[Frontier("INSPECT_ERROR_SITE", 0.52)],
        cfg=_cfg(review_enabled=True),
    )
    assert decision.kind == DecisionKind.REVIEW
    assert "AMBIGUOUS_STATE" in decision.reason_codes


def test_policy_stops_for_bare_arming_directive_without_task():
    decision = decide_next_action(
        _episode(
            directive=ParsedDirective(
                armed=True,
                raw_directive="/as5",
                requested_hops=5,
                effective_hops=5,
                sanitized_message="",
                warnings=[],
            ),
            effective_max_hops=5,
            hops_used=0,
        ),
        _hop(),
        _scores(),
        current_frontier=Frontier("IMPLEMENT_NEXT_SUBTASK", 0.55),
        alternative_frontiers=[Frontier("BROADEN_SEARCH", 0.4)],
        cfg=_cfg(),
    )
    assert decision.kind == DecisionKind.STOP
    assert "NO_ROOT_TASK" in decision.reason_codes
