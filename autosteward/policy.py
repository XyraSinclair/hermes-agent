from __future__ import annotations

from .state import AutoStewardConfig, Decision, DecisionKind, EpisodeState, Frontier, HeuristicScores, HopEvidence


def _stop(reason: str, frontier: Frontier | None = None) -> Decision:
    return Decision(kind=DecisionKind.STOP, reason_codes=[reason], frontier=frontier)


def _cont(reason: str, frontier: Frontier) -> Decision:
    return Decision(kind=DecisionKind.CONTINUE, reason_codes=[reason], frontier=frontier)


def _redirect(reason: str, frontier: Frontier) -> Decision:
    return Decision(kind=DecisionKind.REDIRECT, reason_codes=[reason], frontier=frontier)


def _review(reason: str, frontier: Frontier | None = None) -> Decision:
    return Decision(kind=DecisionKind.REVIEW, reason_codes=[reason], frontier=frontier)


def decide_next_action(
    ep: EpisodeState,
    hop: HopEvidence,
    heuristics: HeuristicScores,
    *,
    current_frontier: Frontier,
    alternative_frontiers: list[Frontier],
    cfg: AutoStewardConfig,
) -> Decision:
    if not ep.armed:
        return _stop("NOT_ARMED", current_frontier)
    if ep.hops_used >= ep.effective_max_hops:
        return _stop("HOP_CAP_REACHED", current_frontier)
    if ep.directive.raw_directive and not (ep.directive.sanitized_message or "").strip():
        return _stop("NO_ROOT_TASK", current_frontier)
    if hop.required_user_input:
        return _stop("REQUIRED_USER_INPUT", current_frontier)
    if hop.interactive_terminal_wait:
        return _stop("INTERACTIVE_TERMINAL", current_frontier)
    if hop.high_risk_action_requires_approval:
        return _stop("APPROVAL_REQUIRED", current_frontier)
    if heuristics.done_confidence >= cfg.done_threshold:
        return _stop("DONE_EVIDENCE_STRONG", current_frontier)

    best_alt = max(alternative_frontiers, key=lambda item: item.score, default=None)

    if heuristics.repetition_score >= 0.85 and heuristics.novelty_score <= 0.2:
        if best_alt and best_alt.score >= current_frontier.score + cfg.redirect_margin:
            return _redirect("LOOP_DETECTED_BETTER_FRONTIER", best_alt)
        if cfg.review_enabled and ep.review_calls < cfg.max_reviews_per_episode:
            return _review("LOOP_DETECTED_AMBIGUOUS", best_alt or current_frontier)
        return _stop("LOOP_DETECTED", current_frontier)

    marginal_utility = (
        0.35 * heuristics.progress_ema
        + 0.20 * heuristics.executability_score
        + 0.15 * heuristics.novelty_score
        + 0.15 * heuristics.confidence_proxy
        + 0.15 * current_frontier.score
        - 0.25 * heuristics.repetition_score
        - 0.20 * heuristics.blocked_score
    )
    dynamic_continue_threshold = cfg.continue_threshold + (0.05 * (ep.hops_used / max(ep.effective_max_hops, 1)))

    if best_alt and best_alt.score >= current_frontier.score + cfg.redirect_margin:
        if heuristics.repetition_score >= 0.60 or marginal_utility < dynamic_continue_threshold:
            return _redirect("BETTER_FRONTIER_AVAILABLE", best_alt)

    if cfg.review_enabled and ep.review_calls < cfg.max_reviews_per_episode:
        ambiguous_utility = abs(marginal_utility - dynamic_continue_threshold) <= cfg.review_band
        ambiguous_frontier = bool(best_alt and abs(best_alt.score - current_frontier.score) <= 0.10)
        expensive_low_yield = ep.hops_used >= cfg.review_start_hop and heuristics.progress_ema < 0.25
        if (ep.hops_used >= cfg.review_start_hop and ambiguous_utility) or ambiguous_frontier or expensive_low_yield:
            return _review("AMBIGUOUS_STATE", best_alt or current_frontier)

    if marginal_utility >= dynamic_continue_threshold:
        return _cont("POSITIVE_MARGINAL_UTILITY", current_frontier)

    if (
        heuristics.low_progress_streak < cfg.low_progress_patience
        and heuristics.executability_score >= 0.70
        and heuristics.repetition_score < 0.60
    ):
        return _cont("GRACE_HOP", current_frontier)

    return _stop("LOW_MARGINAL_UTILITY", current_frontier)
