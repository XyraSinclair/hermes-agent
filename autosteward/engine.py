from __future__ import annotations

import uuid
from typing import Callable, Optional

from .heuristics import compute_heuristics, fingerprint_text, infer_frontiers
from .policy import decide_next_action
from .prompts import build_auto_steward_prompt
from .state import AutoStewardConfig, Decision, EpisodeState, HopEvidence, ParsedDirective
from .storage import append_episode_log


def create_episode_state(
    *,
    session_id: str,
    directive: ParsedDirective,
    previous: EpisodeState | None,
) -> EpisodeState:
    episode_id = f"as-{uuid.uuid4().hex[:12]}"
    return EpisodeState(
        episode_id=episode_id,
        session_id=session_id,
        directive=directive,
        armed=directive.armed,
        effective_max_hops=directive.effective_hops or 0,
        hops_used=0,
        review_calls=0,
        low_progress_streak=0,
        current_frontier=None,
        recent_frontiers=[],
        recent_command_signatures=[],
        recent_error_signatures=[],
        cumulative_tokens=0,
        policy_version="autosteward-v1",
        progress_ema=0.0,
        recent_response_fingerprints=[],
        recent_response_texts=[],
    )


def build_hop_evidence(
    *,
    episode: EpisodeState,
    response: str,
    result: Optional[dict],
    response_requires_user_input: Callable[[str], bool],
) -> HopEvidence:
    text = str(response or "")
    lower = text.lower()
    error_signatures: list[str] = []
    for needle in ("traceback", "error:", "exception", "failed", "timed out"):
        if needle in lower:
            error_signatures.append(needle)
    return HopEvidence(
        hop_index=episode.hops_used + 1,
        assistant_summary=text[:400],
        commands=[],
        files_read=[],
        files_written=[],
        tests=[],
        error_signatures=error_signatures,
        interactive_terminal_wait=bool(result and result.get("interrupted")),
        required_user_input=response_requires_user_input(text),
        high_risk_action_requires_approval=False,
        token_usage=None,
        response_preview=text[:500],
        response_fingerprint=fingerprint_text(text),
    )


def decide_followthrough(
    *,
    episode: EpisodeState,
    response: str,
    result: Optional[dict],
    cfg: AutoStewardConfig,
    response_requires_user_input: Callable[[str], bool],
    response_looks_terminal: Callable[[str], bool],
) -> tuple[Decision, HopEvidence]:
    hop = build_hop_evidence(
        episode=episode,
        response=response,
        result=result,
        response_requires_user_input=response_requires_user_input,
    )
    current_frontier, alternatives = infer_frontiers(episode, hop, response)
    heuristics = compute_heuristics(
        episode,
        hop,
        response,
        response_requires_user_input=response_requires_user_input,
        response_looks_terminal=response_looks_terminal,
    )
    decision = decide_next_action(
        episode,
        hop,
        heuristics,
        current_frontier=current_frontier,
        alternative_frontiers=alternatives,
        cfg=cfg,
    )
    episode.hops_used = hop.hop_index
    episode.low_progress_streak = heuristics.low_progress_streak
    episode.progress_ema = heuristics.progress_ema
    episode.current_frontier = current_frontier.kind
    episode.recent_frontiers = (episode.recent_frontiers + [current_frontier.kind])[-5:]
    episode.recent_error_signatures = (episode.recent_error_signatures + hop.error_signatures)[-10:]
    episode.recent_response_fingerprints = (episode.recent_response_fingerprints + [hop.response_fingerprint])[-5:]
    episode.recent_response_texts = (episode.recent_response_texts + [response or ""])[-3:]
    if decision.kind.value == "review":
        episode.review_calls += 1
    if cfg.log_episodes:
        append_episode_log(
            {
                "episode_id": episode.episode_id,
                "session_id": episode.session_id,
                "hop_index": hop.hop_index,
                "directive": episode.directive.raw_directive,
                "effective_max_hops": episode.effective_max_hops,
                "decision": decision.kind.value,
                "reason_codes": decision.reason_codes,
                "current_frontier": current_frontier.kind,
                "alternative_frontiers": [f.kind for f in alternatives],
                "heuristics": {
                    "progress_ema": heuristics.progress_ema,
                    "novelty_score": heuristics.novelty_score,
                    "repetition_score": heuristics.repetition_score,
                    "executability_score": heuristics.executability_score,
                    "confidence_proxy": heuristics.confidence_proxy,
                    "blocked_score": heuristics.blocked_score,
                    "done_confidence": heuristics.done_confidence,
                },
                "response_fingerprint": hop.response_fingerprint,
            }
        )
    return decision, hop


def build_followthrough_prompt(decision: Decision | None) -> str:
    return build_auto_steward_prompt(decision)
