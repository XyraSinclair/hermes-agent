from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from typing import Callable

from .state import EpisodeState, Frontier, HeuristicScores, HopEvidence


def fingerprint_text(text: str) -> str:
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def response_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=" ".join(a.split()).lower(), b=" ".join(b.split()).lower()).ratio()


def infer_frontiers(ep: EpisodeState, hop: HopEvidence, response: str) -> tuple[Frontier, list[Frontier]]:
    current = Frontier("IMPLEMENT_NEXT_SUBTASK", 0.55)
    alternatives: list[Frontier] = []
    if hop.required_user_input:
        current = Frontier("ASK_USER", 1.0)
    elif hop.high_risk_action_requires_approval:
        current = Frontier("ASK_USER", 1.0)
    elif hop.files_written:
        current = Frontier("VALIDATE_EDIT", 0.9)
        alternatives.append(Frontier("IMPLEMENT_NEXT_SUBTASK", 0.45))
    elif hop.error_signatures:
        current = Frontier("RUN_TARGETED_TEST", 0.55)
        alternatives.append(Frontier("INSPECT_ERROR_SITE", 0.82))
        alternatives.append(Frontier("BROADEN_SEARCH", 0.42))
    elif "remaining: none" in (response or "").lower() or "work is complete" in (response or "").lower():
        current = Frontier("SUMMARIZE_CLOSE", 0.95)
    else:
        alternatives.append(Frontier("BROADEN_SEARCH", 0.4))
    return current, alternatives


def compute_heuristics(
    ep: EpisodeState,
    hop: HopEvidence,
    response: str,
    *,
    response_requires_user_input: Callable[[str], bool],
    response_looks_terminal: Callable[[str], bool],
) -> HeuristicScores:
    similarity = 0.0
    recent_text = ep.recent_response_texts[-1] if ep.recent_response_texts else ""
    if recent_text:
        similarity = response_similarity(recent_text, response)

    novelty = 1.0 - similarity
    if hop.files_written:
        novelty = max(novelty, 0.75)
    elif hop.error_signatures:
        novelty = max(novelty, 0.55)

    repetition = similarity
    if hop.response_fingerprint and hop.response_fingerprint in ep.recent_response_fingerprints:
        repetition = max(repetition, 0.9)
    if hop.error_signatures and any(sig in ep.recent_error_signatures for sig in hop.error_signatures):
        repetition = max(repetition, 0.85)

    blocked = 0.0
    if hop.required_user_input or response_requires_user_input(response):
        blocked = 1.0
    elif hop.interactive_terminal_wait or hop.high_risk_action_requires_approval:
        blocked = 0.85

    done_confidence = 0.0
    if response_looks_terminal(response):
        done_confidence = 0.9
    elif "remaining: none" in (response or "").lower():
        done_confidence = 0.8

    executability = 0.6
    if blocked >= 0.85:
        executability = 0.0
    elif hop.files_written:
        executability = 0.9
    elif hop.error_signatures:
        executability = 0.78
    elif novelty < 0.2:
        executability = 0.35

    progress = 0.45 * novelty + 0.35 * executability + 0.20 * (1.0 - blocked)
    progress_ema = round((0.7 * progress) + (0.3 * ep.progress_ema), 4)
    low_progress_streak = ep.low_progress_streak + 1 if progress_ema < 0.2 else 0

    confidence = max(0.0, min(1.0, 0.4 * progress_ema + 0.35 * executability + 0.25 * (1.0 - blocked) - 0.2 * repetition))

    return HeuristicScores(
        progress_ema=progress_ema,
        novelty_score=max(0.0, min(1.0, novelty)),
        repetition_score=max(0.0, min(1.0, repetition)),
        executability_score=max(0.0, min(1.0, executability)),
        confidence_proxy=confidence,
        blocked_score=max(0.0, min(1.0, blocked)),
        done_confidence=max(0.0, min(1.0, done_confidence)),
        low_progress_streak=low_progress_streak,
    )
