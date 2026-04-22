from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from agent.auxiliary_client import async_call_llm, call_llm, extract_content_or_reasoning

from .state import AutoStewardConfig, Decision, DecisionKind, EpisodeState, Frontier
from .storage import append_episode_log

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ReviewVerdict:
    automate_now: bool
    kind: str
    confidence: float
    reason: str
    frontier_kind: str | None = None
    allowed_scope: str = ""
    forbidden_actions: list[str] | None = None


@dataclass
class ReviewOutcome:
    approved: bool
    replacement: Decision
    verdict: ReviewVerdict | None
    raw_text: str


@dataclass(frozen=True)
class ReviewTrigger:
    code: str
    requires_user_input_boundary: bool = False
    ambiguous_state_boundary: bool = False


def review_trigger_for(cfg: AutoStewardConfig, decision: Decision | None) -> ReviewTrigger | None:
    if not cfg.review_enabled or decision is None:
        return None
    if decision.kind == DecisionKind.REVIEW:
        return ReviewTrigger("AMBIGUOUS_STATE", ambiguous_state_boundary=True)
    if (
        decision.kind == DecisionKind.STOP
        and cfg.review_on_user_input_boundary
        and "REQUIRED_USER_INPUT" in (decision.reason_codes or [])
    ):
        return ReviewTrigger("REQUIRED_USER_INPUT", requires_user_input_boundary=True)
    return None


def _extract_json_blob(text: str) -> str:
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        raise ValueError("reviewer returned no JSON object")
    return match.group(0)


def _parse_verdict(text: str) -> ReviewVerdict:
    blob = _extract_json_blob(text)
    payload = json.loads(blob)
    kind = str(payload.get("kind") or "stop").strip().lower()
    if kind not in {"continue", "stop", "redirect"}:
        raise ValueError(f"invalid reviewer kind: {kind}")
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    forbidden = payload.get("forbidden_actions") or []
    if not isinstance(forbidden, list):
        forbidden = [str(forbidden)]
    return ReviewVerdict(
        automate_now=bool(payload.get("automate_now", False)),
        kind=kind,
        confidence=confidence,
        reason=str(payload.get("reason") or "reviewer gave no reason").strip(),
        frontier_kind=(str(payload.get("frontier_kind") or "").strip() or None),
        allowed_scope=str(payload.get("allowed_scope") or "").strip(),
        forbidden_actions=[str(item).strip() for item in forbidden if str(item).strip()],
    )


def _review_messages(*, episode: EpisodeState, response: str, decision: Decision, trigger: ReviewTrigger) -> list[dict[str, str]]:
    frontier_kind = decision.frontier.kind if decision.frontier else "unknown"
    recent = [text.strip() for text in (episode.recent_response_texts or []) if text and text.strip()]
    recent_blob = "\n\n".join(f"- {item[:500]}" for item in recent[-3:]) or "- none"
    directive = episode.directive.raw_directive or "(none)"
    directive_text = episode.directive.sanitized_message or "(empty after directive stripping)"
    boundary = "required-user-input" if trigger.requires_user_input_boundary else "ambiguous-state"
    return [
        {
            "role": "system",
            "content": (
                "You are a strict auto-steward safety reviewer. "
                "Your job is to decide whether Hermes may automate exactly one more bounded step without asking the user. "
                "Fail closed. Do not authorize destructive actions, permission-sensitive actions, new goals, account changes, purchases, external communications, or anything requiring human preference selection. "
                "Return exactly one JSON object and no prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "Decide whether one additional bounded automated followthrough step is safe.\n\n"
                f"Boundary type: {boundary}\n"
                f"Directive: {directive}\n"
                f"Directive text after stripping: {directive_text}\n"
                f"Current frontier: {frontier_kind}\n"
                f"Policy reason codes: {', '.join(decision.reason_codes or []) or 'none'}\n"
                f"Assistant response that triggered the boundary:\n{response[:4000]}\n\n"
                f"Recent assistant responses in this episode:\n{recent_blob}\n\n"
                "Return JSON with this exact schema:\n"
                "{\n"
                '  "automate_now": true|false,\n'
                '  "kind": "continue"|"stop"|"redirect",\n'
                '  "confidence": 0.0_to_1.0,\n'
                '  "reason": "one sentence",\n'
                '  "frontier_kind": "optional frontier name or null",\n'
                '  "allowed_scope": "very short bounded scope",\n'
                '  "forbidden_actions": ["short strings"]\n'
                "}\n\n"
                "Rules:\n"
                "- If there is any real uncertainty about consent, choose automate_now=false.\n"
                "- If the assistant was merely being polite and the next step is still obvious and bounded, you may choose automate_now=true.\n"
                "- Keep kind=continue unless a redirect is clearly safer.\n"
                "- Never authorize more than one bounded next-step tranche."
            ),
        },
    ]


def _build_review_metadata(
    *,
    prior: Decision,
    trigger: ReviewTrigger,
    cfg: AutoStewardConfig,
    decision_source: str,
    verdict: ReviewVerdict | None = None,
    error: str = "",
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "decision_source": decision_source,
        "decision_origin": "smart_review",
        "review_trigger": trigger.code,
        "review_prior_kind": prior.kind.value,
        "review_prior_reason_codes": list(prior.reason_codes or []),
        "review_provider": cfg.review_provider or "",
        "review_model": cfg.review_model or "",
    }
    if verdict is not None:
        metadata.update(
            {
                "review_confidence": verdict.confidence,
                "review_reason": verdict.reason,
                "review_allowed_scope": verdict.allowed_scope,
                "review_forbidden_actions": verdict.forbidden_actions or [],
                "review_verdict_kind": verdict.kind,
                "review_verdict_frontier_kind": verdict.frontier_kind,
            }
        )
    if error:
        metadata["review_error"] = error[:300]
    return metadata


def _append_review_log(
    *,
    episode: EpisodeState | None,
    cfg: AutoStewardConfig,
    trigger: ReviewTrigger,
    prior: Decision,
    replacement: Decision,
    verdict: ReviewVerdict | None,
) -> None:
    if episode is None or not cfg.log_episodes:
        return
    append_episode_log(
        {
            "event_type": "smart_review",
            "episode_id": episode.episode_id,
            "session_id": episode.session_id,
            "hop_index": episode.hops_used,
            "trigger": trigger.code,
            "prior_decision": prior.kind.value,
            "prior_reason_codes": list(prior.reason_codes or []),
            "final_decision": replacement.kind.value,
            "final_reason_codes": list(replacement.reason_codes or []),
            "decision_source": replacement.metadata.get("decision_source", "smart_review"),
            "review_confidence": replacement.metadata.get("review_confidence"),
            "review_reason": replacement.metadata.get("review_reason"),
            "review_allowed_scope": replacement.metadata.get("review_allowed_scope"),
            "review_forbidden_actions": replacement.metadata.get("review_forbidden_actions", []),
            "review_verdict_kind": verdict.kind if verdict else None,
        }
    )


def _decision_from_verdict(
    *,
    verdict: ReviewVerdict,
    prior: Decision,
    trigger: ReviewTrigger,
    cfg: AutoStewardConfig,
) -> Decision:
    blocked = (not verdict.automate_now) or verdict.confidence < cfg.review_min_confidence
    metadata = _build_review_metadata(
        prior=prior,
        trigger=trigger,
        cfg=cfg,
        decision_source="SMART_REVIEW_BLOCKED" if blocked else "SMART_REVIEW_APPROVED",
        verdict=verdict,
    )
    if blocked:
        reason_codes = ["SMART_REVIEW_BLOCKED", trigger.code]
        if verdict.confidence < cfg.review_min_confidence:
            reason_codes.append("SMART_REVIEW_LOW_CONFIDENCE")
        return Decision(
            kind=DecisionKind.STOP,
            reason_codes=reason_codes,
            frontier=prior.frontier,
            metadata=metadata,
        )
    if verdict.kind == "redirect":
        frontier_kind = verdict.frontier_kind or (prior.frontier.kind if prior.frontier else "REVIEW_REDIRECT")
        frontier_score = prior.frontier.score if prior.frontier else 0.5
        return Decision(
            kind=DecisionKind.REDIRECT,
            reason_codes=["SMART_REVIEW_APPROVED", trigger.code],
            frontier=Frontier(frontier_kind, frontier_score),
            metadata=metadata,
        )
    return Decision(
        kind=DecisionKind.CONTINUE,
        reason_codes=["SMART_REVIEW_APPROVED", trigger.code],
        frontier=prior.frontier,
        metadata=metadata,
    )


def _fail_closed(prior: Decision, trigger: ReviewTrigger, cfg: AutoStewardConfig, detail: str) -> Decision:
    return Decision(
        kind=DecisionKind.STOP,
        reason_codes=["SMART_REVIEW_FAILED_CLOSED", trigger.code],
        frontier=prior.frontier,
        metadata=_build_review_metadata(
            prior=prior,
            trigger=trigger,
            cfg=cfg,
            decision_source="SMART_REVIEW_FAILED_CLOSED",
            error=detail,
        ),
    )


def review_decision_sync(
    *,
    episode: EpisodeState | None,
    response: str,
    decision: Decision,
    cfg: AutoStewardConfig,
) -> ReviewOutcome:
    trigger = review_trigger_for(cfg, decision)
    if trigger is None:
        return ReviewOutcome(False, decision, None, "")
    if episode is None:
        replacement = _fail_closed(decision, trigger, cfg, "missing episode state")
        _append_review_log(
            episode=episode,
            cfg=cfg,
            trigger=trigger,
            prior=decision,
            replacement=replacement,
            verdict=None,
        )
        return ReviewOutcome(True, replacement, None, "missing episode state")
    messages = _review_messages(episode=episode, response=response, decision=decision, trigger=trigger)
    try:
        resp = call_llm(
            provider=cfg.review_provider or None,
            model=cfg.review_model or None,
            base_url=cfg.review_base_url or None,
            api_key=cfg.review_api_key or None,
            messages=messages,
            temperature=0.0,
            max_tokens=400,
            timeout=cfg.review_timeout,
        )
        text = extract_content_or_reasoning(resp)
        verdict = _parse_verdict(text)
        replacement = _decision_from_verdict(verdict=verdict, prior=decision, trigger=trigger, cfg=cfg)
        _append_review_log(
            episode=episode,
            cfg=cfg,
            trigger=trigger,
            prior=decision,
            replacement=replacement,
            verdict=verdict,
        )
        return ReviewOutcome(True, replacement, verdict, text)
    except Exception as exc:
        replacement = _fail_closed(decision, trigger, cfg, str(exc))
        _append_review_log(
            episode=episode,
            cfg=cfg,
            trigger=trigger,
            prior=decision,
            replacement=replacement,
            verdict=None,
        )
        return ReviewOutcome(True, replacement, None, str(exc))


async def review_decision_async(
    *,
    episode: EpisodeState | None,
    response: str,
    decision: Decision,
    cfg: AutoStewardConfig,
) -> ReviewOutcome:
    trigger = review_trigger_for(cfg, decision)
    if trigger is None:
        return ReviewOutcome(False, decision, None, "")
    if episode is None:
        replacement = _fail_closed(decision, trigger, cfg, "missing episode state")
        _append_review_log(
            episode=episode,
            cfg=cfg,
            trigger=trigger,
            prior=decision,
            replacement=replacement,
            verdict=None,
        )
        return ReviewOutcome(True, replacement, None, "missing episode state")
    messages = _review_messages(episode=episode, response=response, decision=decision, trigger=trigger)
    try:
        resp = await async_call_llm(
            provider=cfg.review_provider or None,
            model=cfg.review_model or None,
            base_url=cfg.review_base_url or None,
            api_key=cfg.review_api_key or None,
            messages=messages,
            temperature=0.0,
            max_tokens=400,
            timeout=cfg.review_timeout,
        )
        text = extract_content_or_reasoning(resp)
        verdict = _parse_verdict(text)
        replacement = _decision_from_verdict(verdict=verdict, prior=decision, trigger=trigger, cfg=cfg)
        _append_review_log(
            episode=episode,
            cfg=cfg,
            trigger=trigger,
            prior=decision,
            replacement=replacement,
            verdict=verdict,
        )
        return ReviewOutcome(True, replacement, verdict, text)
    except Exception as exc:
        replacement = _fail_closed(decision, trigger, cfg, str(exc))
        _append_review_log(
            episode=episode,
            cfg=cfg,
            trigger=trigger,
            prior=decision,
            replacement=replacement,
            verdict=None,
        )
        return ReviewOutcome(True, replacement, None, str(exc))
