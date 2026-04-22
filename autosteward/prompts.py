from __future__ import annotations

from .state import Decision, DecisionKind, ParsedDirective

_AUTO_STEWARD_PROMPT_PREFIXES = (
    "This is an automated stewardship followthrough message.",
    "This is an automated stewardship redirect message.",
)


def is_auto_steward_prompt(text: object) -> bool:
    if not isinstance(text, str):
        return False
    normalized = " ".join(text.split())
    return any(normalized.startswith(prefix) for prefix in _AUTO_STEWARD_PROMPT_PREFIXES)


def build_auto_steward_prompt(decision: Decision | None = None) -> str:
    reason_blob = ""
    frontier_blob = ""
    avoid_blob = ""
    if decision and decision.reason_codes:
        reason_blob = " Why continuing: " + ", ".join(decision.reason_codes) + "."
    if decision and decision.frontier and decision.frontier.kind:
        frontier_blob = f" Current best frontier: {decision.frontier.kind}."
    if decision and decision.kind == DecisionKind.REDIRECT and decision.frontier:
        return (
            "This is an automated stewardship redirect message. Reassess the conversation from the perspective of the user's overall strategic intent, not just the last sentence. "
            f"Switch to this frontier instead of repeating the prior line of attack: {decision.frontier.kind}."
            f"{reason_blob} "
            "Before acting, briefly state done / remaining / blocked in <=5 lines. Then take the single highest-leverage safe next step on the redirected frontier. "
            "Do not loop on the prior tactic. Only stop when the work is genuinely complete, concretely blocked, or requires non-retrievable user input."
        )
    return (
        "This is an automated stewardship followthrough message. Reassess the conversation from the perspective of the user's overall strategic intent, not just the last sentence. "
        "Do you understand the overall intent? Does it overall seem safe to continue right now? You have a tendency to stop too early, so if the intent is clear and the next steps are obvious and safe, actually continue and do them now. "
        f"{frontier_blob}{reason_blob}{avoid_blob} "
        "Before acting, briefly state done / remaining / blocked in <=5 lines. Then continue exhaustively through the highest-leverage safe next steps available from the current context. "
        "Do not stop merely to narrate, summarize, or offer obvious next steps. Only stop when the work is genuinely complete, concretely blocked, or requires non-retrievable user input. If you stop, say why in one sentence."
    ).replace("  ", " ").strip()


def build_bare_auto_steward_notice(directive: ParsedDirective) -> str:
    hops = directive.effective_hops or directive.requested_hops or 1
    raw = directive.raw_directive or "/as"
    return (
        f"Auto-steward armed for {hops} hops, but there’s no active task in this chat to continue.\n\n"
        f"Send your actual request with {raw} appended, e.g. “audit this repo and fix the failing tests {raw}”\n\n"
        "Or just give the task now, and I’ll execute it."
    )
