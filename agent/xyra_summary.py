from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from agent.auxiliary_client import async_call_llm, call_llm, extract_content_or_reasoning

_SUMMARY_DIRECTIVE_RE = re.compile(r"(?<!\S)(/sum4xyra)(?:\s*)$", re.IGNORECASE)


@dataclass(frozen=True)
class SummaryDirective:
    armed: bool
    raw_directive: str | None
    sanitized_message: str


DEFAULT_BARE_SUMMARY_REQUEST = "Summarize the last assistant output and current state for Xyra."


def summary_help_entries(
    *,
    token: str = "/sum4xyra",
    enabled: bool = False,
    opt_in_required: bool = True,
) -> list[tuple[str, str]]:
    token = str(token or "/sum4xyra").strip() or "/sum4xyra"
    if not enabled:
        return []

    entries: list[tuple[str, str]] = [
        (
            token,
            "Summarize the most relevant recent state / last assistant output for Xyra.",
        )
    ]
    if opt_in_required:
        entries.append(
            (
                f"<message> {token}",
                "Answer normally, then append a Xyra-tuned summary of the final response.",
            )
        )
    else:
        entries.append(
            (
                "default final responses",
                "Append a Xyra-tuned summary automatically on successful final replies.",
            )
        )
        entries.append(
            (
                f"<message> {token}",
                "Force the opt-in token explicitly even though default summaries are already enabled.",
            )
        )
    return entries


def gateway_summary_help_lines(
    *,
    token: str = "/sum4xyra",
    enabled: bool = False,
    opt_in_required: bool = True,
) -> list[str]:
    lines: list[str] = []
    for label, description in summary_help_entries(
        token=token,
        enabled=enabled,
        opt_in_required=opt_in_required,
    ):
        if label == "default final responses":
            lines.append(f"`{label}` — {description}")
        else:
            lines.append(f"`{label}` — {description}")
    return lines


def parse_summary_directive(
    message: Any,
    token: str = "/sum4xyra",
    *,
    opt_in_required: bool = True,
) -> SummaryDirective:
    text = _normalize_message_text(message)
    token = str(token or "").strip()
    match = None
    if token:
        pattern = re.compile(rf"(?<!\S)({re.escape(token)})(?:\s*)$", re.IGNORECASE)
        match = pattern.search(text)
    if match:
        sanitized = text[:match.start()].rstrip()
        return SummaryDirective(armed=True, raw_directive=match.group(1), sanitized_message=sanitized)
    if not opt_in_required or not token:
        return SummaryDirective(armed=True, raw_directive=None, sanitized_message=text.strip())
    return SummaryDirective(armed=False, raw_directive=None, sanitized_message=text.strip())


def format_summary_block(summary: str, *, heading: str = "Xyra summary") -> str:
    body = (summary or "").strip()
    if not body:
        return ""
    return f"\n\n---\n{heading}\n{body}"


def summarize_for_xyra_sync(
    *,
    user_message: Any,
    assistant_response: str,
    conversation_history: Sequence[dict[str, Any]] | None = None,
    max_context_messages: int = 8,
    max_chars_per_message: int = 1200,
    two_pass: bool = True,
) -> str:
    packet = _build_context_packet(
        user_message=user_message,
        assistant_response=assistant_response,
        conversation_history=conversation_history,
        max_context_messages=max_context_messages,
        max_chars_per_message=max_chars_per_message,
    )
    if not packet["assistant_response"]:
        return ""
    extraction = _extract_pass_sync(packet)
    if not extraction:
        return ""
    if not two_pass:
        return extraction.strip()
    rewritten = _rewrite_pass_sync(packet, extraction)
    return (rewritten or extraction).strip()


async def summarize_for_xyra_async(
    *,
    user_message: Any,
    assistant_response: str,
    conversation_history: Sequence[dict[str, Any]] | None = None,
    max_context_messages: int = 8,
    max_chars_per_message: int = 1200,
    two_pass: bool = True,
) -> str:
    packet = _build_context_packet(
        user_message=user_message,
        assistant_response=assistant_response,
        conversation_history=conversation_history,
        max_context_messages=max_context_messages,
        max_chars_per_message=max_chars_per_message,
    )
    if not packet["assistant_response"]:
        return ""
    extraction = await _extract_pass_async(packet)
    if not extraction:
        return ""
    if not two_pass:
        return extraction.strip()
    rewritten = await _rewrite_pass_async(packet, extraction)
    return (rewritten or extraction).strip()


def _extract_pass_sync(packet: dict[str, Any]) -> str:
    response = call_llm(
        task="xyra_summary",
        messages=_extraction_messages(packet),
        temperature=0.2,
        max_tokens=700,
    )
    return extract_content_or_reasoning(response).strip()


async def _extract_pass_async(packet: dict[str, Any]) -> str:
    response = await async_call_llm(
        task="xyra_summary",
        messages=_extraction_messages(packet),
        temperature=0.2,
        max_tokens=700,
    )
    return extract_content_or_reasoning(response).strip()


def _rewrite_pass_sync(packet: dict[str, Any], extraction: str) -> str:
    response = call_llm(
        task="xyra_summary",
        messages=_rewrite_messages(packet, extraction),
        temperature=0.15,
        max_tokens=600,
    )
    return extract_content_or_reasoning(response).strip()


async def _rewrite_pass_async(packet: dict[str, Any], extraction: str) -> str:
    response = await async_call_llm(
        task="xyra_summary",
        messages=_rewrite_messages(packet, extraction),
        temperature=0.15,
        max_tokens=600,
    )
    return extract_content_or_reasoning(response).strip()


def _extraction_messages(packet: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are building an exceptionally operator-useful summary for Xyra, a human juggling many concurrent agent threads. "
                "Do not write a chronology. Extract strategic intent, actual state, shipped changes, risks, likely confusions, and the next decisive move. "
                "Prefer meaning over mechanics. Distinguish facts from inference."
            ),
        },
        {
            "role": "user",
            "content": (
                "Produce ranked extraction notes for a Xyra-tuned summary.\n\n"
                f"Latest user message:\n{packet['user_message'] or '(empty)'}\n\n"
                f"Recent conversation slice:\n{packet['history_blob']}\n\n"
                f"Assistant output to summarize:\n{packet['assistant_response']}\n\n"
                "Return compact markdown with these sections in this order:\n"
                "1. Strategic intent\n"
                "2. Actual state now\n"
                "3. Important landed changes\n"
                "4. Risks / likely confusion\n"
                "5. Next decisive move\n"
                "6. Ignore for now\n"
                "Keep each section terse and high-signal."
            ),
        },
    ]


def _rewrite_messages(packet: dict[str, Any], extraction: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Rewrite extraction notes into a highly human-readable Xyra-facing summary. "
                "Write to a busy human, not to another model. Use short sentences and concrete wording. "
                "Prioritize orientation, risk, and the next move."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Latest user message:\n{packet['user_message'] or '(empty)'}\n\n"
                f"Extraction notes:\n{extraction}\n\n"
                "Rewrite as markdown using exactly these headings in this order:\n"
                "Bottom line\n"
                "What is in motion\n"
                "What changed / what landed\n"
                "Risks / likely confusion\n"
                "Next best move\n"
                "Ignore for now\n\n"
                "Rules:\n"
                "- 1-3 sentences under Bottom line.\n"
                "- Preserve the real completion boundary.\n"
                "- Say what matters now, not the full chronology.\n"
                "- If something is inference, label it as inference.\n"
                "- Be compact but not cryptic."
            ),
        },
    ]


def _build_context_packet(
    *,
    user_message: Any,
    assistant_response: str,
    conversation_history: Sequence[dict[str, Any]] | None,
    max_context_messages: int,
    max_chars_per_message: int,
) -> dict[str, Any]:
    normalized_user = _truncate(_normalize_message_text(user_message), max_chars_per_message)
    normalized_response = _truncate(_normalize_message_text(assistant_response), max_chars_per_message * 2)
    history_blob = _history_blob(
        conversation_history or [],
        max_messages=max_context_messages,
        max_chars_per_message=max_chars_per_message,
    )
    return {
        "user_message": normalized_user,
        "assistant_response": normalized_response,
        "history_blob": history_blob or "- none",
    }


def _history_blob(
    history: Sequence[dict[str, Any]],
    *,
    max_messages: int,
    max_chars_per_message: int,
) -> str:
    kept: list[str] = []
    relevant = [item for item in history if isinstance(item, dict) and item.get("role") in {"user", "assistant"}]
    for item in relevant[-max_messages:]:
        role = str(item.get("role") or "unknown")
        text = _truncate(_normalize_message_text(item.get("content")), max_chars_per_message)
        if not text:
            continue
        kept.append(f"- {role}: {text}")
    return "\n".join(kept)


def _normalize_message_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts: list[str] = []
        for item in message:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif "content" in item and isinstance(item.get("content"), str):
                    parts.append(str(item.get("content") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part and str(part).strip()).strip()
    return str(message).strip()


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
