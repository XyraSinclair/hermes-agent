from __future__ import annotations

import re
from typing import Any

from .state import AutoStewardConfig, ParsedDirective

_URL_RE = re.compile(r"https?://\S+")
_INLINE_CODE_RE = re.compile(r"`[^`]*`")


def _logical_token(token: str) -> str:
    return (token or "").strip()


def _mask_ignored_regions(text: str) -> str:
    masked = _URL_RE.sub(lambda m: " " * len(m.group(0)), text)
    masked = _INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), masked)
    return masked


def parse_auto_steward_directive(message: Any, cfg: AutoStewardConfig) -> ParsedDirective:
    if not isinstance(message, str):
        return ParsedDirective(
            armed=False,
            raw_directive=None,
            requested_hops=None,
            effective_hops=None,
            sanitized_message="",
            warnings=[],
        )

    text = message.rstrip()
    token = _logical_token(cfg.opt_in_token)
    directive_token = token if token.startswith("/") else "/as"

    masked = _mask_ignored_regions(text)
    bare_re = re.compile(
        rf"(?P<directive>{re.escape(directive_token)}(?P<count>[0-9]{{0,3}})?)(?P<tail>[\)\]\}}\.,!?:;]*)\s*$"
    )
    match = bare_re.fullmatch(masked)
    if match:
        directive_start = match.start("directive")
        directive_end = match.end("directive")
        raw_directive = text[directive_start:directive_end]
        count_text = match.group("count") or ""
        warnings: list[str] = []
        requested_hops = None
        if count_text:
            requested_hops = int(count_text)
            if requested_hops <= 0:
                return ParsedDirective(
                    armed=False,
                    raw_directive=None,
                    requested_hops=None,
                    effective_hops=None,
                    sanitized_message=text,
                    warnings=[f"Ignored invalid auto-steward directive: {raw_directive}"],
                )
        effective = requested_hops if requested_hops is not None else int(cfg.default_hops_when_armed)
        hard_cap = max(0, int(cfg.hard_cap_hops))
        if hard_cap and effective > hard_cap:
            warnings.append(
                f"Clamped auto-steward hops from {effective} to hard cap {hard_cap}"
            )
            effective = hard_cap
        return ParsedDirective(
            armed=True,
            raw_directive=raw_directive,
            requested_hops=requested_hops,
            effective_hops=effective,
            sanitized_message="",
            warnings=warnings,
        )

    suffix_re = re.compile(
        rf"(?P<prefix>.*?)(?P<space>\s+)(?P<directive>{re.escape(directive_token)}(?P<count>[0-9]{{0,3}})?)(?P<tail>[\)\]\}}\.,!?:;]*)\s*$"
    )
    match = suffix_re.fullmatch(masked)
    if not match:
        invalid_re = re.compile(rf".*?(?P<space>\s+)({re.escape(directive_token)}\S+)\s*$")
        invalid = invalid_re.fullmatch(masked)
        warnings = []
        if invalid:
            raw = text[invalid.start(2):].strip()
            if raw.startswith(directive_token):
                warnings.append(f"Ignored invalid auto-steward directive: {raw}")
        if not token:
            return ParsedDirective(
                armed=True,
                raw_directive=None,
                requested_hops=None,
                effective_hops=max(0, int(cfg.default_hops_when_armed)),
                sanitized_message=text,
                warnings=warnings,
            )
        if not token.startswith("/"):
            armed = token in text or not cfg.opt_in_required
            return ParsedDirective(
                armed=armed,
                raw_directive=token if token in text else None,
                requested_hops=None,
                effective_hops=max(0, int(cfg.default_hops_when_armed)) if armed else None,
                sanitized_message=text,
                warnings=warnings,
            )
        return ParsedDirective(
            armed=not cfg.opt_in_required,
            raw_directive=None,
            requested_hops=None,
            effective_hops=max(0, int(cfg.default_hops_when_armed)) if not cfg.opt_in_required else None,
            sanitized_message=text,
            warnings=warnings,
        )

    directive_start = match.start("directive")
    directive_end = match.end("directive")
    raw_directive = text[directive_start:directive_end]
    count_text = match.group("count") or ""
    warnings: list[str] = []
    requested_hops = None
    if count_text:
        requested_hops = int(count_text)
        if requested_hops <= 0:
            return ParsedDirective(
                armed=False,
                raw_directive=None,
                requested_hops=None,
                effective_hops=None,
                sanitized_message=text,
                warnings=[f"Ignored invalid auto-steward directive: {raw_directive}"],
            )
    effective = requested_hops if requested_hops is not None else int(cfg.default_hops_when_armed)
    hard_cap = max(0, int(cfg.hard_cap_hops))
    if hard_cap and effective > hard_cap:
        warnings.append(
            f"Clamped auto-steward hops from {effective} to hard cap {hard_cap}"
        )
        effective = hard_cap

    sanitized = (text[: match.start("space")] + text[match.end("tail"):]).rstrip()
    return ParsedDirective(
        armed=True,
        raw_directive=raw_directive,
        requested_hops=requested_hops,
        effective_hops=effective,
        sanitized_message=sanitized,
        warnings=warnings,
    )
