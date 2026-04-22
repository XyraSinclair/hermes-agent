from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional


@dataclass
class AutoStewardConfig:
    enabled: bool = False
    opt_in_required: bool = True
    opt_in_token: str = "/as"
    default_hops_when_armed: int = 1
    hard_cap_hops: int = 1
    decision_mode: str = "heuristic"
    low_progress_patience: int = 2
    done_threshold: float = 0.8
    continue_threshold: float = 0.45
    stop_threshold: float = 0.3
    redirect_margin: float = 0.2
    review_start_hop: int = 3
    review_band: float = 0.1
    max_reviews_per_episode: int = 2
    review_enabled: bool = False
    review_on_user_input_boundary: bool = True
    review_provider: str = "anthropic"
    review_model: str = "claude-opus-4.6"
    review_base_url: str = ""
    review_api_key: str = ""
    review_timeout: float = 45.0
    review_min_confidence: float = 0.75
    notice: bool = True
    log_episodes: bool = True


@dataclass
class ParsedDirective:
    armed: bool
    raw_directive: Optional[str]
    requested_hops: Optional[int]
    effective_hops: Optional[int]
    sanitized_message: str
    warnings: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ParsedDirective":
        if not isinstance(payload, Mapping):
            raise TypeError("ParsedDirective payload must be a mapping")
        return cls(**dict(payload))


@dataclass
class CommandEvent:
    signature: str
    exit_code: Optional[int] = None
    summary: str = ""


@dataclass
class TestEvent:
    signature: str
    passed: Optional[int] = None
    failed: Optional[int] = None
    summary: str = ""


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class HopEvidence:
    hop_index: int
    assistant_summary: str
    commands: list[CommandEvent] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    tests: list[TestEvent] = field(default_factory=list)
    error_signatures: list[str] = field(default_factory=list)
    interactive_terminal_wait: bool = False
    required_user_input: bool = False
    high_risk_action_requires_approval: bool = False
    token_usage: Optional[TokenUsage] = None
    response_preview: str = ""
    response_fingerprint: str = ""


@dataclass
class Frontier:
    kind: str
    score: float
    prompt_hint: str = ""


@dataclass
class HeuristicScores:
    progress_ema: float
    novelty_score: float
    repetition_score: float
    executability_score: float
    confidence_proxy: float
    blocked_score: float
    done_confidence: float
    low_progress_streak: int


class DecisionKind(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    REDIRECT = "redirect"
    REVIEW = "review"


@dataclass
class Decision:
    kind: DecisionKind
    reason_codes: list[str] = field(default_factory=list)
    frontier: Optional[Frontier] = None
    prompt_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeState:
    episode_id: str
    session_id: str
    directive: ParsedDirective
    armed: bool
    effective_max_hops: int
    hops_used: int = 0
    review_calls: int = 0
    low_progress_streak: int = 0
    current_frontier: Optional[str] = None
    recent_frontiers: list[str] = field(default_factory=list)
    recent_command_signatures: list[str] = field(default_factory=list)
    recent_error_signatures: list[str] = field(default_factory=list)
    cumulative_tokens: int = 0
    policy_version: str = "v1"
    progress_ema: float = 0.0
    recent_response_fingerprints: list[str] = field(default_factory=list)
    recent_response_texts: list[str] = field(default_factory=list)
