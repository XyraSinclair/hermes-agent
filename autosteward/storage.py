from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


def append_episode_log(record: dict[str, Any]) -> None:
    try:
        base = get_hermes_home() / "logs" / "autosteward"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{datetime.utcnow():%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception:
        # Logging must never break the chat path.
        return
