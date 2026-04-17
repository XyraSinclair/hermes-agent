"""Unit tests for tools/dcg_guard.py — DCG subprocess integration.

These tests do not require the real `dcg` binary: they monkey-patch
`shutil.which` and `subprocess.run` to inject deterministic stdout.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure the hermes-agent root is importable when running pytest from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import dcg_guard  # noqa: E402


class _FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def _allow_stdout() -> str:
    # dcg convention: empty stdout means allow
    return ""


def _deny_stdout() -> str:
    obj = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "BLOCKED by dcg\n\nMore detail here.",
            "ruleId": "core.filesystem:rm-rf-general",
            "packId": "core.filesystem",
            "severity": "high",
        }
    }
    banner = "+---+\n|  BLOCKED  |\n+---+"
    return f"{banner}\n{json.dumps(obj)}\n"


def test_soft_allow_when_dcg_missing(monkeypatch):
    monkeypatch.setattr(dcg_guard.shutil, "which", lambda _: None)
    result = dcg_guard.check_with_dcg("rm -rf /")
    assert result["allow"] is True
    assert result["ok"] is False
    assert "not installed" in result["reason"].lower()


def test_allow_on_empty_stdout(monkeypatch):
    monkeypatch.setattr(dcg_guard.shutil, "which", lambda _: "/usr/local/bin/dcg")
    monkeypatch.setattr(dcg_guard.subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout=_allow_stdout()))
    result = dcg_guard.check_with_dcg("ls -la")
    assert result == {"ok": True, "allow": True, "reason": "",
                      "rule_id": "", "raw": {}}


def test_deny_parses_hookSpecificOutput(monkeypatch):
    monkeypatch.setattr(dcg_guard.shutil, "which", lambda _: "/usr/local/bin/dcg")
    monkeypatch.setattr(dcg_guard.subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout=_deny_stdout()))
    result = dcg_guard.check_with_dcg("rm -rf $HOME")
    assert result["ok"] is True
    assert result["allow"] is False
    assert result["rule_id"] == "core.filesystem:rm-rf-general"
    assert "BLOCKED" in result["reason"]


def test_timeout_is_soft_allow(monkeypatch):
    monkeypatch.setattr(dcg_guard.shutil, "which", lambda _: "/usr/local/bin/dcg")

    def _raise(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="dcg", timeout=1)

    monkeypatch.setattr(dcg_guard.subprocess, "run", _raise)
    result = dcg_guard.check_with_dcg("sleep 1000")
    assert result["allow"] is True
    assert result["ok"] is False
    assert "timeout" in result["reason"].lower()


def test_unparseable_stdout_is_soft_allow(monkeypatch):
    monkeypatch.setattr(dcg_guard.shutil, "which", lambda _: "/usr/local/bin/dcg")
    monkeypatch.setattr(dcg_guard.subprocess, "run",
                        lambda *a, **kw: _FakeProc(stdout="garbage output"))
    result = dcg_guard.check_with_dcg("some command")
    assert result["allow"] is True
    assert result["ok"] is False


def test_is_dcg_mode_enabled_respects_env(monkeypatch):
    monkeypatch.setenv("HERMES_DCG_MODE", "1")
    assert dcg_guard.is_dcg_mode_enabled() is True
    monkeypatch.setenv("HERMES_DCG_MODE", "0")
    # Fall through to config; no config mock here, so should be False-ish
    # unless the user's real config has dcg enabled. Guard against that.
    monkeypatch.setattr(dcg_guard, "_dcg_path", lambda: "/usr/local/bin/dcg")
    # don't assert specific value — just confirm it doesn't raise
    _ = dcg_guard.is_dcg_mode_enabled()
