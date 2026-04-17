"""Lightweight self-test for dcg_guard, no pytest required."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import dcg_guard  # type: ignore


class FakeProc:
    def __init__(self, stdout="", rc=0):
        self.stdout = stdout; self.returncode = rc; self.stderr = ""


def run_case(name, patches, expected_allow, expected_ok):
    # patches: list of (module, attr, new_value)
    saved = [(m, a, getattr(m, a)) for m, a, _ in patches]
    try:
        for m, a, v in patches:
            setattr(m, a, v)
        r = dcg_guard.check_with_dcg("rm -rf $HOME")
        ok = r["allow"] == expected_allow and r["ok"] == expected_ok
        print(("PASS" if ok else "FAIL"), name, "->", r)
        return ok
    finally:
        for m, a, v in saved:
            setattr(m, a, v)


results = []

# 1) dcg missing -> soft-allow
results.append(run_case(
    "missing dcg soft-allows",
    [(dcg_guard.shutil, "which", lambda _: None)],
    True, False,
))

# 2) empty stdout -> allow
results.append(run_case(
    "empty stdout allows",
    [
        (dcg_guard.shutil, "which", lambda _: "/usr/bin/dcg"),
        (dcg_guard.subprocess, "run", lambda *a, **kw: FakeProc(stdout="")),
    ],
    True, True,
))

# 3) deny JSON parsed correctly
deny_obj = {"hookSpecificOutput": {
    "permissionDecision": "deny",
    "permissionDecisionReason": "BLOCKED by dcg\n\nmore detail",
    "ruleId": "core.filesystem:rm-rf-general",
}}
results.append(run_case(
    "deny decoded",
    [
        (dcg_guard.shutil, "which", lambda _: "/usr/bin/dcg"),
        (dcg_guard.subprocess, "run",
         lambda *a, **kw: FakeProc(stdout="banner\n" + json.dumps(deny_obj))),
    ],
    False, True,
))

# 4) timeout -> soft-allow
def raise_timeout(*a, **kw):
    raise subprocess.TimeoutExpired(cmd="dcg", timeout=1)

results.append(run_case(
    "timeout soft-allows",
    [
        (dcg_guard.shutil, "which", lambda _: "/usr/bin/dcg"),
        (dcg_guard.subprocess, "run", raise_timeout),
    ],
    True, False,
))

# 5) garbage stdout -> soft-allow
results.append(run_case(
    "garbage soft-allows",
    [
        (dcg_guard.shutil, "which", lambda _: "/usr/bin/dcg"),
        (dcg_guard.subprocess, "run", lambda *a, **kw: FakeProc(stdout="garbage")),
    ],
    True, False,
))

# 6) allow JSON (non-deny) -> allow
allow_obj = {"hookSpecificOutput": {"permissionDecision": "allow"}}
results.append(run_case(
    "explicit allow passes",
    [
        (dcg_guard.shutil, "which", lambda _: "/usr/bin/dcg"),
        (dcg_guard.subprocess, "run",
         lambda *a, **kw: FakeProc(stdout=json.dumps(allow_obj))),
    ],
    True, True,
))

print(f"\n{sum(results)}/{len(results)} tests passed")
sys.exit(0 if all(results) else 1)
