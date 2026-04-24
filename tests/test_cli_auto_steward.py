from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cli as cli_mod
from cli import HermesCLI


class TestCLIAutoSteward:
    def _make_cli(self, enabled=True, max_hops=1, depth=0, opt_in_token="", armed=True):
        cli = HermesCLI.__new__(HermesCLI)
        cli._auto_steward_enabled = enabled
        cli._auto_steward_max_hops = max_hops
        cli._auto_steward_depth = depth
        cli._auto_steward_opt_in_token = opt_in_token
        cli._auto_steward_armed = armed
        return cli

    def test_response_requires_user_input_for_want_me_to(self):
        assert HermesCLI._response_requires_user_input(
            "Feature is done. Want me to switch the default token to something visible?"
        )

    def test_response_requires_user_input_for_confirmation_prompt(self):
        assert HermesCLI._response_requires_user_input(
            "I can keep going, but would you like me to deploy this now?"
        )

    def test_response_requires_user_input_for_missing_credentials(self):
        assert HermesCLI._response_requires_user_input(
            "I cannot proceed without your approval and the missing credential."
        )

    def test_response_requires_user_input_when_only_arming_without_task(self):
        assert HermesCLI._response_requires_user_input(
            "Armed: auto-steward max 5 hops for your next prompted task. Send the task when ready."
        )

    def test_response_without_question_does_not_require_user_input(self):
        assert not HermesCLI._response_requires_user_input(
            "Done: patched the bug, ran the focused test, and updated the docs."
        )

    def test_should_auto_steward_when_enabled_and_not_blocked(self):
        cli = self._make_cli(enabled=True, max_hops=2, depth=0)
        assert cli._should_auto_steward(
            "Done: inspected the repo. Remaining: apply the obvious safe patch and verify it.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_should_not_auto_steward_when_depth_budget_exhausted(self):
        cli = self._make_cli(enabled=True, max_hops=1, depth=1)
        assert not cli._should_auto_steward(
            "Done: inspected the repo. Remaining: apply the obvious safe patch.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_should_not_auto_steward_for_failed_result(self):
        cli = self._make_cli(enabled=True, max_hops=2, depth=0)
        assert not cli._should_auto_steward(
            "Error: provider timed out.",
            {"failed": True, "partial": False, "interrupted": False},
        )

    def test_response_looks_terminal_for_bare_stopping_colon(self):
        assert HermesCLI._response_looks_terminal(
            "Fix landed. Stopping: the feature is complete and remaining moves are preference calls."
        )

    def test_response_looks_terminal_for_completion_language(self):
        assert HermesCLI._response_looks_terminal(
            "Done: prior request already completed exactly as asked. Remaining: none. Stopping because the work is complete."
        )

    def test_should_not_auto_steward_for_terminal_response(self):
        cli = self._make_cli(enabled=True, max_hops=3, depth=0)
        assert not cli._should_auto_steward(
            "Done: no active task beyond the prior request, which was completed. Remaining: none. Stopping because there is no further concrete task to execute.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_build_auto_steward_prompt_mentions_followthrough(self):
        cli = self._make_cli(enabled=True)
        prompt = cli._build_auto_steward_prompt()
        assert "stewardship followthrough" in prompt.lower()
        assert "do not stop" in prompt.lower()

    def test_message_is_auto_steward_prompt_ignores_whitespace_differences(self):
        cli = self._make_cli(enabled=True)
        prompt = cli._build_auto_steward_prompt().replace(" ", "\n")
        assert cli._message_is_auto_steward_prompt(prompt)

    def test_should_drop_unarmed_leaked_auto_steward_prompt(self):
        cli = self._make_cli(enabled=True, opt_in_token="\t", armed=False)
        assert cli._should_drop_unarmed_auto_steward_message(
            cli._build_auto_steward_prompt()
        )

    def test_should_not_drop_armed_auto_steward_prompt(self):
        cli = self._make_cli(enabled=True, opt_in_token="\t", armed=True)
        assert not cli._should_drop_unarmed_auto_steward_message(
            cli._build_auto_steward_prompt()
        )

    def test_should_not_drop_normal_message_without_token(self):
        cli = self._make_cli(enabled=True, opt_in_token="\t", armed=False)
        assert not cli._should_drop_unarmed_auto_steward_message("test")

    def test_should_not_auto_steward_when_token_required_but_missing(self):
        cli = self._make_cli(
            enabled=True, max_hops=2, depth=0, opt_in_token="\t", armed=False
        )
        assert not cli._should_auto_steward(
            "Done: did the thing. Remaining: more obvious safe steps available.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_should_auto_steward_when_token_required_and_armed(self):
        cli = self._make_cli(
            enabled=True, max_hops=2, depth=0, opt_in_token="\t", armed=True
        )
        assert cli._should_auto_steward(
            "Done: did the thing. Remaining: more obvious safe steps available.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_message_contains_opt_in_token_string(self):
        cli = self._make_cli(opt_in_token="\t")
        assert cli._message_contains_opt_in_token("hi\tthere")
        assert not cli._message_contains_opt_in_token("hi there")

    def test_message_contains_visible_as_directive_suffix_and_bare_forms(self):
        cli = self._make_cli(opt_in_token="/as")
        assert cli._message_contains_opt_in_token("fix it /as")
        assert cli._message_contains_opt_in_token("fix it /as5")
        assert cli._message_contains_opt_in_token("/as")
        assert cli._message_contains_opt_in_token("/as5")
        assert not cli._message_contains_opt_in_token("document /as literally in prose")

    def test_coerce_auto_steward_message_fills_bare_directive_gap(self):
        cli = self._make_cli(opt_in_token="/as")
        parsed = cli._parse_auto_steward_directive("/as")
        assert cli._coerce_auto_steward_message(parsed).startswith("Continue from the existing context")

    def test_message_contains_opt_in_token_multimodal_list(self):
        cli = self._make_cli(opt_in_token="\t")
        assert cli._message_contains_opt_in_token([
            {"type": "image", "image_url": "x"},
            {"type": "text", "text": "please continue\tand keep going"},
        ])
        assert not cli._message_contains_opt_in_token([
            {"type": "text", "text": "please continue and keep going"},
        ])

    def test_empty_opt_in_token_disables_gating(self):
        cli = self._make_cli(opt_in_token="", armed=False)
        # Empty token => gating disabled => function returns True regardless
        assert cli._message_contains_opt_in_token("anything")
        # And _should_auto_steward should not require armed=True either
        assert cli._should_auto_steward(
            "Done: did the thing. Remaining: more obvious safe steps available.",
            {"failed": False, "partial": False, "interrupted": False},
        )

    def test_response_looks_terminal_for_should_stop_here_because(self):
        assert HermesCLI._response_looks_terminal(
            "Done: the only explicit task in this conversation was completed exactly. Remaining: no substantive project/task was provided. Blocked: there is no safe next action. I should stop here because continuing would invent work."
        )

    def test_chat_consumes_pending_bare_directive_before_reparse(self):
        cli = self._make_cli(enabled=True, max_hops=8, depth=0, opt_in_token="/as", armed=False)
        cli._auto_steward_default_hops = 3
        cli._auto_steward_hard_cap_hops = 8
        cli._auto_steward_last_directive = None
        cli._auto_steward_effective_hops = 3
        cli._auto_steward_episode = None
        cli._auto_steward_pending_directive = cli._parse_auto_steward_directive("/as5")
        cli._secret_capture_callback = None
        cli._ensure_runtime_credentials = lambda: False
        cli.conversation_history = []
        cli.session_id = "cli-test"
        cli.console = MagicMock()

        result = cli.chat("Continue from the existing context and execute the highest-leverage safe next steps now.")

        assert "Auto-steward armed for 5 hops" in result
        assert cli._auto_steward_pending_directive is None
        assert cli._auto_steward_armed is True
        assert cli._auto_steward_effective_hops == 5
        assert cli._auto_steward_last_directive.raw_directive == "/as5"

    def test_chat_handles_bare_directive_locally_without_model_roundtrip(self):
        cli = self._make_cli(enabled=True, max_hops=8, depth=0, opt_in_token="/as", armed=False)
        cli._auto_steward_default_hops = 3
        cli._auto_steward_hard_cap_hops = 8
        cli._auto_steward_last_directive = None
        cli._auto_steward_effective_hops = 3
        cli._auto_steward_episode = None
        cli._auto_steward_pending_directive = cli._parse_auto_steward_directive("/as5")
        cli._secret_capture_callback = None
        cli._ensure_runtime_credentials = lambda: (_ for _ in ()).throw(AssertionError("model path should not run"))
        cli.conversation_history = []
        cli.session_id = "cli-test"
        cli.console = MagicMock()

        result = cli.chat("Continue from the existing context and execute the highest-leverage safe next steps now.")

        assert isinstance(result, str)
        assert "Auto-steward armed for 5 hops" in result
        assert "actual request with /as5 appended" in result


def test_main_quiet_single_query_bare_directive_uses_chat_local_path(monkeypatch, capsys):
    created = {}

    class FakeCLI:
        def __init__(self, *args, **kwargs):
            self.tool_progress_mode = "all"
            self.session_id = "sess-123"
            self.chat_calls = []
            self.console = SimpleNamespace(print=lambda *a, **k: None)
            created["cli"] = self

        def chat(self, query):
            self.chat_calls.append(query)
            return "Auto-steward armed for 5 hops"

        def _parse_auto_steward_directive(self, query):
            return SimpleNamespace(armed=True, sanitized_message="")

        def _ensure_runtime_credentials(self):
            raise AssertionError("quiet single-query bare /asN should not pre-init credentials")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)

    with pytest.raises(SystemExit) as excinfo:
        cli_mod.main(query="/as5", quiet=True)

    assert excinfo.value.code == 0
    assert created["cli"].chat_calls == ["/as5"]
    output = capsys.readouterr().out
    assert "Auto-steward armed for 5 hops" in output
    assert "session_id: sess-123" in output



def test_main_single_query_bare_directive_prints_local_notice(monkeypatch, capsys):
    created = {}
    console_lines = []

    class FakeCLI:
        def __init__(self, *args, **kwargs):
            self.session_id = "sess-456"
            self.chat_calls = []
            self.console = SimpleNamespace(print=lambda *a, **k: console_lines.append(" ".join(str(x) for x in a)))
            created["cli"] = self

        def show_banner(self):
            console_lines.append("banner")

        def _print_exit_summary(self):
            console_lines.append("summary")

        def chat(self, query):
            self.chat_calls.append(query)
            return "Auto-steward armed for 5 hops"

        def _parse_auto_steward_directive(self, query):
            return SimpleNamespace(armed=True, sanitized_message="")

        def _ensure_runtime_credentials(self):
            raise AssertionError("bare /asN should not pre-init credentials")

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)

    cli_mod.main(query="/as5", quiet=False)

    assert created["cli"].chat_calls == ["/as5"]
    assert console_lines == []
    assert "Auto-steward armed for 5 hops" in capsys.readouterr().out
