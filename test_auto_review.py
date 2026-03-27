"""
test_auto_review.py — Unit tests for the auto code reviewer.

Every test mocks LLM calls via `call_llm` and git/subprocess calls.
No real API requests. No real git operations. Runs in < 2 seconds.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Import the module under test (env vars set in conftest.py)
import scripts.auto_review as ar


# ═══════════════════════════════════════════════════════════════════════════════
# should_ignore
# ═══════════════════════════════════════════════════════════════════════════════


class TestShouldIgnore:
    def test_ignores_git_dir(self):
        assert ar.should_ignore(Path("project/.git/config")) is True

    def test_ignores_pycache(self):
        assert ar.should_ignore(Path("graph/__pycache__/state.cpython-311.pyc")) is True

    def test_ignores_venv(self):
        assert ar.should_ignore(Path(".venv/lib/python3.11/site.py")) is True

    def test_ignores_dotdb_extension(self):
        assert ar.should_ignore(Path("data/prep_agent.db")) is True

    def test_ignores_lock_files(self):
        assert ar.should_ignore(Path("poetry.lock")) is True
        assert ar.should_ignore(Path("package-lock.json")) is False  # not *.lock pattern

    def test_allows_normal_python(self):
        assert ar.should_ignore(Path("graph/nodes/revise.py")) is False

    def test_allows_main(self):
        assert ar.should_ignore(Path("main.py")) is False

    def test_ignores_node_modules(self):
        assert ar.should_ignore(Path("node_modules/foo/index.js")) is True


# ═══════════════════════════════════════════════════════════════════════════════
# _priority (sort ordering)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriority:
    def test_main_first(self):
        assert ar._priority(Path("main.py")) == 0

    def test_state_second(self):
        assert ar._priority(Path("graph/state.py")) == 1

    def test_router_third(self):
        assert ar._priority(Path("graph/router.py")) == 2

    def test_nodes_fourth(self):
        assert ar._priority(Path("graph/nodes/revise.py")) == 3
        assert ar._priority(Path("graph/nodes/star.py")) == 3

    def test_checkpointer_fifth(self):
        assert ar._priority(Path("persistence/checkpointer.py")) == 4

    def test_refresher_sixth(self):
        assert ar._priority(Path("data/topic_refresher.py")) == 5

    def test_other_last(self):
        assert ar._priority(Path("utils/helpers.py")) == 9

    def test_sort_order_correct(self):
        files = [
            Path("utils/helpers.py"),
            Path("graph/nodes/star.py"),
            Path("main.py"),
            Path("graph/state.py"),
        ]
        result = sorted(files, key=ar._priority)
        assert result[0].name == "main.py"
        assert result[1].name == "state.py"
        assert result[2].name == "star.py"
        assert result[3].name == "helpers.py"


# ═══════════════════════════════════════════════════════════════════════════════
# collect_all_files
# ═══════════════════════════════════════════════════════════════════════════════


class TestCollectAllFiles:
    @patch.object(ar, "REPO_ROOT")
    def test_finds_python_files_and_ignores_pycache(self, mock_root, fake_repo):
        mock_root.__class__ = Path
        # Replace REPO_ROOT with the fake repo
        ar.REPO_ROOT = fake_repo

        files = ar.collect_all_files()
        names = [f.name for f in files]

        assert "main.py" in names
        assert "state.py" in names
        assert "revise.py" in names
        assert "router.py" in names
        # Ignored
        assert "cached.pyc" not in names
        assert "something.py" not in names  # in .venv
        assert "README.md" not in names  # not .py
        assert "cs_concepts.json" not in names  # not .py

    @patch.object(ar, "REPO_ROOT")
    def test_priority_ordering(self, mock_root, fake_repo):
        ar.REPO_ROOT = fake_repo
        files = ar.collect_all_files()
        names = [f.name for f in files]

        # main.py should come before state.py which comes before router.py
        assert names.index("main.py") < names.index("state.py")
        assert names.index("state.py") < names.index("router.py")


# ═══════════════════════════════════════════════════════════════════════════════
# collect_changed_files
# ═══════════════════════════════════════════════════════════════════════════════


class TestCollectChangedFiles:
    @patch("scripts.auto_review.subprocess.run")
    @patch.object(ar, "REPO_ROOT")
    def test_uses_diff_output(self, mock_root, mock_run, fake_repo):
        ar.REPO_ROOT = fake_repo

        # First diff strategy succeeds
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="graph/nodes/revise.py\nmain.py\n",
        )

        files = ar.collect_changed_files()
        names = [f.name for f in files]

        assert "main.py" in names
        assert "revise.py" in names
        assert len(files) == 2

    @patch("scripts.auto_review.subprocess.run")
    @patch("scripts.auto_review.collect_all_files")
    @patch.object(ar, "REPO_ROOT")
    def test_falls_back_to_full_scan(self, mock_root, mock_all, mock_run, fake_repo):
        ar.REPO_ROOT = fake_repo

        # Both diff strategies fail
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        mock_all.return_value = [fake_repo / "main.py"]

        files = ar.collect_changed_files()
        assert files == [fake_repo / "main.py"]
        mock_all.assert_called_once()

    @patch("scripts.auto_review.subprocess.run")
    @patch.object(ar, "REPO_ROOT")
    def test_filters_nonexistent_files(self, mock_root, mock_run, fake_repo):
        ar.REPO_ROOT = fake_repo

        # Diff mentions a file that doesn't exist
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="deleted_file.py\nmain.py\n",
        )

        files = ar.collect_changed_files()
        names = [f.name for f in files]
        assert "deleted_file.py" not in names
        assert "main.py" in names


# ═══════════════════════════════════════════════════════════════════════════════
# get_diff_lines
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetDiffLines:
    @patch("scripts.auto_review.subprocess.run")
    def test_parses_multi_line_hunk(self, mock_run, sample_diff_output):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=sample_diff_output,
        )

        result = ar.get_diff_lines()

        # revise.py: +41,5 → lines 41-45 and single-line change at 93
        assert "graph/nodes/revise.py" in result
        revise_lines = result["graph/nodes/revise.py"]
        assert {41, 42, 43, 44, 45} <= revise_lines
        assert 93 in revise_lines

    @patch("scripts.auto_review.subprocess.run")
    def test_parses_multiple_files(self, mock_run, sample_diff_output):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=sample_diff_output,
        )

        result = ar.get_diff_lines()
        assert "main.py" in result
        assert "graph/nodes/revise.py" in result

    @patch("scripts.auto_review.subprocess.run")
    def test_returns_empty_on_diff_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = ar.get_diff_lines()
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════════
# call_llm (retry logic)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCallLlm:
    @patch("scripts.auto_review.time.sleep")  # don't actually wait
    @patch.object(ar, "llm")
    def test_returns_content_on_success(self, mock_llm, mock_sleep):
        mock_llm.invoke.return_value = MagicMock(content="  hello world  ")

        result = ar.call_llm("test prompt")

        assert result == "hello world"
        mock_llm.invoke.assert_called_once()
        mock_sleep.assert_called_once_with(ar.LLM_CALL_DELAY)

    @patch("scripts.auto_review.time.sleep")
    @patch.object(ar, "llm")
    def test_retries_on_429(self, mock_llm, mock_sleep):
        mock_llm.invoke.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            MagicMock(content="success"),
        ]

        result = ar.call_llm("test")

        assert result == "success"
        assert mock_llm.invoke.call_count == 2

    @patch("scripts.auto_review.time.sleep")
    @patch.object(ar, "llm")
    def test_returns_none_after_all_retries(self, mock_llm, mock_sleep):
        mock_llm.invoke.side_effect = Exception("429 RESOURCE_EXHAUSTED")

        result = ar.call_llm("test")

        assert result is None
        assert mock_llm.invoke.call_count == ar.LLM_MAX_RETRIES

    @patch("scripts.auto_review.time.sleep")
    @patch.object(ar, "llm")
    def test_returns_none_on_non_rate_limit_error(self, mock_llm, mock_sleep):
        mock_llm.invoke.side_effect = Exception("Invalid API key")

        result = ar.call_llm("test")

        assert result is None
        # Non-rate-limit errors do NOT retry
        mock_llm.invoke.assert_called_once()

    @patch("scripts.auto_review.time.sleep")
    @patch.object(ar, "llm")
    def test_exponential_backoff_waits(self, mock_llm, mock_sleep):
        mock_llm.invoke.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            Exception("429 RESOURCE_EXHAUSTED"),
            MagicMock(content="ok"),
        ]

        ar.call_llm("test")

        # Check the retry wait times: base * 2^attempt
        wait_calls = [c for c in mock_sleep.call_args_list if c != call(ar.LLM_CALL_DELAY)]
        assert wait_calls[0] == call(ar.LLM_RETRY_BASE_WAIT * 1)   # 10s
        assert wait_calls[1] == call(ar.LLM_RETRY_BASE_WAIT * 2)   # 20s


# ═══════════════════════════════════════════════════════════════════════════════
# review_file
# ═══════════════════════════════════════════════════════════════════════════════


class TestReviewFile:
    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_parses_clean_json(self, mock_root, mock_llm, fake_repo, sample_findings_json):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = sample_findings_json

        findings = ar.review_file(fake_repo / "main.py")

        assert len(findings) == 4
        assert findings[0]["severity"] == "CRITICAL"
        assert findings[0]["file"] == "main.py"

    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_strips_markdown_fences(self, mock_root, mock_llm, fake_repo, sample_findings_fenced):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = sample_findings_fenced

        findings = ar.review_file(fake_repo / "main.py")

        assert len(findings) == 4
        assert findings[0]["severity"] == "CRITICAL"

    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_returns_empty_on_llm_failure(self, mock_root, mock_llm, fake_repo):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = None

        findings = ar.review_file(fake_repo / "main.py")
        assert findings == []

    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_returns_empty_on_invalid_json(self, mock_root, mock_llm, fake_repo):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = "not valid json at all {"

        findings = ar.review_file(fake_repo / "main.py")
        assert findings == []

    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_handles_empty_array(self, mock_root, mock_llm, fake_repo):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = "[]"

        findings = ar.review_file(fake_repo / "main.py")
        assert findings == []


# ═══════════════════════════════════════════════════════════════════════════════
# apply_fix
# ═══════════════════════════════════════════════════════════════════════════════


class TestApplyFix:
    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_writes_fixed_content(self, mock_root, mock_llm, fake_repo, sample_fixed_code):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = sample_fixed_code
        target = fake_repo / "graph" / "nodes" / "revise.py"

        finding = {
            "file": "graph/nodes/revise.py",
            "line": 42,
            "issue": "Missing try/except",
            "fix": "Wrap in try/except",
        }

        result = ar.apply_fix(target, finding)

        assert result is True
        assert "try:" in target.read_text()

    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_strips_python_fences_from_fix(self, mock_root, mock_llm, fake_repo):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = "```python\nprint('fixed')\n```"
        target = fake_repo / "main.py"

        finding = {"file": "main.py", "line": 1, "issue": "x", "fix": "y"}
        result = ar.apply_fix(target, finding)

        assert result is True
        content = target.read_text()
        assert "```" not in content
        assert "print('fixed')" in content

    @patch("scripts.auto_review.call_llm")
    @patch.object(ar, "REPO_ROOT")
    def test_returns_false_on_llm_failure(self, mock_root, mock_llm, fake_repo):
        ar.REPO_ROOT = fake_repo
        mock_llm.return_value = None
        target = fake_repo / "main.py"
        original = target.read_text()

        finding = {"file": "main.py", "line": 1, "issue": "x", "fix": "y"}
        result = ar.apply_fix(target, finding)

        assert result is False
        assert target.read_text() == original  # file untouched


# ═══════════════════════════════════════════════════════════════════════════════
# build_pr_body (push mode)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildPrBody:
    def test_summary_table_counts(self, sample_findings):
        body = ar.build_pr_body(sample_findings, [])

        assert "🔴 CRITICAL | 1 | 0" in body
        assert "🟠 MAJOR | 1 | 0" in body
        assert "🟡 MINOR | 1 | 0" in body
        assert "🔵 NITPICK | 1 | 0" in body

    def test_marks_fixed_items(self, sample_findings):
        fixed = [sample_findings[0]]  # CRITICAL is fixed
        body = ar.build_pr_body(sample_findings, fixed)

        assert "🔴 CRITICAL | 1 | 1" in body
        assert "✅ auto-fixed" in body
        assert "⬜ manual" in body

    def test_includes_file_and_line(self, sample_findings):
        body = ar.build_pr_body(sample_findings, [])
        assert "`graph/nodes/revise.py`" in body
        assert "42" in body
        assert "88" in body


# ═══════════════════════════════════════════════════════════════════════════════
# _build_review_body (PR mode)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildReviewBody:
    def test_empty_findings(self):
        body = ar._build_review_body([], [], [])
        assert "No issues found" in body

    def test_shows_severity_counts(self, sample_findings):
        for f in sample_findings:
            f["file"] = "test.py"

        body = ar._build_review_body(sample_findings, [], sample_findings[:3])

        assert "CRITICAL | 1" in body
        assert "MAJOR | 1" in body

    def test_inlined_count_shown(self, sample_findings):
        for f in sample_findings:
            f["file"] = "test.py"

        inlined = sample_findings[:2]
        body = ar._build_review_body(sample_findings, inlined, [])

        assert "2 finding(s) posted as inline comments" in body

    def test_nitpicks_in_body(self, sample_findings):
        for f in sample_findings:
            f["file"] = "test.py"

        body = ar._build_review_body(sample_findings, [], [])
        assert "Nitpicks" in body
        assert "Missing module docstring" in body


# ═══════════════════════════════════════════════════════════════════════════════
# gh_api
# ═══════════════════════════════════════════════════════════════════════════════


class TestGhApi:
    @patch("scripts.auto_review.subprocess.run")
    def test_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        assert ar.gh_api("repos/x/pulls/1/reviews", payload={"body": "hi"}) is True

    @patch("scripts.auto_review.subprocess.run")
    def test_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="422 Unprocessable")
        assert ar.gh_api("repos/x/pulls/1/reviews", payload={"body": "hi"}) is False

    @patch("scripts.auto_review.subprocess.run")
    def test_passes_json_via_stdin(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        payload = {"body": "test", "event": "COMMENT"}

        ar.gh_api("endpoint", payload=payload)

        called_kwargs = mock_run.call_args
        assert called_kwargs.kwargs.get("input") == json.dumps(payload)


# ═══════════════════════════════════════════════════════════════════════════════
# post_pr_review (integration of PR-mode pieces)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPostPrReview:
    @patch("scripts.auto_review.gh_api")
    @patch("scripts.auto_review.get_diff_lines")
    def test_posts_inline_comments_for_in_diff_findings(
        self, mock_diff, mock_gh, sample_findings
    ):
        for f in sample_findings:
            f["file"] = "graph/nodes/revise.py"

        # Line 42 is in the diff, line 88 is not
        mock_diff.return_value = {"graph/nodes/revise.py": {42, 43, 44}}
        mock_gh.return_value = True

        ar.post_pr_review(sample_findings, "123", "abc123")

        mock_gh.assert_called_once()
        payload = mock_gh.call_args[1]["payload"]
        comments = payload.get("comments", [])

        # Only the CRITICAL at line 42 should be inline (MINOR at 15 is in
        # PR_COMMENT_SEVERITIES but not in diff; MAJOR at 88 not in diff)
        assert len(comments) == 1
        assert comments[0]["line"] == 42
        assert "CRITICAL" in comments[0]["body"]

    @patch("scripts.auto_review.gh_api")
    @patch("scripts.auto_review.get_diff_lines")
    def test_falls_back_on_api_failure(self, mock_diff, mock_gh, sample_findings):
        for f in sample_findings:
            f["file"] = "test.py"

        mock_diff.return_value = {"test.py": {42}}
        # First call fails, second (without inline comments) succeeds
        mock_gh.side_effect = [False, True]

        ar.post_pr_review(sample_findings, "99", "sha456")

        assert mock_gh.call_count == 2
        # Second call should NOT have comments key
        second_payload = mock_gh.call_args_list[1][1]["payload"]
        assert "comments" not in second_payload


# ═══════════════════════════════════════════════════════════════════════════════
# main() mode routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestMainRouting:
    @patch("scripts.auto_review.run_pr_mode")
    @patch("scripts.auto_review.run_push_mode")
    def test_routes_to_pr_mode(self, mock_push, mock_pr):
        with patch.dict("os.environ", {"REVIEW_MODE": "pr"}):
            ar.main()
        mock_pr.assert_called_once()
        mock_push.assert_not_called()

    @patch("scripts.auto_review.run_pr_mode")
    @patch("scripts.auto_review.run_push_mode")
    def test_routes_to_push_mode(self, mock_push, mock_pr):
        with patch.dict("os.environ", {"REVIEW_MODE": "push"}):
            ar.main()
        mock_push.assert_called_once()
        mock_pr.assert_not_called()

    @patch("scripts.auto_review.run_pr_mode")
    @patch("scripts.auto_review.run_push_mode")
    def test_defaults_to_push(self, mock_push, mock_pr):
        with patch.dict("os.environ", {}, clear=False):
            # Remove REVIEW_MODE if present
            import os
            os.environ.pop("REVIEW_MODE", None)
            ar.main()
        mock_push.assert_called_once()
