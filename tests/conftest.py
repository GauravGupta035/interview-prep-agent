"""
conftest.py — Shared fixtures for auto_review unit tests.

Import strategy:
  We add the `scripts/` directory to sys.path so the test file can do
  `import auto_review as ar` — a direct module import that does NOT
  depend on `scripts/__init__.py` existing. This is the most robust
  approach across CI runners, local pytest, and IDE test runners.

  The dummy GOOGLE_API_KEY is set BEFORE the module is imported so the
  ChatGoogleGenerativeAI constructor doesn't crash. All LLM calls are
  mocked in the tests — no real API requests are ever made.
"""

import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Add scripts/ to sys.path so `import auto_review` works directly.
# This removes any dependency on scripts/__init__.py existing.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

for _p in [str(_REPO_ROOT), str(_SCRIPTS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Env vars (must be set BEFORE auto_review is imported) ─────────────────────
os.environ.setdefault("GOOGLE_API_KEY", "fake-key-for-testing")
os.environ.setdefault("GITHUB_REPOSITORY", "testuser/testrepo")
# TavilyClient raises at construction if no key is present, so data.topic_refresher
# can't even be imported without this. All Tavily calls are mocked in the tests.
os.environ.setdefault("TAVILY_API_KEY", "fake-tavily-key-for-testing")

# ── Standard imports ──────────────────────────────────────────────────────────
import json
import textwrap
from unittest.mock import MagicMock

import pytest

# ── Fake repo tree ────────────────────────────────────────────────────────────


@pytest.fixture
def fake_repo(tmp_path):
    """Create a minimal repo tree under tmp_path with reviewable files."""
    # graph/nodes/revise.py
    nodes_dir = tmp_path / "graph" / "nodes"
    nodes_dir.mkdir(parents=True)
    (nodes_dir / "__init__.py").write_text("")
    (nodes_dir / "revise.py").write_text("def revise_node(state):\n    return state\n")

    # graph/state.py
    (tmp_path / "graph" / "__init__.py").write_text("")
    (tmp_path / "graph" / "state.py").write_text(
        "from typing import TypedDict\n\nclass AgentState(TypedDict):\n    messages: list\n"
    )

    # graph/router.py
    (tmp_path / "graph" / "router.py").write_text(
        "def route(state):\n    return 'revise'\n"
    )

    # main.py
    (tmp_path / "main.py").write_text(
        "def main():\n    print('hello')\n\nif __name__ == '__main__':\n    main()\n"
    )

    # persistence/checkpointer.py
    persist_dir = tmp_path / "persistence"
    persist_dir.mkdir()
    (persist_dir / "checkpointer.py").write_text(
        "from langgraph.checkpoint.sqlite import SqliteSaver\n"
    )

    # data/cs_concepts.json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "cs_concepts.json").write_text("[]")

    # files that should be ignored
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.pyc").write_text("binary junk")

    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "something.py").write_text("# venv file")

    # non-Python file (should be skipped)
    (tmp_path / "README.md").write_text("# Hello")

    # scripts dir (where auto_review.py lives)
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()

    return tmp_path


# ── Canned LLM responses ─────────────────────────────────────────────────────


@pytest.fixture
def sample_findings():
    """A typical LLM review response: list of findings."""
    return [
        {
            "severity": "CRITICAL",
            "line": 42,
            "file": "graph/nodes/revise.py",
            "issue": "LLM response parsed as JSON without try/except",
            "fix": "Wrap json.loads() in try/except json.JSONDecodeError",
            "auto_fixable": True,
        },
        {
            "severity": "MAJOR",
            "line": 88,
            "file": "graph/nodes/revise.py",
            "issue": "State dict key 'topic_score' should be 'topic_scores'",
            "fix": "Change the key to 'topic_scores' to match AgentState",
            "auto_fixable": True,
        },
        {
            "severity": "MINOR",
            "line": 15,
            "file": "graph/nodes/revise.py",
            "issue": "Magic string 'revision_main' used as thread ID",
            "fix": "Extract to a constant REVISION_THREAD_ID",
            "auto_fixable": False,
        },
        {
            "severity": "NITPICK",
            "line": 3,
            "file": "graph/nodes/revise.py",
            "issue": "Missing module docstring",
            "fix": "Add a one-line module docstring",
            "auto_fixable": False,
        },
    ]


@pytest.fixture
def sample_findings_json(sample_findings):
    """The findings as a raw JSON string (what call_llm would return)."""
    return json.dumps(sample_findings)


@pytest.fixture
def sample_findings_fenced(sample_findings):
    """Findings wrapped in markdown fences (common LLM quirk)."""
    return f"```json\n{json.dumps(sample_findings, indent=2)}\n```"


@pytest.fixture
def sample_fixed_code():
    """A canned 'fixed' file response from the fix prompt."""
    return textwrap.dedent("""\
        def revise_node(state):
            try:  # fix: wrap JSON parse in try/except
                result = json.loads(state["evaluation"])
            except json.JSONDecodeError:
                result = {}
            return state
    """)


# ── Git diff output ───────────────────────────────────────────────────────────


@pytest.fixture
def sample_diff_output():
    """A realistic unified diff (--unified=0) output."""
    return textwrap.dedent("""\
        diff --git a/graph/nodes/revise.py b/graph/nodes/revise.py
        --- a/graph/nodes/revise.py
        +++ b/graph/nodes/revise.py
        @@ -40,0 +41,5 @@
        +    try:
        +        result = json.loads(raw)
        +    except json.JSONDecodeError:
        +        result = {}
        +        print("parse error")
        @@ -88 +93 @@
        -    return {"topic_score": score}
        +    return {"topic_scores": score}
        diff --git a/main.py b/main.py
        --- a/main.py
        +++ b/main.py
        @@ -10,2 +10,3 @@
        +    print("new line 10")
        +    print("new line 11")
         some context
    """)


@pytest.fixture
def sample_diff_changed_files():
    """Output of git diff --name-only --diff-filter=ACMR."""
    return "graph/nodes/revise.py\nmain.py\n"
