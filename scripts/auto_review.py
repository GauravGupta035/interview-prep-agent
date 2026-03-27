"""
auto_review.py — Standalone code reviewer for GitHub Actions.

Two modes controlled by the REVIEW_MODE env var:

  "pr"   → Reviews changed files, posts inline comments on the existing PR.
            No auto-fixes. Triggered on pull_request events.

  "push" → Reviews changed files, applies CRITICAL/MAJOR auto-fixes,
            creates a fix branch and signals the workflow to open a new PR.
            Triggered on push-to-main events.

Env vars expected (set by the workflow):
  GOOGLE_API_KEY       — Gemini API key
  REVIEW_MODE          — "pr" or "push"
  PR_NUMBER            — (pr mode) the pull request number to comment on
  PR_HEAD_SHA          — (pr mode) HEAD commit SHA of the PR branch
  GITHUB_REPOSITORY    — owner/repo

Outputs (push mode only, read by the workflow):
  .fix_branch_name  — name of the branch fixes were committed to (if any)
  .pr_body.md       — PR description with full findings table
"""

import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
FIX_BRANCH = f"code-review/auto-fixes-{date.today().strftime('%Y%m%d')}"

# Files/dirs to never read
IGNORE_PATTERNS = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "coverage",
    ".egg-info",
    "*.lock",
    "*.db",
    "*.log",
    "*.pyc",
    "*.patch",
    "venv",
}

# Only review these extensions
REVIEWABLE_EXTENSIONS = {".py"}

# Files too large to review (bytes)
MAX_FILE_SIZE = 100_000

# Rate-limit settings
LLM_CALL_DELAY = 4  # seconds between every LLM call
LLM_MAX_RETRIES = 3  # retries on 429 / RESOURCE_EXHAUSTED
LLM_RETRY_BASE_WAIT = 10  # base wait in seconds (doubles each retry)

# PR comment severities
PR_COMMENT_SEVERITIES = {"CRITICAL", "MAJOR", "MINOR"}

SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "MAJOR": "🟠",
    "MINOR": "🟡",
    "NITPICK": "🔵",
}

# ── LLM setup ─────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)


def call_llm(content: str) -> str | None:
    """Call the LLM with retry + exponential backoff on rate-limit errors.

    Returns the response text on success, or None after all retries fail.
    Adds a throttle delay after every successful call to avoid bursting.
    """
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = llm.invoke([HumanMessage(content=content)])
            time.sleep(LLM_CALL_DELAY)
            return response.content.strip()

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = LLM_RETRY_BASE_WAIT * (2**attempt)
                print(
                    f"    ⏳ Rate limited. Waiting {wait}s "
                    f"(attempt {attempt + 1}/{LLM_MAX_RETRIES})...",
                    flush=True,
                )
                time.sleep(wait)
            else:
                print(f"    LLM error: {e}", flush=True)
                return None

    print("    ❌ All retries exhausted.", flush=True)
    return None


# ── File collection ───────────────────────────────────────────────────────────


def should_ignore(path: Path) -> bool:
    for part in path.parts:
        for pattern in IGNORE_PATTERNS:
            if pattern.startswith("*"):
                if part.endswith(pattern[1:]):
                    return True
            elif part == pattern:
                return True
    return False


def _is_reviewable(path: Path) -> bool:
    return (
        path.is_file()
        and not should_ignore(path)
        and path.suffix in REVIEWABLE_EXTENSIONS
        and path.stat().st_size <= MAX_FILE_SIZE
    )


def _priority(p: Path) -> int:
    name = p.name
    parts = str(p)
    if name == "main.py":
        return 0
    if name == "state.py":
        return 1
    if name == "router.py":
        return 2
    if "nodes/" in parts:
        return 3
    if name == "checkpointer.py":
        return 4
    if name == "topic_refresher.py":
        return 5
    return 9


def collect_changed_files() -> list[Path]:
    """Return only files changed in this push/PR, filtered to reviewable Python."""
    diff_strategies = [
        ["diff", "--name-only", "--diff-filter=ACMR", "HEAD~1"],
        ["diff", "--name-only", "--diff-filter=ACMR", "origin/main...HEAD"],
    ]

    for diff_args in diff_strategies:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT)] + diff_args,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            changed = []
            for line in result.stdout.strip().splitlines():
                p = REPO_ROOT / line.strip()
                if p.exists() and _is_reviewable(p):
                    changed.append(p)
            if changed:
                print(f"  (reviewing {len(changed)} changed file(s) only)", flush=True)
                return sorted(changed, key=_priority)

    print("  (could not determine changed files — scanning full repo)", flush=True)
    return collect_all_files()


def collect_all_files() -> list[Path]:
    all_files = [f for f in REPO_ROOT.rglob("*") if _is_reviewable(f)]
    return sorted(all_files, key=_priority)


# ── Diff line parsing (for PR inline comments) ───────────────────────────────


def get_diff_lines() -> dict[str, set[int]]:
    """Parse git diff to find which lines (new-side) are part of the diff.

    Returns a dict mapping relative file paths to sets of line numbers.
    GitHub rejects inline review comments on lines outside the diff,
    so this is used to decide which findings can be posted inline.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--unified=0", "origin/main...HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "--unified=0", "HEAD~1"],
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        return {}

    diff_map: dict[str, set[int]] = {}
    current_file = None

    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:]
            diff_map.setdefault(current_file, set())
        elif line.startswith("@@") and current_file:
            parts = line.split("+", 1)
            if len(parts) < 2:
                continue
            new_part = parts[1].split("@@")[0].strip()

            if "," in new_part:
                start_str, count_str = new_part.split(",", 1)
                start = int(start_str)
                count = int(count_str)
            else:
                start = int(new_part)
                count = 1

            if count > 0:
                for ln in range(start, start + count):
                    diff_map[current_file].add(ln)

    return diff_map


# ── Review ────────────────────────────────────────────────────────────────────

REVIEW_PROMPT = """You are an expert Python code reviewer specialising in LangGraph agents.

Review the following file and return ONLY a valid JSON array of findings.
Each finding must have exactly these keys:
  - "severity":     "CRITICAL" | "MAJOR" | "MINOR" | "NITPICK"
  - "line":         integer line number (best estimate)
  - "issue":        one sentence describing the problem
  - "fix":          one sentence describing the correct approach
  - "auto_fixable": true if you can apply a safe, minimal fix without needing business context

Severity guide:
  CRITICAL  — will crash or silently break at runtime (unhandled exception,
               wrong dict key, typo in state field name, broken control flow)
  MAJOR     — bug-prone pattern (missing try/except on LLM JSON parse,
               state mutated in-place, connection leak, wrong moving average)
  MINOR     — code smell (magic string, function too long, missing error handling on I/O)
  NITPICK   — style only (missing docstring, line length, quote style)

LangGraph-specific things to flag as CRITICAL or MAJOR:
  - TypedDict fields using = instead of : (breaks Annotated reducers)
  - Dict key typos when reading from state (wrong key → always gets default)
  - Typos in returned state dict keys (updates silently dropped)
  - LLM response parsed as JSON without try/except
  - prompt variable built but never passed to llm.invoke()
  - interrupt_before missing on nodes that require human input

Return ONLY the JSON array. No markdown fences. No explanation.
If there are no findings, return an empty array: []

File: {filename}

```python
{code}
```"""


def review_file(path: Path) -> list[dict]:
    """Send one file to the LLM and return its findings list."""
    relative = path.relative_to(REPO_ROOT)
    code = path.read_text()

    print(f"  Reviewing {relative}...", flush=True)

    raw = call_llm(REVIEW_PROMPT.format(filename=str(relative), code=code))

    if raw is None:
        print(f"  Warning: could not get review for {relative}", flush=True)
        return []

    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        findings = json.loads(raw)
        for f in findings:
            f["file"] = str(relative)
        return findings

    except json.JSONDecodeError as e:
        print(f"  Warning: could not parse JSON for {relative}: {e}", flush=True)
        return []


# ── PR comment mode ───────────────────────────────────────────────────────────


def gh_api(endpoint: str, method: str = "POST", payload: dict | None = None) -> bool:
    """Call the GitHub API via gh CLI. Returns True on success."""
    cmd = ["gh", "api", endpoint, "--method", method]
    input_data = None
    if payload is not None:
        cmd.append("--input=-")
        input_data = json.dumps(payload)

    result = subprocess.run(
        cmd,
        input=input_data,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"    gh api error: {result.stderr[:200]}", flush=True)
        return False
    return True


def post_pr_review(all_findings: list[dict], pr_number: str, commit_sha: str):
    """Post findings as a single GitHub PR review with inline comments.

    Findings on lines within the diff get inline comments.
    All findings also appear in the review summary body.
    """
    repo = os.environ["GITHUB_REPOSITORY"]
    diff_lines = get_diff_lines()

    # Split findings: inlineable (in diff + right severity) vs body-only
    commentable = [
        f
        for f in all_findings
        if f["severity"] in PR_COMMENT_SEVERITIES
        and f["file"] in diff_lines
        and f["line"] in diff_lines[f["file"]]
    ]
    body_only = [
        f
        for f in all_findings
        if f not in commentable and f["severity"] in PR_COMMENT_SEVERITIES
    ]

    body = _build_review_body(all_findings, commentable, body_only)

    # Build inline comments
    comments = []
    for f in commentable:
        emoji = SEVERITY_EMOJI.get(f["severity"], "")
        comment_body = (
            f"{emoji} **{f['severity']}**\n\n"
            f"**Issue:** {f['issue']}\n\n"
            f"**Suggested fix:** {f['fix']}"
        )
        comments.append(
            {
                "path": f["file"],
                "line": f["line"],
                "side": "RIGHT",
                "body": comment_body,
            }
        )

    payload = {
        "commit_id": commit_sha,
        "body": body,
        "event": "COMMENT",
    }
    if comments:
        payload["comments"] = comments

    endpoint = f"repos/{repo}/pulls/{pr_number}/reviews"

    print(
        f"\nPosting review: {len(comments)} inline comment(s) + summary...",
        flush=True,
    )

    if gh_api(endpoint, payload=payload):
        print("✅ PR review posted.", flush=True)
        return

    # Inline comments may have failed (lines not actually in diff).
    # Retry without them — everything goes in the body instead.
    print("  Retrying without inline comments...", flush=True)
    all_body = [f for f in all_findings if f["severity"] in PR_COMMENT_SEVERITIES]
    payload["body"] = _build_review_body(all_findings, [], all_body)
    payload.pop("comments", None)

    if gh_api(endpoint, payload=payload):
        print("✅ PR review posted (summary only).", flush=True)
        return

    # Last resort: plain PR comment
    print("  Falling back to plain PR comment...", flush=True)
    subprocess.run(
        ["gh", "pr", "comment", pr_number, "--body", payload["body"]],
        capture_output=True,
        text=True,
    )
    print("✅ PR comment posted.", flush=True)


def _build_review_body(
    all_findings: list[dict],
    inlined: list[dict],
    body_only: list[dict],
) -> str:
    """Build the markdown body for the PR review."""
    sections = {}
    for f in all_findings:
        sections.setdefault(f["severity"], []).append(f)

    lines = [
        "## 🤖 Auto Code Review\n",
        "| Severity | Found |",
        "|---|---|",
    ]
    for sev in ["CRITICAL", "MAJOR", "MINOR", "NITPICK"]:
        items = sections.get(sev, [])
        if items:
            lines.append(f"| {SEVERITY_EMOJI[sev]} {sev} | {len(items)} |")

    if not all_findings:
        lines.append("\n✅ No issues found!")
        return "\n".join(lines)

    # Findings that couldn't be posted inline
    if body_only:
        lines.append("\n### Findings not posted inline\n")
        lines.append(
            "*These are on lines outside the diff, or the inline post failed.*\n"
        )
        lines.append("| File | Line | Severity | Issue | Fix |")
        lines.append("|---|---|---|---|---|")
        for f in body_only:
            emoji = SEVERITY_EMOJI.get(f["severity"], "")
            lines.append(
                f"| `{f['file']}` | {f['line']} "
                f"| {emoji} {f['severity']} "
                f"| {f['issue'][:80]} "
                f"| {f['fix'][:80]} |"
            )

    # NITPICKs listed in body only (never inlined)
    nitpicks = sections.get("NITPICK", [])
    if nitpicks:
        lines.append("\n### 🔵 Nitpicks (informational)\n")
        for f in nitpicks:
            lines.append(f"- `{f['file']}:{f['line']}` — {f['issue']}")

    if inlined:
        lines.append(f"\n*{len(inlined)} finding(s) posted as inline comments above.*")

    return "\n".join(lines)


# ── Auto-fix (push mode only) ────────────────────────────────────────────────

FIX_PROMPT = """You are applying a minimal, safe code fix.

File: {filename}
Issue (line {line}): {issue}
Fix to apply: {fix}

Return ONLY the complete corrected file content — no markdown fences,
no explanation, just the raw Python. Make the smallest possible change
that addresses the issue. Add a comment `# fix: <one line>` on the changed line."""


def apply_fix(path: Path, finding: dict) -> bool:
    """Ask the LLM to apply one fix and overwrite the file. Returns True on success."""
    relative = path.relative_to(REPO_ROOT)
    original = path.read_text()

    print(f"    Fixing line {finding['line']}: {finding['issue'][:60]}...", flush=True)

    fixed = call_llm(
        FIX_PROMPT.format(
            filename=str(relative),
            line=finding["line"],
            issue=finding["issue"],
            fix=finding["fix"],
            code=original,
        )
    )

    if fixed is None:
        print(f"    Could not apply fix for {relative}:{finding['line']}", flush=True)
        return False

    try:
        if fixed.startswith("```"):
            fixed = fixed.split("```")[1]
            if fixed.startswith("python"):
                fixed = fixed[6:]
            fixed = fixed.strip()

        path.write_text(fixed)
        return True

    except Exception as e:
        print(f"    Error writing fix: {e}", flush=True)
        return False


# ── PR body builder (push mode only) ─────────────────────────────────────────


def build_pr_body(all_findings: list[dict], fixed: list[dict]) -> str:
    fixed_keys = {(f["file"], f["line"]) for f in fixed}

    def was_fixed(f):
        return "✅ auto-fixed" if (f["file"], f["line"]) in fixed_keys else "⬜ manual"

    sections = {}
    for f in all_findings:
        sections.setdefault(f["severity"], []).append(f)

    lines = [
        "## Automated Code Review\n",
        "### Summary",
        "| Severity | Found | Auto-fixed |",
        "|---|---|---|",
    ]
    for sev in ["CRITICAL", "MAJOR", "MINOR", "NITPICK"]:
        emoji = SEVERITY_EMOJI[sev]
        items = sections.get(sev, [])
        n_fixed = sum(1 for f in items if (f["file"], f["line"]) in fixed_keys)
        lines.append(f"| {emoji} {sev} | {len(items)} | {n_fixed} |")

    lines.append("")

    for sev in ["CRITICAL", "MAJOR", "MINOR", "NITPICK"]:
        emoji = SEVERITY_EMOJI[sev]
        items = sections.get(sev, [])
        if not items:
            continue
        lines.append(f"\n### {emoji} {sev}\n")
        lines.append("| File | Line | Issue | Status |")
        lines.append("|---|---|---|---|")
        for f in items:
            issue_short = f["issue"][:80]
            lines.append(
                f"| `{f['file']}` | {f['line']} | {issue_short} | {was_fixed(f)} |"
            )

    return "\n".join(lines)


# ── Git helper ────────────────────────────────────────────────────────────────


def git(args: list[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT)] + args,
        capture_output=True,
        text=True,
        check=check,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def run_pr_mode():
    """PR mode: review changed files → post inline comments. No auto-fixes."""
    pr_number = os.environ["PR_NUMBER"]
    commit_sha = os.environ["PR_HEAD_SHA"]

    print(f"Mode: PR review (PR #{pr_number})\n", flush=True)

    files = collect_changed_files()
    print(f"Found {len(files)} file(s) to review", flush=True)

    if not files:
        print("No reviewable files changed. Exiting.", flush=True)
        return

    print("\nReviewing files...", flush=True)
    all_findings = []
    for f in files:
        all_findings.extend(review_file(f))

    for sev in ["CRITICAL", "MAJOR", "MINOR", "NITPICK"]:
        count = sum(1 for f in all_findings if f["severity"] == sev)
        if count:
            print(f"  {SEVERITY_EMOJI[sev]} {count} {sev}", flush=True)

    post_pr_review(all_findings, pr_number, commit_sha)


def run_push_mode():
    """Push mode: review → auto-fix → create branch → signal workflow for PR."""
    print("Mode: push to main (auto-fix)\n", flush=True)

    git(["config", "user.email", "github-actions[bot]@users.noreply.github.com"])
    git(["config", "user.name", "github-actions[bot]"])

    files = collect_changed_files()
    print(f"Found {len(files)} file(s) to review", flush=True)

    if not files:
        print("No reviewable files found. Exiting.", flush=True)
        sys.exit(0)

    print("\nReviewing files...", flush=True)
    all_findings = []
    for f in files:
        all_findings.extend(review_file(f))

    critical = [f for f in all_findings if f["severity"] == "CRITICAL"]
    major = [f for f in all_findings if f["severity"] == "MAJOR"]
    minor = [f for f in all_findings if f["severity"] == "MINOR"]
    nitpick = [f for f in all_findings if f["severity"] == "NITPICK"]

    print(
        f"\nFindings: 🔴 {len(critical)} CRITICAL  🟠 {len(major)} MAJOR  "
        f"🟡 {len(minor)} MINOR  🔵 {len(nitpick)} NITPICK",
        flush=True,
    )

    fixable = [f for f in critical + major if f.get("auto_fixable")]

    if not fixable:
        print("\nNo auto-fixable CRITICAL/MAJOR issues found. Exiting.", flush=True)
        Path(REPO_ROOT / ".pr_body.md").write_text(build_pr_body(all_findings, []))
        sys.exit(0)

    print(f"\nApplying {len(fixable)} fixes...", flush=True)

    git(["checkout", "-b", FIX_BRANCH])

    fixed = []
    for finding in fixable:
        path = REPO_ROOT / finding["file"]
        if apply_fix(path, finding):
            fixed.append(finding)

    if not fixed:
        print("No fixes were successfully applied. Exiting.", flush=True)
        sys.exit(0)

    git(["add", "-A"])
    git(
        [
            "commit",
            "-m",
            f"fix: {len(fixed)} auto-fix(es) from code review ({date.today()})\n\n"
            + "\n".join(
                f"- [{f['severity']}] {f['file']}:{f['line']} — {f['issue']}"
                for f in fixed
            ),
        ]
    )

    pr_body = build_pr_body(all_findings, fixed)
    (REPO_ROOT / ".pr_body.md").write_text(pr_body)
    (REPO_ROOT / ".fix_branch_name").write_text(FIX_BRANCH)

    print(f"\n✅ Done. Branch '{FIX_BRANCH}' ready. Workflow will open PR.", flush=True)


def main():
    print("=== Auto Code Reviewer ===", flush=True)

    mode = os.environ.get("REVIEW_MODE", "push")

    if mode == "pr":
        run_pr_mode()
    else:
        run_push_mode()


if __name__ == "__main__":
    main()
