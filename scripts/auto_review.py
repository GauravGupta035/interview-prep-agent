"""
auto_review.py — Standalone code reviewer for GitHub Actions.

Called by .github/workflows/auto-code-review.yml on every push/PR to main.
Walks the repo, sends each file to the LLM for review, applies CRITICAL and
MAJOR auto-fixes, then signals the workflow to open a PR if changes were made.

Outputs (read by the workflow):
  .fix_branch_name  — name of the branch fixes were committed to (if any)
  .pr_body.md       — PR description with full findings table
"""

import json
import os
import subprocess
import sys
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

# ── LLM setup ─────────────────────────────────────────────────────────────────

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

# ── Phase 1: Map ──────────────────────────────────────────────────────────────


def should_ignore(path: Path) -> bool:
    for part in path.parts:
        for pattern in IGNORE_PATTERNS:
            if pattern.startswith("*"):
                if part.endswith(pattern[1:]):
                    return True
            elif part == pattern:
                return True
    return False


def collect_files() -> list[Path]:
    """Walk repo and return reviewable Python files, priority-ordered."""
    all_files = []
    for f in REPO_ROOT.rglob("*"):
        if f.is_file() and not should_ignore(f) and f.suffix in REVIEWABLE_EXTENSIONS:
            if f.stat().st_size <= MAX_FILE_SIZE:
                all_files.append(f)

    # Priority order: main.py → state → router → nodes/* → rest
    def priority(p: Path) -> int:
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

    return sorted(all_files, key=priority)


# ── Phase 3: Review ───────────────────────────────────────────────────────────

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

    try:
        response = llm.invoke(
            [
                HumanMessage(
                    content=REVIEW_PROMPT.format(
                        filename=str(relative),
                        code=code,
                    )
                )
            ]
        )
        raw = response.content.strip()

        # Strip markdown fences if the LLM adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        findings = json.loads(raw)

        # Attach file path to each finding
        for f in findings:
            f["file"] = str(relative)

        return findings

    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: could not parse review for {relative}: {e}", flush=True)
        return []


# ── Phase 4: Fix ──────────────────────────────────────────────────────────────

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

    try:
        response = llm.invoke(
            [
                HumanMessage(
                    content=FIX_PROMPT.format(
                        filename=str(relative),
                        line=finding["line"],
                        issue=finding["issue"],
                        fix=finding["fix"],
                        code=original,
                    )
                )
            ]
        )
        fixed = response.content.strip()

        # Strip markdown fences if present
        if fixed.startswith("```"):
            fixed = fixed.split("```")[1]
            if fixed.startswith("python"):
                fixed = fixed[6:]
            fixed = fixed.strip()

        path.write_text(fixed)
        return True

    except Exception as e:
        print(f"    Could not apply fix: {e}", flush=True)
        return False


# ── Phase 5: PR body & git ops ────────────────────────────────────────────────


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
    for sev, emoji in [
        ("CRITICAL", "🔴"),
        ("MAJOR", "🟠"),
        ("MINOR", "🟡"),
        ("NITPICK", "🔵"),
    ]:
        items = sections.get(sev, [])
        n_fixed = sum(1 for f in items if (f["file"], f["line"]) in fixed_keys)
        lines.append(f"| {emoji} {sev} | {len(items)} | {n_fixed} |")

    lines.append("")

    for sev, emoji in [
        ("CRITICAL", "🔴"),
        ("MAJOR", "🟠"),
        ("MINOR", "🟡"),
        ("NITPICK", "🔵"),
    ]:
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


def git(args: list[str], check=True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT)] + args,
        capture_output=True,
        text=True,
        check=check,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    print("=== Auto Code Reviewer ===", flush=True)

    # Configure git identity for commits made by the action
    git(["config", "user.email", "github-actions[bot]@users.noreply.github.com"])
    git(["config", "user.name", "github-actions[bot]"])

    # Phase 1 + 2: Map and triage
    files = collect_files()
    print(f"\nPhase 1-2: Found {len(files)} files to review", flush=True)

    # Phase 3: Review all files
    print("\nPhase 3: Reviewing files...", flush=True)
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

    # Phase 4: Apply fixes for CRITICAL + MAJOR auto_fixable findings
    fixable = [f for f in critical + major if f.get("auto_fixable")]

    if not fixable:
        print("\nNo auto-fixable CRITICAL/MAJOR issues found. Exiting.", flush=True)
        # Write empty pr_body so workflow 'if' condition is false
        Path(REPO_ROOT / ".pr_body.md").write_text(build_pr_body(all_findings, []))
        sys.exit(0)

    print(f"\nPhase 4: Applying {len(fixable)} fixes...", flush=True)

    # Create the fix branch before touching any files
    git(["checkout", "-b", FIX_BRANCH])

    fixed = []
    for finding in fixable:
        path = REPO_ROOT / finding["file"]
        if apply_fix(path, finding):
            fixed.append(finding)

    if not fixed:
        print("No fixes were successfully applied. Exiting.", flush=True)
        sys.exit(0)

    # Commit all fixes in one commit
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

    # Phase 5: Write outputs for the workflow to pick up
    pr_body = build_pr_body(all_findings, fixed)
    (REPO_ROOT / ".pr_body.md").write_text(pr_body)
    (REPO_ROOT / ".fix_branch_name").write_text(FIX_BRANCH)

    print(f"\n✅ Done. Branch '{FIX_BRANCH}' ready. Workflow will open PR.", flush=True)


if __name__ == "__main__":
    main()
