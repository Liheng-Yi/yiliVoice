#!/usr/bin/env python3
"""Run the Codex CLI headlessly against a self-hosted LLM to review a PR diff,
then post the result as a formal GitHub review with inline comments.

Driven by the .github/workflows/codex-review.yml workflow. Expects these env vars:
  GH_TOKEN            - token with pull-requests:write (the workflow's GITHUB_TOKEN)
  FLEXAI_API_KEY      - API key for the Codex model provider
  GITHUB_REPOSITORY   - "owner/repo"
  PR_NUMBER           - pull request number
Optional overrides:
  CODEX_MODEL         - model id (default: Qwen3-Coder-30B-A3B-Instruct-FP8)
  CODEX_BASE_URL      - provider base url (default: https://tokens.flexsystems.ai/v1)
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

MODEL = os.environ.get("CODEX_MODEL", "Qwen3-Coder-30B-A3B-Instruct-FP8")
BASE_URL = os.environ.get("CODEX_BASE_URL", "https://tokens.flexsystems.ai/v1")
REPO = os.environ["GITHUB_REPOSITORY"]
PR = os.environ["PR_NUMBER"]

# Keep the prompt within the model's context window; huge diffs are truncated.
MAX_DIFF_CHARS = 80_000
CODEX_TIMEOUT_S = 480


def run(cmd, **kw):
    """Run a command, raising with captured output on failure."""
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def get_diff() -> str:
    out = run(["gh", "pr", "diff", str(PR), "--repo", REPO]).stdout
    return out


def valid_positions(diff: str) -> dict[str, set[int]]:
    """Map file path -> set of new-file line numbers that are commentable (RIGHT side).

    GitHub's review API only accepts inline comments anchored to lines that appear
    in the diff hunks. We count added ('+') and context (' ') lines by their new-file
    line number, which is what side="RIGHT" comments reference.
    """
    positions: dict[str, set[int]] = defaultdict(set)
    path: str | None = None
    new_line: int | None = None
    in_hunk = False

    for line in diff.splitlines():
        if line.startswith("diff --git"):
            path, new_line, in_hunk = None, None, False
            continue
        if not in_hunk and line.startswith("+++ "):
            p = line[4:].strip()
            if p == "/dev/null":
                path = None
            elif p.startswith("b/"):
                path = p[2:]
            else:
                path = p
            continue
        if line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            if m:
                new_line = int(m.group(1))
                in_hunk = True
            continue
        if in_hunk and path is not None and new_line is not None:
            c = line[:1]
            if c == "+":
                positions[path].add(new_line)
                new_line += 1
            elif c == "-":
                pass
            elif c == "\\":  # "\ No newline at end of file"
                pass
            else:  # context line
                positions[path].add(new_line)
                new_line += 1
    return positions


PROMPT_TEMPLATE = """You are an expert code reviewer. Review the following unified diff \
from a GitHub pull request. Focus only on the changed lines. Look for bugs, security \
issues, race conditions, incorrect logic, and clear quality problems. Do not nitpick \
style. Do not modify any files.

Respond with ONLY a single JSON object inside a ```json fenced code block, and nothing \
else. Use exactly this schema:

{{
  "summary": "a short markdown overview of the PR and your overall assessment",
  "comments": [
    {{"path": "path/relative/to/repo/root", "line": <new-file line number>, "body": "markdown comment"}}
  ]
}}

Rules:
- Only reference lines that appear as added (+) or context lines in the diff below.
- `line` is the line number in the NEW version of the file.
- If you find no issues, return an empty "comments" array but still write a "summary".

DIFF:
{diff}
"""


def run_codex(prompt: str) -> str:
    cmd = [
        "codex", "exec", "--sandbox", "read-only",
        "-c", f"model={MODEL}",
        "-c", "model_provider=flexai",
        "-c", "model_providers.flexai.name=flexai-staging",
        "-c", f"model_providers.flexai.base_url={BASE_URL}",
        "-c", "model_providers.flexai.env_key=FLEXAI_API_KEY",
        "-c", "model_providers.flexai.wire_api=responses",
        # Codex has no built-in metadata for this model; declare the real window so
        # token budgeting is correct on large diffs (the "metadata not found" warning
        # on stderr is cosmetic and can be ignored).
        "-c", "model_context_window=131072",
        "-c", "model_max_output_tokens=32768",
    ]
    proc = subprocess.run(
        cmd, input=prompt, text=True, capture_output=True, timeout=CODEX_TIMEOUT_S
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"codex exec failed (exit {proc.returncode})")
    return proc.stdout


def extract_json(text: str) -> dict:
    # Prefer the last ```json ... ``` fenced block.
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = list(blocks)
    # Fallback: the widest {...} span in the output.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    raise SystemExit("Could not parse JSON from Codex output:\n" + text[:2000])


def submit_review(body: str, comments: list[dict]) -> None:
    payload = {"event": "COMMENT", "body": body}
    if comments:
        payload["comments"] = comments
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    api = ["gh", "api", "--method", "POST",
           f"repos/{REPO}/pulls/{PR}/reviews", "--input", path]
    try:
        run(api)
    except subprocess.CalledProcessError as e:
        # Most likely an inline comment anchored to a line GitHub rejects. Retry
        # with the body only so the review still lands.
        sys.stderr.write(f"Review with inline comments failed:\n{e.stderr}\n")
        if comments:
            sys.stderr.write("Retrying with summary only...\n")
            submit_review(
                body + "\n\n> ⚠️ Inline comments could not be anchored to the diff; "
                "see summary above.",
                [],
            )
        else:
            raise


def main() -> None:
    diff = get_diff()
    if not diff.strip():
        print("Empty diff; nothing to review.")
        return

    truncated = diff[:MAX_DIFF_CHARS]
    note = "" if len(diff) <= MAX_DIFF_CHARS else "\n\n[diff truncated for length]"

    review = extract_json(run_codex(PROMPT_TEMPLATE.format(diff=truncated + note)))

    positions = valid_positions(diff)
    summary = review.get("summary", "Automated review.")
    kept, dropped = [], []
    for c in review.get("comments", []):
        p, ln, b = c.get("path"), c.get("line"), c.get("body")
        if not (p and isinstance(ln, int) and b):
            continue
        if ln in positions.get(p, set()):
            kept.append({"path": p, "line": ln, "side": "RIGHT", "body": b})
        else:
            dropped.append(f"- `{p}:{ln}` — {b}")

    body = f"## 🤖 Codex review ({MODEL})\n\n{summary}"
    if dropped:
        body += "\n\n<details><summary>Comments that couldn't be anchored to the diff</summary>\n\n" \
                + "\n".join(dropped) + "\n\n</details>"

    submit_review(body, kept)
    print(f"Posted review: {len(kept)} inline comment(s), {len(dropped)} unanchored.")


if __name__ == "__main__":
    main()
