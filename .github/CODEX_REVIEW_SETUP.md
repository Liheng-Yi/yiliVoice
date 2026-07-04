# Codex PR Review bot — setup

Automated code review on every pull request. On PR open/update, a **self-hosted GitHub
Actions runner** runs the **Codex CLI headlessly** against a self-hosted LLM
(Flex AI staging + `Qwen3-Coder-30B-A3B-Instruct-FP8` by default) and posts a formal
GitHub review with inline comments.

Files:
- [`.github/workflows/codex-review.yml`](workflows/codex-review.yml) — the trigger + job
- [`.github/scripts/codex_review.py`](scripts/codex_review.py) — runs Codex, maps its
  findings onto diff lines, posts the review

## One-time setup

### 1. Add the API key as a repo secret
Settings → Secrets and variables → Actions → **New repository secret**
- Name: `FLEXAI_API_KEY`
- Value: your `sk-flexai-...` key

> ⚠️ Rotate the key that was shared in chat first — treat it as compromised.

### 2. Register a self-hosted runner
Chosen because the staging endpoint (`tokens.flexsystems.ai`) may not be reachable from
GitHub's cloud runners. Settings → Actions → Runners → **New self-hosted runner**, then
run the provided script on a machine that:
- can reach `https://tokens.flexsystems.ai/v1`, and
- has `codex` (or Node/npm to install `@openai/codex`), `gh`, `git`, and `python3`.

On macOS with Homebrew these live in `/opt/homebrew/bin`, which the workflow adds to PATH.
Keep the runner service running (`./run.sh` or install it as a service).

### 3. Done
Open a PR from a branch **in this repo** and the review appears within a minute or two.

## Notes & knobs
- **Model**: change `CODEX_MODEL` in the workflow env (e.g. `gpt-oss-120b`,
  `Nemotron-3-Super-120B-A12B`). `GLM-5` and the 480B Qwen are disabled on staging.
- **Fork PRs are skipped** by design — running untrusted code on a self-hosted runner is
  unsafe, and forks can't read the secret. Remove the `if:` guard in the workflow only if
  you understand that risk.
- **Inline comment placement**: GitHub only allows comments on lines present in the diff.
  The script filters the model's comments to valid lines; any it can't anchor are listed
  in a collapsed "couldn't be anchored" section of the review body instead of being lost.
- **Review type** is `COMMENT` (non-blocking) — the bot never approves or requests changes.
- **Cost/latency**: large diffs are truncated to ~80k chars before being sent to the model.
