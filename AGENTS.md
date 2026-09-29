# AGENTS.md

Instructions for coding agents working in this repository.

This file is **not** the OpenClaw workspace bootstrap. Benchmarked agents
read `AGENTS.md` from `~/.openclaw/workspace/` (copied into each task
workspace at run time). Edits here do not change agent behavior during a
benchmark. See [Workspace bootstrap](#workspace-bootstrap).

## What this repo is

PinchBench is a benchmark **skill**: a task suite plus a Python runner that
drives an [OpenClaw](https://github.com/openclaw/openclaw) agent and grades
the result. Public scores live at [pinchbench.com](https://pinchbench.com).

This repository is **not** the source of official leaderboard model lists.
Those live in [pinchbench/scripts `default-models.yml`](https://github.com/pinchbench/scripts/blob/main/default-models.yml).

Human-facing docs: `README.md` (operators) and `SKILL.md` (skill metadata /
quick start). `SKILL.md` can lag the task list; treat `tasks/manifest.yaml`
as the source of truth.

## Layout

| Path | Role |
| --- | --- |
| `scripts/benchmark.py` | CLI entry. Orchestrates load → execute → grade → upload. |
| `scripts/lib_agent.py` | OpenClaw session, workspace prep, transcripts, fws. |
| `scripts/lib_tasks.py` | Parse `task_*.md` + `manifest.yaml`. |
| `scripts/lib_grading.py` | Automated / LLM-judge / hybrid scoring. |
| `scripts/lib_upload.py` | Leaderboard token + upload. |
| `scripts/lib_axiom.py` | Optional Axiom telemetry (`AXIOM_API_TOKEN`). |
| `scripts/lib_train_recorder.py` | `--record-train` JSONL capture. |
| `scripts/lib_trend.py` | `--trend` regression across prior result files. |
| `scripts/lib_fws.py` | Fake Workspace Server for GWS/GitHub-style tasks. |
| `scripts/lint_manifest.py` | Manifest ↔ files ↔ frontmatter `id`. |
| `scripts/lint_argparse_help.py` | Unescaped `%` in argparse `help=`. |
| `scripts/run.sh` | `uv run scripts/benchmark.py "$@"`. |
| `tasks/manifest.yaml` | Category order, `run_first`, `core`. |
| `tasks/TASK_TEMPLATE.md` | Required task format. |
| `tasks/task_*.md` | One task per file. |
| `assets/` | Fixtures referenced by `workspace_files`. |
| `tests/` | Unit tests (add `scripts/` to `sys.path` themselves). |
| `BENCHMARK_VERSION` | Bumped on GitHub release (`v*` tag). |
| `Dockerfile.benchmark` | CI image with OpenClaw + uv. |

Python 3.10+, [uv](https://docs.astral.sh/uv/). Line length 100 (`ruff` /
`black` in `pyproject.toml`).

## Everyday commands

```bash
# Lint (matches CI: ruff 0.15.6 + local linters)
ruff check .
python scripts/lint_manifest.py
python scripts/lint_argparse_help.py

# Unit tests
uv run --extra dev pytest

# One task, no upload (needs a running OpenClaw)
./scripts/run.sh --model openrouter/anthropic/claude-sonnet-4 \
  --suite task_sanity --no-upload

# Category, core subset, or automated-only
./scripts/run.sh --model MODEL --suite coding --no-upload
./scripts/run.sh --model MODEL --core --no-upload
./scripts/run.sh --model MODEL --suite automated-only --no-upload
```

`--suite` accepts `all`, `automated-only`, a category name, `cat+cat`
(e.g. `coding+research`), or comma-separated task IDs.

Full operator flags are in `README.md`. Do not duplicate that table here.

## How a run works

1. `TaskLoader` reads `tasks/manifest.yaml`. `run_first` (currently
   `task_sanity`) is prepended; remaining tasks follow category order.
2. For each selected task (and each `--runs` iteration) the runner creates
   a short unique OpenClaw agent and calls `execute_openclaw_task`.
   Auto-created agents, including the judge, deny `process`,
   `sessions_spawn`, `update_plan`, and tavily (`tavily`, `tavily_search`,
   `tavily_extract`). Their `alsoAllow` whitelist includes built-in
   `web_search` and `web_fetch` plus `task-guard`. The tavily plugin entry
   is set to disabled. The interactive `main` agent entry is left unchanged.
3. `prepare_task_workspace` wipes the agent workspace, copies OpenClaw
   bootstrap files, then copies `workspace_files` from `assets/`.
4. The task prompt (or `sessions:` sequence) is sent with `--local` so
   dynamically created bench agents skip the gateway.
5. Transcripts are collected from the agent session store, including
   spawned child sessions. Multi-session steps with `new_session: true`
   reset conversation history; the workspace is never reset.
6. `grade_task` scores the transcript + workspace. Judge calls can run in
   parallel unless `--no-parallel-judge`.
7. Results JSON lands in `results/` (gitignored). Transcripts go to
   `results/{run_id}_transcripts/`. Runtime workspaces live under
   `/tmp/pinchbench/`.
8. Unless `--no-upload`, results are posted to `https://api.pinchbench.com`.

`task_sanity` is fail-fast: a 0% score aborts the rest of the suite unless
`--no-fail-fast`. Missing transcripts on sanity are treated as infra and
do **not** abort.

`--continue FILE` resumes a partial results JSON. `--model` must match the
file; other flags apply only to remaining tasks.

GWS/GitHub-style tasks (`prerequisites` mentioning `fws`, or frontmatter
category `gws`/`github`) start `@juppytt/fws` and rewrite env so `gws`
hits the mock.

## Tasks

Every task is `tasks/task_<id>.md`. The `id` in frontmatter **must** equal
the filename without `.md`, and the file **must** appear in exactly one
`categories:` list in `tasks/manifest.yaml`. Append new IDs at the end of
the category. CI (`scripts/lint_manifest.py`) enforces this.

Frontmatter fields that matter at runtime:

| Field | Notes |
| --- | --- |
| `id`, `name`, `category` | `category` should match the manifest key. |
| `grading_type` | `automated` \| `llm_judge` \| `hybrid`. |
| `timeout_seconds` | Scaled by `--timeout-multiplier`. Raise for multi-session. |
| `workspace_files` | `{source, dest}` relative to `assets/` → workspace root, or inline `{path, content}`. |
| `grading_weights` | Hybrid only. Default `{automated: 0.5, llm_judge: 0.5}`. |
| `multi_session` / `sessions` | Ordered prompts. `new_session: true` starts a fresh OpenClaw session. |
| `prerequisites` | e.g. `npm:@juppytt/fws`, `cli:gws` — used to decide fws startup. |

Required markdown sections: `## Prompt`, `## Expected Behavior`,
`## Grading Criteria`. Add `## Automated Checks` and/or
`## LLM Judge Rubric` to match `grading_type`. Follow
`tasks/TASK_TEMPLATE.md`.

Good tasks are real user requests, independently checkable, and stable
across re-runs. Prefer fixtures in `assets/` over live network when the
thing under test is analysis or file work.

Categories in the manifest today: `productivity`, `research`, `writing`,
`coding`, `analysis`, `csv_analysis`, `log_analysis`, `meeting_analysis`,
`memory`, `skills`, `integrations`. `core` is a ~20-task smoke subset
(`--core`).

After adding `tasks/task_*.md` on `main`,
`.github/workflows/update-task-count.yml` rewrites the README task-count
badge. Do not hand-edit those `<!-- task-count-* -->` spans.

## Grading

- **automated** — `exec` the `grade(transcript, workspace_path) -> dict`
  function inside `## Automated Checks`. Mean of numeric values is the
  score. Stdlib + `pathlib` only. Missing files/data → `0.0`, not an
  exception.
- **llm_judge** — rubric in `## LLM Judge Rubric`. Weights must sum to
  100%. Default (no `--judge`): OpenClaw judge session with
  `openrouter/anthropic/claude-haiku-4.5`. `--judge MODEL` calls that
  model’s API (or `claude` / `grok` CLIs) and skips OpenClaw personality
  injection. Judge workspace bootstrap files are stripped so the judge
  does not inherit the bench agent’s persona.
- **hybrid** — weighted combination of the two (`grading_weights`).

Automated `grade` keys should stay stable; they become breakdown fields.
Partial credit is `0.0`–`1.0`.

Judge results cache under `results/.judge_cache/` keyed by task,
transcript, rubric, model, workspace content, and API shape. Use
`--no-judge-cache` or `--clear-judge-cache` when changing judge logic.

`task_image_identification` uses a private answer key staged to
`/tmp/pinchbench/judge/private/` — do not put ground-truth labels in the
agent-visible workspace.

`lib_grading._read_workspace_files` skips bootstrap names
(`AGENTS.md`, `SOUL.md`, …) so the judge is not scored on those files.

## Workspace bootstrap

`prepare_task_workspace` copies, in order of preference, from
`~/.openclaw/workspace/` then from whatever was already in the agent
workspace:

`AGENTS.md`, `SOUL.md`, `BOOTSTRAP.md`, `USER.md`, `IDENTITY.md`,
`HEARTBEAT.md`, `TOOLS.md`

That **host** `AGENTS.md` is the operating manual for the *benchmarked*
OpenClaw agent. Changing it changes every subsequent local run. Do not
confuse it with this repository file.

Judge sessions delete `BOOTSTRAP.md`, `SOUL.md`, `USER.md`,
`IDENTITY.md`, and `HEARTBEAT.md` from the judge workspace on purpose.

## Conventions

- Keep runner modules in `scripts/` as flat `lib_*.py` (imported via that
  directory on `sys.path`). Do not introduce a package layout unless the
  import story is updated everywhere, including inlined `grade()` code.
- Argparse `help=` cannot contain a raw `%`. Write `%%` (CI checks this).
- Prefer `pathlib`, explicit encodings (`utf-8`), and structured logs
  already used in `scripts/`.
- Do not expand `SKILL.md`’s task table for every new task; it is a skill
  card, not the catalog.
- Official submissions need `PINCHBENCH_OFFICIAL_KEY` / `--official-key`.
  Never commit tokens or official keys.
- Training capture (`--record-train`) records successful **benchmarked**
  model calls only (`openai-completions` / `anthropic-messages`). Judge
  calls are excluded.
- `task_git_rescue_recovery` assumes unsigned commits in the fixture;
  do not re-enable GPG signing in that asset.

## Tests and CI

`.github/workflows/lint.yml` on `main` and PRs: `ruff check .`,
`python scripts/lint_argparse_help.py`, `python scripts/lint_manifest.py`.

`.github/workflows/release.yml` on `v*` GitHub releases writes
`BENCHMARK_VERSION`.

Add tests next to the module they cover (`tests/test_lib_*.py`). Existing
coverage includes grading normalization, multi-session archive, train
recorder, trend, and multi-run result aggregation.

## Do not

- Hand-edit official model lists here (wrong repo).
- Commit `results/`, `benchmark.log`, `.pinchbench/`, or `uv.lock`.
- Put secrets or image-classification answers in agent-visible fixtures.
- Assume `SKILL.md` or the README badge is the live task count — count
  `tasks/task_*.md` or read the manifest.
- Edit this file’s sibling OpenClaw bootstrap copy under `~/.openclaw/`
  unless the user asked to change **agent** behavior.

## Keep this file current

Update `AGENTS.md` in the same PR when you change: CLI flags, manifest
schema, grading types, workspace/bootstrap copy rules, fail-fast,
upload/auth, or the scripts layout. Do not list individual tasks here.
