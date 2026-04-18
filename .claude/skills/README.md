# Meta Watcher agent skills

These repo-local skills teach future LLM sessions how to work on Meta Watcher without re-reading the full codebase every time. They are concise, procedural, and point back at the canonical docs under `docs/`.

## When to pick which skill

| Skill | Use when the task touches ... |
| ----- | ----------------------------- |
| [pipeline-core](pipeline-core/SKILL.md) | `meta_watcher/core.py`, `pipeline.py`, `overlay.py`, `sources.py` — capture, tracking, state machine, recorder, overlay rendering |
| [web-surface](web-surface/SKILL.md) | `meta_watcher/web/state.py`, `meta_watcher/web/server.py` — HTTP endpoints, runtime state, MJPEG stream |
| [frontend-astro](frontend-astro/SKILL.md) | `web/src/` — Astro components, `app.ts`, `global.css`, committed static bundle |
| [inference-backends](inference-backends/SKILL.md) | `meta_watcher/inference.py` — MLX subprocess, CUDA SAM 3.1 provider, fp32 patch |

## Shared expectations

Every skill assumes:

- Changes that affect behavior get docs updates in the same commit. The docs map is in [`../../docs/README.md`](../../docs/README.md).
- A meaningful change gets a CHANGELOG entry in [`../../CHANGELOG.md`](../../CHANGELOG.md).
- Tests live in `tests/` and run with `python3 -m unittest discover -s tests`. Add coverage when you add behavior; fakes (`FakeProvider`, `_ScriptedSource`, `MemorySink`, `MemorySinkFactory`) are already in the suite.
- No skill should claim a task is done without running the full test suite locally.

## Entry docs for ingestion

- [`../../llms.txt`](../../llms.txt) — concise LLM index.
- [`../../llms-full.txt`](../../llms-full.txt) — full documentation bundle.
- [`../../docs/README.md`](../../docs/README.md) — browsable docs root.
