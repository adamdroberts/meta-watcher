---
name: web-surface
description: Use when adding or changing Meta Watcher HTTP endpoints, `RuntimeState` methods, MJPEG/JPEG behavior, or any API consumed by the Astro operator UI. Trigger on edits to `meta_watcher/web/server.py` or `meta_watcher/web/state.py`.
---

# Web surface skill

## When to use

- Adding, removing, or changing an endpoint in `meta_watcher/web/server.py`.
- Changing `RuntimeState` lifecycle (`start`, `stop`, `rescan`, `set_recording_enabled`, `update_config`, `snapshot_payload`, JPEG buffering).
- Touching MJPEG or `/frame.jpg` streaming behavior.
- Changing the wire shape of `/api/snapshot` or `/api/config`.

## When not to use

- Changes inside `StreamProcessor` or `StreamRuntime` — use [pipeline-core](../pipeline-core/SKILL.md).
- Changes in the Astro source — use [frontend-astro](../frontend-astro/SKILL.md). If the UI needs a new field, you usually update *both* skills' outputs in one commit.

## Rules

1. **Thread safety is load-bearing.** Every read/write against `RuntimeState` is guarded by `self._lock` or `self._frame_cond`. New fields must follow the same pattern. Do not expose raw mutable references outside the lock.
2. **Patches are additive and conservative.** `update_config` only merges top-level keys that already exist on the dataclasses. Do not loosen this — it is relied on by the UI and tests.
3. **Errors flow through `_on_error`.** Any new background task must report failures via `_on_error(str)` so the UI banner stays consistent.
4. **Bump the JPEG version when something changes.** If you introduce a new writer path, make sure it calls `self._latest_jpeg_version += 1` inside the cond so MJPEG waiters wake up.
5. **Endpoints return JSON.** Except for `/stream.mjpg`, `/frame.jpg`, and static content, every endpoint returns `JSONResponse`. Error responses use `{"error": "..."}` with an appropriate HTTP status.
6. **Keep FastAPI disabled from auto-docs.** `docs_url=None, redoc_url=None`. The authoritative docs are [docs/reference/http-api.md](../../docs/reference/http-api.md).

## Workflow

1. Read [docs/reference/http-api.md](../../docs/reference/http-api.md) and [docs/internals/web-state.md](../../docs/internals/web-state.md) first.
2. Add or modify the endpoint in `build_app`. Add the matching `RuntimeState` method when the action requires locked state.
3. Extend `tests/test_app.py::WebServerTests` and/or `RuntimeStateTests`. Reuse `_build_client` with `_FakeProvider` / `_ScriptedSource` to avoid real hardware.
4. Update [docs/reference/http-api.md](../../docs/reference/http-api.md) with the new shape. If the snapshot grows a field, also update [docs/guides/operator-ui.md](../../docs/guides/operator-ui.md).
5. If the UI needs to consume the change, follow [frontend-astro](../frontend-astro/SKILL.md) to wire it in and rebuild the bundle.
6. Append a `CHANGELOG.md` entry under `[Unreleased]`.

## Quick reference

- Default bind: `127.0.0.1:8765`, `docs_url=None`, `redoc_url=None`.
- MJPEG boundary: `meta-watcher-frame`.
- Static content: served from `meta_watcher/web/static/` via `StaticFiles(html=True)` if the directory exists; otherwise a minimal HTML fallback.
- Config patch shape: dict keyed by `source`, `models`, `thresholds`, `timings`, `inventory`, `output`; unknown keys are dropped silently.

## Docs this skill routes into

- [docs/reference/http-api.md](../../docs/reference/http-api.md)
- [docs/reference/python-api.md](../../docs/reference/python-api.md)
- [docs/internals/web-state.md](../../docs/internals/web-state.md)
- [docs/guides/operator-ui.md](../../docs/guides/operator-ui.md)
