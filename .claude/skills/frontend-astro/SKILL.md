---
name: frontend-astro
description: Use when editing the Meta Watcher operator UI under `web/src/` — Astro components, `app.ts`, or `global.css` — or when rebuilding the committed static bundle at `meta_watcher/web/static/`.
---

# Frontend skill

## When to use

- Editing `web/src/pages/index.astro`, `web/src/components/*.astro`, `web/src/scripts/app.ts`, or `web/src/styles/global.css`.
- Rebuilding `meta_watcher/web/static/` after a UI change.
- Wiring a new `/api/*` field into the control panel or snapshot polling.

## When not to use

- Changes to the HTTP surface that the UI depends on — use [web-surface](../web-surface/SKILL.md) first, then come back here.
- Changes inside the Python pipeline — use [pipeline-core](../pipeline-core/SKILL.md).

## Rules

1. **The committed bundle matters.** `meta_watcher/web/static/` is what `pip install -e .` ships. After any Astro source change, run `npm run build` in `web/` and commit the resulting diff along with the source change.
2. **Use the dev proxy.** While iterating, run `npm run dev` in `web/` (Astro on `:4321`) alongside `meta-watcher`. The proxy in `astro.config.mjs` forwards `/api`, `/stream.mjpg`, `/frame.jpg` to `http://127.0.0.1:8765` (override with `META_WATCHER_BACKEND`). Do not hardcode absolute backend URLs.
3. **Keep `SnapshotPayload` honest.** The TypeScript type in `app.ts` only lists fields the UI reads. Extend it whenever you start reading a new field, and do not access fields that aren't in the type — use `as unknown as …` only when absolutely necessary.
4. **Reuse CSS variables.** All colors come from `global.css` custom properties (`--bg`, `--text`, `--accent`, etc.). Add new properties at the top rather than inlining hex colors in new rules.
5. **MJPEG reconnect pattern.** When you add a feature that requires restarting the video stream, reuse `reloadStream()` which appends a cache-busting query string.
6. **Accessibility.** Keep `aria-live` on inventory and clip lists, preserve the form (`<form id="controls">`) so Enter submits, and keep the video `<img>` `alt` in sync.

## Workflow

1. Read [docs/guides/frontend.md](../../docs/guides/frontend.md) and [docs/guides/operator-ui.md](../../docs/guides/operator-ui.md).
2. Confirm the backend fields you need actually exist at `/api/config`, `/api/snapshot`, or a runtime endpoint. If not, implement them first per the [web-surface](../web-surface/SKILL.md) skill.
3. Edit the Astro source. Prefer `app.ts` over inline `<script>`s.
4. Run `npm run dev` from `web/` and verify manually in the browser against a running backend.
5. Run `npm run build` and commit the resulting `meta_watcher/web/static/*` updates together with the `web/src/*` changes.
6. Run `python3 -m unittest discover -s tests` to confirm the Python-side tests still pass.
7. Update [docs/guides/operator-ui.md](../../docs/guides/operator-ui.md) if controls or status readouts changed.
8. Append a `CHANGELOG.md` entry under `[Unreleased]` when the change is user-facing.

## Quick reference

- Endpoints the UI calls: `GET /api/config`, `PUT /api/config`, `GET /api/snapshot`, `GET /api/devices/webcams`, `POST /api/runtime/start`, `POST /api/runtime/stop`, `POST /api/runtime/rescan`, `POST /api/runtime/recording`, `GET /stream.mjpg`.
- Snapshot fields consumed today: `running`, `mode`, `status_text`, `recording_active`, `inventory_items`, `completed_clips`, `error`.
- Poll cadence: 500 ms in `pollLoop`.

## Docs this skill routes into

- [docs/guides/frontend.md](../../docs/guides/frontend.md)
- [docs/guides/operator-ui.md](../../docs/guides/operator-ui.md)
- [docs/reference/http-api.md](../../docs/reference/http-api.md)
