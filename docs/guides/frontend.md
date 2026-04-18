# Frontend development

The operator UI is an Astro 5 static site. Source lives in `web/`; the build output is committed to `meta_watcher/web/static/` so `pip install -e .` produces a working UI without requiring a Node toolchain.

## Project layout

```
web/
├── astro.config.mjs     Output target + dev-server proxy
├── package.json         pnpm / npm scripts
├── tsconfig.json
└── src/
    ├── pages/index.astro           Root page, loads VideoPanel and ControlPanel, imports app.ts
    ├── components/
    │   ├── VideoPanel.astro        <img src="/stream.mjpg">
    │   └── ControlPanel.astro      Status rows, form controls, inventory/clip lists
    ├── scripts/app.ts              All client logic: fetch config, start/stop, poll snapshot
    └── styles/global.css           Dark-theme styles, grid layout, status pill colors
```

## Install

From the `web/` directory:

```bash
cd web
npm install
```

A `pnpm-lock.yaml` is checked in, so `pnpm install` also works and is noticeably faster for cold installs.

## Build

```bash
cd web
npm run build
```

Astro writes the build to `outDir: "../meta_watcher/web/static"`. The Python server mounts that directory via `StaticFiles(html=True)`, so after a build `meta-watcher` immediately serves the new bundle.

Commit the resulting `meta_watcher/web/static/` diff together with the `web/src/` change. The repo is set up to ship the compiled bundle so the Python install does not need Node.

## Dev loop

```bash
# Terminal 1: backend
meta-watcher

# Terminal 2: frontend with HMR
cd web
npm run dev
# Visit http://127.0.0.1:4321
```

`astro.config.mjs` wires Vite's proxy so `/api`, `/stream.mjpg`, and `/frame.jpg` go to `http://127.0.0.1:8765`. Point elsewhere with the `META_WATCHER_BACKEND` env var before starting `npm run dev`:

```bash
META_WATCHER_BACKEND=http://192.168.1.100:8765 npm run dev
```

## Preview

`npm run preview` serves the static build locally but does *not* proxy to a backend, so `/api/*` calls will 404 unless you run the Python server on the same origin (by binding to `127.0.0.1:4321` with `--port 4321`, for example).

## Styling conventions

- All colors and fonts are declared as CSS custom properties at the top of `global.css` (`--bg`, `--text`, `--accent`, etc.). Keep new styles consistent with those variables so dark mode stays coherent.
- Status values (`idle`, `inventory`, `person_present`, `cooldown`) become `data-mode` attributes on the `#status-mode` pill. Color rules key off those attributes in `global.css`.

## API contract

The TypeScript side assumes the shapes documented in [`http-api.md`](../reference/http-api.md). The `SnapshotPayload` type in `app.ts` is intentionally narrow — it lists just the fields the UI actually reads. If you add a new field to `/api/snapshot`, extend that type before relying on it in the UI.

## Accessibility notes

- The control panel uses a form (`<form id="controls">`) so Enter submits and Start is the default.
- `aria-live="polite"` is set on the inventory and clip lists so screen readers announce updates without overwhelming.
- The video `<img>` has an `alt` of `"Meta Watcher video stream"`; reconnect logic preserves the element so focus survives restarts.

## Shipping changes

1. Edit the Astro / TS source.
2. Run `npm run build` in `web/`.
3. Commit both the `web/src/*` change and the updated `meta_watcher/web/static/*` files.
4. Run `python3 -m unittest discover -s tests` to confirm the Python `test_app.py::WebServerTests` still pass (they exercise the endpoints your UI talks to).
