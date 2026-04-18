# CLI reference

Entry point: `meta_watcher.app.main`, registered as the console script `meta-watcher` in `pyproject.toml`.

```bash
meta-watcher [--config PATH] [--host HOST] [--port PORT] [--open-browser]
```

## Flags

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--config PATH` | unset (use built-in defaults) | Path to a JSON or YAML config file. See the [configuration reference](configuration.md) for the schema. |
| `--host HOST` | `127.0.0.1` | Address to bind the HTTP server to. Use `0.0.0.0` to listen on all interfaces. |
| `--port PORT` | `8765` | Port to bind the HTTP server to. |
| `--open-browser` | off | Open the system browser at `http://{host}:{port}` after uvicorn starts. |

## Exit codes

- `0` — uvicorn shut down cleanly (stop signal or `ctrl-c`).
- Any other exit originates from uvicorn / the runtime; the state is stopped from a `finally` block before propagation.

## Startup behavior

1. `argparse` parses flags.
2. `load_config(args.config)` returns an `AppConfig`.
3. `RuntimeState(config)` constructs the thread-safe runtime container.
4. `build_app(state)` returns the FastAPI app.
5. If `--open-browser` was passed, a daemon thread sleeps 0.4 seconds and then calls `webbrowser.open(url)`.
6. `uvicorn.run(app, host, port, log_level="warning", access_log=False)` takes over the main thread.
7. On exit, `state.stop()` tears down the runtime (joins producer + consumer threads, releases the source, shuts down the inference provider).

## Running from the repo

The repo includes `bin/meta-watcher`, a Bash launcher that prepends the repo root and user site-packages to `PYTHONPATH` and then execs `python -m meta_watcher.app`. Use it when you want to run without installing the package:

```bash
./bin/meta-watcher --config ./config.default.json
```

It picks `$CONDA_PREFIX/bin/python` first, then `python`, then `python3`. On macOS it also adds the user framework site-packages path so `mlx-vlm` installed with `pip install --user` is visible.

## uvicorn logging

Meta Watcher forces `log_level="warning"` and `access_log=False` to keep the terminal quiet. Errors still surface via `on_error` callbacks into `RuntimeState`, where they are exposed through `/api/snapshot`'s `error` field.

## Environment variables

`meta_watcher/__init__.py` sets a couple of OpenCV environment defaults before `cv2` is ever imported, to reduce backend probing noise. These can still be overridden:

- `OPENCV_LOG_LEVEL=ERROR` — quiets OpenCV's own logger.
- `OPENCV_VIDEOIO_PRIORITY_OBSENSOR=0` — disables the Orbbec backend probe that otherwise spams `can't open camera by index` lines.
- `META_WATCHER_BACKEND` — read by the Astro dev server (`web/astro.config.mjs`) to proxy `/api`, `/stream.mjpg`, and `/frame.jpg` to a non-default backend URL.
