from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import threading
import time
import webbrowser

from .config import load_config
from .web import RuntimeState, build_app


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Meta Watcher local web server")
    parser.add_argument("--config", default=None, help="Optional path to a JSON or YAML config file")
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind the HTTP server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the HTTP server to (default: 8765)")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the web UI in a browser on startup (off by default).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help=argparse.SUPPRESS,  # kept for backward compatibility; default is already no-browser.
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    state = RuntimeState(config)
    app = build_app(state)

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn is required to run the Meta Watcher web server. "
            'Reinstall with `python3 -m pip install -e ".[desktop]"`.'
        ) from exc

    url = f"http://{args.host}:{args.port}"
    print(f"[meta-watcher] web UI: {url}", file=sys.stderr, flush=True)
    if args.open_browser and not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
            access_log=False,
        )
    finally:
        state.stop()
    return 0


def _open_browser_when_ready(url: str) -> None:
    # Give uvicorn a short head start so the browser does not race the bind.
    time.sleep(0.4)
    try:
        webbrowser.open(url, new=1, autoraise=True)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
