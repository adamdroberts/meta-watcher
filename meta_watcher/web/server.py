from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI is required to serve the Meta Watcher web UI. "
        'Install it with `python3 -m pip install -e ".[desktop]"`.'
    ) from exc

from ..sources import list_webcams
from .state import ConfigPathError, ConfigValidationError, RuntimeState, placeholder_jpeg


STATIC_DIR = Path(__file__).resolve().parent / "static"
MJPEG_BOUNDARY = "meta-watcher-frame"


def build_app(state: RuntimeState) -> FastAPI:
    app = FastAPI(title="Meta Watcher", docs_url=None, redoc_url=None)

    @app.get("/api/config")
    def get_config() -> JSONResponse:
        return JSONResponse(state.config_dict())

    @app.put("/api/config")
    async def put_config(request: Request) -> JSONResponse:
        patch = await request.json()
        if not isinstance(patch, dict):
            return JSONResponse({"error": "Config patch must be a JSON object."}, status_code=400)
        try:
            updated = state.update_config(patch)
        except ConfigValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(asdict(updated))

    @app.get("/api/config/files")
    def list_config_files() -> JSONResponse:
        active = state.active_config_path()
        return JSONResponse(
            {
                "active": str(active) if active is not None else None,
                "files": state.list_config_files(),
            }
        )

    @app.post("/api/config/switch")
    async def switch_config(request: Request) -> JSONResponse:
        body = await request.json() if request.headers.get("content-length") else {}
        if not isinstance(body, dict) or "path" not in body:
            return JSONResponse(
                {"error": "Body must be a JSON object with a 'path' field."}, status_code=400
            )
        try:
            loaded = state.reload_config(body["path"])
        except FileNotFoundError as exc:
            return JSONResponse({"error": f"Config file not found: {exc}"}, status_code=404)
        except ConfigPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        except ConfigValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        active = state.active_config_path()
        running = state.is_running()
        return JSONResponse(
            {
                "config": asdict(loaded),
                "active": str(active) if active is not None else None,
                "running": running,
                "requires_restart": running,
            }
        )

    @app.post("/api/config/save")
    async def save_config_endpoint(request: Request) -> JSONResponse:
        body: Any = {}
        if request.headers.get("content-length"):
            try:
                body = await request.json()
            except Exception:
                body = {}
        path_override = body.get("path") if isinstance(body, dict) else None
        try:
            written = state.save_active_config(path_override)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except ConfigPathError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        except OSError as exc:
            return JSONResponse({"error": f"Failed to write config: {exc}"}, status_code=500)
        return JSONResponse(
            {
                "path": str(written),
                "bytes_written": written.stat().st_size,
            }
        )

    @app.get("/api/snapshot")
    def get_snapshot() -> JSONResponse:
        return JSONResponse(state.snapshot_payload())

    @app.get("/api/devices/webcams")
    def get_webcams() -> JSONResponse:
        devices = list_webcams()
        return JSONResponse(
            {
                "devices": [
                    {"value": str(device.index), "label": device.label, "path": device.path}
                    for device in devices
                ],
            }
        )

    @app.post("/api/runtime/start")
    async def start_runtime(request: Request) -> JSONResponse:
        patch: dict[str, Any] | None = None
        if request.headers.get("content-length"):
            try:
                body = await request.json()
                if isinstance(body, dict):
                    patch = body
            except Exception:
                patch = None
        try:
            state.start(patch)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        return JSONResponse({"running": True})

    @app.post("/api/runtime/stop")
    def stop_runtime() -> JSONResponse:
        state.stop()
        return JSONResponse({"running": False})

    @app.post("/api/runtime/rescan")
    def rescan() -> JSONResponse:
        state.rescan()
        return JSONResponse({"ok": True})

    @app.post("/api/runtime/recording")
    async def set_recording(request: Request) -> JSONResponse:
        body: Any = {}
        if request.headers.get("content-length"):
            try:
                body = await request.json()
            except Exception:
                body = {}
        enabled = bool(body.get("enabled", True)) if isinstance(body, dict) else True
        state.set_recording_enabled(enabled)
        return JSONResponse({"recording_enabled": enabled})

    @app.get("/api/storage/health")
    def storage_health() -> JSONResponse:
        return JSONResponse(state.storage_health())

    @app.get("/api/recordings")
    def list_recordings() -> JSONResponse:
        return JSONResponse(state.list_recordings())

    @app.get("/api/recordings/{event_id}")
    def recording_detail(event_id: str) -> JSONResponse:
        try:
            return JSONResponse(state.recording_detail(event_id))
        except KeyError:
            return JSONResponse(
                {"error": f"event not found: {event_id}"}, status_code=404
            )
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

    @app.get("/api/recordings/{event_id}/artifact")
    def get_artifact(event_id: str, key: str, request: Request) -> Response:
        event_segment = f"/{event_id}/"
        if event_segment not in key and not key.endswith(f"/{event_id}"):
            return JSONResponse(
                {"error": "key does not belong to event"}, status_code=400
            )

        byte_range: tuple[int, int] | None = None
        total_hint: int | None = None
        range_header = request.headers.get("range") or request.headers.get("Range")
        if range_header:
            import re

            m = re.match(r"bytes=(\d+)-(\d*)$", range_header.strip())
            if not m:
                return Response(status_code=416)
            start = int(m.group(1))
            end: int | None = int(m.group(2)) if m.group(2) else None

            try:
                detail = state.recording_detail(event_id)
            except KeyError:
                return JSONResponse({"error": "event not found"}, status_code=404)
            total_hint = next(
                (a["size"] for a in detail["artifacts"] if a["key"] == key), None
            )
            if total_hint is None:
                return JSONResponse({"error": "artifact not found"}, status_code=404)
            if end is None or end >= total_hint:
                end = total_hint - 1
            byte_range = (start, end)

        try:
            chunks, length, content_type = state.stream_artifact(
                key, byte_range=byte_range
            )
        except PermissionError as exc:
            return JSONResponse({"error": str(exc)}, status_code=403)
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

        headers: dict[str, str] = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        }
        status_code = 200
        if byte_range is not None and total_hint is not None:
            start, end = byte_range
            headers["Content-Range"] = f"bytes {start}-{end}/{total_hint}"
            status_code = 206

        return StreamingResponse(
            chunks,
            media_type=content_type,
            headers=headers,
            status_code=status_code,
        )

    @app.post("/api/recordings/{event_id}/verify")
    def verify_recording(event_id: str) -> JSONResponse:
        try:
            return JSONResponse(state.verify_recording(event_id))
        except KeyError:
            return JSONResponse(
                {"error": f"event not found: {event_id}"}, status_code=404
            )
        except RuntimeError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)

    @app.get("/stream.mjpg")
    def stream_mjpg() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg_generator(state),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )

    @app.get("/frame.jpg")
    def latest_frame() -> Response:
        jpeg = state.latest_jpeg() or placeholder_jpeg()
        return Response(content=jpeg, media_type="image/jpeg")

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    else:
        @app.get("/")
        def fallback_index() -> Response:
            message = (
                "<html><body><h1>Meta Watcher</h1>"
                "<p>The web UI bundle is missing. Run <code>cd web && npm install && npm run build</code> to produce it.</p>"
                "<p>The REST API is available at <code>/api/config</code>, <code>/api/snapshot</code>, and related endpoints.</p>"
                "</body></html>"
            )
            return Response(content=message, media_type="text/html")

    return app


def _mjpeg_generator(state: RuntimeState) -> Iterator[bytes]:
    boundary = MJPEG_BOUNDARY.encode("ascii")
    last_version = -1
    while True:
        jpeg, version = state.wait_for_new_jpeg(last_version, timeout=1.0)
        if version == last_version:
            jpeg = state.latest_jpeg() or placeholder_jpeg()
        last_version = version
        payload = jpeg if jpeg is not None else placeholder_jpeg()
        yield (
            b"--" + boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(payload)).encode("ascii") + b"\r\n\r\n"
            + payload + b"\r\n"
        )
