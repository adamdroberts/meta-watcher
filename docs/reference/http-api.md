# HTTP API reference

All endpoints are defined in `meta_watcher/web/server.py::build_app`. The server listens on the `--host` / `--port` passed to `meta-watcher` (default `http://127.0.0.1:8765`). FastAPI's interactive docs are disabled (`docs_url=None`, `redoc_url=None`); use this page as the source of truth.

Content type is `application/json` for API endpoints unless noted.

## Configuration

### `GET /api/config`

Returns the current `AppConfig` as a JSON dict (dataclass-serialized via `asdict`). Shape matches [`configuration.md`](configuration.md).

Response `200`:

```json
{
  "source":     { "kind": "webcam", "value": "0", "width": 1280, "height": 720, "fps": 30 },
  "models":     { "mac_sam_model_id": "...", "linux_sam_model_id": "...", "inference_max_side": 960, "inference_interval_ms": 250 },
  "thresholds": { "person_confidence": 0.3, "inventory_confidence": 0.25, "overlap_iou": 0.5, "tracking_iou": 0.3, "min_area_ratio": 0.002 },
  "timings":    { "empty_scene_rescan_seconds": 5.0, "person_confirmation_frames": 3, "pre_roll_seconds": 3.0, "post_roll_seconds": 2.0 },
  "inventory":  { "labels": [], "auto_rescan": true },
  "output":     { "directory": "recordings" },
  "upload":     { "enabled": false, "provider": "gcp", "bucket": "", ... },
  "timestamps": { "enabled": false, "ots_binary": "ots", "stamp_videos": true, "stamp_snapshots": true, ... }
}
```

### `PUT /api/config`

Merges a patch into the running config and returns the result. Only keys that already exist on the dataclass block (for example `thresholds.person_confidence`) are applied; unknown keys are silently dropped.

Request body:

```json
{ "thresholds": { "person_confidence": 0.55 } }
```

Response `200`: updated `AppConfig` as JSON, same shape as `GET /api/config`.

Response `400`: `{"error": "Config patch must be a JSON object."}` if the body is not a JSON object, or `{"error": "Section '<name>' must be a JSON object; got <type>."}` if a known top-level section carries a non-dict value.

### `GET /api/config/files`

Enumerates `*.json` / `*.yaml` / `*.yml` files under the search dirs (by default the repo root). Hidden files are skipped; duplicates (by resolved path) are collapsed. The currently-active path is always included even if it lives outside the search dirs.

Response `200`:

```json
{
  "active": "/abs/path/to/config.oci.json",
  "files": [
    { "path": "/abs/path/to/config.default.json", "name": "config.default.json", "writable": true, "active": false },
    { "path": "/abs/path/to/config.oci.json",     "name": "config.oci.json",     "writable": true, "active": true }
  ]
}
```

### `POST /api/config/switch`

Load a config file from disk, replace the in-memory `AppConfig`, and update the tracked active path. Does **not** restart the runtime — the `requires_restart` flag in the response tells the caller whether the running pipeline is still on the old config.

Request body:

```json
{ "path": "/abs/path/to/config.oci.json" }
```

Relative paths are resolved against the first search directory.

Responses:

- `200`: `{ "config": { ... }, "active": "/abs/path", "running": true, "requires_restart": true }`.
- `400`: `{ "error": "..." }` for malformed body or validation failure.
- `403`: `{ "error": "Config path … is outside the allowed search dirs." }` — path traversal guard.
- `404`: `{ "error": "Config file not found: ..." }`.

### `POST /api/config/save`

Persist the current in-memory config to the active file via `meta_watcher.config.save_config` (atomic: tmp file + `os.replace`, JSON with `indent=2` and a trailing newline). If the active file is YAML (`.yaml` / `.yml`), a sibling JSON file is written instead to avoid destroying comments/ordering, and the active path is updated to that JSON.

Request body (optional):

```json
{ "path": "/abs/path/to/config.json" }
```

Omit the body to save to the current active path. Any override path must live under an allowed search directory.

Responses:

- `200`: `{ "path": "/abs/path", "bytes_written": 1234 }`.
- `400`: `{ "error": "No active config path set — cannot save." }` — boot without `--config`, no override provided.
- `403`: `{ "error": "..." }` — path traversal guard.
- `500`: `{ "error": "Failed to write config: ..." }` — underlying `OSError`.

## Runtime lifecycle

### `POST /api/runtime/start`

Starts the capture pipeline. Optionally accepts a config patch in the body, applied before start (same rules as `PUT /api/config`). If the runtime is already running, this is a no-op.

Request body (optional):

```json
{ "source": { "kind": "webcam", "value": "0" } }
```

Responses:

- `200` `{ "running": true }` on success.
- `500` `{ "error": "..." }` if provider or source construction fails (for example, the camera is unavailable).

### `POST /api/runtime/stop`

Stops the runtime (joins producer and consumer threads, releases the source, shuts down the provider). Safe to call when already stopped.

Response `200`: `{ "running": false }`.

### `POST /api/runtime/rescan`

Requests a manual inventory rescan. Latched and applied by `StreamStateMachine` on the next safe transition (see the [state machine doc](../architecture/state-machine.md) for latching rules).

Response `200`: `{ "ok": true }`. This returns `200` even if the runtime is not currently running (it becomes a no-op).

### `POST /api/runtime/recording`

Enable or disable clip recording at runtime. The producer stops calling `push_frame` and the processor stops starting events when disabled. Existing open events are not forcibly closed.

Request body: `{ "enabled": true }` or `{ "enabled": false }`. If the body is missing or malformed, recording is enabled.

Response `200`: `{ "recording_enabled": true }`.

## Pipeline state

### `GET /api/snapshot`

Returns the current pipeline state. Used by the operator UI to refresh once per 500 ms.

Response `200`:

```json
{
  "running": true,
  "mode": "inventory",
  "status_text": "scene objects: 3",
  "frame_index": 1287,
  "source_id": "webcam",
  "inventory_active": true,
  "recording_active": false,
  "inventory_items": [
    { "label": "chair", "confidence": 0.82, "samples": 1, "last_seen": 1234.5 }
  ],
  "completed_clips": [
    "/abs/path/to/recordings/webcam_20260418T003658Z.mp4"
  ],
  "error": null
}
```

- `mode` — `idle`, `inventory`, `person_present`, or `cooldown`.
- `inventory_items` — the configured labels normalized into `InventoryItem` entries. Confidence and samples are placeholders from config (SAM 3.1 is the authority in the per-frame overlays).
- `completed_clips` — cumulative list of MP4 paths produced since the runtime was last started.
- `error` — last captured error message from the producer or consumer thread, or `null`.

### `GET /api/devices/webcams`

Enumerates local webcams. On Linux, uses `/dev/video*` + V4L2 capability bits. On macOS, shells out to `system_profiler SPCameraDataType -json` and falls back to probing OpenCV indices `0..3`. Other platforms fall back to probing too.

Response `200`:

```json
{
  "devices": [
    { "value": "0", "label": "FaceTime HD Camera (index 0)", "path": null }
  ]
}
```

`value` is the string form of the OpenCV index that `source.value` expects when `source.kind == "webcam"`.

## Video streams

### `GET /stream.mjpg`

Multipart MJPEG stream (`multipart/x-mixed-replace; boundary=meta-watcher-frame`). Frames are driven by `RuntimeState.wait_for_new_jpeg`:

- While the pipeline is not producing snapshots yet, the producer thread's raw camera frames are encoded and pushed so the preview goes live as soon as the camera opens.
- After the first `PipelineSnapshot`, the stream emits the annotated overlays.
- A placeholder "Waiting for camera…" frame is sent if no frames are available.

Each multipart chunk looks like:

```
--meta-watcher-frame\r\n
Content-Type: image/jpeg\r\n
Content-Length: <N>\r\n
\r\n
<JPEG bytes>\r\n
```

The endpoint has no explicit end state; clients should close the connection when done.

### `GET /frame.jpg`

Latest single JPEG, or the "Waiting for camera…" placeholder if nothing has been produced yet. Useful for scripts and snapshots.

Response `200`: `image/jpeg` body.

## Static content

### `GET /`

Serves `meta_watcher/web/static/index.html` (the committed Astro build) via `StaticFiles(html=True)`. If the static directory is missing, the app falls back to a simple HTML page telling the operator to run `cd web && npm install && npm run build`.

Other paths under `/_astro/`, `/content-assets.mjs`, etc., are served by `StaticFiles`.
