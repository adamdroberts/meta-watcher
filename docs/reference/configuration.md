# Configuration reference

All config fields map to dataclasses in `meta_watcher/config.py`. `load_config(path)` reads JSON or YAML (selected by extension), merges top-level objects onto the dataclass defaults, and returns an `AppConfig`. Missing keys fall back to defaults; unknown keys are silently ignored.

`config.default.json` is the canonical example; tests use `default_config()` directly.

## Top-level shape

```json
{
  "source":     { ... SourceConfig ... },
  "models":     { ... ModelConfig ... },
  "thresholds": { ... ThresholdConfig ... },
  "timings":    { ... TimingConfig ... },
  "inventory":  { ... InventoryConfig ... },
  "output":     { ... OutputConfig ... },
  "upload":     { ... UploadConfig ... }
}
```

## `source` (`SourceConfig`)

Defines the capture source fed to `meta_watcher.sources.build_source`.

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `kind` | string | `"webcam"` | One of `webcam`, `rtsp`, `file`. Anything else raises `ValueError`. |
| `value` | string | `"0"` | Kind-specific. Webcams: integer index as a string, or `"auto"`. RTSP: full URL. File: absolute or relative path. |
| `width` | int | `1280` | Requested frame width. Only the webcam source tries to set this on the device; cameras may ignore it. |
| `height` | int | `720` | Requested frame height. Same caveat. |
| `fps` | int | `30` | Requested capture fps. Same caveat; actual camera fps is logged at startup. |

## `models` (`ModelConfig`)

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `provider` | string | `"sam3.1"` | Object detection backend. `"sam3.1"` uses the platform-based SAM 3.1 routing below. `"detr-resnet-50"` and `"rt-detrv2"` select the cross-platform `TransformersObjectDetectionProvider` (requires `.[detr]`). Any other value raises `ValueError` in `build_provider`. |
| `mac_sam_model_id` | string | `"mlx-community/sam3.1-bf16"` | Hugging Face or local model ID used by `MlxSam31Provider`. Only consulted when `provider == "sam3.1"`. |
| `linux_sam_model_id` | string | `"facebook/sam3.1"` | Model ID passed to `sam3.model_builder.build_sam3_image_model`. Only consulted when `provider == "sam3.1"`. |
| `detr_model_id` | string | `"facebook/detr-resnet-50"` | HF model ID loaded via `AutoModelForObjectDetection` when `provider == "detr-resnet-50"`. |
| `rt_detr_model_id` | string | `"jadechoghari/RT-DETRv2"` | HF model ID loaded via `AutoModelForObjectDetection` when `provider == "rt-detrv2"`. Loaded with `trust_remote_code=True` because the repo ships custom modeling code. |
| `inference_max_side` | int | `960` | Longest side (px) the frame is downscaled to before being passed to the provider. `<=0` disables downscaling. Detection bboxes are upscaled back to the original resolution. |
| `inference_interval_ms` | int | `250` (`200` in `config.default.json`) | Minimum wall-clock spacing between provider calls. Frames between calls reuse the last detections so the overlay and recording still advance at full rate. |

## `thresholds` (`ThresholdConfig`)

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `person_confidence` | float | `0.3` | Minimum SAM 3.1 score to accept a person detection. Applied in `StreamProcessor._normalize_people`. |
| `inventory_confidence` | float | `0.25` | Minimum score to keep an inventory detection. |
| `overlap_iou` | float | `0.5` | IoU above which duplicate person detections are merged by `merge_overlapping_detections` before tracking. |
| `tracking_iou` | float | `0.3` | Minimum IoU with an existing track for `TrackManager` to reuse that track ID. |
| `min_area_ratio` | float | `0.002` | Detections smaller than this fraction of the frame area are dropped by `TrackManager.update`. |

## `timings` (`TimingConfig`)

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `empty_scene_rescan_seconds` | float | `5.0` | Dwell time after the last person leaves before `StreamStateMachine` exits `cooldown` back to `inventory`. |
| `person_confirmation_frames` | int | `3` | Consecutive "people present" frames required to transition from `inventory` to `person_present`. |
| `pre_roll_seconds` | float | `3.0` (`0.5` in `config.default.json`) | Seconds of frames `ClipRecorder` keeps in its ring buffer so events include the moments before someone enters. |
| `post_roll_seconds` | float | `2.0` | Seconds of frames written after `finish_event` before the sink closes. |

## `inventory` (`InventoryConfig`)

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `labels` | `list[str]` | `[]` | Open-vocabulary labels SAM 3.1 is asked to validate in `inventory` mode. Empty list disables inventory detection entirely. |
| `auto_rescan` | bool | `true` | When `true`, the state machine re-triggers an `inventory_reset` after each occupancy event. When `false`, the machine returns to `inventory` mode but leaves the last inventory snapshot in place until a manual rescan. |

## `output` (`OutputConfig`)

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `directory` | string | `"recordings"` | Directory where `ClipRecorder` writes `{source_id}_{UTC}.mp4` + `.json` + `.jpg`. Created automatically on startup. |
| `recording_mode` | string | `"raw"` | One of `"raw"`, `"overlay"`, `"both"`. `"raw"` records the camera stream as-is (default, matches prior behavior). `"overlay"` records the annotated frames the operator UI shows. `"both"` writes two MP4s per event: `{stem}.mp4` (raw, canonical) and `{stem}.overlay.mp4` (overlay). Any other value raises `ValueError`. |

A JPEG snapshot of the frame that triggered the event is saved alongside the MP4(s) and `.json` (same stem, `.jpg` suffix). It's written synchronously in `ClipRecorder.start_event` using the shared `meta_watcher.jpeg.encode_jpeg` helper.

## `upload` (`UploadConfig`)

Optional per-event upload of the clip + snapshot + metadata to a cloud object store. Off by default. Requires the matching `pip` extra — `.[gcp]`, `.[aws]`, or `.[oci]` — because each SDK is heavy and lazy-imported only when the provider is selected.

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `enabled` | bool | `false` | Master switch. When false, no provider SDK is imported. |
| `provider` | string | `"gcp"` | One of `gcp`, `aws`, `oci`. Anything else raises `ValueError`. |
| `bucket` | string | `""` | Destination bucket *name* (not OCID — OCI object APIs take the friendly name). Empty bucket = uploads skipped (same as `enabled: false`). |
| `prefix` | string | `"meta-watcher/"` | Prefix prepended to each object key. Object key = `{prefix}{file.name}`. |
| `credentials_path` | string | `""` | Path to provider credentials. GCP: service-account JSON. AWS: credentials file (sets `AWS_SHARED_CREDENTIALS_FILE`). OCI: `oci` config file (default `~/.oci/config`; `~` is expanded). Empty = SDK's default credential chain. |
| `region` | string | `""` | AWS region / OCI region. Ignored for GCP. |
| `namespace` | string | `""` | **OCI only.** Tenancy Object Storage namespace. Empty = auto-detected via `ObjectStorageClient.get_namespace()` on startup. |
| `profile` | string | `""` | **OCI only.** Profile name inside the `oci` config file. Empty = SDK default (`DEFAULT`). |
| `upload_videos` | bool | `true` | Upload the `.mp4` clip. |
| `upload_snapshots` | bool | `true` | Upload the `.jpg` event snapshot. |
| `upload_metadata` | bool | `true` | Upload the `.json` metadata. |
| `delete_after_upload` | bool | `false` | When true, successfully-uploaded files are deleted from `output.directory`. Failed uploads are always retained. |
| `queue_size` | int | `32` | Upload worker queue size. When full, the oldest pending upload is dropped (logged to stderr). |

Uploads run on a dedicated background thread in `meta_watcher.upload.EventUploader`. Failures are logged and dropped — there is no built-in retry.

### Object layout

Each event is grouped under its own folder in the destination bucket, keyed by the clip's filename stem (the event id):

```
{prefix}{event_id}/{event_id}.jpg            # snapshot — uploaded as soon as the event starts
{prefix}{event_id}/frames/{event_id}_…Z.jpg  # live overlay frame, enqueued every 0.5s while the event is still recording
{prefix}{event_id}/{event_id}.mp4            # finalized clip (raw)
{prefix}{event_id}/{event_id}.overlay.mp4    # finalized overlay clip (only when recording_mode = "both")
{prefix}{event_id}/{event_id}.json           # finalized metadata
```

The snapshot is always sent on the `start_event` transition — operators watching the bucket see a still frame within seconds of the event firing, long before the clip finalizes. Live frames are written to `output.directory/{event_id}/frames/` next to the clip + snapshot so the local filesystem mirrors the bucket layout; they follow the same `delete_after_upload` policy as every other artifact (so they stay on disk by default). If `output.directory` is unwritable, the uploader falls back to `/tmp/meta-watcher-frames/` so enqueueing never hard-fails.

## `timestamps` (`TimestampConfig`)

Optional OpenTimestamps sidecars for every uploaded artifact — a cryptographic proof committing each file's SHA-256 to public calendar servers. For each uploaded file tagged as stampable (videos, snapshots, frames, metadata — configurable per-type) the worker runs `ots stamp <local>` after the upload completes, then uploads the resulting `<local>.ots` sidecar to `<remote_key>.ots` in the same bucket folder. Failures are non-fatal — the primary upload is already safe.

Requires the `ots` CLI from the `opentimestamps-client` pip package. Off by default.

| Field | Type | Default | Purpose |
| ----- | ---- | ------- | ------- |
| `enabled` | bool | `false` | Master switch. When false, no sidecars are generated. |
| `ots_binary` | string | `"ots"` | Path (or name on `$PATH`) of the `ots` CLI. |
| `calendar_urls` | list[string] | `[]` | Override calendar servers. Empty = use `ots` built-in defaults (Alice, Bob, Finney, PTB). |
| `timeout_seconds` | float | `30.0` | Max time to wait for a single `ots stamp` invocation. |
| `stamp_videos` | bool | `true` | Stamp `.mp4` clips (raw + overlay). |
| `stamp_snapshots` | bool | `true` | Stamp the event snapshot `.jpg`. |
| `stamp_frames` | bool | `false` | Stamp live frames (off by default — hundreds per event would be noisy on public calendars). |
| `stamp_metadata` | bool | `false` | Stamp the `.json` metadata sidecar. |

### Verifying

Download both the file and its `.ots` sibling and run `ots verify <file>` (or `ots upgrade <file>.ots` first to pin the proof to a Bitcoin block). The sidecar is interoperable with any OpenTimestamps-compatible verifier.

## Loading rules

- `load_config(None)` returns the dataclass defaults.
- `load_config(path)` picks YAML when the extension is `.yaml` / `.yml` (requires PyYAML, included in the `[desktop]` extra), otherwise JSON.
- Top-level keys that aren't one of `source`, `models`, `thresholds`, `timings`, `inventory`, `output`, `upload` are ignored.
- Inside each top-level block, unknown keys are ignored by `_merge_dataclass`. Known keys overwrite the default value as-is (no type coercion is performed).

## Runtime patches

`RuntimeState.update_config(patch)` performs a shallow merge over each block — only keys that already exist on the dataclass are copied. The HTTP API exposes this as `PUT /api/config` (see the [HTTP API reference](http-api.md)). `POST /api/runtime/start` optionally accepts the same patch shape in its body.
