# Python API reference

Meta Watcher is primarily a CLI + web app, but its modules are importable as a library. This page lists the public surfaces that embedders should treat as stable within a `0.1.x` release. Internal helpers (names starting with `_`) are not covered here; see `meta_watcher/*.py` directly if you need them.

## `meta_watcher.config`

Config dataclasses and loader.

```python
from meta_watcher.config import AppConfig, default_config, load_config
```

- `AppConfig(source, models, thresholds, timings, inventory, output)` — root dataclass. `to_dict()` returns an `asdict`-serialized dict.
- `SourceConfig`, `ModelConfig`, `ThresholdConfig`, `TimingConfig`, `InventoryConfig`, `OutputConfig` — block-level dataclasses. Field-level defaults are listed in [configuration.md](configuration.md).
- `default_config() -> AppConfig` — built-in defaults, used when no config path is provided.
- `load_config(path: str | Path | None) -> AppConfig` — loads JSON (anything not ending in `.yaml`/`.yml`) or YAML. Raises `FileNotFoundError` if `path` is set and missing.

## `meta_watcher.core`

Shared types and logic.

- `VideoFrame(image, timestamp, frame_index, source_id, fps=None)` — canonical frame type. `.width`, `.height` come from `image.shape`.
- `Detection(label, confidence, bbox, mask=None, track_id=None, metadata={})` — canonical detection type. `bbox` is `(x1, y1, x2, y2)` in pixel space.
- `InventoryItem(label, confidence, samples, last_seen)` — inventory list entry.
- `PipelineSnapshot(mode, frame_index, source_id, overlay, people, inventory_detections, inventory_items, inventory_active, recording_active, completed_clips, status_text, timings, effective_fps)` — output of `StreamProcessor.process_frame`.
- `TrackManager(min_iou=0.3, max_missed_frames=12, smoothing=0.35, min_area_ratio=0.002)` — assigns stable IDs to per-frame detections. `update(detections, frame)` returns the tracked detections for that frame; `reset()` clears state.
- `StreamStateMachine(person_confirmation_frames=3, empty_scene_rescan_seconds=5.0)` — mode transition logic. `observe(timestamp, people_present, *, auto_rescan_enabled)` returns a `StreamDecision(mode, event_started, event_finished, inventory_active, inventory_reset)`. `request_manual_rescan()` latches a rescan.
- `ClipRecorder(output_dir, *, pre_roll_seconds=3.0, post_roll_seconds=2.0, sink_factory=None, mode="raw")` — pre/post-roll video recorder. `mode` ∈ `{"raw", "overlay", "both"}` selects which channels are maintained. Exposes `push_frame` (raw channel, producer thread), `push_overlay_frame` (overlay channel, consumer thread), `start_event`, `finish_event`, `add_person_ids`, `drain_completed`, and `recording_active`. Either `push_*` call is a no-op when its channel is disabled. Swappable sinks via `sink_factory(path, size, fps) -> ClipSink`.
- `OpenCvClipSink(path, size, fps)` — default sink using OpenCV's `VideoWriter`.
- `EventArtifact(clip_path, metadata_path, started_at, ended_at, person_ids, snapshot_path=None, overlay_clip_path=None)` — returned by `drain_completed`. `overlay_clip_path` is only set when the recorder ran in `"both"` mode; in `"overlay"` mode the overlay MP4 is the canonical `clip_path`.
- `clamp_bbox`, `box_area`, `iou`, `normalize_label`, `is_people_label`, `merge_overlapping_detections` — helpers. `PEOPLE_PROMPTS` and `PEOPLE_LABELS` define the label vocabulary for person detection.

## `meta_watcher.sources`

Video capture.

- `VideoSource` (ABC) with `source_id`, `open()`, `read() -> VideoFrame | None`, `close()`.
- `OpenCvVideoSource(source_id, capture_value, *, live)` — base class wrapping `cv2.VideoCapture`. RGB-converted on every read.
- `WebcamSource(value, *, width, height, fps)` — retries opening across a candidate list, requests the desired resolution, and logs actual vs. requested.
- `RtspSource(url)`, `FileSource(path)` — thin subclasses for RTSP URLs and local files.
- `build_source(config: SourceConfig) -> VideoSource` — factory used by the runtime.
- `list_webcams() -> list[WebcamDevice]` — platform-aware enumeration with V4L2 + macOS `system_profiler` paths.

## `meta_watcher.inference`

SAM 3.1 providers.

- `InferenceProvider` (ABC) — `warmup`, `detect_text_prompts(frame, prompts)`, `start_tracking(frame, prompts)`, `track_next(frame)`, `shutdown`.
- `MlxSam31Provider(model_id)` — in-process MLX implementation; imports `mlx_vlm` lazily inside `warmup`.
- `MlxSubprocessSam31Provider(model_id, timeout_seconds=120.0)` — spawns a `multiprocessing` worker running `MlxSam31Provider`. Default on Apple Silicon.
- `CudaSam31Provider(model_id)` — CUDA implementation using `facebookresearch/sam3`. Applies the `sam3.perflib.fused.addmm_act` fp32 patch on first warmup.
- `build_provider(models: ModelConfig) -> InferenceProvider` — platform router used by the runtime.

## `meta_watcher.overlay`

- `render_overlay(frame, *, detections, inventory, mode, inventory_active, recording_active, status_text, hud_timings=None, hud_fps=0.0) -> np.ndarray` — draws masks, bboxes, labels, HUD, and the inventory list on an RGB frame. Returns RGB.

## `meta_watcher.pipeline`

- `StreamProcessor(config, provider, recorder)` — pure per-frame processor. `process_frame(frame) -> PipelineSnapshot`. Supports `request_manual_rescan()`, `set_recording_enabled(bool)`, and `update_labels(list[str])`.
- `StreamRuntime(source, processor, *, on_snapshot, on_error, on_raw_frame=None, queue_size=2)` — producer/consumer threads wrapping a source and a processor. `start()`, `stop()`, `request_manual_rescan()`, `set_recording_enabled(bool)`.

## `meta_watcher.web`

- `RuntimeState(config=None, *, provider_factory=None, source_factory=None, jpeg_quality=80)` — thread-safe wrapper owning the `StreamRuntime`, the latest JPEG, and the cumulative completed-clip list. Factory overrides are how tests inject fake providers and sources.
- `build_app(state: RuntimeState) -> FastAPI` — constructs the FastAPI app with all endpoints documented in [http-api.md](http-api.md).

## `meta_watcher.app`

- `main(argv: list[str] | None = None) -> int` — CLI entry. Used by the `meta-watcher` console script.
