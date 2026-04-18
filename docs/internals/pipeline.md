# Pipeline internals

This page dives into `meta_watcher/pipeline.py`. Read the [architecture overview](../architecture/overview.md) first for the component picture.

## `StreamProcessor.process_frame`

Each call performs the following stages, in order, timed in milliseconds and reported on `PipelineSnapshot.timings`:

1. **Inference gating.** If `(frame.timestamp - _last_inference_ts) >= inference_interval_ms / 1000`, run the provider; otherwise reuse the previous detections. Frames between inferences still get a fresh tracker update using the cached detections, which is how the recorded overlay keeps animating at full frame rate.
2. **Downscaling.** `_downscale_for_inference` uses OpenCV `cv2.resize` (INTER_AREA) to fit the longest side of the frame into `inference_max_side`. The scale factor is recorded so `_upscale_detections` can map bboxes back into the original coordinate space.
3. **Provider call.**
   - In `inventory` mode: `detect_text_prompts(frame, PEOPLE_PROMPTS)` first. Inventory labels are a separate call (`detect_text_prompts(frame, inventory_labels)`) only if no people are present.
   - In `person_present` / `cooldown`: `track_next(frame)` is preferred once `_tracking_live` is true, otherwise the person detection call.
4. **Person normalization + tracking.** `_normalize_people` drops anything below `thresholds.person_confidence`, renames the label to `person`, and merges duplicate boxes. `TrackManager.update` assigns stable IDs.
5. **State machine observation.** `StreamStateMachine.observe(timestamp, people_present, auto_rescan_enabled=config.inventory.auto_rescan)` returns the decision for this frame.
6. **Event start hook.** If `decision.event_started`, the processor calls `provider.start_tracking(frame, PEOPLE_PROMPTS)` so providers can reset their internal tracking state. It also clears `_event_person_ids` and resets the inference interval so the next frame re-runs inference immediately.
7. **Inventory detection.** Runs only if we're in inventory mode, inventory is active, no people are present, and there are configured labels. Filters by `thresholds.inventory_confidence`.
8. **Overlay.** `render_overlay` receives the current detections, inventory list, mode flags, HUD timings, and effective fps.
9. **Recorder drive.** On `event_started` → `recorder.start_event(frame, inventory_labels)`. On every frame with an active event → `recorder.add_person_ids(sorted(_event_person_ids))`. On `event_finished` → `recorder.finish_event(timestamp, sorted(_event_person_ids))`. `drain_completed` pulls any artifacts that have been closed in the meantime.
10. **Timings accounting.** Rolling windows of the last 30 samples per stage; average latencies and effective fps are surfaced on every snapshot and logged once per second.

## Downscaling and upscaling

```python
def _downscale_for_inference(frame, max_side):
    if max_side <= 0 or max(frame.height, frame.width) <= max_side:
        return frame, 1.0
    ratio = max_side / max(frame.height, frame.width)
    return VideoFrame(image=resized, ...), 1.0 / ratio
```

- `cv2` is imported lazily so the module still works without OpenCV (tests that don't need downscaling won't pay the import).
- `_upscale_detections` multiplies bbox coordinates by the returned scale factor and clamps to frame bounds.

## Rate limiting

- `inference_interval_ms` is a *lower bound* between provider calls, not a fixed cadence. If inference itself takes longer than the interval, the next call happens as soon as inference finishes.
- Between calls, `_cached_people_candidates` and `_cached_inventory_detections` are reused. `_cached_inventory_detections` is cleared when we leave `inventory` mode to avoid stale overlays.
- Tests (`test_pipeline.py::test_inference_rate_limited_by_interval`) assert the gating math at 100 ms frame spacing and 500 ms interval: 6 frames produce 2 provider calls.

## `StreamRuntime`

- Two daemon threads, `meta-watcher-source` (producer) and `meta-watcher-pipeline` (consumer).
- `_put_latest` replaces the queued frame if the queue is full (size 2). This keeps consumer latency proportional to inference time instead of building up a backlog.
- `on_raw_frame` is invoked on the producer thread before the queue push so `RuntimeState` can display a live preview during model warmup.
- `stop()` sets the stop event, joins both threads with a 2-second timeout, then closes the source and calls `processor.provider.shutdown()`. The finally guarantees provider shutdown even if the consumer raised.
- If the source reports `live=False` (file playback), the producer sleeps `1/fps` seconds between frames so playback runs at the file's recorded rate.

## Timing windows

`_stage_samples` and `_frame_wall_times` are `collections.deque(maxlen=30)`. Short windows keep the HUD responsive after transitions (you see the real cost of the current mode quickly), at the cost of ignoring older history. If you need longer-horizon metrics, tee the timings off `PipelineSnapshot.timings` in a consumer rather than enlarging the windows.

## Extending

- New modes: extend `StreamStateMachine` with explicit transitions and surface the new mode through `decision.mode`. `StreamProcessor` uses the mode string directly in `visible_detections = tracked_people if decision.mode in {"person_present", "cooldown"} else inventory_detections` — update both the state machine and the processor to keep them in sync.
- New detection types: add a branch in `StreamProcessor.process_frame` for the extra provider call, then extend `PipelineSnapshot` with a new field so consumers can read it. Remember to update `render_overlay` for display and `RuntimeState.snapshot_payload` for the HTTP surface.
- Swapping inference providers: pass a custom `provider_factory` to `RuntimeState` (see `tests/test_app.py::WebServerTests._build_client` for the pattern).
