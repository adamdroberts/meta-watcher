---
name: pipeline-core
description: Use when modifying Meta Watcher's capture, tracking, state machine, recorder, or overlay logic. Trigger on changes to `meta_watcher/core.py`, `pipeline.py`, `overlay.py`, or `sources.py`, or when the user asks to change modes, recording behavior, detection filtering, or frame flow.
---

# Pipeline core skill

## When to use

- Editing `meta_watcher/core.py`, `meta_watcher/pipeline.py`, `meta_watcher/overlay.py`, or `meta_watcher/sources.py`.
- Adjusting inventory / person-present / cooldown transitions, track IDs, pre-roll/post-roll behavior, or overlay rendering.
- Wiring a new per-frame signal into `PipelineSnapshot` or `render_overlay`.

## When not to use

- Changes that are purely in the HTTP/JSON surface — use [web-surface](../web-surface/SKILL.md).
- Changes to the Astro UI — use [frontend-astro](../frontend-astro/SKILL.md).
- Changes to SAM 3.1 providers — use [inference-backends](../inference-backends/SKILL.md).

## Rules

1. **Preserve the producer/consumer contract.** `StreamRuntime` runs two daemon threads. `push_frame` belongs on the producer; `start_event`, `finish_event`, `drain_completed`, `process_frame` belong on the consumer. Do not call consumer methods from the producer or vice versa.
2. **Downscale before inference, upscale after.** New inference call sites must use `_downscale_for_inference` + `_upscale_detections` so configurable `inference_max_side` keeps working.
3. **Rate-limit new inference calls.** New provider calls gated by `inference_interval_ms` should reuse the cached detections between calls, the same way `_cached_people_candidates` and `_cached_inventory_detections` do.
4. **Events end on the producer.** Do not call `sink.close()` directly. Use `ClipRecorder.finish_event` + `end_requested_at` so the producer flushes the post-roll.
5. **Surface new signals on `PipelineSnapshot`.** Extending the processor means extending both `PipelineSnapshot` (dataclass in `core.py`) and `RuntimeState.snapshot_payload` (`web/state.py`) if the UI needs the signal. Update the test fakes to populate the new field.
6. **Keep logging one line per second.** The stderr HUD line in `_maybe_log_timings` is intentionally throttled to 1 Hz — do not change the cadence without a good reason.

## Workflow

1. Read [pipeline internals](../../docs/internals/pipeline.md) and [state machine](../../docs/architecture/state-machine.md).
2. Locate the narrow change site — usually `StreamProcessor.process_frame`, `StreamStateMachine.observe`, `ClipRecorder`, `TrackManager.update`, or `render_overlay`.
3. Add or update a test in `tests/test_pipeline.py` or `tests/test_core.py` using the existing `FakeProvider` / `MemorySink` patterns.
4. Run `python3 -m unittest discover -s tests` and make sure the related tests pass.
5. Update docs that describe the changed behavior: [pipeline internals](../../docs/internals/pipeline.md), [state machine](../../docs/architecture/state-machine.md), [recording](../../docs/architecture/recording.md), or [overview](../../docs/architecture/overview.md).
6. Append a `CHANGELOG.md` entry under `[Unreleased]`.

## Quick reference

- Mode values: `"inventory"`, `"person_present"`, `"cooldown"`. Snapshot exposes `mode = "idle"` only when nothing has produced a frame yet (handled by `RuntimeState`, not the processor).
- `Detection.bbox` is `(x1, y1, x2, y2)` in the *original frame's* pixel space.
- Track IDs look like `person-<N>`; `TrackManager.reset()` is called at the end of each event.
- `PipelineSnapshot.timings` stage keys: `inference`, `overlay`, `recorder`, `total`. Add a new key only with a matching `_TIMING_STAGES` update and a tests update.

## Docs this skill routes into

- [docs/architecture/overview.md](../../docs/architecture/overview.md)
- [docs/architecture/state-machine.md](../../docs/architecture/state-machine.md)
- [docs/architecture/recording.md](../../docs/architecture/recording.md)
- [docs/internals/pipeline.md](../../docs/internals/pipeline.md)
- [docs/reference/configuration.md](../../docs/reference/configuration.md)
