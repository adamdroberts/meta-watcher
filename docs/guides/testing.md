# Testing

The repository uses the stdlib `unittest` runner so tests run without pytest installed. Model and camera dependencies are stubbed with fakes so the suite works on any machine.

## Run

```bash
python3 -m unittest discover -s tests
```

Optional, if you installed the `[dev]` extra:

```bash
pytest tests
```

The `[dev]` extra only adds `pytest`; there are no pytest-specific fixtures in use, so both runners exercise the same tests.

## Modules

### `tests/test_core.py`

Covers the non-ML building blocks:

- `CoreTests.test_normalize_label` — label normalization rules (stemming, article stripping).
- `CoreTests.test_track_manager_reuses_ids` — `TrackManager` keeps stable IDs across near-overlapping boxes.
- `CoreTests.test_state_machine_confirm_cooldown_and_manual_rescan` — full `inventory → person_present → cooldown → inventory` cycle plus the manual-rescan latch.
- `CoreTests.test_clip_recorder_uses_preroll_and_postroll` — pre-roll flush, post-roll delay, and metadata sidecar contents, using an in-memory sink.
- `CoreTests.test_webcam_source_only_probes_requested_index` — when the user selects a specific webcam index, the source trusts it and does not probe other devices.
- `CoreTests.test_webcam_source_auto_uses_enumeration` — when the source value is `auto`, the Linux path uses `_list_linux_webcams` to pick candidates.

### `tests/test_pipeline.py`

Exercises `StreamProcessor` end-to-end with a `FakeProvider` and the in-memory sink:

- `PipelineTests.test_configured_labels_drive_inventory_detection` — inventory labels come from config and the provider is asked for them during inventory mode; the event flow records a single clip.
- `PipelineTests.test_empty_label_config_skips_inventory_detection` — empty label list never triggers a provider call for inventory prompts.
- `PipelineTests.test_update_labels_takes_effect_immediately` — `StreamProcessor.update_labels` affects the very next inventory pass.
- `PipelineTests.test_inference_receives_downscaled_frame_and_bboxes_upscale` — `inference_max_side` downscales the frame fed to the provider and detections come back in original-resolution coordinates.
- `PipelineTests.test_inference_rate_limited_by_interval` — `inference_interval_ms` gates how often the provider is called per second.
- `PipelineTests.test_snapshot_carries_stage_timings` — every snapshot has non-negative `inference`/`overlay`/`recorder`/`total` timings and an `effective_fps` that becomes positive after at least two samples.

### `tests/test_inference.py`

Validates the Linux CUDA provider's install-time and warmup contracts without actually loading SAM 3.1:

- `LinuxInferenceInstallTests.test_base_dependencies_keep_numpy_in_sam3_compatible_range` — guards against someone bumping numpy to 2.x.
- `LinuxInferenceInstallTests.test_linux_extra_installs_sam3_from_main` — guards the Linux extra declaration.
- `LinuxInferenceInstallTests.test_cuda_provider_reports_linux_extra_when_sam3_is_missing` — the import-failure error message must tell the operator to reinstall.
- `LinuxInferenceInstallTests.test_cuda_provider_forces_float32_on_warmup` — floating tensors are cast to fp32, complex tensors keep their dtype, and the model is moved to `train(False)`.
- `LinuxInferenceInstallTests.test_move_inputs_to_model_casts_only_floating_tensors` — input-movement helper respects `is_floating_point()`.

### `tests/test_app.py`

Tests the web layer with fakes:

- `RuntimeStateTests` — initial snapshot shape, config patch merging, snapshot-callback JPEG encoding, and start/stop against a scripted source + fake provider.
- `WebServerTests` — FastAPI endpoints via `fastapi.testclient.TestClient`. Skipped if `fastapi` is not installed. Covers `GET /api/config`, `PUT /api/config`, `GET /api/snapshot`, `POST /api/runtime/rescan`, and `GET /frame.jpg`.

## Writing new tests

- Do not add tests that require a real camera, GPU, or gated Hugging Face access. Use `FakeProvider` / `MemorySink` / `_ScriptedSource` patterns that already exist.
- Keep tests deterministic. Pass explicit timestamps (`make_frame(i, timestamp=...)`) rather than relying on wall time.
- When testing new recorder behavior, use `ClipRecorder(..., sink_factory=MemorySinkFactory())` so you get hold of the in-memory sink list for assertions.
- For new HTTP endpoints, extend `WebServerTests` rather than spinning up uvicorn. Use `_build_client()` to get a `(TestClient, RuntimeState)` tuple with fakes already installed.
