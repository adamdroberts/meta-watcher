# Changelog

Notable changes to Meta Watcher. Dates are ISO-8601.

## [Unreleased]

### Added

- Dedicated `/settings` page rendering every `AppConfig` field grouped by section (`source`, `models`, `thresholds`, `timings`, `inventory`, `output`, `upload`), plus a config-file selector that enumerates `*.json` / `*.yaml` / `*.yml` under the repo root and lets operators swap between them from the UI.
- Three new config endpoints: `GET /api/config/files`, `POST /api/config/switch`, `POST /api/config/save`. Saves are atomic (`tempfile + os.replace`) with `indent=2` and a trailing newline. YAML save targets redirect to a JSON sibling to avoid destroying comments/ordering.
- `meta_watcher.config.save_config`, `list_config_files`, `build_config_from_dict`, and `repo_root` helpers.
- OpenTimestamps integration via a new `timestamps` config section and `meta_watcher/timestamp.py` helper. When enabled, the upload worker runs `ots stamp` on each uploaded artifact (per-type toggles for videos, snapshots, frames, metadata) and pushes the resulting `.ots` sidecar to `<remote_key>.ots` in the same bucket folder. Requires the `ots` CLI from the `opentimestamps-client` pip package. The uploader logs the sidecar's local and remote paths loudly so it's obvious where the proof lives — a deliberate fix for the CLI's silent-sidecar surprise.
- Eager per-event uploads: the event snapshot JPEG is uploaded the moment the event starts, and a live overlay frame is enqueued every 0.5s while the pipeline is recording. All event artifacts upload under `{prefix}{event_id}/...` so each event gets its own bucket folder.
- OCI upload config supports `~/.oci/config` expansion, `namespace`, and `profile` fields so operators don't need to hand-edit bucket URIs.
- `output.recording_mode` config field with values `"raw"` (default, matches prior behavior), `"overlay"`, and `"both"`. `"overlay"` records the annotated frames from `render_overlay`; `"both"` writes two MP4s per event — `{stem}.mp4` (raw) and `{stem}.overlay.mp4`. `ClipRecorder` now maintains independent pre-roll buffers per channel and finalizes the event only after every enabled channel has closed. `EventArtifact` gained an `overlay_clip_path` field and the JSON sidecar gained `raw_clip_path` / `overlay_clip_path`.
- Optional `facebook/detr-resnet-50` and `jadechoghari/RT-DETRv2` object detection backends via a new `models.provider` config field (`"sam3.1"` default, `"detr-resnet-50"`, `"rt-detrv2"`) and a new `[detr]` install extra. The transformers-backed provider is cross-platform (CUDA / MPS / CPU), filters COCO class outputs against the caller's prompts via `_matches_any_prompt`, and does not produce segmentation masks.
- Documentation set under `docs/` with guides, reference, architecture, and internals pages, plus Mermaid diagrams for component layout, frame lifecycle, state machine, recording, UI start/stop, and MLX subprocess RPC.
- `llms.txt` LLM-facing index and `llms-full.txt` aggregated bundle at the repo root.
- Repo-local agent skills under `.claude/skills/` for the core pipeline, HTTP surface, frontend, and inference backends, with an index routing each skill to its docs.

### Changed

- Live frames + their OpenTimestamps `.ots` sidecars are now written to `output.directory/{event_id}/frames/` (previously `/tmp/meta-watcher-frames/`) and respect the global `upload.delete_after_upload` policy (default: keep). The local filesystem now mirrors the bucket layout, so operators retain a full on-disk record of each event alongside the clip, snapshot, and metadata.
- `PUT /api/config` now rejects a known top-level section carrying a non-dict value with HTTP 400 and `{"error": "..."}`. Unknown top-level keys still silently drop (previous behavior).
- `README.md` now links to the docs set and LLM artifacts.

### Removed

- Dead `--no-browser` CLI flag (the default is already no browser).
