# Getting started

This guide walks from a fresh checkout to a running Meta Watcher instance with a custom config.

## 1. Prerequisites

- Python 3.12 or newer (`pyproject.toml` requires `>=3.12`).
- A supported platform:
  - Apple Silicon macOS (arm64) with [`mlx-vlm`](https://pypi.org/project/mlx-vlm/) available, or
  - Linux x86_64 with an NVIDIA GPU and a CUDA-capable PyTorch install.
- A webcam, RTSP/IP stream, or local video file.
- Access to the SAM 3.1 model weights:
  - `facebook/sam3.1` on Hugging Face (gated — request access).
  - `mlx-community/sam3.1-bf16` for MLX on macOS.

## 2. Install

Pick the extras that match your platform. Both sets layer on top of the `desktop` extra, which installs FastAPI, uvicorn, OpenCV, and PyYAML.

```bash
# Base package only (no GUI, no inference)
python3 -m pip install -e .

# Desktop runtime on any platform (adds FastAPI + OpenCV)
python3 -m pip install -e ".[desktop]"

# Apple Silicon macOS
python3 -m pip install -e ".[desktop,mac]"

# Linux with NVIDIA CUDA (pulls facebookresearch/sam3 from main)
python3 -m pip install -e ".[desktop,linux]"
```

See the [configuration reference](../reference/configuration.md) for how the Linux extra is pinned.

## 3. Run with defaults

```bash
meta-watcher
```

This starts uvicorn at `http://127.0.0.1:8765` with the built-in defaults, which mean:

- Webcam source at index `0`, 1280×720 @ 30 fps.
- MLX provider on macOS, CUDA provider elsewhere.
- Empty inventory labels (so inventory mode is effectively a no-op until you set some).
- Recordings written to `./recordings/`.

Open the URL in your browser and use the right-hand control panel to set source, thresholds, and inventory labels, then click **Start**.

## 4. Run with a config file

```bash
meta-watcher --config ./config.default.json
```

The shipped `config.default.json` is the canonical example:

```json
{
  "source": { "kind": "webcam", "value": "0", "width": 1280, "height": 720, "fps": 30 },
  "models": { "mac_sam_model_id": "mlx-community/sam3.1-bf16",
              "linux_sam_model_id": "facebook/sam3.1",
              "inference_max_side": 960,
              "inference_interval_ms": 200 },
  "thresholds": { "person_confidence": 0.3, "inventory_confidence": 0.25,
                  "overlap_iou": 0.5, "tracking_iou": 0.3, "min_area_ratio": 0.002 },
  "timings": { "empty_scene_rescan_seconds": 5.0, "person_confirmation_frames": 3,
               "pre_roll_seconds": 0.5, "post_roll_seconds": 2.0 },
  "inventory": { "labels": ["chair", "table", "laptop", "book", "cup", "bottle", "plant"],
                 "auto_rescan": true },
  "output": { "directory": "recordings" }
}
```

YAML with the same top-level keys is also accepted.

## 5. Read the overlay

When the pipeline is running, the video frame carries an in-frame HUD:

- **Mode** — `inventory`, `person_present`, or `cooldown`.
- **Inventory scan** — `on` while SAM 3.1 is validating inventory labels, `frozen` otherwise.
- **Recording** — `active` whenever a clip is being written.
- **HUD timings** — a compact `fps / i:<infer> o:<overlay> r:<recorder>` line (see the [pipeline internals](../internals/pipeline.md)).

The same state is mirrored in the right-hand operator panel in the UI and the `/api/snapshot` HTTP endpoint. See the [operator UI guide](operator-ui.md) and the [HTTP API reference](../reference/http-api.md).

## 6. Inspect recordings

Every confirmed occupancy event becomes two files in the output directory:

- `{source_id}_{UTC timestamp}.mp4` — the annotated clip, including the pre-roll buffer.
- `{source_id}_{UTC timestamp}.json` — metadata: `clip_path`, `source_id`, `inventory` (labels at event start), `started_at`, `ended_at`, and `person_ids` (track IDs that appeared during the event).

The UI lists completed clip paths under **Completed clips** in the control panel.

## 7. Next steps

- [Operate the web UI](operator-ui.md) — end-to-end walk through the controls.
- [Architecture overview](../architecture/overview.md) — understand the threading and frame flow before tuning.
- [Troubleshooting](troubleshooting.md) — for webcam permission issues, model gating errors, or low FPS.
