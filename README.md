# Meta Watcher

Meta Watcher is a person-first video overlay application for Apple Silicon macOS and NVIDIA Linux. It supports USB webcams, RTSP/IP streams, and local video files, keeps a stable empty-scene inventory of objects, switches to people-only overlays when someone enters the scene, and records one annotated clip per occupancy event.

- **Backend (all platforms):** FastAPI + uvicorn on `http://127.0.0.1:8765`, a two-thread capture/inference pipeline, and an MJPEG stream at `/stream.mjpg`.
- **Apple Silicon inference:** `mlx-community/sam3.1-bf16` loaded through `mlx-vlm` in a spawned worker process.
- **Linux CUDA inference:** `facebook/sam3.1` loaded through the official `facebookresearch/sam3` package.
- **Operator UI:** static Astro bundle shipped inside the Python package at `meta_watcher/web/static/`.

## Documentation

- Browse the full set: [`docs/README.md`](docs/README.md).
- Getting started: [`docs/guides/getting-started.md`](docs/guides/getting-started.md).
- Architecture: [`docs/architecture/overview.md`](docs/architecture/overview.md).
- HTTP API reference: [`docs/reference/http-api.md`](docs/reference/http-api.md).
- Python API reference: [`docs/reference/python-api.md`](docs/reference/python-api.md).
- Configuration reference: [`docs/reference/configuration.md`](docs/reference/configuration.md).
- CLI reference: [`docs/reference/cli.md`](docs/reference/cli.md).
- Frontend guide: [`docs/guides/frontend.md`](docs/guides/frontend.md).
- Testing: [`docs/guides/testing.md`](docs/guides/testing.md).
- Changelog: [`CHANGELOG.md`](CHANGELOG.md).
- LLM-facing index: [`llms.txt`](llms.txt) and bundle: [`llms-full.txt`](llms-full.txt).

## Install

Base package:

```bash
python3 -m pip install -e .
```

Meta Watcher currently targets `numpy>=1.26,<2` so the Linux CUDA path remains compatible with the official `facebookresearch/sam3` package.

Desktop runtime (FastAPI server + OpenCV capture + PyYAML):

```bash
python3 -m pip install -e ".[desktop]"
```

Apple Silicon Mac (adds `mlx-vlm`):

```bash
python3 -m pip install -e ".[desktop,mac]"
```

Linux with NVIDIA CUDA (adds upstream `facebookresearch/sam3` from `main`, plus `einops`, `pycocotools`, `torch`, `torchvision`):

```bash
python3 -m pip install -e ".[desktop,linux]"
```

The Linux extra pulls `facebookresearch/sam3` from its upstream `main` branch so Meta Watcher stays aligned with the latest SAM 3.1 code, but it is not a reproducible pinned install. Linux CUDA inference also expects a CUDA-capable PyTorch environment; if you need PyTorch separately, follow the current CUDA wheel instructions from PyTorch before installing `.[desktop,linux]`.

Cross-platform alternative detection backends (adds `transformers`, `torch`, `torchvision`, `timm`):

```bash
python3 -m pip install -e ".[desktop,detr]"
```

With the `[detr]` extra installed, set `models.provider` to `"detr-resnet-50"` or `"rt-detrv2"` in the config to swap SAM 3.1 for `facebook/detr-resnet-50` or `jadechoghari/RT-DETRv2`. This is lighter, runs on CUDA / MPS / CPU, but is closed-vocabulary (COCO classes only) and emits bounding boxes without segmentation masks. See [inference backends](docs/architecture/inference-backends.md#transformers-detection-detr--rt-detrv2) for details.

Optional cloud upload (GCP, AWS, OCI). Each extra pulls only the SDK for that provider, so you install just the one you use:

```bash
python3 -m pip install -e ".[desktop,linux,gcp]"   # google-cloud-storage
python3 -m pip install -e ".[desktop,linux,aws]"   # boto3
python3 -m pip install -e ".[desktop,linux,oci]"   # oci
```

Uploads are off by default. Enable them by setting `upload.enabled: true` and `upload.bucket: ...` in the config (see [configuration reference](docs/reference/configuration.md#upload-uploadconfig)).

### OCI Object Storage retention

For security-camera / evidence use cases, pair the upload bucket with an Oracle Cloud Object Storage **retention rule** so uploaded clips, snapshots, metadata, and OpenTimestamps `.ots` sidecars cannot be altered or deleted inside the retention window — not even by users with write permissions, and not by Meta Watcher itself if `delete_after_upload` is somehow flipped on.

Create a 90-day retention rule on the bucket with the OCI CLI:

```bash
oci os retention-rule create \
  --bucket-name "security-camera-retention" \
  --display-name "compliance-hold" \
  --time-amount 90 \
  --time-unit DAYS
```

Once the rule is in place, every object Meta Watcher uploads to that bucket — `{prefix}{event_id}/{event_id}.mp4`, `.jpg`, `.json`, `.ots`, and the live-frame JPEGs under `frames/` — is locked against modification or deletion for 90 days from the time it was written. Combined with the OpenTimestamps sidecars (`timestamps.enabled: true`), the bucket becomes a tamper-evident + tamper-resistant archive: the `.ots` proof pins each file's SHA-256 to Bitcoin, and the retention rule prevents anyone from replacing the bytes those proofs refer to.

Other useful options:

- Add `--time-unit YEARS` (or `DAYS`) to match your legal-hold window.
- Omit `--time-amount` and `--time-unit` to create an **indefinite** rule — locks objects until the rule itself is deleted. Indefinite rules can later be converted to time-bound rules with `oci os retention-rule update`.
- List the rules on a bucket with `oci os retention-rule list --bucket-name <name>`.
- See the OCI docs on [retention rules](https://docs.oracle.com/en-us/iaas/Content/Object/Tasks/usingretentionrules.htm) for concepts like the 14-day "lock delay" before a rule becomes immutable.

The retention rule is configured on the bucket in OCI — there is no corresponding Meta Watcher config field. Meta Watcher's uploader just `PUT`s objects; the rule takes effect on Oracle's side.

The `[desktop]` extra expects the prebuilt Astro bundle at `meta_watcher/web/static/index.html` to exist. The bundle is committed to the repo so `pip install -e .` works out of the box. To rebuild it, see "Frontend development" below.

## Model access

- Request gated access to [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1).
- On macOS, install the MLX conversion [`mlx-community/sam3.1-bf16`](https://huggingface.co/mlx-community/sam3.1-bf16).
- Inventory labels come from the `inventory.labels` config field and SAM 3.1 validates them with open-vocabulary text prompts.

## Run

```bash
meta-watcher --config ./config.default.json
```

If no config is supplied, the app uses built-in defaults. The web UI is served at `http://127.0.0.1:8765`. Useful flags:

- `--host 0.0.0.0` / `--port 9000` — bind elsewhere. Defaults are `127.0.0.1:8765`.
- `--open-browser` — auto-open the system browser once uvicorn is listening (off by default).

### Capture resolution

The webcam source defaults to 1280×720 at 30 fps. Higher resolutions make every downstream stage more expensive (inference, mask compositing, H.264 recording, JPEG streaming), so lower resolutions are usually the right tradeoff. Override via config:

```json
{ "source": { "kind": "webcam", "value": "0", "width": 1920, "height": 1080, "fps": 30 } }
```

On startup the webcam source logs the resolution the camera actually reported back (some devices silently ignore the request). The web HUD and stderr also emit per-stage timings and effective FPS once per second so you can see which stage is the bottleneck.

To run headless (no GUI session available) just omit `--open-browser` and point your own browser at the bound URL.

## Frontend development

The Astro source lives under `web/`. Rebuild the static bundle (output lands in `meta_watcher/web/static/`) with:

```bash
cd web
npm install
npm run build
```

For an interactive dev loop, run `npm run dev` (Astro on `:4321`) alongside `meta-watcher` and browse to `http://127.0.0.1:4321`. The Astro dev server proxies `/api`, `/stream.mjpg`, and `/frame.jpg` to the backend at `http://127.0.0.1:8765` (override with `META_WATCHER_BACKEND`).

See [`docs/guides/frontend.md`](docs/guides/frontend.md) for the full frontend workflow.

## Tests

The repository uses stdlib `unittest` for logic tests so they can run without the GUI or model dependencies:

```bash
python3 -m unittest discover -s tests
```

See [`docs/guides/testing.md`](docs/guides/testing.md) for what each test module covers and how to extend them.

## Repository layout

```
meta_watcher/        Python package (capture → inference → overlay → recording → web)
  app.py             meta-watcher CLI entry point
  config.py          AppConfig dataclasses and load_config
  core.py            Shared types, TrackManager, StreamStateMachine, ClipRecorder
  inference.py       InferenceProvider ABC + MLX/CUDA implementations
  sources.py         VideoSource implementations for webcam, RTSP, file
  overlay.py         render_overlay: draws masks, bboxes, HUD
  pipeline.py        StreamProcessor + StreamRuntime (producer/consumer threads)
  web/
    server.py        FastAPI app (build_app)
    state.py         RuntimeState thread-safe container for the runtime + MJPEG buffer
    static/          Prebuilt Astro bundle (committed)
web/                 Astro source for the operator UI (build output goes into meta_watcher/web/static/)
tests/               Stdlib unittest suite (core, pipeline, inference, app/web)
bin/meta-watcher     Bash launcher (used when running from the repo without install)
config.default.json  Example config used by docs and manual runs
```

## License

MIT. See [`pyproject.toml`](pyproject.toml) for package metadata.
