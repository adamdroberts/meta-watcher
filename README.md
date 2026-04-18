# Meta Watcher

Meta Watcher is a person-first video overlay application for Apple Silicon macOS and NVIDIA Linux. It supports USB webcams, RTSP/IP streams, and local video files, keeps a stable empty-scene inventory of objects, switches to people-only overlays when someone enters the scene, and records one annotated clip per occupancy event.

## Architecture

- Local web UI served by FastAPI/uvicorn on `http://127.0.0.1:8765`; the operator console is an Astro static bundle rendered in the browser.
- MJPEG video stream at `/stream.mjpg` displayed inline in the page.
- Shared Python capture and state pipeline.
- Apple Silicon backend: `mlx-community/sam3.1-bf16` through `mlx-vlm`.
- Linux backend: `facebook/sam3.1` through the official `facebookresearch/sam3` package.
- Empty-scene inventory: Qwen2.5-VL proposes labels, SAM 3.1 validates them.
- Person mode: SAM 3.1 detections plus temporal association provide stable person IDs and event recording.

## Install

Base package:

```bash
python3 -m pip install -e .
```

Meta Watcher currently targets `numpy>=1.26,<2` so the Linux CUDA path remains compatible with the official `facebookresearch/sam3` package.

Desktop runtime:

```bash
python3 -m pip install -e ".[desktop]"
```

Apple Silicon Mac:

```bash
python3 -m pip install -e ".[desktop,mac]"
```

Linux with NVIDIA CUDA:

```bash
python3 -m pip install -e ".[desktop,linux]"
```

The Linux extra installs the official [`facebookresearch/sam3`](https://github.com/facebookresearch/sam3) package from the upstream `main` branch, plus supporting runtime dependencies that current `sam3` main expects at import time. That keeps Meta Watcher aligned with the latest SAM 3.1 code, but it is not a reproducible pinned install.

Linux CUDA inference also expects a CUDA-capable PyTorch environment. If you need to install PyTorch separately, follow the current CUDA wheel instructions from PyTorch before installing `.[desktop,linux]`.

The `[desktop]` extra ships the FastAPI backend. It expects the prebuilt Astro bundle at `meta_watcher/web/static/index.html` to exist — it is committed to the repo so `pip install -e .` works out of the box. To rebuild it see "Frontend development" below.

## Model access

- Request gated access to [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1).
- On macOS, install the MLX conversion [`mlx-community/sam3.1-bf16`](https://huggingface.co/mlx-community/sam3.1-bf16).
- For scene inventory, install [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) or an MLX-compatible conversion.

## Run

```bash
meta-watcher --config ./config.default.json
```

If no config is supplied, the app uses built-in defaults. A browser window opens automatically at `http://127.0.0.1:8765`. Useful flags:

- `--no-browser` — do not launch the system browser (useful over SSH).
- `--host 0.0.0.0` / `--port 9000` — bind elsewhere. Defaults are `127.0.0.1:8765`.

### Capture resolution

The webcam source defaults to 1280×720 at 30 fps. Higher resolutions make every downstream stage more expensive (inference, mask compositing, H.264 recording, JPEG streaming), so lower resolutions are usually the right tradeoff. Override via config:

```json
{ "source": { "kind": "webcam", "value": "0", "width": 1920, "height": 1080, "fps": 30 } }
```

On startup the app logs the resolution the camera actually reported back (some devices silently ignore the request). The web HUD and stderr also emit per-stage timings and effective FPS once per second so you can see which stage is the bottleneck.

To run headless (no GUI session available) just use `--no-browser` and point your own browser at the bound URL.

## Frontend development

The Astro source lives under `web/`. Rebuild the static bundle (output lands in `meta_watcher/web/static/`) with:

```bash
cd web
npm install
npm run build
```

For an interactive dev loop, run `npm run dev` (Astro on `:4321`) alongside `meta-watcher --no-browser` and browse to `http://127.0.0.1:4321`. The Astro dev server proxies `/api`, `/stream.mjpg`, and `/frame.jpg` to the backend at `http://127.0.0.1:8765` (override with `META_WATCHER_BACKEND`).

## Tests

The repository uses stdlib `unittest` for logic tests so they can run without the GUI or model dependencies:

```bash
python3 -m unittest discover -s tests
```
