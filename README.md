# Meta Watcher

Meta Watcher is a person-first desktop video overlay application for Apple Silicon macOS and NVIDIA Linux. It supports USB webcams, RTSP/IP streams, and local video files, keeps a stable empty-scene inventory of objects, switches to people-only overlays when someone enters the scene, and records one annotated clip per occupancy event.

## Architecture

- PySide6 desktop shell with a live video panel and operator controls.
- Shared Python capture and state pipeline.
- Apple Silicon backend: `mlx-community/sam3.1-bf16` through `mlx-vlm`.
- Linux backend: `facebook/sam3.1` through the official `sam3` codebase.
- Empty-scene inventory: Qwen2.5-VL proposes labels, SAM 3.1 validates them.
- Person mode: SAM 3.1 detections plus temporal association provide stable person IDs and event recording.

## Install

Base package:

```bash
python3 -m pip install -e .
```

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

## Model access

- Request gated access to [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1).
- On macOS, install the MLX conversion [`mlx-community/sam3.1-bf16`](https://huggingface.co/mlx-community/sam3.1-bf16).
- For scene inventory, install [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) or an MLX-compatible conversion.

## Run

```bash
meta-watcher --config ./config.default.json
```

If no config is supplied, the app uses built-in defaults.

## Tests

The repository uses stdlib `unittest` for logic tests so they can run without the GUI or model dependencies:

```bash
python3 -m unittest discover -s tests
```
