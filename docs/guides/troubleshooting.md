# Troubleshooting

## Cameras

### No webcam is detected on the dropdown

- Click the ↻ button next to the webcam select to re-call `/api/devices/webcams`.
- On Linux, confirm `/dev/video*` entries exist and that your user is in the `video` group. `list_webcams()` filters out metadata-only devices by reading `/sys/class/video4linux/videoN/device/capabilities`; check the bitfield if your device is missing.
- On macOS, Camera permission is enforced per-binary. If the enumeration returns nothing, the Python launcher may not have Camera access. Reset it under *System Settings → Privacy & Security → Camera* (toggle off and back on for Python). The backend also emits a hint message on failure.

### "Failed to open any webcam device" on macOS

`WebcamSource` raises this when every candidate index fails `isOpened()` or the initial `.read()` retries exhaust. Causes seen in practice:

- Camera is in use by another app.
- macOS permission prompt was dismissed; reset it as above.
- The index provided to `source.value` refers to a virtual or metadata-only device.

### Webcam opens but frames look wrong or too low-res

The camera may silently ignore the requested `width`/`height`/`fps`. Meta Watcher logs actual values at startup:

```
[meta-watcher] webcam index=0 requested=1920x1080@30 actual=1280x720@30.0
```

If `actual` is smaller than `requested`, drop the request to match the device's supported modes.

## Inference / model access

### `mlx-vlm is required for Apple Silicon inference.`

Raised by `MlxSam31Provider.warmup`. Install the Mac extra:

```bash
python3 -m pip install -e ".[desktop,mac]"
```

### `Linux CUDA inference requires the official facebookresearch/sam3 package.`

Raised by `CudaSam31Provider.warmup` when `import sam3` fails. Install the Linux extra:

```bash
python3 -m pip install -e ".[desktop,linux]"
```

This pulls from `facebookresearch/sam3` main, so a clean `pip install` is required after the upstream branch updates if you want the latest code.

### `mat1 and mat2 must have the same dtype: float vs bfloat16`

This is the fused `addmm_act` mismatch described in [inference backends](../architecture/inference-backends.md). If you see it, the `_patch_sam3_fused_addmm_to_fp32` patch did not run before the model's first forward. Check that `CudaSam31Provider.warmup` actually ran on your call path — the patch lives inside warmup — and that no other code path is loading `sam3` before the provider does.

### Hugging Face 403 / gated model

`facebook/sam3.1` requires model access approval. Request it from the model page, then authenticate with `huggingface-cli login` or set `HF_TOKEN` in your environment before starting the server.

## Performance

### FPS is low

Read the per-second log line Meta Watcher prints to stderr:

```
[meta-watcher] fps=17.5 infer=58.0ms overlay=6.0ms recorder=1.0ms total=64.9ms mode=inventory frame=1280x720
```

- If `infer` is the dominant stage, drop the capture resolution or lower `models.inference_max_side`, or raise `models.inference_interval_ms`.
- If `overlay` dominates, the frame is probably very large; the overlay does PIL compositing at full resolution.
- If `recorder` dominates, the OpenCV `VideoWriter` is bottlenecked on disk or encoding. Switch the output to a faster disk.
- If FPS is lower than your capture frame rate but each stage is cheap, your consumer thread is CPU-starved. The producer will drop frames — the MJPEG stream and the recording still get every camera frame.

### MJPEG is smooth but annotations lag

This is by design. The producer pushes camera frames into the MJPEG buffer and the recorder at camera rate. The consumer is rate-limited by `inference_interval_ms` and inference cost. Overlays only update when the consumer finishes a frame. If you need annotations to lead, drop the inference interval or downscale more aggressively.

## UI / server

### "The web UI bundle is missing"

Shown when `meta_watcher/web/static/index.html` does not exist. Rebuild the bundle:

```bash
cd web && npm install && npm run build
```

### Browser did not open automatically

The CLI opens a browser only when `--open-browser` is passed. Without that flag, point your browser at the bound URL yourself. This is the safer default for SSH / headless use.

### Errors surface in the UI banner

`/api/snapshot` returns the last error captured by `RuntimeState._on_error` in the `error` field. Producer errors (camera read failures) and consumer errors (provider crashes) both route through this field. The red banner in the control panel shows the latest message and clears once a subsequent successful frame arrives.
