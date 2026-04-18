# Inference internals

This page documents the two things about `meta_watcher/inference.py` that are easy to break: the MLX subprocess RPC protocol and the `sam3.perflib` fp32 patch.

## MLX subprocess RPC

`MlxSubprocessSam31Provider` wraps an in-process `MlxSam31Provider` in a dedicated `multiprocessing` process using the `spawn` start method. This is the default on macOS arm64.

### Protocol

Request messages pushed onto `request_queue`:

```python
{"method": "warmup" | "detect" | "start_tracking" | "track_next" | "shutdown",
 "payload": { ... }}
```

Response messages pushed onto `response_queue`:

```python
{"ok": True, "payload": <method-specific>}                       # success
{"ok": False, "error": "<name>: <msg>", "traceback": "<text>"}   # failure
```

Method payloads:

| Method            | Request payload                                  | Response payload               |
| ----------------- | ------------------------------------------------ | ------------------------------ |
| `warmup`          | `{}`                                             | `None`                         |
| `detect`          | `{"frame": serialized, "prompts": list[str]}`    | `list[serialized detection]`   |
| `start_tracking`  | `{"frame": serialized, "prompts": list[str]}`    | `list[serialized detection]`   |
| `track_next`      | `{"frame": serialized}`                          | `list[serialized detection]`   |
| `shutdown`        | `{}`                                             | no response (worker exits)     |

`_serialize_frame` turns a `VideoFrame` into a dict with a numpy `image`, `timestamp`, `frame_index`, `source_id`, and optional `fps`. `_serialize_detections` walks each `Detection` into a dict of pure primitives plus a numpy mask when present. Deserializers mirror these transforms.

### Timeouts and failure

`_rpc` polls `response_queue.get(timeout=0.2)` in a loop until the overall `timeout_seconds` deadline (default 120 s). If the worker process dies while a call is pending, the next poll detects `not self._process.is_alive()` and raises `RuntimeError("The MLX worker process exited unexpectedly.")`.

On `shutdown`, the provider sends the message, joins the process with a 5-second grace period, and calls `terminate()` if it's still alive. This sequence is also run in `RuntimeState.stop()` → `StreamRuntime.stop()` → `provider.shutdown()`.

### Why not shared memory

Frames and detections are small enough that the cost of marshaling through `multiprocessing.Queue` is not the bottleneck, and staying on plain queues keeps the serialization story readable. If a future version needs large per-frame buffers (raw masks, say), consider switching to `multiprocessing.shared_memory` for the image payload; the rest of the dict can stay on the queue.

## `sam3.perflib` fp32 patch

`CudaSam31Provider.warmup` calls `_patch_sam3_fused_addmm_to_fp32()` before building the model.

### The upstream issue

`facebookresearch/sam3` ships a fused addmm+activation helper:

```python
# sam3/perflib/fused.py
def addmm_act(activation, linear, mat1):
    # hard-casts mat1, linear.weight, linear.bias to bfloat16
    # calls torch.ops.aten._addmm_activation(...)
    ...
```

It is bound by name into `sam3/model/vitdet.py` (`from sam3.perflib.fused import addmm_act`). When the rest of the graph is kept in `fp32`, the fused op returns bf16 outputs that then feed an fp32 Linear and PyTorch raises:

```
RuntimeError: mat1 and mat2 must have the same dtype: float vs bfloat16
```

### The patch

```python
def _addmm_act_fp32(activation, linear, mat1):
    y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
    if activation in (torch.nn.functional.gelu, torch.nn.GELU):
        return torch.nn.functional.gelu(y)
    if activation in (torch.nn.functional.relu, torch.nn.ReLU):
        return torch.nn.functional.relu(y)
    raise ValueError(f"Unexpected activation {activation}")
```

The patch rebinds both `sam3.perflib.fused.addmm_act` (the module-level symbol) *and* `sam3.model.vitdet.addmm_act` (the symbol already imported at module load time). Without the second binding, `vitdet` would keep using the original fused op via its local alias and the patch would have no effect.

### Idempotency

`_SAM3_FUSED_PATCHED` guards against re-patching. Subsequent warmups (for example after a reconfigure that rebuilds the provider) skip the patch.

### What if upstream changes

The patch is tightly coupled to upstream. If upstream renames `addmm_act`, moves it out of `perflib`, fixes the dtype handling, or adds a third activation (swiglu, etc.), the patch needs to follow. The failure mode is explicit: either an `ImportError` inside `_patch_sam3_fused_addmm_to_fp32` (silently ignored — we assume the patch target is gone) or a `ValueError("Unexpected activation ...")` on the next forward.

When bumping the Linux extra, run at least one real CUDA forward to catch drift:

```bash
python3 -c "
from meta_watcher.config import default_config
from meta_watcher.inference import build_provider, CudaSam31Provider
provider = build_provider(default_config().models)
provider.warmup()
"
```

## Transformers detection provider

`TransformersObjectDetectionProvider` is selected when `ModelConfig.provider` is `"detr-resnet-50"` or `"rt-detrv2"`. It loads the model via `transformers.AutoModelForObjectDetection.from_pretrained(model_id, trust_remote_code=True)` and the processor via `AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)`. `trust_remote_code=True` is required for community repos like `jadechoghari/RT-DETRv2` that ship their own modeling files.

Device selection at warmup: CUDA if `torch.cuda.is_available()`, otherwise MPS on darwin if `torch.backends.mps.is_available()`, otherwise CPU. The model is moved with `model.to(device)` and put in inference mode with `model.train(False)` — same idiom as `CudaSam31Provider`.

### Prompt matching

Both DETR and RT-DETRv2 are closed-vocabulary detectors: they emit COCO class names from `self._model.config.id2label`, not free-form prompts. `_matches_any_prompt(label, prompts)` runs `normalize_label` over both sides and accepts a match when either normalized string contains the other, or they are equal. That is what lets short configured prompts like `"table"` or `"plant"` line up with COCO's `"dining table"` and `"potted plant"`, and what collapses the multi-prompt `PEOPLE_PROMPTS` onto COCO's single `"person"` class. Detections whose labels match no prompt are filtered out before the pipeline sees them; `Detection.mask` is always `None`.

### Tracking stubs

`start_tracking(frame, prompts)` stores `prompts` in `self._tracking_prompts` and calls `detect_text_prompts`. `track_next(frame)` re-issues `detect_text_prompts(frame, self._tracking_prompts)`. There is no SAM-style per-image caching — the pipeline's `inference_interval_ms` gate already rate-limits calls, and `TrackManager` does the per-frame IoU bookkeeping.

## Input movement helper

`_move_inputs_to_model` is a generic helper shared between providers and tests:

- Reads `device` and `dtype` from the model (preferring explicit attributes, then the first parameter).
- Moves every tensor to that device.
- Casts floating-point tensors to that dtype; leaves non-floating tensors alone.
- Non-tensor values pass through unchanged.

The CUDA path does not currently call this helper on its way into the Sam3 processor (it calls `model._apply(_cast)` instead), but it's kept available for future providers and is unit-tested in `tests/test_inference.py`.
