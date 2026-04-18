---
name: inference-backends
description: Use when modifying Meta Watcher's SAM 3.1 providers — the MLX subprocess path, the Linux CUDA path, the `sam3.perflib` fp32 patch, or the `InferenceProvider` interface. Trigger on edits to `meta_watcher/inference.py`, or when the user asks about provider swapping, model IDs, GPU dtypes, or the MLX worker.
---

# Inference backends skill

## When to use

- Editing `meta_watcher/inference.py` or adding a new `InferenceProvider` implementation.
- Updating how the CUDA provider handles dtypes, patches upstream SAM 3.1, or moves tensors.
- Changing the MLX subprocess RPC protocol or timeouts.
- Adjusting the platform routing in `build_provider`.

## When not to use

- Changes to the pipeline that *calls* the provider — use [pipeline-core](../pipeline-core/SKILL.md).
- Changes to how the web layer exposes provider state — use [web-surface](../web-surface/SKILL.md).

## Rules

1. **Preserve the `InferenceProvider` contract.** Any new provider must implement `warmup`, `detect_text_prompts`, `start_tracking`, `track_next`, and `shutdown`. Both test suites and `StreamProcessor` rely on all five methods.
2. **Lazy-import heavy deps.** Imports of `mlx_vlm`, `sam3`, `torch`, etc. live inside `warmup` (or deeper). The module must import cleanly on any platform — the tests depend on that. Raise a helpful `RuntimeError` inside warmup when a dep is missing.
3. **Keep the MLX worker isolated.** `MlxSubprocessSam31Provider` uses `mp.get_context("spawn")`; do not switch to `fork` on macOS. Do not move `mlx_vlm` imports out of the worker body.
4. **The fp32 patch must patch both sites.** `_patch_sam3_fused_addmm_to_fp32` has to rewrite `sam3.perflib.fused.addmm_act` *and* `sam3.model.vitdet.addmm_act`. Skipping the second binding silently breaks inference. Keep the `_SAM3_FUSED_PATCHED` idempotency guard.
5. **Return real numpy data across process boundaries.** Use `_serialize_frame` / `_serialize_detections` helpers (or new equivalents). Never send torch/mlx tensors through `multiprocessing.Queue`.
6. **Update the install tests.** `tests/test_inference.py` pins numpy range (`numpy>=1.26,<2`), the Linux extra (`sam3 @ git+...@main`), the missing-module error message, and the fp32 cast behavior. Any change to those assertions must be motivated by a matching code change.

## Workflow

1. Read [docs/architecture/inference-backends.md](../../docs/architecture/inference-backends.md) and [docs/internals/inference.md](../../docs/internals/inference.md).
2. Make the code change in `meta_watcher/inference.py`. Keep imports lazy and keep error messages explicit about how to fix them.
3. Update or add a test in `tests/test_inference.py`. The existing tests show how to stub `torch` / `sam3.*` with `types.ModuleType` + `mock.patch.dict(sys.modules, ...)`.
4. Run `python3 -m unittest discover -s tests`.
5. If you are bumping the Linux extra or the upstream branch, also run a real CUDA warmup as described in [docs/internals/inference.md](../../docs/internals/inference.md) to catch upstream drift.
6. Update [docs/architecture/inference-backends.md](../../docs/architecture/inference-backends.md) and [docs/internals/inference.md](../../docs/internals/inference.md) to reflect the new behavior.
7. Append a `CHANGELOG.md` entry under `[Unreleased]`.

## Quick reference

- Apple Silicon routing: `sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}`.
- MLX worker start method: `mp.get_context("spawn")`.
- MLX RPC timeout: `MlxSubprocessSam31Provider.timeout_seconds = 120.0`.
- CUDA dtype target: `torch.float32` for floating tensors; complex tensors keep their dtype and just move device.
- CUDA inference context: `torch.inference_mode()` + `torch.autocast(device_type=..., enabled=True)`.

## Docs this skill routes into

- [docs/architecture/inference-backends.md](../../docs/architecture/inference-backends.md)
- [docs/internals/inference.md](../../docs/internals/inference.md)
- [docs/reference/configuration.md](../../docs/reference/configuration.md)
- [docs/guides/troubleshooting.md](../../docs/guides/troubleshooting.md)
