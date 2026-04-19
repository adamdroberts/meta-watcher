from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
import multiprocessing as mp
import platform
import sys
from typing import Any

import numpy as np
from PIL import Image

from .config import ModelConfig
from .core import Detection, PEOPLE_PROMPTS, VideoFrame, clamp_bbox, normalize_label


class InferenceProvider(ABC):
    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def detect_text_prompts(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]: ...

    @abstractmethod
    def start_tracking(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]: ...

    @abstractmethod
    def track_next(self, frame: VideoFrame) -> list[Detection]: ...

    @abstractmethod
    def shutdown(self) -> None: ...


def _model_execution_context(model: Any) -> tuple[Any | None, Any | None]:
    device = getattr(model, "device", None)
    dtype = getattr(model, "dtype", None)

    if device is not None and dtype is not None:
        return device, dtype

    try:
        parameters: Iterable[Any] = model.parameters()
        first_parameter = next(iter(parameters))
    except (AttributeError, StopIteration, TypeError):
        return device, dtype

    if device is None:
        device = getattr(first_parameter, "device", None)
    if dtype is None:
        dtype = getattr(first_parameter, "dtype", None)
    return device, dtype


def _move_inputs_to_model(inputs: Mapping[str, Any], model: Any) -> dict[str, Any]:
    device, dtype = _model_execution_context(model)
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue

        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None and hasattr(value, "is_floating_point") and value.is_floating_point():
            kwargs["dtype"] = dtype
        moved[key] = value.to(**kwargs) if kwargs else value
    return moved


class MlxSam31Provider(InferenceProvider):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._predictor: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._tracking_prompts: list[str] = list(PEOPLE_PROMPTS)

    def warmup(self) -> None:
        if self._predictor is not None:
            return
        try:
            from mlx_vlm.utils import get_model_path, load_model
            from mlx_vlm.models.sam3.generate import Sam3Predictor
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        except ImportError as exc:
            raise RuntimeError("mlx-vlm is required for Apple Silicon inference.") from exc

        model_path = get_model_path(self.model_id)
        self._model = load_model(model_path)
        self._processor = Sam31Processor.from_pretrained(str(model_path))
        self._predictor = Sam3Predictor(self._model, self._processor, score_threshold=0.0)

    def detect_text_prompts(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]:
        self.warmup()
        assert self._predictor is not None
        image = Image.fromarray(frame.image)
        detections: list[Detection] = []
        try:
            from mlx_vlm.models.sam3_1.generate import predict_multi

            result = predict_multi(self._predictor, image, list(prompts))
            detections.extend(_detections_from_generic_result(frame, result))
        except Exception:
            for prompt in prompts:
                result = self._predictor.predict(image, text_prompt=prompt)
                detections.extend(_detections_from_generic_result(frame, result, prompt_override=prompt))
        _synchronize_mlx()
        return detections

    def start_tracking(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]:
        self._tracking_prompts = list(prompts)
        return self.detect_text_prompts(frame, prompts)

    def track_next(self, frame: VideoFrame) -> list[Detection]:
        return self.detect_text_prompts(frame, self._tracking_prompts)

    def shutdown(self) -> None:
        self._predictor = None
        self._processor = None
        self._model = None


_SAM3_FUSED_PATCHED = False


def _patch_sam3_fused_addmm_to_fp32() -> None:
    """Replace sam3.perflib.fused.addmm_act with an fp32-safe linear+activation.

    Upstream's fused op casts mat1/mat2/bias to bfloat16 and uses the
    `torch.ops.aten._addmm_activation` kernel, which leaves the output in
    bf16. When we run the rest of the graph in fp32, subsequent Linear
    layers see bf16 activations against fp32 weights and torch raises
    `mat1 and mat2 must have the same dtype`. Swapping the fused op for a
    plain torch.nn.functional.linear + activation keeps everything in fp32
    at a small perf cost. Both the original module and the vitdet call
    site (which bound the symbol at import time) are patched.
    """
    global _SAM3_FUSED_PATCHED
    if _SAM3_FUSED_PATCHED:
        return
    try:
        import torch
        from sam3.perflib import fused as _fused
        from sam3.model import vitdet as _vitdet
    except ImportError:
        return

    def _addmm_act_fp32(activation, linear, mat1):
        y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
        if activation in (torch.nn.functional.gelu, torch.nn.GELU):
            return torch.nn.functional.gelu(y)
        if activation in (torch.nn.functional.relu, torch.nn.ReLU):
            return torch.nn.functional.relu(y)
        raise ValueError(f"Unexpected activation {activation}")

    _fused.addmm_act = _addmm_act_fp32
    _vitdet.addmm_act = _addmm_act_fp32
    _SAM3_FUSED_PATCHED = True


class CudaSam31Provider(InferenceProvider):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._processor: Any | None = None
        self._device: Any | None = None
        self._tracking_prompts: list[str] = list(PEOPLE_PROMPTS)

    def warmup(self) -> None:
        if self._processor is not None:
            return
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "Linux CUDA inference requires the official facebookresearch/sam3 package. "
                'Reinstall with `python3 -m pip install -e ".[desktop,linux]"` to pull the upstream main branch.'
            ) from exc

        import torch

        # sam3's perflib ships a fused addmm+activation kernel (vitdet.Mlp.fc1
        # path) that hard-casts its inputs and the Linear weight/bias to
        # bfloat16 on every call. Its output then feeds a plain fp32 Linear
        # (fc2) and triggers `mat1 and mat2 must have the same dtype: float
        # vs bfloat16`. Replace the fused op with a simple fp32 linear+act so
        # the whole graph stays fp32. Must patch both the defining module and
        # the call site (vitdet already did `from ... import addmm_act`).
        _patch_sam3_fused_addmm_to_fp32()

        try:
            model = build_sam3_image_model(model_id=self.model_id)
        except TypeError:
            model = build_sam3_image_model()

        # Upstream sam3 can leave the vision backbone in bf16 while the text
        # path stays fp32. Force a single floating dtype on the whole graph
        # and switch to inference mode before handing the model to the
        # processor. Use _apply so complex-valued tensors (e.g. rotary
        # embeddings) keep their dtype instead of being silently truncated
        # to real.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target_device = self._device
        target_dtype = torch.float32

        def _cast(tensor: "torch.Tensor") -> "torch.Tensor":
            moved = tensor.to(device=target_device)
            if moved.is_floating_point():
                return moved.to(dtype=target_dtype)
            return moved

        model = model._apply(_cast)
        model.train(False)
        self._processor = Sam3Processor(model)

    def detect_text_prompts(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]:
        self.warmup()
        assert self._processor is not None
        import torch

        device_type = self._device.type if self._device is not None else "cpu"
        # print(f"Inference device: {device_type}")
        detections: list[Detection] = []
        with torch.inference_mode(), torch.autocast(device_type=device_type, enabled=True):
            state = self._processor.set_image(Image.fromarray(frame.image))
            for prompt in prompts:
                output = self._processor.set_text_prompt(state=state, prompt=prompt)
                detections.extend(_detections_from_output_dict(frame, output, prompt))
        return detections

    def start_tracking(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]:
        self._tracking_prompts = list(prompts)
        return self.detect_text_prompts(frame, prompts)

    def track_next(self, frame: VideoFrame) -> list[Detection]:
        return self.detect_text_prompts(frame, self._tracking_prompts)

    def shutdown(self) -> None:
        self._processor = None
        self._device = None


def _matches_any_prompt(label: str, prompts: Iterable[str]) -> bool:
    """Loose match a closed-vocabulary class name (e.g. "dining table")
    against user-supplied prompts (e.g. "table"). Both sides are run through
    ``normalize_label`` and then compared in both directions so that
    multi-word COCO labels line up with short prompts.
    """
    normalized_label = normalize_label(label)
    if not normalized_label:
        return False
    for prompt in prompts:
        normalized_prompt = normalize_label(prompt)
        if not normalized_prompt:
            continue
        if normalized_prompt == normalized_label:
            return True
        if normalized_prompt in normalized_label:
            return True
        if normalized_label in normalized_prompt:
            return True
    return False


class TransformersObjectDetectionProvider(InferenceProvider):
    """Object detection backed by a HuggingFace ``transformers`` model.

    Works with closed-vocabulary COCO detectors such as
    ``facebook/detr-resnet-50`` and ``jadechoghari/RT-DETRv2``. Detections
    are filtered against the caller's prompt list using
    :func:`_matches_any_prompt` since these models do not accept free-form
    text prompts. No segmentation masks are produced.
    """

    MISSING_DEPENDENCY_MESSAGE = (
        "Transformers-based object detection requires the `transformers` and "
        "`torch` packages. Reinstall with "
        '`python3 -m pip install -e ".[desktop,detr]"`.'
    )

    def __init__(self, model_id: str, *, score_threshold: float = 0.05) -> None:
        self.model_id = model_id
        self.score_threshold = float(score_threshold)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: Any | None = None
        self._tracking_prompts: list[str] = list(PEOPLE_PROMPTS)

    def warmup(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            import torch
        except ImportError as exc:
            raise RuntimeError(self.MISSING_DEPENDENCY_MESSAGE) from exc

        # ``trust_remote_code=True`` is required for community repos like
        # ``jadechoghari/RT-DETRv2`` that ship their own modeling files.
        # Users opted into this backend via config, so enabling it here is
        # appropriate.
        processor = AutoImageProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        model = AutoModelForObjectDetection.from_pretrained(self.model_id, trust_remote_code=True)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            sys.platform == "darwin"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        model.train(False)

        print(
            f"[meta-watcher] inference provider={self.model_id} device={device.type}",
            file=sys.stderr,
            flush=True,
        )

        self._processor = processor
        self._model = model
        self._device = device

    def detect_text_prompts(
        self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]
    ) -> list[Detection]:
        self.warmup()
        assert self._model is not None and self._processor is not None
        import torch

        image = Image.fromarray(frame.image)
        inputs = self._processor(images=image, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)

        device_type = self._device.type if self._device is not None else "cpu"
        with torch.inference_mode():
            if device_type == "cuda" and hasattr(torch, "autocast"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self._model(**inputs)
            else:
                outputs = self._model(**inputs)

        target_sizes = [(frame.height, frame.width)]
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.score_threshold,
            target_sizes=target_sizes,
        )
        if not results:
            return []
        result = results[0]

        id2label = getattr(self._model.config, "id2label", {}) or {}
        scores = _to_numpy(result.get("scores", []), dtype=float)
        label_ids = _to_numpy(result.get("labels", []))
        boxes = _to_numpy(result.get("boxes", []))

        detections: list[Detection] = []
        for index, score in enumerate(scores.tolist()):
            if index >= len(label_ids) or boxes.size == 0 or index >= len(boxes):
                continue
            label_id = int(label_ids[index])
            label_name = str(id2label.get(label_id, id2label.get(str(label_id), label_id)))
            if not _matches_any_prompt(label_name, prompts):
                continue
            detections.append(
                Detection(
                    label=label_name,
                    confidence=float(score),
                    bbox=_box_tuple(boxes[index], frame),
                    mask=None,
                )
            )
        return detections

    def start_tracking(
        self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]
    ) -> list[Detection]:
        self._tracking_prompts = list(prompts)
        return self.detect_text_prompts(frame, prompts)

    def track_next(self, frame: VideoFrame) -> list[Detection]:
        return self.detect_text_prompts(frame, self._tracking_prompts)

    def shutdown(self) -> None:
        self._model = None
        self._processor = None
        self._device = None


def build_provider(models: ModelConfig) -> InferenceProvider:
    provider = getattr(models, "provider", "sam3.1")
    if provider == "detr-resnet-50":
        return TransformersObjectDetectionProvider(models.detr_model_id)
    if provider == "rt-detrv2":
        return TransformersObjectDetectionProvider(models.rt_detr_model_id)
    if provider != "sam3.1":
        raise ValueError(
            f"Unknown inference provider {provider!r}. Expected one of: "
            "'sam3.1', 'detr-resnet-50', 'rt-detrv2'."
        )
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return MlxSubprocessSam31Provider(models.mac_sam_model_id)
    return CudaSam31Provider(models.linux_sam_model_id)


class MlxSubprocessSam31Provider(InferenceProvider):
    def __init__(self, model_id: str, timeout_seconds: float = 120.0) -> None:
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self._ctx = mp.get_context("spawn")
        self._request_queue: Any | None = None
        self._response_queue: Any | None = None
        self._process: mp.Process | None = None

    def warmup(self) -> None:
        self._ensure_worker()
        self._rpc("warmup")

    def detect_text_prompts(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]:
        self._ensure_worker()
        payload = self._rpc("detect", frame=_serialize_frame(frame), prompts=list(prompts))
        return _deserialize_detections(payload)

    def start_tracking(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]:
        self._ensure_worker()
        payload = self._rpc("start_tracking", frame=_serialize_frame(frame), prompts=list(prompts))
        return _deserialize_detections(payload)

    def track_next(self, frame: VideoFrame) -> list[Detection]:
        self._ensure_worker()
        payload = self._rpc("track_next", frame=_serialize_frame(frame))
        return _deserialize_detections(payload)

    def shutdown(self) -> None:
        if self._process is None:
            return
        try:
            self._rpc("shutdown", expect_response=False)
        finally:
            if self._process is not None:
                self._process.join(timeout=5.0)
                if self._process.is_alive():
                    self._process.terminate()
                    self._process.join(timeout=2.0)
            self._process = None
            self._request_queue = None
            self._response_queue = None

    def _ensure_worker(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._process = self._ctx.Process(
            target=_mlx_worker_main,
            args=(self.model_id, self._request_queue, self._response_queue),
            daemon=True,
        )
        self._process.start()

    def _rpc(self, method: str, expect_response: bool = True, **payload: Any) -> Any:
        if self._request_queue is None or self._response_queue is None:
            raise RuntimeError("MLX worker is not available.")
        self._request_queue.put({"method": method, "payload": payload})
        if not expect_response:
            return None

        deadline = _monotonic_seconds() + self.timeout_seconds
        while _monotonic_seconds() < deadline:
            if self._process is None or not self._process.is_alive():
                raise RuntimeError("The MLX worker process exited unexpectedly.")
            try:
                response = self._response_queue.get(timeout=0.2)
            except Exception:
                continue
            if response.get("ok"):
                return response.get("payload")
            error = response.get("error", "Unknown MLX worker error")
            trace = response.get("traceback")
            if trace:
                raise RuntimeError(f"{error}\n{trace}")
            raise RuntimeError(error)
        raise RuntimeError(f"Timed out waiting for MLX worker method {method!r}.")


def _mlx_worker_main(model_id: str, request_queue: Any, response_queue: Any) -> None:
    provider = MlxSam31Provider(model_id)
    try:
        while True:
            request = request_queue.get()
            method = request.get("method")
            payload = request.get("payload", {})
            if method == "shutdown":
                provider.shutdown()
                break
            try:
                if method == "warmup":
                    provider.warmup()
                    response_queue.put({"ok": True, "payload": None})
                elif method == "detect":
                    frame = _deserialize_frame(payload["frame"])
                    detections = provider.detect_text_prompts(frame, payload["prompts"])
                    response_queue.put({"ok": True, "payload": _serialize_detections(detections)})
                elif method == "start_tracking":
                    frame = _deserialize_frame(payload["frame"])
                    detections = provider.start_tracking(frame, payload["prompts"])
                    response_queue.put({"ok": True, "payload": _serialize_detections(detections)})
                elif method == "track_next":
                    frame = _deserialize_frame(payload["frame"])
                    detections = provider.track_next(frame)
                    response_queue.put({"ok": True, "payload": _serialize_detections(detections)})
                else:
                    response_queue.put({"ok": False, "error": f"Unknown MLX worker method: {method}"})
            except Exception as exc:
                import traceback

                response_queue.put(
                    {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        provider.shutdown()


def _serialize_frame(frame: VideoFrame) -> dict[str, Any]:
    return {
        "image": np.asarray(frame.image),
        "timestamp": float(frame.timestamp),
        "frame_index": int(frame.frame_index),
        "source_id": frame.source_id,
        "fps": frame.fps,
    }


def _deserialize_frame(payload: dict[str, Any]) -> VideoFrame:
    return VideoFrame(
        image=np.asarray(payload["image"]),
        timestamp=float(payload["timestamp"]),
        frame_index=int(payload["frame_index"]),
        source_id=str(payload["source_id"]),
        fps=(float(payload["fps"]) if payload.get("fps") is not None else None),
    )


def _serialize_detections(detections: list[Detection]) -> list[dict[str, Any]]:
    return [
        {
            "label": detection.label,
            "confidence": float(detection.confidence),
            "bbox": tuple(int(value) for value in detection.bbox),
            "mask": np.asarray(detection.mask) if detection.mask is not None else None,
            "track_id": detection.track_id,
            "metadata": dict(detection.metadata),
        }
        for detection in detections
    ]


def _deserialize_detections(payload: Any) -> list[Detection]:
    items = payload or []
    return [
        Detection(
            label=str(item["label"]),
            confidence=float(item["confidence"]),
            bbox=tuple(int(value) for value in item["bbox"]),
            mask=(np.asarray(item["mask"]) if item.get("mask") is not None else None),
            track_id=item.get("track_id"),
            metadata=dict(item.get("metadata", {})),
        )
        for item in items
    ]


def _synchronize_mlx() -> None:
    try:
        import mlx.core as mx
    except Exception:
        return
    try:
        mx.synchronize()
    except Exception:
        return


def _monotonic_seconds() -> float:
    import time

    return time.monotonic()


def _to_numpy(value: Any, dtype: Any = None) -> np.ndarray:
    """Convert tensor-like values to numpy, handling CUDA tensors."""
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        array = value.detach().cpu().numpy()
        if dtype is not None:
            array = array.astype(dtype)
        return array
    if dtype is not None:
        return np.asarray(value, dtype=dtype)
    return np.asarray(value)


def _detections_from_output_dict(frame: VideoFrame, output: Any, prompt: str) -> list[Detection]:
    if not isinstance(output, dict):
        return []
    boxes = _to_numpy(output.get("boxes", []))
    scores = _to_numpy(output.get("scores", []), dtype=float)
    raw_masks = output.get("masks", [])
    labels = output.get("labels", [prompt] * len(scores))
    detections: list[Detection] = []
    for index, score in enumerate(scores.tolist()):
        if boxes.size == 0:
            continue
        box = _box_tuple(boxes[index], frame)
        mask = raw_masks[index] if index < len(raw_masks) else None
        label = labels[index] if index < len(labels) else prompt
        detections.append(Detection(label=str(label), confidence=float(score), bbox=box, mask=_coerce_mask(mask)))
    return detections


def _detections_from_generic_result(frame: VideoFrame, result: Any, prompt_override: str | None = None) -> list[Detection]:
    boxes = _to_numpy(getattr(result, "boxes", []))
    scores = _to_numpy(getattr(result, "scores", []), dtype=float)
    raw_masks = getattr(result, "masks", [])
    labels = getattr(result, "labels", None)
    detections: list[Detection] = []
    for index, score in enumerate(scores.tolist()):
        if boxes.size == 0:
            continue
        label = prompt_override or (labels[index] if labels is not None and index < len(labels) else "object")
        mask = raw_masks[index] if index < len(raw_masks) else None
        detections.append(
            Detection(
                label=str(label),
                confidence=float(score),
                bbox=_box_tuple(boxes[index], frame),
                mask=_coerce_mask(mask),
            )
        )
    return detections


def _box_tuple(box: Any, frame: VideoFrame) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(value))) for value in box[:4]]
    return clamp_bbox((x1, y1, x2, y2), frame.width, frame.height)


def _coerce_mask(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        return mask
    if hasattr(mask, "detach") and hasattr(mask, "cpu") and hasattr(mask, "numpy"):
        try:
            return mask.detach().cpu().numpy()
        except Exception:
            return None
    try:
        return np.asarray(mask)
    except Exception:
        return None


