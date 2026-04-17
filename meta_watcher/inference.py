from __future__ import annotations

from abc import ABC, abstractmethod
import importlib.util
import json
import multiprocessing as mp
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import ModelConfig
from .core import Detection, PEOPLE_PROMPTS, VideoFrame, clamp_bbox


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


class SceneLabelProposer(ABC):
    @abstractmethod
    def propose(self, frame: np.ndarray) -> dict[str, float]: ...


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


class CudaSam31Provider(InferenceProvider):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._processor: Any | None = None
        self._tracking_prompts: list[str] = list(PEOPLE_PROMPTS)

    def warmup(self) -> None:
        if self._processor is not None:
            return
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise RuntimeError("The official facebookresearch/sam3 package is required for Linux CUDA inference.") from exc

        try:
            model = build_sam3_image_model(model_id=self.model_id)
        except TypeError:
            model = build_sam3_image_model()
        self._processor = Sam3Processor(model)

    def detect_text_prompts(self, frame: VideoFrame, prompts: list[str] | tuple[str, ...]) -> list[Detection]:
        self.warmup()
        assert self._processor is not None
        state = self._processor.set_image(Image.fromarray(frame.image))
        detections: list[Detection] = []
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


class QwenSceneLabelProposer(SceneLabelProposer):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._transformers_bundle: tuple[Any, Any] | None = None

    def propose(self, frame: np.ndarray) -> dict[str, float]:
        if importlib.util.find_spec("transformers"):
            return self._propose_with_transformers(frame)
        if sys.platform == "darwin" and importlib.util.find_spec("mlx_vlm"):
            return self._propose_with_mlx_cli(frame)
        raise RuntimeError("No scene label proposer backend is installed. Install mlx-vlm or transformers.")

    def _prompt(self) -> str:
        return (
            "List the distinct non-human physical objects visible in this scene. "
            "Exclude people, body parts, reflections, screens with content, and actions. "
            "Return strict JSON as an array of short lowercase noun phrases, max 12 items."
        )

    def _propose_with_mlx_cli(self, frame: np.ndarray) -> dict[str, float]:
        image = Image.fromarray(frame)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            image.save(handle.name)
            temp_path = Path(handle.name)
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlx_vlm.generate",
                    "--model",
                    self.model_id,
                    "--max-tokens",
                    "120",
                    "--temp",
                    "0.0",
                    "--prompt",
                    self._prompt(),
                    "--image",
                    str(temp_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return _parse_label_payload(proc.stdout)

    def _propose_with_transformers(self, frame: np.ndarray) -> dict[str, float]:
        if self._transformers_bundle is None:
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError("transformers is required for Qwen scene label proposing.") from exc

            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForImageTextToText.from_pretrained(self.model_id, device_map="auto")
            self._transformers_bundle = (processor, model)

        processor, model = self._transformers_bundle
        image = Image.fromarray(frame)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt()},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generated = model.generate(**inputs, max_new_tokens=128)
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return _parse_label_payload(output)


def build_provider(models: ModelConfig) -> InferenceProvider:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return MlxSubprocessSam31Provider(models.mac_sam_model_id)
    return CudaSam31Provider(models.linux_sam_model_id)


def build_label_proposer(models: ModelConfig) -> SceneLabelProposer:
    return QwenSceneLabelProposer(models.label_model_id)


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


def _detections_from_output_dict(frame: VideoFrame, output: Any, prompt: str) -> list[Detection]:
    if not isinstance(output, dict):
        return []
    boxes = np.asarray(output.get("boxes", []))
    scores = np.asarray(output.get("scores", []), dtype=float)
    masks = output.get("masks", [])
    labels = output.get("labels", [prompt] * len(scores))
    detections: list[Detection] = []
    for index, score in enumerate(scores.tolist()):
        if boxes.size == 0:
            continue
        box = _box_tuple(boxes[index], frame)
        mask = masks[index] if index < len(masks) else None
        label = labels[index] if index < len(labels) else prompt
        detections.append(Detection(label=str(label), confidence=float(score), bbox=box, mask=_coerce_mask(mask)))
    return detections


def _detections_from_generic_result(frame: VideoFrame, result: Any, prompt_override: str | None = None) -> list[Detection]:
    boxes = np.asarray(getattr(result, "boxes", []))
    scores = np.asarray(getattr(result, "scores", []), dtype=float)
    masks = getattr(result, "masks", [])
    labels = getattr(result, "labels", None)
    detections: list[Detection] = []
    for index, score in enumerate(scores.tolist()):
        if boxes.size == 0:
            continue
        label = prompt_override or (labels[index] if labels is not None and index < len(labels) else "object")
        mask = masks[index] if index < len(masks) else None
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
    try:
        return np.asarray(mask)
    except Exception:
        return None


def _parse_label_payload(payload: str) -> dict[str, float]:
    payload = payload.strip()
    for candidate in _json_candidates(payload):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return {str(item).strip().lower(): 0.6 for item in parsed if str(item).strip()}
            if isinstance(parsed, dict):
                labels = parsed.get("labels") or parsed.get("objects") or parsed.get("items")
                if isinstance(labels, list):
                    return {str(item).strip().lower(): 0.6 for item in labels if str(item).strip()}
        except json.JSONDecodeError:
            continue

    fallback = {}
    for token in re.split(r"[\n,]", payload):
        label = token.strip(" -•\t").lower()
        if label:
            fallback[label] = 0.5
    return fallback


def _json_candidates(payload: str) -> list[str]:
    candidates: list[str] = []
    for start_char, end_char in (("[", "]"), ("{", "}")):
        start = payload.find(start_char)
        end = payload.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidates.append(payload[start : end + 1])
    return candidates
