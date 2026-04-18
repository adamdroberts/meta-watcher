from __future__ import annotations

from collections import deque
from dataclasses import asdict
import io
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ..config import AppConfig, default_config
from ..core import ClipRecorder, PipelineSnapshot
from ..inference import InferenceProvider, build_provider
from ..pipeline import StreamProcessor, StreamRuntime
from ..sources import VideoSource, build_source


ProviderFactory = Callable[[AppConfig], InferenceProvider]
SourceFactory = Callable[[AppConfig], VideoSource]


def _default_provider(config: AppConfig) -> InferenceProvider:
    return build_provider(config.models)


def _default_source(config: AppConfig) -> VideoSource:
    return build_source(config.source)


class RuntimeState:
    """Thread-safe container owning the StreamRuntime, latest snapshot, and MJPEG buffer."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        provider_factory: ProviderFactory | None = None,
        source_factory: SourceFactory | None = None,
        jpeg_quality: int = 80,
    ) -> None:
        self._config = config if config is not None else default_config()
        self._provider_factory = provider_factory or _default_provider
        self._source_factory = source_factory or _default_source
        self._jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._runtime: StreamRuntime | None = None
        self._recording_enabled = True
        self._latest_snapshot: PipelineSnapshot | None = None
        self._latest_jpeg: bytes | None = None
        self._latest_jpeg_version = 0
        self._completed_clips: list[str] = []
        self._error: str | None = None
        self._pipeline_live = False
        self._jpeg_samples: deque[float] = deque(maxlen=30)
        self._jpeg_last_log = 0.0

    # ------------------------------------------------------------------ config

    def config(self) -> AppConfig:
        with self._lock:
            return self._config

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config())

    def update_config(self, patch: dict[str, Any]) -> AppConfig:
        with self._lock:
            updated = _merge_config(self._config, patch)
            self._config = updated
            return updated

    def set_recording_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._recording_enabled = bool(enabled)
            if self._runtime is not None:
                self._runtime.set_recording_enabled(self._recording_enabled)

    # ---------------------------------------------------------------- lifecycle

    def is_running(self) -> bool:
        with self._lock:
            return self._runtime is not None

    def start(self, patch: dict[str, Any] | None = None) -> None:
        if patch:
            self.update_config(patch)
        with self._lock:
            if self._runtime is not None:
                return
            self._error = None
            self._completed_clips = []
            self._pipeline_live = False
            self._latest_snapshot = None
            config = self._config
            provider = self._provider_factory(config)
            recorder = ClipRecorder(
                config.output.directory,
                pre_roll_seconds=config.timings.pre_roll_seconds,
                post_roll_seconds=config.timings.post_roll_seconds,
            )
            processor = StreamProcessor(config, provider, recorder)
            processor.set_recording_enabled(self._recording_enabled)
            source = self._source_factory(config)
            runtime = StreamRuntime(
                source,
                processor,
                on_snapshot=self._on_snapshot,
                on_error=self._on_error,
                on_raw_frame=self._on_raw_frame,
            )
            runtime.start()
            self._runtime = runtime

    def stop(self) -> None:
        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            runtime.stop()

    def rescan(self) -> None:
        with self._lock:
            runtime = self._runtime
        if runtime is not None:
            runtime.request_manual_rescan()

    # ------------------------------------------------------------------ frames

    def _on_snapshot(self, snapshot: PipelineSnapshot) -> None:
        jpeg_start = time.perf_counter()
        jpeg = _encode_jpeg(snapshot.overlay, self._jpeg_quality)
        jpeg_ms = (time.perf_counter() - jpeg_start) * 1000.0
        with self._frame_cond:
            self._latest_snapshot = snapshot
            self._latest_jpeg = jpeg
            self._latest_jpeg_version += 1
            self._pipeline_live = True
            self._jpeg_samples.append(jpeg_ms)
            for clip in snapshot.completed_clips:
                self._completed_clips.append(str(clip))
            self._frame_cond.notify_all()
        self._maybe_log_jpeg()

    def _maybe_log_jpeg(self) -> None:
        now = time.perf_counter()
        with self._lock:
            if now - self._jpeg_last_log < 1.0 or not self._jpeg_samples:
                return
            self._jpeg_last_log = now
            avg = sum(self._jpeg_samples) / len(self._jpeg_samples)
        print(f"[meta-watcher] jpeg_encode_avg={avg:.1f}ms", file=sys.stderr, flush=True)

    def _on_raw_frame(self, frame) -> None:
        # Show the live camera as soon as the producer thread starts reading,
        # even while the heavy model warmup runs in the consumer thread. Once
        # the pipeline starts producing annotated overlays, stop overwriting
        # them with raw frames so detections do not flicker off.
        with self._frame_cond:
            if self._pipeline_live:
                return
        jpeg = _encode_jpeg(frame.image, self._jpeg_quality)
        with self._frame_cond:
            if self._pipeline_live:
                return
            self._latest_jpeg = jpeg
            self._latest_jpeg_version += 1
            self._frame_cond.notify_all()

    def _on_error(self, message: str) -> None:
        with self._frame_cond:
            self._error = message
            self._runtime = None
            self._latest_jpeg_version += 1
            self._frame_cond.notify_all()

    def snapshot_payload(self) -> dict[str, Any]:
        with self._lock:
            snapshot = self._latest_snapshot
            running = self._runtime is not None
            error = self._error
            completed = list(self._completed_clips)
        if snapshot is None:
            return {
                "running": running,
                "mode": "idle",
                "status_text": "idle",
                "frame_index": None,
                "source_id": None,
                "inventory_active": False,
                "recording_active": False,
                "inventory_items": [],
                "completed_clips": completed,
                "error": error,
            }
        return {
            "running": running,
            "mode": snapshot.mode,
            "status_text": snapshot.status_text,
            "frame_index": snapshot.frame_index,
            "source_id": snapshot.source_id,
            "inventory_active": snapshot.inventory_active,
            "recording_active": snapshot.recording_active,
            "inventory_items": [
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "samples": item.samples,
                    "last_seen": item.last_seen,
                }
                for item in snapshot.inventory_items
            ],
            "completed_clips": completed,
            "error": error,
        }

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def wait_for_new_jpeg(self, last_version: int, timeout: float) -> tuple[bytes | None, int]:
        with self._frame_cond:
            self._frame_cond.wait_for(
                lambda: self._latest_jpeg_version != last_version,
                timeout=timeout,
            )
            return self._latest_jpeg, self._latest_jpeg_version


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    pil = Image.fromarray(image)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _merge_config(base: AppConfig, patch: dict[str, Any]) -> AppConfig:
    current = asdict(base)
    for key, value in patch.items():
        if key not in current or not isinstance(current[key], dict) or not isinstance(value, dict):
            continue
        current[key].update({k: v for k, v in value.items() if k in current[key]})
    from ..config import (
        InventoryConfig,
        ModelConfig,
        OutputConfig,
        SourceConfig,
        ThresholdConfig,
        TimingConfig,
    )

    return AppConfig(
        source=SourceConfig(**current["source"]),
        models=ModelConfig(**current["models"]),
        thresholds=ThresholdConfig(**current["thresholds"]),
        timings=TimingConfig(**current["timings"]),
        inventory=InventoryConfig(**current["inventory"]),
        output=OutputConfig(**current["output"]),
    )


_PLACEHOLDER_JPEG: bytes | None = None


def placeholder_jpeg() -> bytes:
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is None:
        image = Image.new("RGB", (1280, 720), color=(17, 17, 17))
        try:
            from PIL import ImageDraw, ImageFont

            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 28)
            except Exception:
                font = ImageFont.load_default()
            message = "Waiting for camera…"
            bbox = draw.textbbox((0, 0), message, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            draw.text(
                ((1280 - tw) / 2, (720 - th) / 2),
                message,
                fill=(170, 180, 190),
                font=font,
            )
        except Exception:
            pass
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70)
        _PLACEHOLDER_JPEG = buffer.getvalue()
    return _PLACEHOLDER_JPEG


def clip_list(path: Path | str) -> list[str]:
    directory = Path(path)
    if not directory.exists():
        return []
    return sorted(str(p) for p in directory.glob("*.mp4"))
