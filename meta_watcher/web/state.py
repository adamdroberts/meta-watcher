from __future__ import annotations

from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import io
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ..config import (
    AppConfig,
    build_config_from_dict,
    default_config,
    list_config_files,
    load_config,
    repo_root,
    save_config,
)
from ..core import ClipRecorder, PipelineSnapshot
from ..inference import InferenceProvider, build_provider
from ..pipeline import StreamProcessor, StreamRuntime
from ..sources import VideoSource, build_source
from ..upload import EventUploader, UploadProvider, build_upload_provider


LIVE_FRAME_INTERVAL_SECONDS = 0.5


class ConfigValidationError(ValueError):
    """Raised by update_config/reload_config when a patch/file cannot be coerced
    into AppConfig. The server maps this to HTTP 400."""


class ConfigPathError(ValueError):
    """Raised when a requested config path escapes the allowed search dirs.
    The server maps this to HTTP 403."""


ProviderFactory = Callable[[AppConfig], InferenceProvider]
SourceFactory = Callable[[AppConfig], VideoSource]
UploaderFactory = Callable[[AppConfig], "EventUploader | None"]


def _default_uploader(config: AppConfig) -> "EventUploader | None":
    try:
        provider: UploadProvider | None = build_upload_provider(config.upload)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[meta-watcher] upload provider init failed: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None
    if provider is None:
        return None
    return EventUploader(provider, config.upload, timestamps=config.timestamps)


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
        uploader_factory: UploaderFactory | None = None,
        jpeg_quality: int = 80,
        active_config_path: Path | None = None,
        search_dirs: list[Path] | None = None,
    ) -> None:
        self._config = config if config is not None else default_config()
        self._provider_factory = provider_factory or _default_provider
        self._source_factory = source_factory or _default_source
        self._uploader_factory = uploader_factory or _default_uploader
        self._jpeg_quality = jpeg_quality
        self._active_config_path: Path | None = (
            Path(active_config_path).resolve() if active_config_path is not None else None
        )
        self._search_dirs: list[Path] = [
            Path(p).resolve() for p in (search_dirs if search_dirs is not None else [repo_root()])
        ]

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
        self._uploader: EventUploader | None = None
        self._recorder: ClipRecorder | None = None
        self._uploaded_snapshot_event_ids: set[str] = set()
        self._live_frame_last_mono: float = 0.0
        self._live_frame_event_id: str | None = None

    # ------------------------------------------------------------------ config

    def config(self) -> AppConfig:
        with self._lock:
            return self._config

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config())

    def update_config(self, patch: dict[str, Any]) -> AppConfig:
        with self._lock:
            try:
                updated = _merge_config(self._config, patch)
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(str(exc)) from exc
            self._config = updated
            return updated

    def active_config_path(self) -> Path | None:
        with self._lock:
            return self._active_config_path

    def search_dirs(self) -> list[Path]:
        with self._lock:
            return list(self._search_dirs)

    def list_config_files(self) -> list[dict[str, Any]]:
        """Return descriptors for every discoverable config file.

        Includes the current `active_config_path` even if it falls outside the
        configured search dirs (so the UI can always show what's loaded).
        """
        with self._lock:
            dirs = list(self._search_dirs)
            active = self._active_config_path
        paths = list_config_files(dirs)
        if active is not None and active not in [p.resolve() for p in paths]:
            paths.append(active)
            paths.sort(key=lambda p: p.name.lower())
        out: list[dict[str, Any]] = []
        for path in paths:
            resolved = path.resolve()
            out.append(
                {
                    "path": str(resolved),
                    "name": resolved.name,
                    "writable": os.access(resolved.parent, os.W_OK) and (
                        not resolved.exists() or os.access(resolved, os.W_OK)
                    ),
                    "active": active is not None and resolved == active,
                }
            )
        return out

    def reload_config(self, path: str | Path) -> AppConfig:
        """Load a config file from disk, replace the in-memory config, and
        update `active_config_path`. Does NOT restart the pipeline — the caller
        decides whether a restart is warranted.
        """
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            # Resolve relative paths against the first writable search dir,
            # falling back to CWD.
            base = next((d for d in self._search_dirs if d.is_dir()), Path.cwd())
            candidate = base / candidate
        resolved = candidate.resolve()
        self._ensure_path_allowed(resolved)
        if not resolved.exists():
            raise FileNotFoundError(str(resolved))
        try:
            loaded = load_config(resolved)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(str(exc)) from exc
        with self._lock:
            self._config = loaded
            self._active_config_path = resolved
        return loaded

    def save_active_config(self, path: str | Path | None = None) -> Path:
        """Persist the in-memory config to `path` (or to the active path if
        omitted). Returns the actual path written — YAML targets are redirected
        to a JSON sibling by `save_config`, so the active path may shift.
        """
        with self._lock:
            target = Path(path).expanduser() if path is not None else self._active_config_path
            config = self._config
        if target is None:
            raise ValueError("No active config path set — cannot save.")
        if not target.is_absolute():
            base = next((d for d in self._search_dirs if d.is_dir()), Path.cwd())
            target = (base / target)
        target = target.resolve()
        self._ensure_path_allowed(target)
        written = save_config(target, config)
        with self._lock:
            self._active_config_path = written.resolve()
        return written

    def _ensure_path_allowed(self, candidate: Path) -> None:
        """Guard against path traversal: `candidate` must live under a search
        dir, or equal the currently-active path. Relative callers go through
        here after resolution.
        """
        resolved = candidate.resolve() if candidate.is_absolute() else candidate
        if self._active_config_path is not None and resolved == self._active_config_path:
            return
        for base in self._search_dirs:
            try:
                resolved.relative_to(base)
                return
            except ValueError:
                continue
        raise ConfigPathError(
            f"Config path {resolved} is outside the allowed search dirs."
        )

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
                mode=config.output.recording_mode,
            )
            processor = StreamProcessor(config, provider, recorder)
            processor.set_recording_enabled(self._recording_enabled)
            source = self._source_factory(config)
            uploader = self._uploader_factory(config)
            if uploader is not None:
                uploader.start()
            self._uploader = uploader
            self._recorder = recorder
            self._uploaded_snapshot_event_ids = set()
            self._live_frame_last_mono = 0.0
            self._live_frame_event_id = None
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
            uploader = self._uploader
            self._runtime = None
            self._uploader = None
            self._recorder = None
            self._uploaded_snapshot_event_ids = set()
            self._live_frame_last_mono = 0.0
            self._live_frame_event_id = None
        if runtime is not None:
            runtime.stop()
        if uploader is not None:
            uploader.stop()

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
            uploader = self._uploader
            recorder = self._recorder
            for artifact in snapshot.completed_clips:
                self._completed_clips.append(str(artifact.clip_path))
                if uploader is not None:
                    already_sent_snapshot = (
                        artifact.clip_path.stem in self._uploaded_snapshot_event_ids
                    )
                    uploader.enqueue_artifact(artifact, skip_snapshot=already_sent_snapshot)
                self._uploaded_snapshot_event_ids.discard(artifact.clip_path.stem)
                if self._live_frame_event_id == artifact.clip_path.stem:
                    self._live_frame_event_id = None
                    self._live_frame_last_mono = 0.0
            self._frame_cond.notify_all()
        self._pump_live_uploads(jpeg, uploader, recorder)
        self._maybe_log_jpeg()

    def _pump_live_uploads(
        self,
        overlay_jpeg: bytes,
        uploader: EventUploader | None,
        recorder: ClipRecorder | None,
    ) -> None:
        """Upload the event snapshot ASAP and push a frame every 0.5s mid-event."""
        if uploader is None or recorder is None:
            return
        info = recorder.active_event_info()
        if info is None:
            return
        event_id, snapshot_path = info
        with self._lock:
            should_send_snapshot = (
                snapshot_path is not None
                and event_id not in self._uploaded_snapshot_event_ids
            )
            if should_send_snapshot:
                self._uploaded_snapshot_event_ids.add(event_id)

            now_mono = time.monotonic()
            if self._live_frame_event_id != event_id:
                self._live_frame_event_id = event_id
                self._live_frame_last_mono = 0.0
            should_send_frame = (
                now_mono - self._live_frame_last_mono
            ) >= LIVE_FRAME_INTERVAL_SECONDS
            if should_send_frame:
                self._live_frame_last_mono = now_mono

        if should_send_snapshot and snapshot_path is not None:
            try:
                uploader.enqueue_snapshot(snapshot_path, event_id)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[meta-watcher] snapshot enqueue failed for {event_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        if should_send_frame:
            frame_path = self._write_live_frame(overlay_jpeg, event_id)
            if frame_path is not None:
                try:
                    uploader.enqueue_frame(frame_path, event_id)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[meta-watcher] frame enqueue failed for {event_id}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

    def _write_live_frame(self, jpeg_bytes: bytes, event_id: str) -> Path | None:
        """Persist a live frame JPEG next to the event's clip + snapshot so
        local operators have a full on-disk record of the event, not just the
        uploaded copies. Falls back to the system tempdir if the configured
        output directory is unwritable (so enqueueing never hard-fails)."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        name = f"{event_id}_{stamp}_{uuid.uuid4().hex[:6]}.jpg"
        output_dir = Path(self.config().output.directory)
        frames_dir = output_dir / event_id / "frames"
        try:
            frames_dir.mkdir(parents=True, exist_ok=True)
            target = frames_dir / name
            with target.open("wb") as handle:
                handle.write(jpeg_bytes)
            return target
        except OSError as exc:
            print(
                f"[meta-watcher] live frame write to {frames_dir} failed: {exc}; "
                "falling back to system tempdir",
                file=sys.stderr,
                flush=True,
            )
        fallback_dir = Path(tempfile.gettempdir()) / "meta-watcher-frames"
        try:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            target = fallback_dir / name
            with target.open("wb") as handle:
                handle.write(jpeg_bytes)
            return target
        except OSError as exc:
            print(
                f"[meta-watcher] live frame fallback write failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

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


from ..jpeg import encode_jpeg as _encode_jpeg_impl


def _encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    return _encode_jpeg_impl(image, quality)


def _merge_config(base: AppConfig, patch: dict[str, Any]) -> AppConfig:
    current = asdict(base)
    for key, value in patch.items():
        if key not in current or not isinstance(current[key], dict):
            # Unknown top-level keys are intentionally tolerated (silent drop).
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"Section '{key}' must be a JSON object; got {type(value).__name__}."
            )
        current[key].update({k: v for k, v in value.items() if k in current[key]})
    from ..config import (
        InventoryConfig,
        ModelConfig,
        OutputConfig,
        SourceConfig,
        ThresholdConfig,
        TimestampConfig,
        TimingConfig,
        UploadConfig,
    )

    return AppConfig(
        source=SourceConfig(**current["source"]),
        models=ModelConfig(**current["models"]),
        thresholds=ThresholdConfig(**current["thresholds"]),
        timings=TimingConfig(**current["timings"]),
        inventory=InventoryConfig(**current["inventory"]),
        output=OutputConfig(**current["output"]),
        upload=UploadConfig(**current["upload"]),
        timestamps=TimestampConfig(**current["timestamps"]),
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
