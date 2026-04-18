from __future__ import annotations

import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .core import (
    PEOPLE_PROMPTS,
    ClipRecorder,
    Detection,
    InventoryItem,
    PipelineSnapshot,
    StreamStateMachine,
    TrackManager,
    VideoFrame,
    clamp_bbox,
    is_people_label,
    merge_overlapping_detections,
    normalize_label,
)
from .inference import InferenceProvider
from .overlay import render_overlay
from .sources import VideoSource


_TIMING_WINDOW = 30
_TIMING_STAGES = ("inference", "overlay", "recorder", "total")


def _downscale_for_inference(frame: VideoFrame, max_side: int) -> tuple[VideoFrame, float]:
    """Return (possibly downscaled frame, scale factor to map bboxes back to original)."""
    height, width = frame.image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return frame, 1.0
    ratio = max_side / longest
    new_w = max(1, int(round(width * ratio)))
    new_h = max(1, int(round(height * ratio)))
    try:
        import cv2
    except ImportError:
        return frame, 1.0
    resized = cv2.resize(frame.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled = VideoFrame(
        image=resized,
        timestamp=frame.timestamp,
        frame_index=frame.frame_index,
        source_id=frame.source_id,
        fps=frame.fps,
    )
    return scaled, 1.0 / ratio


def _upscale_detections(
    detections: list[Detection],
    scale: float,
    width: int,
    height: int,
) -> list[Detection]:
    if scale == 1.0 or not detections:
        return detections
    out: list[Detection] = []
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bbox = (
            int(round(x1 * scale)),
            int(round(y1 * scale)),
            int(round(x2 * scale)),
            int(round(y2 * scale)),
        )
        bbox = clamp_bbox(bbox, width, height)
        out.append(
            Detection(
                label=det.label,
                confidence=det.confidence,
                bbox=bbox,
                mask=det.mask,
                track_id=det.track_id,
                metadata=det.metadata,
            )
        )
    return out


class StreamProcessor:
    def __init__(
        self,
        config: AppConfig,
        provider: InferenceProvider,
        recorder: ClipRecorder | None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.recorder = recorder
        self.recording_enabled = True

        self._tracker = TrackManager(
            min_iou=config.thresholds.tracking_iou,
            min_area_ratio=config.thresholds.min_area_ratio,
        )
        self._state_machine = StreamStateMachine(
            person_confirmation_frames=config.timings.person_confirmation_frames,
            empty_scene_rescan_seconds=config.timings.empty_scene_rescan_seconds,
        )
        self._inventory_items: list[InventoryItem] = _inventory_items_from_labels(
            config.inventory.labels
        )
        self._tracking_live = False
        self._event_person_ids: set[str] = set()

        self._stage_samples: dict[str, deque[float]] = {
            stage: deque(maxlen=_TIMING_WINDOW) for stage in _TIMING_STAGES
        }
        self._frame_wall_times: deque[float] = deque(maxlen=_TIMING_WINDOW)
        self._last_log_time = 0.0

        # Inference rate-limiter: run the provider at most once per interval,
        # reusing the last detections on in-between frames so the overlay and
        # recording still advance at the consumer's full rate.
        self._last_inference_ts: float = -1.0e9
        self._cached_people_candidates: list[Detection] = []
        self._cached_inventory_detections: list[Detection] = []

    def request_manual_rescan(self) -> None:
        self._state_machine.request_manual_rescan()

    def set_recording_enabled(self, enabled: bool) -> None:
        self.recording_enabled = enabled

    def update_labels(self, labels: list[str]) -> None:
        self.config.inventory.labels = list(labels)
        self._inventory_items = _inventory_items_from_labels(labels)

    def process_frame(self, frame: VideoFrame) -> PipelineSnapshot:
        frame_start = time.perf_counter()

        inference_start = time.perf_counter()
        interval_s = max(0.0, self.config.models.inference_interval_ms / 1000.0)
        should_run_inference = (frame.timestamp - self._last_inference_ts) >= interval_s

        inf_frame: VideoFrame | None = None
        scale = 1.0
        if should_run_inference:
            inf_frame, scale = _downscale_for_inference(
                frame, self.config.models.inference_max_side
            )
            if self._tracking_live and self._state_machine.mode in {"person_present", "cooldown"}:
                people_candidates = self.provider.track_next(inf_frame)
            else:
                people_candidates = self.provider.detect_text_prompts(inf_frame, list(PEOPLE_PROMPTS))
            people_candidates = _upscale_detections(people_candidates, scale, frame.width, frame.height)
            self._cached_people_candidates = list(people_candidates)
            self._last_inference_ts = frame.timestamp
        else:
            people_candidates = list(self._cached_people_candidates)

        people_candidates = self._normalize_people(people_candidates)
        tracked_people = self._tracker.update(people_candidates, frame)
        people_present = bool(tracked_people)

        decision = self._state_machine.observe(
            frame.timestamp,
            people_present,
            auto_rescan_enabled=self.config.inventory.auto_rescan,
        )

        if decision.event_started:
            self._tracking_live = True
            self._event_person_ids.clear()
            # start_tracking itself triggers an inference pass; ensure the frame
            # fed in matches the downscaling contract and reset the interval timer.
            start_frame = inf_frame
            if start_frame is None:
                start_frame, _ = _downscale_for_inference(
                    frame, self.config.models.inference_max_side
                )
            self.provider.start_tracking(start_frame, list(PEOPLE_PROMPTS))
            self._last_inference_ts = frame.timestamp

        inventory_labels = [item.label for item in self._inventory_items]
        inventory_detections: list[Detection] = []
        inventory_eligible = (
            decision.mode == "inventory"
            and decision.inventory_active
            and not people_present
            and bool(inventory_labels)
        )
        if inventory_eligible:
            if should_run_inference and inf_frame is not None:
                inventory_raw = self.provider.detect_text_prompts(inf_frame, inventory_labels)
                inventory_raw = _upscale_detections(
                    inventory_raw, scale, frame.width, frame.height
                )
                inventory_detections = [
                    detection
                    for detection in inventory_raw
                    if detection.confidence >= self.config.thresholds.inventory_confidence
                ]
                self._cached_inventory_detections = list(inventory_detections)
            else:
                inventory_detections = list(self._cached_inventory_detections)
        else:
            self._cached_inventory_detections = []
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        if tracked_people:
            self._event_person_ids.update(
                detection.track_id for detection in tracked_people if detection.track_id is not None
            )

        visible_detections = tracked_people if decision.mode in {"person_present", "cooldown"} else inventory_detections
        status_text = self._status_text(decision.mode, tracked_people, self._inventory_items)

        # Build an early snapshot of timings so the HUD shows recent history.
        hud_timings = self._current_timings()
        hud_fps = self._current_fps()

        overlay_start = time.perf_counter()
        overlay = render_overlay(
            frame.image,
            detections=visible_detections,
            inventory=self._inventory_items,
            mode=decision.mode,
            inventory_active=decision.inventory_active,
            recording_active=bool(self.recorder and self.recorder.recording_active),
            status_text=status_text,
            hud_timings=hud_timings,
            hud_fps=hud_fps,
        )
        overlay_ms = (time.perf_counter() - overlay_start) * 1000.0

        recorder_start = time.perf_counter()
        completed_clips: list[Path] = []
        if self.recorder is not None and self.recording_enabled:
            if decision.event_started:
                self.recorder.start_event(frame, [item.label for item in self._inventory_items])
            self.recorder.add_person_ids(sorted(self._event_person_ids))
            if decision.event_finished:
                self.recorder.finish_event(frame.timestamp, sorted(self._event_person_ids))
                self._tracking_live = False
                self._tracker.reset()
            # Frame write happens on the producer thread (see StreamRuntime._produce_loop);
            # here we only pick up clips that have finalized in the meantime.
            artifacts = self.recorder.drain_completed()
            completed_clips = [artifact.clip_path for artifact in artifacts]
            if completed_clips:
                self._event_person_ids.clear()
        elif decision.event_finished:
            self._tracking_live = False
            self._tracker.reset()
            self._event_person_ids.clear()
        recorder_ms = (time.perf_counter() - recorder_start) * 1000.0

        total_ms = (time.perf_counter() - frame_start) * 1000.0
        self._record_frame(
            wall_time=time.perf_counter(),
            inference_ms=inference_ms,
            overlay_ms=overlay_ms,
            recorder_ms=recorder_ms,
            total_ms=total_ms,
        )
        timings = self._current_timings()
        effective_fps = self._current_fps()
        self._maybe_log_timings(
            timings,
            effective_fps,
            mode=decision.mode,
            frame_shape=frame.image.shape,
        )

        return PipelineSnapshot(
            mode=decision.mode,
            frame_index=frame.frame_index,
            source_id=frame.source_id,
            overlay=overlay,
            people=tracked_people,
            inventory_detections=inventory_detections,
            inventory_items=list(self._inventory_items),
            inventory_active=decision.inventory_active,
            recording_active=bool(self.recorder and self.recorder.recording_active),
            completed_clips=completed_clips,
            status_text=status_text,
            timings=timings,
            effective_fps=effective_fps,
        )

    def _record_frame(
        self,
        *,
        wall_time: float,
        inference_ms: float,
        overlay_ms: float,
        recorder_ms: float,
        total_ms: float,
    ) -> None:
        self._frame_wall_times.append(wall_time)
        self._stage_samples["inference"].append(inference_ms)
        self._stage_samples["overlay"].append(overlay_ms)
        self._stage_samples["recorder"].append(recorder_ms)
        self._stage_samples["total"].append(total_ms)

    def _current_timings(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for stage, samples in self._stage_samples.items():
            out[stage] = (sum(samples) / len(samples)) if samples else 0.0
        return out

    def _current_fps(self) -> float:
        if len(self._frame_wall_times) < 2:
            return 0.0
        span = self._frame_wall_times[-1] - self._frame_wall_times[0]
        if span <= 0.0:
            return 0.0
        return (len(self._frame_wall_times) - 1) / span

    def _maybe_log_timings(
        self,
        timings: dict[str, float],
        fps: float,
        *,
        mode: str,
        frame_shape: tuple[int, ...],
    ) -> None:
        now = time.perf_counter()
        if now - self._last_log_time < 1.0:
            return
        self._last_log_time = now
        height, width = frame_shape[0], frame_shape[1]
        print(
            f"[meta-watcher] fps={fps:.1f} "
            f"infer={timings['inference']:.1f}ms "
            f"overlay={timings['overlay']:.1f}ms "
            f"recorder={timings['recorder']:.1f}ms "
            f"total={timings['total']:.1f}ms "
            f"mode={mode} frame={width}x{height}",
            file=sys.stderr,
            flush=True,
        )

    def _normalize_people(self, detections: list[Detection]) -> list[Detection]:
        normalized: list[Detection] = []
        for detection in detections:
            label = normalize_label(detection.label)
            if not is_people_label(label):
                continue
            if detection.confidence < self.config.thresholds.person_confidence:
                continue
            normalized.append(
                Detection(
                    label="person",
                    confidence=detection.confidence,
                    bbox=detection.bbox,
                    mask=detection.mask,
                    metadata=detection.metadata,
                )
            )
        return merge_overlapping_detections(normalized, self.config.thresholds.overlap_iou)

    def _status_text(self, mode: str, people: list[Detection], inventory: list[InventoryItem]) -> str:
        if mode == "inventory":
            return f"scene objects: {len(inventory)}"
        return f"tracked people: {len(people)}"


def _inventory_items_from_labels(labels: list[str]) -> list[InventoryItem]:
    items: list[InventoryItem] = []
    for raw in labels:
        label = normalize_label(raw)
        if not label or is_people_label(label):
            continue
        items.append(InventoryItem(label=label, confidence=1.0, samples=1, last_seen=0.0))
    return items


class StreamRuntime:
    def __init__(
        self,
        source: VideoSource,
        processor: StreamProcessor,
        *,
        on_snapshot: Callable[[PipelineSnapshot], None],
        on_error: Callable[[str], None],
        on_raw_frame: Callable[[VideoFrame], None] | None = None,
        queue_size: int = 2,
    ) -> None:
        self.source = source
        self.processor = processor
        self.on_snapshot = on_snapshot
        self.on_error = on_error
        self.on_raw_frame = on_raw_frame
        self.queue: queue.Queue[VideoFrame] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._producer: threading.Thread | None = None
        self._consumer: threading.Thread | None = None

    def start(self) -> None:
        if self._producer is not None:
            return
        self._stop.clear()
        self._producer = threading.Thread(target=self._produce_loop, name="meta-watcher-source", daemon=True)
        self._consumer = threading.Thread(target=self._consume_loop, name="meta-watcher-pipeline", daemon=True)
        self._producer.start()
        self._consumer.start()

    def stop(self) -> None:
        self._stop.set()
        if self._producer is not None:
            self._producer.join(timeout=2.0)
        if self._consumer is not None:
            self._consumer.join(timeout=2.0)
        self._producer = None
        self._consumer = None
        try:
            self.source.close()
        finally:
            self.processor.provider.shutdown()

    def request_manual_rescan(self) -> None:
        self.processor.request_manual_rescan()

    def set_recording_enabled(self, enabled: bool) -> None:
        self.processor.set_recording_enabled(enabled)

    def _produce_loop(self) -> None:
        try:
            self.source.open()
            while not self._stop.is_set():
                frame = self.source.read()
                if frame is None:
                    break
                if self.on_raw_frame is not None:
                    try:
                        self.on_raw_frame(frame)
                    except Exception:
                        pass
                # Recording tap: capture every camera frame at camera rate,
                # independent of the inference-gated consumer thread.
                recorder = self.processor.recorder
                if recorder is not None and self.processor.recording_enabled:
                    try:
                        recorder.push_frame(frame)
                    except Exception:
                        pass
                self._put_latest(frame)
                if getattr(self.source, "live", True) is False and frame.fps:
                    time.sleep(max(0.0, 1.0 / frame.fps))
        except Exception as exc:
            self.on_error(str(exc))
        finally:
            self._stop.set()
            self.source.close()

    def _consume_loop(self) -> None:
        try:
            self.processor.provider.warmup()
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    frame = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                snapshot = self.processor.process_frame(frame)
                self.on_snapshot(snapshot)
        except Exception as exc:
            self.on_error(str(exc))
            self._stop.set()

    def _put_latest(self, frame: VideoFrame) -> None:
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(frame)
