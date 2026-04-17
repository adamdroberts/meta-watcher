from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .core import (
    PEOPLE_PROMPTS,
    ClipRecorder,
    Detection,
    InventoryItem,
    LabelStabilizer,
    PipelineSnapshot,
    StreamStateMachine,
    TrackManager,
    VideoFrame,
    is_people_label,
    merge_overlapping_detections,
    normalize_label,
)
from .inference import InferenceProvider, SceneLabelProposer
from .overlay import render_overlay
from .sources import VideoSource


class StreamProcessor:
    def __init__(
        self,
        config: AppConfig,
        provider: InferenceProvider,
        label_proposer: SceneLabelProposer | None,
        recorder: ClipRecorder | None,
    ) -> None:
        self.config = config
        self.provider = provider
        self.label_proposer = label_proposer
        self.recorder = recorder
        self.recording_enabled = True

        self._inventory = LabelStabilizer(
            required_samples=config.inventory.required_samples,
            max_labels=config.inventory.max_labels,
        )
        self._tracker = TrackManager(
            min_iou=config.thresholds.tracking_iou,
            min_area_ratio=config.thresholds.min_area_ratio,
        )
        self._state_machine = StreamStateMachine(
            person_confirmation_frames=config.timings.person_confirmation_frames,
            empty_scene_rescan_seconds=config.timings.empty_scene_rescan_seconds,
        )
        self._inventory_items: list[InventoryItem] = []
        self._last_inventory_sample = float("-inf")
        self._tracking_live = False
        self._event_person_ids: set[str] = set()

    def request_manual_rescan(self) -> None:
        self._state_machine.request_manual_rescan()

    def set_recording_enabled(self, enabled: bool) -> None:
        self.recording_enabled = enabled

    def process_frame(self, frame: VideoFrame) -> PipelineSnapshot:
        if self._tracking_live and self._state_machine.mode in {"person_present", "cooldown"}:
            people_candidates = self.provider.track_next(frame)
        else:
            people_candidates = self.provider.detect_text_prompts(frame, list(PEOPLE_PROMPTS))

        people_candidates = self._normalize_people(people_candidates)
        tracked_people = self._tracker.update(people_candidates, frame)
        people_present = bool(tracked_people)

        decision = self._state_machine.observe(
            frame.timestamp,
            people_present,
            auto_rescan_enabled=self.config.inventory.auto_rescan,
        )

        if decision.inventory_reset:
            self._inventory.reset()
            self._inventory_items = []
            self._last_inventory_sample = float("-inf")

        if decision.event_started:
            self._tracking_live = True
            self._event_person_ids.clear()
            self.provider.start_tracking(frame, list(PEOPLE_PROMPTS))

        inventory_detections: list[Detection] = []
        if decision.mode == "inventory" and decision.inventory_active and not people_present:
            if (
                self.label_proposer is not None
                and (frame.timestamp - self._last_inventory_sample) >= self.config.timings.inventory_sample_seconds
            ):
                proposals = self.label_proposer.propose(frame.image)
                self._inventory_items = self._inventory.observe(proposals, frame.timestamp)
                self._last_inventory_sample = frame.timestamp

            if self._inventory.labels:
                inventory_detections = self.provider.detect_text_prompts(frame, self._inventory.labels)
                inventory_detections = [
                    detection
                    for detection in inventory_detections
                    if detection.confidence >= self.config.thresholds.inventory_confidence
                ]
        else:
            inventory_detections = []

        if tracked_people:
            self._event_person_ids.update(
                detection.track_id for detection in tracked_people if detection.track_id is not None
            )

        visible_detections = tracked_people if decision.mode in {"person_present", "cooldown"} else inventory_detections
        status_text = self._status_text(decision.mode, tracked_people, self._inventory_items)

        overlay = render_overlay(
            frame.image,
            detections=visible_detections,
            inventory=self._inventory_items,
            mode=decision.mode,
            inventory_active=decision.inventory_active,
            recording_active=bool(self.recorder and self.recorder.recording_active),
            status_text=status_text,
        )

        completed_clips: list[Path] = []
        if self.recorder is not None and self.recording_enabled:
            if decision.event_started:
                self.recorder.start_event(frame, [item.label for item in self._inventory_items])
            self.recorder.add_person_ids(sorted(self._event_person_ids))
            if decision.event_finished:
                self.recorder.finish_event(frame.timestamp, sorted(self._event_person_ids))
                self._tracking_live = False
                self._tracker.reset()
            artifacts = self.recorder.push_frame(frame, overlay)
            completed_clips = [artifact.clip_path for artifact in artifacts]
            if completed_clips:
                self._event_person_ids.clear()
        elif decision.event_finished:
            self._tracking_live = False
            self._tracker.reset()
            self._event_person_ids.clear()

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


class StreamRuntime:
    def __init__(
        self,
        source: VideoSource,
        processor: StreamProcessor,
        *,
        on_snapshot: Callable[[PipelineSnapshot], None],
        on_error: Callable[[str], None],
        queue_size: int = 2,
    ) -> None:
        self.source = source
        self.processor = processor
        self.on_snapshot = on_snapshot
        self.on_error = on_error
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
