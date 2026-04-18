from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from typing import Any, Protocol

import numpy as np


PEOPLE_PROMPTS = ("person", "human", "adult", "child")
PEOPLE_LABELS = {"person", "people", "human", "adult", "child", "man", "woman", "boy", "girl"}


BBox = tuple[int, int, int, int]


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return (x1, y1, x2, y2)


def box_area(bbox: BBox) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(lhs: BBox, rhs: BBox) -> float:
    ax1, ay1, ax2, ay2 = lhs
    bx1, by1, bx2, by2 = rhs
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = box_area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    union = box_area(lhs) + box_area(rhs) - inter
    return inter / union if union else 0.0


def normalize_label(label: str) -> str:
    value = label.strip().lower()
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if value.endswith("es") and len(value) > 4 and not value.endswith(("ses", "xes")):
        value = value[:-2]
    elif value.endswith("s") and len(value) > 3 and not value.endswith(("ss", "us")):
        value = value[:-1]
    return value


def is_people_label(label: str) -> bool:
    return normalize_label(label) in PEOPLE_LABELS


@dataclass(slots=True)
class VideoFrame:
    image: np.ndarray
    timestamp: float
    frame_index: int
    source_id: str
    fps: float | None = None

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox
    mask: np.ndarray | None = None
    track_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InventoryItem:
    label: str
    confidence: float
    samples: int
    last_seen: float


@dataclass(slots=True)
class PipelineSnapshot:
    mode: str
    frame_index: int
    source_id: str
    overlay: np.ndarray
    people: list[Detection]
    inventory_detections: list[Detection]
    inventory_items: list[InventoryItem]
    inventory_active: bool
    recording_active: bool
    completed_clips: list[Path]
    status_text: str
    timings: dict[str, float] = field(default_factory=dict)
    effective_fps: float = 0.0


@dataclass(slots=True)
class Track:
    track_id: str
    label: str
    bbox: BBox
    confidence: float
    last_seen: float
    missed_frames: int = 0


class TrackManager:
    def __init__(
        self,
        *,
        min_iou: float = 0.3,
        max_missed_frames: int = 12,
        smoothing: float = 0.35,
        min_area_ratio: float = 0.002,
    ) -> None:
        self.min_iou = min_iou
        self.max_missed_frames = max_missed_frames
        self.smoothing = smoothing
        self.min_area_ratio = min_area_ratio
        self._tracks: dict[str, Track] = {}
        self._counter = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._counter = 0

    def _next_track_id(self, label: str) -> str:
        self._counter += 1
        return f"{label}-{self._counter}"

    def update(self, detections: list[Detection], frame: VideoFrame) -> list[Detection]:
        min_area = frame.width * frame.height * self.min_area_ratio
        filtered = [d for d in detections if box_area(d.bbox) >= min_area]
        matched_tracks: set[str] = set()
        visible: list[Detection] = []

        for detection in sorted(filtered, key=lambda item: item.confidence, reverse=True):
            best_track_id: str | None = None
            best_score = 0.0
            for track_id, track in self._tracks.items():
                if track.label != detection.label or track_id in matched_tracks:
                    continue
                score = iou(track.bbox, detection.bbox)
                if score >= self.min_iou and score > best_score:
                    best_track_id = track_id
                    best_score = score

            if best_track_id is None:
                track_id = self._next_track_id(detection.label)
                track = Track(track_id=track_id, label=detection.label, bbox=detection.bbox, confidence=detection.confidence, last_seen=frame.timestamp)
                self._tracks[track_id] = track
            else:
                track = self._tracks[best_track_id]
                track.bbox = _smooth_bbox(track.bbox, detection.bbox, self.smoothing)
                track.confidence = (track.confidence * (1.0 - self.smoothing)) + (detection.confidence * self.smoothing)
                track.last_seen = frame.timestamp
                track.missed_frames = 0
                track_id = track.track_id

            matched_tracks.add(track_id)
            visible.append(
                Detection(
                    label=detection.label,
                    confidence=track.confidence,
                    bbox=clamp_bbox(track.bbox, frame.width, frame.height),
                    mask=detection.mask,
                    track_id=track_id,
                    metadata=dict(detection.metadata),
                )
            )

        expired: list[str] = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    expired.append(track_id)
        for track_id in expired:
            self._tracks.pop(track_id, None)

        return visible

    def active_ids(self, label: str | None = None) -> list[str]:
        ids = [track_id for track_id, track in self._tracks.items() if label is None or track.label == label]
        return sorted(ids)


def _smooth_bbox(previous: BBox, current: BBox, factor: float) -> BBox:
    coords = []
    for prev, cur in zip(previous, current):
        coords.append(int(round((prev * (1.0 - factor)) + (cur * factor))))
    return tuple(coords)  # type: ignore[return-value]


def merge_overlapping_detections(detections: list[Detection], overlap_iou: float) -> list[Detection]:
    merged: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        candidate = detection
        was_merged = False
        for index, existing in enumerate(merged):
            if existing.label == candidate.label and iou(existing.bbox, candidate.bbox) >= overlap_iou:
                x1 = min(existing.bbox[0], candidate.bbox[0])
                y1 = min(existing.bbox[1], candidate.bbox[1])
                x2 = max(existing.bbox[2], candidate.bbox[2])
                y2 = max(existing.bbox[3], candidate.bbox[3])
                merged[index] = Detection(
                    label=existing.label,
                    confidence=max(existing.confidence, candidate.confidence),
                    bbox=(x1, y1, x2, y2),
                    mask=existing.mask if existing.confidence >= candidate.confidence else candidate.mask,
                    metadata={**existing.metadata, **candidate.metadata},
                )
                was_merged = True
                break
        if not was_merged:
            merged.append(candidate)
    return merged


@dataclass(slots=True)
class StreamDecision:
    mode: str
    event_started: bool = False
    event_finished: bool = False
    inventory_active: bool = True
    inventory_reset: bool = False


class StreamStateMachine:
    def __init__(self, *, person_confirmation_frames: int = 3, empty_scene_rescan_seconds: float = 5.0) -> None:
        self.person_confirmation_frames = person_confirmation_frames
        self.empty_scene_rescan_seconds = empty_scene_rescan_seconds
        self.mode = "inventory"
        self._positive_frames = 0
        self._empty_since: float | None = None
        self._manual_rescan_requested = False

    def request_manual_rescan(self) -> None:
        self._manual_rescan_requested = True

    def observe(self, timestamp: float, people_present: bool, *, auto_rescan_enabled: bool) -> StreamDecision:
        decision = StreamDecision(mode=self.mode, inventory_active=(self.mode == "inventory"))

        if self.mode == "inventory":
            if self._manual_rescan_requested and not people_present:
                self._manual_rescan_requested = False
                decision.inventory_reset = True
                decision.inventory_active = True

            self._positive_frames = self._positive_frames + 1 if people_present else 0
            if self._positive_frames >= self.person_confirmation_frames:
                self.mode = "person_present"
                self._positive_frames = 0
                self._empty_since = None
                decision.mode = self.mode
                decision.event_started = True
                decision.inventory_active = False
            return decision

        if self.mode == "person_present":
            decision.inventory_active = False
            if not people_present:
                self.mode = "cooldown"
                self._empty_since = timestamp
                decision.mode = self.mode
            return decision

        decision.inventory_active = False
        if people_present:
            self.mode = "person_present"
            self._empty_since = None
            decision.mode = self.mode
            return decision

        if self._empty_since is None:
            self._empty_since = timestamp

        manual_rescan = self._manual_rescan_requested
        dwell_elapsed = (timestamp - self._empty_since) >= self.empty_scene_rescan_seconds
        if manual_rescan or dwell_elapsed:
            self.mode = "inventory"
            self._manual_rescan_requested = False
            self._positive_frames = 0
            self._empty_since = None
            decision.mode = self.mode
            decision.event_finished = True
            decision.inventory_active = auto_rescan_enabled or manual_rescan
            decision.inventory_reset = auto_rescan_enabled or manual_rescan
        return decision


class ClipSink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class OpenCvClipSink:
    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for video recording.") from exc
        self._cv2 = cv2
        self._path = path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer at {path}")

    def write(self, frame: np.ndarray) -> None:
        bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def close(self) -> None:
        self._writer.release()


@dataclass(slots=True)
class EventArtifact:
    clip_path: Path
    metadata_path: Path
    started_at: float
    ended_at: float
    person_ids: list[str]


@dataclass(slots=True)
class _BufferedFrame:
    timestamp: float
    image: np.ndarray


@dataclass(slots=True)
class _OpenEvent:
    clip_path: Path
    metadata_path: Path
    started_at: float
    inventory: list[str]
    source_id: str
    sink: ClipSink
    end_requested_at: float | None = None
    person_ids: set[str] = field(default_factory=set)


class ClipRecorder:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        pre_roll_seconds: float = 3.0,
        post_roll_seconds: float = 2.0,
        sink_factory: Any | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pre_roll_seconds = pre_roll_seconds
        self.post_roll_seconds = post_roll_seconds
        self.sink_factory = sink_factory or self._default_sink_factory
        self._buffer: deque[_BufferedFrame] = deque()
        self._event: _OpenEvent | None = None
        self._completed: list[EventArtifact] = []
        # Producer thread calls push_frame; consumer thread calls start/finish/drain.
        # Use an RLock so start_event can delegate to helper methods without deadlocking.
        self._lock = threading.RLock()

    def _default_sink_factory(self, path: Path, size: tuple[int, int], fps: float) -> ClipSink:
        return OpenCvClipSink(path, size, fps)

    @property
    def recording_active(self) -> bool:
        with self._lock:
            return self._event is not None

    def push_frame(
        self,
        frame: VideoFrame,
        image: np.ndarray | None = None,
    ) -> list[EventArtifact]:
        """Buffer the frame and, if an event is open, write it to the sink.

        Safe to call from any thread. Returns the artifact list for anything
        closed by *this* call. Items are still accumulated on the internal
        completed list so consumers can pick them up via `drain_completed`.
        The optional `image` override exists for tests and legacy callers;
        default is to write the raw camera pixels (`frame.image`).
        """
        payload = image if image is not None else frame.image
        newly_closed: list[EventArtifact] = []
        with self._lock:
            self._buffer.append(
                _BufferedFrame(timestamp=frame.timestamp, image=np.array(payload, copy=True))
            )
            self._prune_buffer(frame.timestamp)

            if self._event is not None:
                self._event.sink.write(payload)
                if (
                    self._event.end_requested_at is not None
                    and frame.timestamp >= self._event.end_requested_at
                ):
                    artifact = self._close_event(frame.timestamp)
                    if artifact is not None:
                        newly_closed.append(artifact)
        return newly_closed

    def _prune_buffer(self, now: float) -> None:
        while self._buffer and (now - self._buffer[0].timestamp) > self.pre_roll_seconds:
            self._buffer.popleft()

    def _estimate_buffer_fps(self, *, fallback: float) -> float:
        if len(self._buffer) >= 2:
            span = self._buffer[-1].timestamp - self._buffer[0].timestamp
            if span > 0:
                return (len(self._buffer) - 1) / span
        return max(1.0, fallback)

    def start_event(self, frame: VideoFrame, inventory: list[str]) -> None:
        with self._lock:
            if self._event is not None:
                return

            started_at = frame.timestamp
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            clip_path = self.output_dir / f"{frame.source_id}_{stamp}.mp4"
            metadata_path = clip_path.with_suffix(".json")
            fps = self._estimate_buffer_fps(fallback=frame.fps or 12.0)
            sink = self.sink_factory(clip_path, (frame.width, frame.height), fps)
            self._event = _OpenEvent(
                clip_path=clip_path,
                metadata_path=metadata_path,
                started_at=started_at,
                inventory=list(inventory),
                source_id=frame.source_id,
                sink=sink,
            )
            for buffered in self._buffer:
                sink.write(buffered.image)

    def finish_event(self, timestamp: float, person_ids: list[str]) -> None:
        with self._lock:
            if self._event is None:
                return
            self._event.person_ids.update(person_ids)
            if self._event.end_requested_at is None:
                self._event.end_requested_at = timestamp + self.post_roll_seconds

    def add_person_ids(self, person_ids: list[str]) -> None:
        with self._lock:
            if self._event is None:
                return
            self._event.person_ids.update(person_ids)

    def drain_completed(self) -> list[EventArtifact]:
        with self._lock:
            return self._drain_completed_locked()

    def _drain_completed_locked(self) -> list[EventArtifact]:
        completed = list(self._completed)
        self._completed.clear()
        return completed

    def _close_event(self, ended_at: float) -> EventArtifact | None:
        if self._event is None:
            return None
        event = self._event
        event.sink.close()
        payload = {
            "clip_path": str(event.clip_path),
            "source_id": event.source_id,
            "inventory": event.inventory,
            "started_at": event.started_at,
            "ended_at": ended_at,
            "person_ids": sorted(event.person_ids),
        }
        with event.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        artifact = EventArtifact(
            clip_path=event.clip_path,
            metadata_path=event.metadata_path,
            started_at=event.started_at,
            ended_at=ended_at,
            person_ids=sorted(event.person_ids),
        )
        self._completed.append(artifact)
        self._event = None
        return artifact
