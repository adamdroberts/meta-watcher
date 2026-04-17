from __future__ import annotations

from abc import ABC, abstractmethod
import sys
import time

from .config import SourceConfig
from .core import VideoFrame


class VideoSource(ABC):
    @property
    @abstractmethod
    def source_id(self) -> str: ...

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def read(self) -> VideoFrame | None: ...

    @abstractmethod
    def close(self) -> None: ...


class OpenCvVideoSource(VideoSource):
    def __init__(self, source_id: str, capture_value: str | int, *, live: bool) -> None:
        self._source_id = source_id
        self.capture_value = capture_value
        self.live = live
        self._capture = None
        self._frame_index = 0
        self._fps = 0.0
        self._cv2 = None

    @property
    def source_id(self) -> str:
        return self._source_id

    def _load_cv2(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required for webcam, RTSP, and file sources.") from exc
        self._cv2 = cv2
        return cv2

    def _open_capture(self, value: str | int):
        cv2 = self._load_cv2()
        if sys.platform == "darwin" and self.source_id == "webcam":
            return cv2.VideoCapture(value, cv2.CAP_AVFOUNDATION)
        return cv2.VideoCapture(value)

    def _finalize_capture(self, capture) -> None:
        self._capture = capture
        self._fps = float(self._capture.get(self._cv2.CAP_PROP_FPS) or 0.0)

    def open(self) -> None:
        capture = self._open_capture(self.capture_value)
        self._capture = capture
        if not self._capture.isOpened():
            raise RuntimeError(f"Failed to open source {self.capture_value!r}")
        self._finalize_capture(self._capture)

    def read(self) -> VideoFrame | None:
        if self._capture is None:
            raise RuntimeError("Source is not open.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        timestamp = time.monotonic() if self.live else float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        packet = VideoFrame(
            image=rgb,
            timestamp=timestamp,
            frame_index=self._frame_index,
            source_id=self._source_id,
            fps=self._fps or None,
        )
        self._frame_index += 1
        return packet

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class WebcamSource(OpenCvVideoSource):
    def __init__(self, value: str) -> None:
        super().__init__(source_id="webcam", capture_value=value, live=True)

    def open(self) -> None:
        cv2 = self._load_cv2()
        configured = self._parse_requested_index(self.capture_value)
        candidates = self._candidate_indices(configured)
        failures: list[int] = []

        for index in candidates:
            capture = self._open_capture(index)
            if not capture.isOpened():
                capture.release()
                failures.append(index)
                continue

            # Prime the device once so we reject camera slots that "open" but never produce frames.
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                failures.append(index)
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(cv2.CAP_PROP_FPS, 30)
            self.capture_value = index
            self._finalize_capture(capture)
            return

        message = f"Failed to open any webcam device from candidates {candidates}."
        if sys.platform == "darwin":
            message += (
                " macOS camera access may be blocked for the Python launcher. "
                "Reset Camera permission for org.python.python or re-enable camera access in Privacy & Security > Camera."
            )
        raise RuntimeError(message)

    def _parse_requested_index(self, value: str | int) -> int | None:
        if isinstance(value, int):
            return value
        text = str(value).strip().lower()
        if not text or text == "auto":
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _candidate_indices(self, configured: int | None) -> list[int]:
        candidates: list[int] = []
        if configured is not None:
            candidates.append(configured)
        for index in range(8):
            if index not in candidates:
                candidates.append(index)
        return candidates


class RtspSource(OpenCvVideoSource):
    def __init__(self, value: str) -> None:
        super().__init__(source_id="rtsp", capture_value=value, live=True)


class FileSource(OpenCvVideoSource):
    def __init__(self, value: str) -> None:
        super().__init__(source_id="file", capture_value=value, live=False)


def build_source(config: SourceConfig) -> VideoSource:
    kind = config.kind.lower()
    if kind == "webcam":
        return WebcamSource(config.value)
    if kind == "rtsp":
        return RtspSource(config.value)
    if kind == "file":
        return FileSource(config.value)
    raise ValueError(f"Unsupported source kind: {config.kind}")
