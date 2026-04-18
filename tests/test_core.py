from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from meta_watcher.core import (
    ClipRecorder,
    Detection,
    StreamStateMachine,
    TrackManager,
    VideoFrame,
    normalize_label,
)
from meta_watcher.sources import WebcamSource


def make_frame(index: int, timestamp: float | None = None) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        timestamp=float(index if timestamp is None else timestamp),
        frame_index=index,
        source_id="test",
        fps=1.0,
    )


class MemorySink:
    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        self.path = path
        self.size = size
        self.fps = fps
        self.frames: list[np.ndarray] = []
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(np.array(frame, copy=True))

    def close(self) -> None:
        self.closed = True


class MemorySinkFactory:
    def __init__(self) -> None:
        self.sinks: list[MemorySink] = []

    def __call__(self, path: Path, size: tuple[int, int], fps: float) -> MemorySink:
        sink = MemorySink(path, size, fps)
        self.sinks.append(sink)
        return sink


class CoreTests(unittest.TestCase):
    def test_normalize_label(self) -> None:
        self.assertEqual(normalize_label("The Chairs"), "chair")
        self.assertEqual(normalize_label("An office lamp"), "office lamp")

    def test_track_manager_reuses_ids(self) -> None:
        tracker = TrackManager(min_iou=0.1, max_missed_frames=2)
        frame0 = make_frame(0)
        first = tracker.update([Detection(label="person", confidence=0.9, bbox=(10, 10, 30, 40))], frame0)
        frame1 = make_frame(1)
        second = tracker.update([Detection(label="person", confidence=0.88, bbox=(11, 10, 31, 40))], frame1)
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_state_machine_confirm_cooldown_and_manual_rescan(self) -> None:
        machine = StreamStateMachine(person_confirmation_frames=3, empty_scene_rescan_seconds=5.0)
        self.assertEqual(machine.observe(0.0, False, auto_rescan_enabled=False).mode, "inventory")
        machine.observe(1.0, True, auto_rescan_enabled=False)
        machine.observe(2.0, True, auto_rescan_enabled=False)
        started = machine.observe(3.0, True, auto_rescan_enabled=False)
        self.assertTrue(started.event_started)
        self.assertEqual(started.mode, "person_present")
        cooldown = machine.observe(4.0, False, auto_rescan_enabled=False)
        self.assertEqual(cooldown.mode, "cooldown")
        finished = machine.observe(9.0, False, auto_rescan_enabled=False)
        self.assertTrue(finished.event_finished)
        self.assertFalse(finished.inventory_active)
        machine.request_manual_rescan()
        rescanned = machine.observe(10.0, False, auto_rescan_enabled=False)
        self.assertTrue(rescanned.inventory_reset)
        self.assertTrue(rescanned.inventory_active)

    def test_clip_recorder_uses_preroll_and_postroll(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            sink_factory = MemorySinkFactory()
            recorder = ClipRecorder(tempdir, pre_roll_seconds=3.0, post_roll_seconds=2.0, sink_factory=sink_factory)

            for index in range(3):
                recorder.push_frame(make_frame(index), np.zeros((64, 64, 3), dtype=np.uint8))

            event_frame = make_frame(3)
            recorder.start_event(event_frame, ["chair", "lamp"])
            recorder.push_frame(event_frame, np.zeros((64, 64, 3), dtype=np.uint8))
            recorder.finish_event(4.0, ["person-1"])
            recorder.push_frame(make_frame(4), np.zeros((64, 64, 3), dtype=np.uint8))
            recorder.push_frame(make_frame(5), np.zeros((64, 64, 3), dtype=np.uint8))
            artifacts = recorder.push_frame(make_frame(6), np.zeros((64, 64, 3), dtype=np.uint8))

            self.assertEqual(len(sink_factory.sinks), 1)
            self.assertEqual(len(sink_factory.sinks[0].frames), 7)
            self.assertEqual(len(artifacts), 1)
            metadata = json.loads(Path(artifacts[0].metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["inventory"], ["chair", "lamp"])
            self.assertEqual(metadata["person_ids"], ["person-1"])

    def test_webcam_source_only_probes_requested_index(self) -> None:
        class FakeCapture:
            def __init__(self, index: int) -> None:
                self.index = index

            def isOpened(self) -> bool:
                return True

            def read(self):
                return True, np.zeros((32, 32, 3), dtype=np.uint8)

            def set(self, prop, value) -> bool:
                return True

            def get(self, prop) -> float:
                return 30.0

            def release(self) -> None:
                return None

        source = WebcamSource("3")
        fake_cv2 = mock.Mock(CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5)
        source._cv2 = fake_cv2
        source._load_cv2 = mock.Mock(return_value=fake_cv2)
        probed: list[int] = []

        def opener(index: int) -> FakeCapture:
            probed.append(index)
            return FakeCapture(index)

        source._open_capture = opener
        source.open()

        self.assertEqual(probed, [3])
        self.assertEqual(source.capture_value, 3)
        self.assertIsNotNone(source._capture)

    def test_webcam_source_auto_uses_enumeration(self) -> None:
        class FakeCapture:
            def __init__(self, index: int, works: bool) -> None:
                self.index = index
                self.works = works

            def isOpened(self) -> bool:
                return self.works

            def read(self):
                if self.works:
                    return True, np.zeros((32, 32, 3), dtype=np.uint8)
                return False, None

            def set(self, prop, value) -> bool:
                return True

            def get(self, prop) -> float:
                return 30.0

            def release(self) -> None:
                return None

        source = WebcamSource("auto")
        fake_cv2 = mock.Mock(CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5)
        source._cv2 = fake_cv2
        source._load_cv2 = mock.Mock(return_value=fake_cv2)

        from meta_watcher import sources as sources_module

        with mock.patch.object(
            sources_module,
            "_list_linux_webcams",
            return_value=[sources_module.WebcamDevice(index=7, label="cam", path="/dev/video7")],
        ), mock.patch("meta_watcher.sources.sys.platform", "linux"):
            source._open_capture = mock.Mock(side_effect=lambda index: FakeCapture(index, index == 7))
            source.open()

        self.assertEqual(source.capture_value, 7)
        self.assertIsNotNone(source._capture)


if __name__ == "__main__":
    unittest.main()
