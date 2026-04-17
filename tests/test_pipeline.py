from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from meta_watcher.config import default_config
from meta_watcher.core import ClipRecorder, Detection, VideoFrame
from meta_watcher.pipeline import StreamProcessor


def frame(index: int, timestamp: float | None = None) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((96, 96, 3), dtype=np.uint8),
        timestamp=float(index if timestamp is None else timestamp),
        frame_index=index,
        source_id="demo",
        fps=1.0,
    )


class MemorySink:
    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        self.path = path
        self.frames: list[np.ndarray] = []

    def write(self, image: np.ndarray) -> None:
        self.frames.append(np.array(image, copy=True))

    def close(self) -> None:
        return None


class FakeProvider:
    def __init__(self, people_script: dict[int, list[Detection]], inventory_script: dict[int, list[Detection]]) -> None:
        self.people_script = people_script
        self.inventory_script = inventory_script
        self.started = False

    def warmup(self) -> None:
        return None

    def detect_text_prompts(self, frame: VideoFrame, prompts) -> list[Detection]:
        if "person" in prompts:
            return [self._clone(det) for det in self.people_script.get(frame.frame_index, [])]
        return [self._clone(det) for det in self.inventory_script.get(frame.frame_index, [])]

    def start_tracking(self, frame: VideoFrame, prompts) -> list[Detection]:
        self.started = True
        return self.detect_text_prompts(frame, prompts)

    def track_next(self, frame: VideoFrame) -> list[Detection]:
        return [self._clone(det) for det in self.people_script.get(frame.frame_index, [])]

    def shutdown(self) -> None:
        return None

    def _clone(self, detection: Detection) -> Detection:
        return Detection(label=detection.label, confidence=detection.confidence, bbox=detection.bbox)


class FakeProposer:
    def __init__(self, labels_by_call: dict[int, dict[str, float]]) -> None:
        self.labels_by_call = labels_by_call
        self.call_count = 0

    def propose(self, image: np.ndarray) -> dict[str, float]:
        value = self.labels_by_call.get(self.call_count, {})
        self.call_count += 1
        return value


class PipelineTests(unittest.TestCase):
    def _build_processor(self, provider: FakeProvider, proposer: FakeProposer, tempdir: str, *, auto_rescan: bool) -> StreamProcessor:
        config = default_config()
        config.inventory.auto_rescan = auto_rescan
        config.inventory.required_samples = 2
        config.timings.person_confirmation_frames = 3
        config.timings.empty_scene_rescan_seconds = 2.0
        config.timings.pre_roll_seconds = 0.0
        config.timings.post_roll_seconds = 0.0
        config.thresholds.person_confidence = 0.1
        recorder = ClipRecorder(tempdir, pre_roll_seconds=0.0, post_roll_seconds=0.0, sink_factory=MemorySink)
        return StreamProcessor(config, provider, proposer, recorder)

    def test_inventory_then_person_event_then_autorescan(self) -> None:
        people = {
            3: [Detection("person", 0.9, (20, 10, 50, 70))],
            4: [Detection("person", 0.92, (21, 10, 51, 70))],
            5: [Detection("person", 0.93, (22, 10, 52, 70))],
        }
        inventory = {
            1: [Detection("chair", 0.8, (10, 20, 30, 50))],
            2: [Detection("chair", 0.82, (10, 20, 30, 50))],
            8: [Detection("plant", 0.75, (60, 10, 80, 50))],
            9: [Detection("plant", 0.77, (60, 10, 80, 50))],
        }
        proposer = FakeProposer(
            {
                0: {"chair": 0.8},
                1: {"chair": 0.82},
                3: {"plant": 0.75},
                4: {"plant": 0.77},
            }
        )
        provider = FakeProvider(people, inventory)
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(provider, proposer, tempdir, auto_rescan=True)
            snapshots = [processor.process_frame(frame(i)) for i in range(10)]

        self.assertEqual([item.label for item in snapshots[2].inventory_items], ["chair"])
        self.assertEqual(snapshots[5].mode, "person_present")
        self.assertTrue(provider.started)
        self.assertEqual(snapshots[7].mode, "cooldown")
        self.assertEqual(snapshots[8].mode, "inventory")
        self.assertEqual([item.label for item in snapshots[9].inventory_items], ["plant"])
        self.assertEqual(len(snapshots[8].completed_clips), 1)

    def test_manual_rescan_reenables_inventory_when_auto_rescan_is_off(self) -> None:
        people = {
            2: [Detection("person", 0.9, (20, 10, 50, 70))],
            3: [Detection("person", 0.92, (21, 10, 51, 70))],
            4: [Detection("person", 0.93, (22, 10, 52, 70))],
        }
        proposer = FakeProposer(
            {
                0: {"chair": 0.8},
                1: {"chair": 0.82},
                2: {"lamp": 0.8},
                3: {"lamp": 0.84},
            }
        )
        provider = FakeProvider(people, {})
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(provider, proposer, tempdir, auto_rescan=False)
            snapshots = [processor.process_frame(frame(i)) for i in range(8)]
            self.assertFalse(snapshots[-1].inventory_active)
            processor.request_manual_rescan()
            manual = processor.process_frame(frame(8))
            stable = processor.process_frame(frame(9))

        self.assertTrue(manual.inventory_active)
        self.assertEqual([item.label for item in stable.inventory_items], ["lamp"])


if __name__ == "__main__":
    unittest.main()
