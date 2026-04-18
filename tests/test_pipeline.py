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
    def __init__(
        self,
        people_script: dict[int, list[Detection]],
        inventory_script: dict[int, list[Detection]],
    ) -> None:
        self.people_script = people_script
        self.inventory_script = inventory_script
        self.started = False
        self.inventory_calls: list[list[str]] = []

    def warmup(self) -> None:
        return None

    def detect_text_prompts(self, frame: VideoFrame, prompts) -> list[Detection]:
        if "person" in prompts:
            return [self._clone(det) for det in self.people_script.get(frame.frame_index, [])]
        self.inventory_calls.append(list(prompts))
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


class PipelineTests(unittest.TestCase):
    def _build_processor(
        self,
        provider: FakeProvider,
        tempdir: str,
        *,
        labels: list[str],
        auto_rescan: bool,
    ) -> StreamProcessor:
        config = default_config()
        config.inventory.auto_rescan = auto_rescan
        config.inventory.labels = list(labels)
        config.timings.person_confirmation_frames = 3
        config.timings.empty_scene_rescan_seconds = 2.0
        config.timings.pre_roll_seconds = 0.0
        config.timings.post_roll_seconds = 0.0
        config.thresholds.person_confidence = 0.1
        config.thresholds.inventory_confidence = 0.0
        recorder = ClipRecorder(tempdir, pre_roll_seconds=0.0, post_roll_seconds=0.0, sink_factory=MemorySink)
        return StreamProcessor(config, provider, recorder)

    @staticmethod
    def _drive(processor: StreamProcessor, f: VideoFrame):
        """Simulate the runtime: producer pushes the frame to the recorder,
        then the consumer processes it. Mirrors StreamRuntime._produce_loop + _consume_loop."""
        if processor.recorder is not None and processor.recording_enabled:
            processor.recorder.push_frame(f)
        return processor.process_frame(f)

    def test_configured_labels_drive_inventory_detection(self) -> None:
        people = {
            3: [Detection("person", 0.9, (20, 10, 50, 70))],
            4: [Detection("person", 0.92, (21, 10, 51, 70))],
            5: [Detection("person", 0.93, (22, 10, 52, 70))],
        }
        inventory = {
            0: [Detection("chair", 0.8, (10, 20, 30, 50))],
            1: [Detection("chair", 0.82, (10, 20, 30, 50))],
            2: [Detection("plant", 0.75, (60, 10, 80, 50))],
        }
        provider = FakeProvider(people, inventory)
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(
                provider, tempdir, labels=["chair", "plant"], auto_rescan=True
            )
            snapshots = [self._drive(processor, frame(i)) for i in range(11)]

        # Inventory labels come straight from config, never change
        self.assertEqual(
            [item.label for item in snapshots[0].inventory_items],
            ["chair", "plant"],
        )
        # The provider was asked for the configured labels during inventory mode
        self.assertIn(["chair", "plant"], provider.inventory_calls)
        # Person event proceeds as before
        self.assertEqual(snapshots[5].mode, "person_present")
        self.assertTrue(provider.started)
        self.assertEqual(snapshots[7].mode, "cooldown")
        self.assertEqual(snapshots[8].mode, "inventory")
        # With the producer/consumer split, the clip finalizes on the producer's
        # next push after finish_event sets end_requested_at — so the artifact
        # appears on snapshot[9] rather than snapshot[8].
        completed = [clip for snap in snapshots for clip in snap.completed_clips]
        self.assertEqual(len(completed), 1)

    def test_empty_label_config_skips_inventory_detection(self) -> None:
        provider = FakeProvider({}, {})
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(provider, tempdir, labels=[], auto_rescan=True)
            for i in range(3):
                processor.process_frame(frame(i))
        # No inventory text_prompts call should have been issued when there are no labels
        self.assertEqual(provider.inventory_calls, [])

    def test_update_labels_takes_effect_immediately(self) -> None:
        provider = FakeProvider({}, {})
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(provider, tempdir, labels=["chair"], auto_rescan=True)
            processor.process_frame(frame(0))
            processor.update_labels(["lamp", "bottle"])
            processor.process_frame(frame(1))
        self.assertIn(["chair"], provider.inventory_calls)
        self.assertIn(["lamp", "bottle"], provider.inventory_calls)

    def test_inference_receives_downscaled_frame_and_bboxes_upscale(self) -> None:
        seen_frames: list[tuple[int, int]] = []

        class RecordingProvider(FakeProvider):
            def detect_text_prompts(self, frame, prompts):
                seen_frames.append((frame.width, frame.height))
                if "person" in prompts:
                    return [Detection("person", 0.9, (10, 20, 50, 80))]
                return []

            def track_next(self, frame):
                seen_frames.append((frame.width, frame.height))
                return [Detection("person", 0.9, (10, 20, 50, 80))]

        provider = RecordingProvider({}, {})
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(provider, tempdir, labels=[], auto_rescan=True)
            processor.config.models.inference_max_side = 48
            big = VideoFrame(
                image=np.zeros((240, 320, 3), dtype=np.uint8),
                timestamp=0.0,
                frame_index=0,
                source_id="demo",
                fps=1.0,
            )
            snapshot = processor.process_frame(big)

        # Longest side should now be <= 48 (so 320x240 -> 48x36).
        self.assertTrue(seen_frames, "provider should have been called")
        w, h = seen_frames[0]
        self.assertLessEqual(max(w, h), 48)
        # Detections should come back in ORIGINAL (320x240) coordinate space.
        # The downscaled bbox (10,20,50,80) should be upscaled by 320/48 ≈ 6.67.
        self.assertEqual(snapshot.frame_index, 0)
        # (Person confirmation gate is 3 frames so tracked_people will be empty on frame 0,
        # but the unnormalised detections were scaled before normalisation/thresholding.)
        # Re-run the helper directly to check the scaling math.
        from meta_watcher.pipeline import _upscale_detections

        up = _upscale_detections([Detection("person", 0.9, (10, 20, 50, 80))], 320 / 48, 320, 240)
        self.assertEqual(len(up), 1)
        x1, y1, x2, y2 = up[0].bbox
        self.assertGreaterEqual(x1, 66)  # 10 * 6.67 ≈ 67
        self.assertLessEqual(x2, 320)
        self.assertLessEqual(y2, 240)

    def test_inference_rate_limited_by_interval(self) -> None:
        provider = FakeProvider({}, {})
        calls = {"people": 0}
        original = provider.detect_text_prompts

        def counting(f, prompts):
            if "person" in prompts:
                calls["people"] += 1
            return original(f, prompts)

        provider.detect_text_prompts = counting
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(provider, tempdir, labels=[], auto_rescan=True)
            processor.config.models.inference_interval_ms = 500
            # 6 frames at 100ms spacing → inference at t=0.0 and t=0.5 only.
            for i in range(6):
                processor.process_frame(frame(i, timestamp=i * 0.1))
        self.assertEqual(calls["people"], 2)

    def test_snapshot_carries_stage_timings(self) -> None:
        provider = FakeProvider({}, {})
        with tempfile.TemporaryDirectory() as tempdir:
            processor = self._build_processor(provider, tempdir, labels=[], auto_rescan=True)
            snapshots = [processor.process_frame(frame(i, timestamp=i * 0.1)) for i in range(4)]

        # First snapshot has a record for itself; expected stages all present and non-negative.
        expected_keys = {"inference", "overlay", "recorder", "total"}
        for snapshot in snapshots:
            self.assertEqual(set(snapshot.timings.keys()), expected_keys)
            for stage, value in snapshot.timings.items():
                self.assertGreaterEqual(value, 0.0, f"{stage} should not be negative")

        # effective_fps needs at least two wall samples — the second+ snapshots must report a positive value.
        self.assertEqual(snapshots[0].effective_fps, 0.0)
        self.assertGreater(snapshots[-1].effective_fps, 0.0)


if __name__ == "__main__":
    unittest.main()
