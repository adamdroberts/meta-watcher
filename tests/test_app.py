from __future__ import annotations

import importlib
import json
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from meta_watcher.config import default_config
from meta_watcher.core import (
    Detection,
    InventoryItem,
    PipelineSnapshot,
    VideoFrame,
)
from meta_watcher.inference import InferenceProvider
from meta_watcher.sources import VideoSource
from meta_watcher.web import RuntimeState, build_app


class _FakeProvider(InferenceProvider):
    def warmup(self) -> None:
        pass

    def detect_text_prompts(self, frame, prompts):
        return []

    def start_tracking(self, frame, prompts):
        return []

    def track_next(self, frame):
        return []

    def shutdown(self) -> None:
        pass


class _ScriptedSource(VideoSource):
    def __init__(self) -> None:
        self._frames: "queue.Queue[VideoFrame | None]" = queue.Queue()

    @property
    def source_id(self) -> str:
        return "fake"

    def open(self) -> None:
        pass

    def read(self) -> VideoFrame | None:
        return self._frames.get()

    def close(self) -> None:
        pass

    def push(self, frame: VideoFrame | None) -> None:
        self._frames.put(frame)


def _blank_frame(index: int = 0) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((48, 64, 3), dtype=np.uint8),
        timestamp=float(index) / 10.0,
        frame_index=index,
        source_id="fake",
        fps=10.0,
    )


def _make_snapshot(mode: str = "inventory", frame_index: int = 7) -> PipelineSnapshot:
    overlay = np.zeros((48, 64, 3), dtype=np.uint8)
    return PipelineSnapshot(
        mode=mode,
        frame_index=frame_index,
        source_id="fake",
        overlay=overlay,
        people=[],
        inventory_detections=[],
        inventory_items=[
            InventoryItem(label="chair", confidence=0.7, samples=3, last_seen=1.0),
        ],
        inventory_active=True,
        recording_active=False,
        completed_clips=[],
        status_text="scene objects: 1",
    )


class RuntimeStateTests(unittest.TestCase):
    def test_initial_snapshot_payload_is_idle(self) -> None:
        state = RuntimeState(default_config())
        payload = state.snapshot_payload()
        self.assertFalse(payload["running"])
        self.assertEqual(payload["mode"], "idle")
        self.assertEqual(payload["inventory_items"], [])
        self.assertIsNone(payload["error"])

    def test_config_patch_updates_nested_fields(self) -> None:
        state = RuntimeState(default_config())
        patched = state.update_config({
            "source": {"kind": "file", "value": "/tmp/example.mp4"},
            "thresholds": {"person_confidence": 0.42},
            "inventory": {"auto_rescan": False},
        })
        self.assertEqual(patched.source.kind, "file")
        self.assertEqual(patched.source.value, "/tmp/example.mp4")
        self.assertAlmostEqual(patched.thresholds.person_confidence, 0.42)
        self.assertFalse(patched.inventory.auto_rescan)

    def test_snapshot_callback_updates_payload_and_jpeg(self) -> None:
        state = RuntimeState(default_config())
        snapshot = _make_snapshot(mode="person_present", frame_index=3)
        state._on_snapshot(snapshot)  # exercise internal callback path
        payload = state.snapshot_payload()
        self.assertEqual(payload["mode"], "person_present")
        self.assertEqual(payload["frame_index"], 3)
        self.assertEqual(payload["inventory_items"][0]["label"], "chair")
        jpeg = state.latest_jpeg()
        self.assertIsNotNone(jpeg)
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_start_and_stop_runtime_with_fakes(self) -> None:
        state = RuntimeState(default_config())
        source = _ScriptedSource()
        provider = _FakeProvider()

        state._provider_factory = lambda config: provider
        state._source_factory = lambda config: source

        state.start()
        self.assertTrue(state.is_running())

        source.push(_blank_frame(0))
        deadline = time.monotonic() + 1.0
        while state.snapshot_payload()["frame_index"] is None and time.monotonic() < deadline:
            time.sleep(0.02)

        self.assertIsNotNone(state.snapshot_payload()["frame_index"])

        source.push(None)  # signal producer to exit
        state.stop()
        self.assertFalse(state.is_running())


class WebServerTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi is not installed; skipping web tests.")
        self.TestClient = TestClient

    def _build_client(self) -> tuple[Any, RuntimeState]:
        state = RuntimeState(default_config())
        state._provider_factory = lambda config: _FakeProvider()
        state._source_factory = lambda config: _ScriptedSource()
        app = build_app(state)
        return self.TestClient(app), state

    def test_get_config_returns_defaults(self) -> None:
        client, _state = self._build_client()
        response = client.get("/api/config")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source"]["kind"], "webcam")
        self.assertIn("thresholds", payload)

    def test_put_config_merges_patch(self) -> None:
        client, state = self._build_client()
        response = client.put(
            "/api/config",
            json={"thresholds": {"person_confidence": 0.55}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertAlmostEqual(state.config().thresholds.person_confidence, 0.55)

    def test_get_snapshot_returns_idle_state(self) -> None:
        client, _state = self._build_client()
        response = client.get("/api/snapshot")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["running"])
        self.assertEqual(payload["mode"], "idle")

    def test_rescan_is_a_noop_when_idle(self) -> None:
        client, _state = self._build_client()
        response = client.post("/api/runtime/rescan")
        self.assertEqual(response.status_code, 200)

    def test_frame_jpg_returns_placeholder_when_no_frames(self) -> None:
        client, _state = self._build_client()
        response = client.get("/frame.jpg")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertTrue(response.content.startswith(b"\xff\xd8"))


class SettingsApiTests(unittest.TestCase):
    """End-to-end tests for the /api/config/files, /switch, /save endpoints."""

    def setUp(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi is not installed; skipping web tests.")
        self.TestClient = TestClient
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.root = Path(self._tempdir.name)

    def _write_config(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.root / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def _build_client(
        self,
        *,
        active: Path | None = None,
    ) -> tuple[Any, RuntimeState]:
        state = RuntimeState(
            default_config(),
            active_config_path=active,
            search_dirs=[self.root],
        )
        state._provider_factory = lambda config: _FakeProvider()
        state._source_factory = lambda config: _ScriptedSource()
        app = build_app(state)
        return self.TestClient(app), state

    def test_list_config_files_returns_active_and_entries(self) -> None:
        a = self._write_config("a.json", {"source": {"kind": "rtsp"}})
        b = self._write_config("b.json", {"source": {"kind": "file"}})
        client, state = self._build_client(active=a)
        response = client.get("/api/config/files")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active"], str(a.resolve()))
        names = sorted(entry["name"] for entry in payload["files"])
        self.assertEqual(names, ["a.json", "b.json"])
        active_entries = [e for e in payload["files"] if e["active"]]
        self.assertEqual(len(active_entries), 1)
        self.assertEqual(active_entries[0]["name"], "a.json")

    def test_switch_config_loads_new_values(self) -> None:
        a = self._write_config("a.json", {"source": {"kind": "rtsp"}})
        b = self._write_config(
            "b.json", {"source": {"kind": "file", "value": "/tmp/v.mp4"}}
        )
        client, state = self._build_client(active=a)
        response = client.post("/api/config/switch", json={"path": str(b)})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["active"], str(b.resolve()))
        self.assertEqual(body["config"]["source"]["kind"], "file")
        self.assertEqual(body["config"]["source"]["value"], "/tmp/v.mp4")
        self.assertFalse(body["requires_restart"])
        # Active path updated on the state too.
        self.assertEqual(state.active_config_path(), b.resolve())

    def test_switch_config_flags_requires_restart_when_running(self) -> None:
        a = self._write_config("a.json", {"source": {"kind": "rtsp"}})
        b = self._write_config("b.json", {"source": {"kind": "file"}})
        client, state = self._build_client(active=a)
        # Cheap mock for running state — tests don't need a full StreamRuntime.
        state._runtime = object()  # type: ignore[assignment]
        try:
            response = client.post("/api/config/switch", json={"path": str(b)})
        finally:
            state._runtime = None
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["running"])
        self.assertTrue(body["requires_restart"])

    def test_switch_config_rejects_path_outside_search_dirs(self) -> None:
        a = self._write_config("a.json", {"source": {"kind": "rtsp"}})
        client, _state = self._build_client(active=a)
        response = client.post("/api/config/switch", json={"path": "/etc/passwd"})
        self.assertEqual(response.status_code, 403)
        self.assertIn("outside", response.json()["error"])

    def test_switch_config_missing_file_returns_404(self) -> None:
        a = self._write_config("a.json", {"source": {"kind": "rtsp"}})
        client, _state = self._build_client(active=a)
        response = client.post(
            "/api/config/switch", json={"path": str(self.root / "nope.json")}
        )
        self.assertEqual(response.status_code, 404)

    def test_save_config_writes_atomically(self) -> None:
        a = self._write_config("a.json", {"source": {"kind": "rtsp"}})
        client, state = self._build_client(active=a)
        put = client.put(
            "/api/config",
            json={"thresholds": {"person_confidence": 0.42}},
        )
        self.assertEqual(put.status_code, 200)
        save = client.post("/api/config/save")
        self.assertEqual(save.status_code, 200)
        body = save.json()
        self.assertEqual(body["path"], str(a.resolve()))
        self.assertGreater(body["bytes_written"], 0)

        # Round-trip the persisted file.
        persisted = json.loads(a.read_text(encoding="utf-8"))
        self.assertAlmostEqual(persisted["thresholds"]["person_confidence"], 0.42)
        self.assertTrue(a.read_text(encoding="utf-8").endswith("\n"))
        # No tmp file left behind.
        self.assertFalse((self.root / "a.json.tmp").exists())

    def test_save_config_without_active_path_returns_400(self) -> None:
        client, _state = self._build_client(active=None)
        response = client.post("/api/config/save")
        self.assertEqual(response.status_code, 400)
        self.assertIn("active", response.json()["error"].lower())

    def test_put_config_invalid_section_shape_returns_400(self) -> None:
        a = self._write_config("a.json", {})
        client, _state = self._build_client(active=a)
        # A known top-level section must be a JSON object; sending a scalar
        # used to silently drop; now it surfaces as 400 so UIs catch bugs.
        response = client.put("/api/config", json={"thresholds": 42})
        self.assertEqual(response.status_code, 400)
        self.assertIn("thresholds", response.json()["error"])


if __name__ == "__main__":
    unittest.main()
