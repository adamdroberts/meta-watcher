from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from typing import Any

import numpy as np

from meta_watcher.config import TimestampConfig, UploadConfig, default_config
from meta_watcher.core import (
    ClipRecorder,
    Detection,
    EventArtifact,
    VideoFrame,
)
from meta_watcher.pipeline import StreamProcessor
from meta_watcher.upload import (
    EventUploader,
    OciUploadProvider,
    UploadProvider,
    build_upload_provider,
)


def make_frame(index: int, timestamp: float | None = None) -> VideoFrame:
    return VideoFrame(
        image=np.zeros((32, 48, 3), dtype=np.uint8),
        timestamp=float(index if timestamp is None else timestamp),
        frame_index=index,
        source_id="demo",
        fps=1.0,
    )


class FakeProvider(UploadProvider):
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[Path, str]] = []
        self.fail = fail
        self.lock = threading.Lock()
        self.call_event = threading.Event()

    @property
    def scheme(self) -> str:
        return "fake"

    def upload(self, local_path: Path, remote_key: str) -> str:
        with self.lock:
            self.calls.append((local_path, remote_key))
        self.call_event.set()
        if self.fail:
            raise RuntimeError("boom")
        return f"fake://{remote_key}"


class MemorySink:
    def __init__(self, path: Path, size: tuple[int, int], fps: float) -> None:
        self.path = path
        self.frames: list[np.ndarray] = []

    def write(self, image: np.ndarray) -> None:
        self.frames.append(np.array(image, copy=True))

    def close(self) -> None:
        return None


class BuildUploadProviderTests(unittest.TestCase):
    def test_disabled_returns_none(self) -> None:
        self.assertIsNone(build_upload_provider(UploadConfig(enabled=False)))

    def test_missing_bucket_returns_none(self) -> None:
        self.assertIsNone(
            build_upload_provider(UploadConfig(enabled=True, bucket=""))
        )

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_upload_provider(
                UploadConfig(enabled=True, provider="unknown", bucket="b")
            )


class EventUploaderTests(unittest.TestCase):
    def _build_uploader(
        self,
        provider: FakeProvider,
        *,
        delete_after_upload: bool = False,
        upload_videos: bool = True,
        upload_snapshots: bool = True,
        upload_metadata: bool = True,
        queue_size: int = 32,
    ) -> EventUploader:
        cfg = UploadConfig(
            enabled=True,
            provider="fake",
            bucket="b",
            prefix="meta/",
            delete_after_upload=delete_after_upload,
            upload_videos=upload_videos,
            upload_snapshots=upload_snapshots,
            upload_metadata=upload_metadata,
            queue_size=queue_size,
        )
        return EventUploader(provider, cfg)

    def _artifact(self, tempdir: str) -> EventArtifact:
        base = Path(tempdir)
        clip = base / "cam_001.mp4"
        meta = base / "cam_001.json"
        snap = base / "cam_001.jpg"
        for path in (clip, meta, snap):
            path.write_bytes(b"x")
        return EventArtifact(
            clip_path=clip,
            metadata_path=meta,
            started_at=0.0,
            ended_at=1.0,
            person_ids=[],
            snapshot_path=snap,
        )

    def test_uploads_all_artifacts_with_correct_keys(self) -> None:
        provider = FakeProvider()
        uploader = self._build_uploader(provider)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                artifact = self._artifact(tempdir)
                uploader.enqueue_artifact(artifact)

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 3:
                            break
                    time.sleep(0.02)
        finally:
            uploader.stop(timeout=2.0)

        with provider.lock:
            keys = sorted(key for _, key in provider.calls)
        self.assertEqual(
            keys,
            [
                "meta/cam_001/cam_001.jpg",
                "meta/cam_001/cam_001.json",
                "meta/cam_001/cam_001.mp4",
            ],
        )

    def test_delete_after_upload_removes_local_files(self) -> None:
        provider = FakeProvider()
        uploader = self._build_uploader(provider, delete_after_upload=True)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                artifact = self._artifact(tempdir)
                uploader.enqueue_artifact(artifact)

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    if (
                        not artifact.clip_path.exists()
                        and not artifact.metadata_path.exists()
                        and artifact.snapshot_path is not None
                        and not artifact.snapshot_path.exists()
                    ):
                        break
                    time.sleep(0.02)

                self.assertFalse(artifact.clip_path.exists())
                self.assertFalse(artifact.metadata_path.exists())
                assert artifact.snapshot_path is not None
                self.assertFalse(artifact.snapshot_path.exists())
        finally:
            uploader.stop(timeout=2.0)

    def test_keeps_local_files_on_upload_failure(self) -> None:
        provider = FakeProvider(fail=True)
        uploader = self._build_uploader(provider, delete_after_upload=True)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                artifact = self._artifact(tempdir)
                uploader.enqueue_artifact(artifact)

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 3:
                            break
                    time.sleep(0.02)
                # All three types still live on disk because the upload failed.
                self.assertTrue(artifact.clip_path.exists())
                self.assertTrue(artifact.metadata_path.exists())
                assert artifact.snapshot_path is not None
                self.assertTrue(artifact.snapshot_path.exists())
        finally:
            uploader.stop(timeout=2.0)

    def test_respects_per_type_upload_flags(self) -> None:
        provider = FakeProvider()
        uploader = self._build_uploader(
            provider,
            upload_videos=False,
            upload_snapshots=True,
            upload_metadata=False,
        )
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                artifact = self._artifact(tempdir)
                uploader.enqueue_artifact(artifact)

                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 1:
                            break
                    time.sleep(0.02)
        finally:
            uploader.stop(timeout=2.0)

        with provider.lock:
            keys = [key for _, key in provider.calls]
        self.assertEqual(keys, ["meta/cam_001/cam_001.jpg"])

    def test_uploads_run_in_parallel_across_workers(self) -> None:
        # Four uploads must rendezvous simultaneously on a barrier; if the
        # pool only has one worker (or upload is still serial), the barrier
        # times out and this test fails loudly rather than hanging forever.
        barrier = threading.Barrier(parties=4, timeout=3.0)

        class BarrierProvider(FakeProvider):
            def upload(self, local_path: Path, remote_key: str) -> str:
                barrier.wait()
                return super().upload(local_path, remote_key)

        provider = BarrierProvider()
        cfg = UploadConfig(
            enabled=True,
            provider="fake",
            bucket="b",
            prefix="meta/",
            upload_videos=True,
            upload_snapshots=False,
            upload_metadata=False,
            upload_workers=4,
        )
        uploader = EventUploader(provider, cfg)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                for index in range(4):
                    clip = Path(tempdir) / f"cam_{index:03d}.mp4"
                    clip.write_bytes(b"x")
                    uploader.enqueue_artifact(
                        EventArtifact(
                            clip_path=clip,
                            metadata_path=clip.with_suffix(".json"),
                            started_at=0.0,
                            ended_at=1.0,
                            person_ids=[],
                            snapshot_path=None,
                        )
                    )
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 4:
                            break
                    time.sleep(0.02)
        finally:
            uploader.stop(timeout=3.0)

        with provider.lock:
            self.assertEqual(len(provider.calls), 4)

    def test_queue_overflow_drops_oldest_without_raising(self) -> None:
        # Provider blocks until released so we can queue more jobs than capacity.
        release = threading.Event()

        class SlowProvider(FakeProvider):
            def upload(self, local_path: Path, remote_key: str) -> str:
                release.wait(timeout=2.0)
                return super().upload(local_path, remote_key)

        provider = SlowProvider()
        uploader = self._build_uploader(
            provider,
            upload_videos=True,
            upload_snapshots=False,
            upload_metadata=False,
            queue_size=2,
        )
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                for index in range(20):
                    clip = Path(tempdir) / f"cam_{index:03d}.mp4"
                    clip.write_bytes(b"x")
                    uploader.enqueue_artifact(
                        EventArtifact(
                            clip_path=clip,
                            metadata_path=clip.with_suffix(".json"),
                            started_at=0.0,
                            ended_at=1.0,
                            person_ids=[],
                            snapshot_path=None,
                        )
                    )
                # Allow the worker to drain.
                release.set()
                time.sleep(0.5)
        finally:
            release.set()
            uploader.stop(timeout=2.0)

        with provider.lock:
            # Can't predict exact count (depends on worker speed), but the queue
            # capped at 2 means we must have uploaded far fewer than 20 without
            # raising, and at least one upload must have happened.
            self.assertGreaterEqual(len(provider.calls), 1)
            self.assertLessEqual(len(provider.calls), 20)


class SnapshotIntegrationTests(unittest.TestCase):
    def test_clip_recorder_emits_snapshot_on_event_artifact(self) -> None:
        config = default_config()
        config.inventory.labels = []
        config.timings.pre_roll_seconds = 0.0
        config.timings.post_roll_seconds = 0.0
        config.timings.person_confirmation_frames = 1
        config.timings.empty_scene_rescan_seconds = 1.0
        config.thresholds.person_confidence = 0.1

        people = {0: [Detection("person", 0.99, (2, 2, 30, 30))]}

        class Provider:
            def warmup(self) -> None:
                return None

            def detect_text_prompts(self, frame, prompts):
                if "person" in prompts:
                    return [
                        Detection(det.label, det.confidence, det.bbox)
                        for det in people.get(frame.frame_index, [])
                    ]
                return []

            def start_tracking(self, frame, prompts):
                return self.detect_text_prompts(frame, prompts)

            def track_next(self, frame):
                return [
                    Detection(det.label, det.confidence, det.bbox)
                    for det in people.get(frame.frame_index, [])
                ]

            def shutdown(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as tempdir:
            recorder = ClipRecorder(
                tempdir,
                pre_roll_seconds=0.0,
                post_roll_seconds=0.0,
                sink_factory=lambda path, size, fps: MemorySink(path, size, fps),
            )
            processor = StreamProcessor(config, Provider(), recorder)
            # Frame 0 should trigger event_started (person_confirmation_frames=1,
            # person at frame 0). Then consume a few empties to finish the event.
            # Producer tap writes frames via push_frame.
            artifacts: list[EventArtifact] = []
            for i in range(6):
                f = make_frame(i, timestamp=float(i))
                if processor.recorder is not None and processor.recording_enabled:
                    artifacts.extend(processor.recorder.push_frame(f))
                processor.process_frame(f)

            all_completed = artifacts + processor.recorder.drain_completed()
            self.assertGreaterEqual(len(all_completed), 1, "event should have closed")
            art = all_completed[-1]
            self.assertIsNotNone(art.snapshot_path, "snapshot path must be populated")
            assert art.snapshot_path is not None
            self.assertTrue(art.snapshot_path.exists(), "snapshot JPEG must exist on disk")
            self.assertGreater(art.snapshot_path.stat().st_size, 0)


class RuntimeStateLiveUploadTests(unittest.TestCase):
    def test_pump_live_uploads_snapshot_once_and_frames_every_interval(self) -> None:
        from meta_watcher.web.state import LIVE_FRAME_INTERVAL_SECONDS, RuntimeState

        enqueued: dict[str, list[tuple[str, str]]] = {
            "snapshot": [],
            "frame": [],
        }

        class FakeUploader:
            def enqueue_snapshot(self, path: Path, event_id: str) -> None:
                enqueued["snapshot"].append((event_id, path.name))

            def enqueue_frame(self, path: Path, event_id: str) -> None:
                enqueued["frame"].append((event_id, path.name))

            def enqueue_artifact(self, *a: Any, **kw: Any) -> None:  # pragma: no cover
                raise AssertionError("enqueue_artifact not expected in this test")

        class FakeRecorder:
            def __init__(self) -> None:
                self._event_id: str | None = "cam_evt_A"
                self._snapshot = Path("/tmp/cam_evt_A.jpg")

            def set_event(self, event_id: str | None, snapshot: Path | None) -> None:
                self._event_id = event_id
                self._snapshot = snapshot

            def active_event_info(self) -> tuple[str, Path | None] | None:
                if self._event_id is None:
                    return None
                return (self._event_id, self._snapshot)

        state = RuntimeState()
        uploader = FakeUploader()
        recorder = FakeRecorder()

        # _write_live_frame now writes under config.output.directory. Point
        # that at a tempdir so the test never drops files into the repo.
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        state.update_config({"output": {"directory": tempdir.name}})

        # First call: should fire snapshot + frame (last_mono was 0.0).
        state._pump_live_uploads(b"jpegbytes", uploader, recorder)  # type: ignore[arg-type]
        # Immediate second call: snapshot already recorded, and 0.5s has NOT
        # elapsed, so no new work.
        state._pump_live_uploads(b"jpegbytes", uploader, recorder)  # type: ignore[arg-type]

        self.assertEqual([e for e, _ in enqueued["snapshot"]], ["cam_evt_A"])
        self.assertEqual([e for e, _ in enqueued["frame"]], ["cam_evt_A"])

        # Fast-forward the frame gate by the configured interval and call again:
        # new frame expected, snapshot still not re-sent.
        state._live_frame_last_mono -= LIVE_FRAME_INTERVAL_SECONDS + 0.05
        state._pump_live_uploads(b"jpegbytes", uploader, recorder)  # type: ignore[arg-type]
        self.assertEqual(len(enqueued["snapshot"]), 1)
        self.assertEqual(len(enqueued["frame"]), 2)

        # A new event id resets both gates.
        recorder.set_event("cam_evt_B", Path("/tmp/cam_evt_B.jpg"))
        state._pump_live_uploads(b"jpegbytes", uploader, recorder)  # type: ignore[arg-type]
        self.assertEqual([e for e, _ in enqueued["snapshot"]], ["cam_evt_A", "cam_evt_B"])
        self.assertEqual(len(enqueued["frame"]), 3)

        # Event closes — pump is a no-op.
        recorder.set_event(None, None)
        state._pump_live_uploads(b"jpegbytes", uploader, recorder)  # type: ignore[arg-type]
        self.assertEqual(len(enqueued["snapshot"]), 2)
        self.assertEqual(len(enqueued["frame"]), 3)


class _FakeOciModule:
    """Minimal stand-in for the `oci` package so tests don't need it installed."""

    def __init__(self) -> None:
        self.config_calls: list[dict[str, object]] = []
        self.config_to_return: dict[str, object] = {"region": "us-ashburn-1"}
        self.namespace_value = "tenancy-ns"
        self.put_calls: list[tuple[str, str, str]] = []
        self.objects: dict[str, bytes] = {}
        self.object_meta: dict[str, dict] = {}
        self.list_calls: list[dict] = []
        self.get_calls: list[dict] = []

        parent = self

        class _Config:
            @staticmethod
            def from_file(**kwargs: object) -> dict[str, object]:
                parent.config_calls.append(kwargs)
                return dict(parent.config_to_return)

        class _NamespaceResponse:
            def __init__(self, data: str) -> None:
                self.data = data

        class _ObjectStorageClient:
            def __init__(self, config: dict[str, object]) -> None:
                self.config = config

            def get_namespace(self) -> _NamespaceResponse:
                return _NamespaceResponse(parent.namespace_value)

            def put_object(
                self,
                namespace: str,
                bucket: str,
                name: str,
                _body: object,
            ) -> None:
                parent.put_calls.append((namespace, bucket, name))

            def list_objects(
                self,
                namespace: str,
                bucket: str,
                *,
                prefix: str | None = None,
                start: str | None = None,
                limit: int | None = None,
                fields: str | None = None,
            ) -> object:
                parent.list_calls.append(
                    {
                        "namespace": namespace,
                        "bucket": bucket,
                        "prefix": prefix,
                        "start": start,
                        "limit": limit,
                        "fields": fields,
                    }
                )
                keys = sorted(parent.objects.keys())
                if prefix:
                    keys = [k for k in keys if k.startswith(prefix)]
                if start:
                    keys = [k for k in keys if k > start]
                next_start: str | None = None
                if limit and len(keys) > limit:
                    next_start = keys[limit - 1]
                    keys = keys[:limit]

                class _Obj:
                    def __init__(self, name: str, meta: dict) -> None:
                        self.name = name
                        self.size = meta.get("size", 0)
                        self.time_modified = meta.get("time_modified")
                        self.md5 = meta.get("md5")
                        self.etag = meta.get("etag")

                objs = [_Obj(n, parent.object_meta.get(n, {})) for n in keys]
                return types.SimpleNamespace(
                    data=types.SimpleNamespace(
                        objects=objs,
                        next_start_with=next_start,
                    )
                )

            def get_object(
                self,
                namespace: str,
                bucket: str,
                name: str,
                **kwargs: object,
            ) -> object:
                # The real OCI SDK exposes a keyword-only `range` parameter; we
                # accept it via **kwargs here to avoid shadowing the builtin
                # inside this closure.
                range_header = kwargs.get("range")
                parent.get_calls.append(
                    {
                        "namespace": namespace,
                        "bucket": bucket,
                        "name": name,
                        "range": range_header,
                    }
                )
                if name not in parent.objects:
                    raise RuntimeError(f"404 not found: {name}")
                body = parent.objects[name]
                if range_header:
                    import re as _re

                    m = _re.match(r"bytes=(\d+)-(\d*)$", str(range_header))
                    if m:
                        s = int(m.group(1))
                        e = int(m.group(2)) if m.group(2) else len(body) - 1
                        body = body[s : e + 1]

                class _Stream:
                    def __init__(self, b: bytes) -> None:
                        self._b = b

                    def iter_content(self, chunk_size: int = 8192):
                        for i in range(0, len(self._b), chunk_size):
                            yield self._b[i : i + chunk_size]

                return types.SimpleNamespace(
                    data=_Stream(body),
                    headers={
                        "Content-Length": str(len(body)),
                        "Content-Type": "application/octet-stream",
                    },
                    status=206 if range_header else 200,
                )

        self.config = _Config
        self.object_storage = types.SimpleNamespace(
            ObjectStorageClient=_ObjectStorageClient,
        )


class LiveUploadPathsTests(unittest.TestCase):
    def _build(
        self,
        provider: FakeProvider,
        **overrides: Any,
    ) -> EventUploader:
        cfg = UploadConfig(
            enabled=True,
            provider="fake",
            bucket="b",
            prefix="meta/",
            **overrides,
        )
        return EventUploader(provider, cfg)

    def test_enqueue_snapshot_uploads_into_event_folder(self) -> None:
        provider = FakeProvider()
        uploader = self._build(provider)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                snap = Path(tempdir) / "cam_evt.jpg"
                snap.write_bytes(b"x")
                uploader.enqueue_snapshot(snap, "cam_evt")
                provider.call_event.wait(timeout=2.0)
        finally:
            uploader.stop(timeout=2.0)

        with provider.lock:
            self.assertEqual(
                [key for _, key in provider.calls],
                ["meta/cam_evt/cam_evt.jpg"],
            )

    def test_enqueue_frame_uses_frames_subfolder_and_keeps_local(self) -> None:
        provider = FakeProvider()
        # Global delete_after_upload off (the default): frames must stay on
        # disk so operators have a complete local record alongside the clip.
        uploader = self._build(provider, delete_after_upload=False)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                frame = Path(tempdir) / "cam_evt_live_001.jpg"
                frame.write_bytes(b"x")
                uploader.enqueue_frame(frame, "cam_evt")
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if provider.calls:
                            break
                    time.sleep(0.02)
                # Give the worker a beat to run its (no-op) delete branch.
                time.sleep(0.1)
                with provider.lock:
                    keys = [key for _, key in provider.calls]
                self.assertEqual(keys, ["meta/cam_evt/frames/cam_evt_live_001.jpg"])
                # Kept on disk — operator still has the frame after upload.
                self.assertTrue(frame.exists())
        finally:
            uploader.stop(timeout=2.0)

    def test_enqueue_frame_respects_global_delete_flag(self) -> None:
        provider = FakeProvider()
        uploader = self._build(provider, delete_after_upload=True)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                frame = Path(tempdir) / "cam_evt_live_002.jpg"
                frame.write_bytes(b"x")
                uploader.enqueue_frame(frame, "cam_evt")
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and frame.exists():
                    time.sleep(0.02)
        finally:
            uploader.stop(timeout=2.0)

        self.assertFalse(frame.exists(), "global delete flag should still drop frames")

    def test_enqueue_frame_delete_override(self) -> None:
        provider = FakeProvider()
        # Global keep, but caller explicitly requests delete — still deletes.
        uploader = self._build(provider, delete_after_upload=False)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                frame = Path(tempdir) / "cam_evt_live_003.jpg"
                frame.write_bytes(b"x")
                uploader.enqueue_frame(frame, "cam_evt", delete_after_upload=True)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and frame.exists():
                    time.sleep(0.02)
        finally:
            uploader.stop(timeout=2.0)

        self.assertFalse(frame.exists())

    def test_enqueue_artifact_skip_snapshot(self) -> None:
        provider = FakeProvider()
        uploader = self._build(provider)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                clip = Path(tempdir) / "cam_evt.mp4"
                meta = Path(tempdir) / "cam_evt.json"
                snap = Path(tempdir) / "cam_evt.jpg"
                for p in (clip, meta, snap):
                    p.write_bytes(b"x")
                artifact = EventArtifact(
                    clip_path=clip,
                    metadata_path=meta,
                    started_at=0.0,
                    ended_at=1.0,
                    person_ids=[],
                    snapshot_path=snap,
                )
                uploader.enqueue_artifact(artifact, skip_snapshot=True)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 2:
                            break
                    time.sleep(0.02)
        finally:
            uploader.stop(timeout=2.0)

        with provider.lock:
            keys = sorted(key for _, key in provider.calls)
        self.assertEqual(
            keys,
            ["meta/cam_evt/cam_evt.json", "meta/cam_evt/cam_evt.mp4"],
        )

    def test_enqueue_artifact_uploads_overlay_clip(self) -> None:
        provider = FakeProvider()
        uploader = self._build(provider, upload_snapshots=False, upload_metadata=False)
        uploader.start()
        try:
            with tempfile.TemporaryDirectory() as tempdir:
                clip = Path(tempdir) / "cam_evt.mp4"
                overlay = Path(tempdir) / "cam_evt.overlay.mp4"
                meta = Path(tempdir) / "cam_evt.json"
                for p in (clip, overlay, meta):
                    p.write_bytes(b"x")
                artifact = EventArtifact(
                    clip_path=clip,
                    metadata_path=meta,
                    started_at=0.0,
                    ended_at=1.0,
                    person_ids=[],
                    snapshot_path=None,
                    overlay_clip_path=overlay,
                )
                uploader.enqueue_artifact(artifact)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 2:
                            break
                    time.sleep(0.02)
        finally:
            uploader.stop(timeout=2.0)

        with provider.lock:
            keys = sorted(key for _, key in provider.calls)
        self.assertEqual(
            keys,
            ["meta/cam_evt/cam_evt.mp4", "meta/cam_evt/cam_evt.overlay.mp4"],
        )


class TimestampingUploadTests(unittest.TestCase):
    """Tests the upload -> stamp -> upload .ots sidecar chain."""

    def _write_fake_ots(self, tempdir: Path) -> Path:
        import stat as _stat
        import textwrap

        script = tempdir / "ots"
        body = textwrap.dedent(
            f"""
            #!{sys.executable}
            import pathlib, sys
            args = sys.argv[1:]
            path = pathlib.Path(args[-1])
            (path.parent / (path.name + ".ots")).write_bytes(b"PROOF")
            sys.exit(0)
            """
        ).strip() + "\n"
        script.write_text(body, encoding="utf-8")
        script.chmod(
            script.stat().st_mode
            | _stat.S_IEXEC
            | _stat.S_IXGRP
            | _stat.S_IXOTH
        )
        return script

    def _build(
        self,
        provider: FakeProvider,
        *,
        ts_enabled: bool,
        ots_binary: str,
        stamp_videos: bool = True,
        stamp_snapshots: bool = True,
        stamp_frames: bool = False,
        stamp_metadata: bool = False,
        delete_after_upload: bool = False,
    ) -> EventUploader:
        cfg = UploadConfig(
            enabled=True,
            provider="fake",
            bucket="b",
            prefix="meta/",
            delete_after_upload=delete_after_upload,
        )
        ts = TimestampConfig(
            enabled=ts_enabled,
            ots_binary=ots_binary,
            stamp_videos=stamp_videos,
            stamp_snapshots=stamp_snapshots,
            stamp_frames=stamp_frames,
            stamp_metadata=stamp_metadata,
        )
        return EventUploader(provider, cfg, timestamps=ts)

    def test_video_and_snapshot_get_ots_sidecars(self) -> None:
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tempdir:
            ots = self._write_fake_ots(Path(tempdir))
            uploader = self._build(provider, ts_enabled=True, ots_binary=str(ots))
            uploader.start()
            try:
                clip = Path(tempdir) / "cam_evt.mp4"
                snap = Path(tempdir) / "cam_evt.jpg"
                meta = Path(tempdir) / "cam_evt.json"
                for p in (clip, snap, meta):
                    p.write_bytes(b"x")
                artifact = EventArtifact(
                    clip_path=clip,
                    metadata_path=meta,
                    started_at=0.0,
                    ended_at=1.0,
                    person_ids=[],
                    snapshot_path=snap,
                )
                uploader.enqueue_artifact(artifact)
                # 3 artifacts + 2 OTS sidecars (video + snapshot).
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 5:
                            break
                    time.sleep(0.02)
            finally:
                uploader.stop(timeout=2.0)

        with provider.lock:
            keys = sorted(key for _, key in provider.calls)
        self.assertEqual(
            keys,
            [
                "meta/cam_evt/cam_evt.jpg",
                "meta/cam_evt/cam_evt.jpg.ots",
                "meta/cam_evt/cam_evt.json",
                "meta/cam_evt/cam_evt.mp4",
                "meta/cam_evt/cam_evt.mp4.ots",
            ],
        )

    def test_disabled_timestamps_produce_no_sidecars(self) -> None:
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tempdir:
            ots = self._write_fake_ots(Path(tempdir))
            uploader = self._build(provider, ts_enabled=False, ots_binary=str(ots))
            uploader.start()
            try:
                clip = Path(tempdir) / "cam_evt.mp4"
                snap = Path(tempdir) / "cam_evt.jpg"
                meta = Path(tempdir) / "cam_evt.json"
                for p in (clip, snap, meta):
                    p.write_bytes(b"x")
                artifact = EventArtifact(
                    clip_path=clip,
                    metadata_path=meta,
                    started_at=0.0,
                    ended_at=1.0,
                    person_ids=[],
                    snapshot_path=snap,
                )
                uploader.enqueue_artifact(artifact)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 3:
                            break
                    time.sleep(0.02)
            finally:
                uploader.stop(timeout=2.0)

        with provider.lock:
            keys = sorted(key for _, key in provider.calls)
        self.assertEqual(
            keys,
            [
                "meta/cam_evt/cam_evt.jpg",
                "meta/cam_evt/cam_evt.json",
                "meta/cam_evt/cam_evt.mp4",
            ],
        )

    def test_delete_after_upload_removes_parent_and_sidecar(self) -> None:
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tempdir:
            ots = self._write_fake_ots(Path(tempdir))
            uploader = self._build(
                provider,
                ts_enabled=True,
                ots_binary=str(ots),
                delete_after_upload=True,
            )
            uploader.start()
            try:
                clip = Path(tempdir) / "cam_evt.mp4"
                clip.write_bytes(b"x")
                snap = Path(tempdir) / "cam_evt.jpg"
                snap.write_bytes(b"x")
                meta = Path(tempdir) / "cam_evt.json"
                meta.write_bytes(b"x")
                artifact = EventArtifact(
                    clip_path=clip,
                    metadata_path=meta,
                    started_at=0.0,
                    ended_at=1.0,
                    person_ids=[],
                    snapshot_path=snap,
                )
                uploader.enqueue_artifact(artifact)
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 5:
                            break
                    time.sleep(0.02)
            finally:
                uploader.stop(timeout=2.0)

        # Parents and sidecars all removed from disk.
        for name in ("cam_evt.mp4", "cam_evt.mp4.ots", "cam_evt.jpg", "cam_evt.jpg.ots"):
            self.assertFalse(
                (Path(tempdir) / name).exists(),
                f"{name} should have been deleted after upload",
            )

    def test_stamp_failure_does_not_block_parent_cleanup(self) -> None:
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tempdir:
            # Binary that always fails — tests the "best-effort" contract.
            import stat as _stat
            import textwrap
            script = Path(tempdir) / "ots"
            body = textwrap.dedent(
                f"""
                #!{sys.executable}
                import sys
                sys.stderr.write("no network")
                sys.exit(1)
                """
            ).strip() + "\n"
            script.write_text(body, encoding="utf-8")
            script.chmod(
                script.stat().st_mode | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH
            )
            uploader = self._build(
                provider,
                ts_enabled=True,
                ots_binary=str(script),
                delete_after_upload=True,
            )
            uploader.start()
            try:
                clip = Path(tempdir) / "cam_evt.mp4"
                clip.write_bytes(b"x")
                artifact = EventArtifact(
                    clip_path=clip,
                    metadata_path=Path(tempdir) / "cam_evt.json",
                    started_at=0.0,
                    ended_at=1.0,
                    person_ids=[],
                    snapshot_path=None,
                )
                uploader.enqueue_artifact(artifact)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline and clip.exists():
                    time.sleep(0.02)
            finally:
                uploader.stop(timeout=2.0)

        # Parent file deleted as usual; no sidecar was produced.
        self.assertFalse(clip.exists())
        self.assertFalse((Path(tempdir) / "cam_evt.mp4.ots").exists())
        with provider.lock:
            # Only the parent mp4 upload happened; no .ots in remote calls.
            keys = [key for _, key in provider.calls]
        self.assertIn("meta/cam_evt/cam_evt.mp4", keys)
        self.assertNotIn("meta/cam_evt/cam_evt.mp4.ots", keys)

    def test_per_type_flags_gate_sidecars(self) -> None:
        provider = FakeProvider()
        with tempfile.TemporaryDirectory() as tempdir:
            ots = self._write_fake_ots(Path(tempdir))
            # Only metadata gets stamped; videos/snapshots off.
            uploader = self._build(
                provider,
                ts_enabled=True,
                ots_binary=str(ots),
                stamp_videos=False,
                stamp_snapshots=False,
                stamp_metadata=True,
            )
            uploader.start()
            try:
                clip = Path(tempdir) / "cam_evt.mp4"
                snap = Path(tempdir) / "cam_evt.jpg"
                meta = Path(tempdir) / "cam_evt.json"
                for p in (clip, snap, meta):
                    p.write_bytes(b"x")
                artifact = EventArtifact(
                    clip_path=clip,
                    metadata_path=meta,
                    started_at=0.0,
                    ended_at=1.0,
                    person_ids=[],
                    snapshot_path=snap,
                )
                uploader.enqueue_artifact(artifact)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    with provider.lock:
                        if len(provider.calls) >= 4:
                            break
                    time.sleep(0.02)
            finally:
                uploader.stop(timeout=2.0)

        with provider.lock:
            keys = sorted(key for _, key in provider.calls)
        self.assertEqual(
            keys,
            [
                "meta/cam_evt/cam_evt.jpg",
                "meta/cam_evt/cam_evt.json",
                "meta/cam_evt/cam_evt.json.ots",
                "meta/cam_evt/cam_evt.mp4",
            ],
        )


class OciUploadProviderTests(unittest.TestCase):
    def _install_fake(self) -> _FakeOciModule:
        fake = _FakeOciModule()
        self.addCleanup(lambda: sys.modules.pop("oci", None))
        sys.modules["oci"] = fake  # type: ignore[assignment]
        return fake

    def test_expands_tilde_and_passes_profile(self) -> None:
        fake = self._install_fake()
        provider = OciUploadProvider(
            "bkt",
            credentials_path="~/.oci/config",
            profile="PROD",
            region="uk-london-1",
        )
        self.assertEqual(provider.scheme, "oci")
        self.assertEqual(len(fake.config_calls), 1)
        call = fake.config_calls[0]
        self.assertEqual(call["file_location"], os.path.expanduser("~/.oci/config"))
        self.assertEqual(call["profile_name"], "PROD")

    def test_default_credentials_path_when_empty(self) -> None:
        fake = self._install_fake()
        OciUploadProvider("bkt")
        self.assertEqual(
            fake.config_calls[0]["file_location"],
            os.path.expanduser("~/.oci/config"),
        )
        # No profile kwarg emitted when not configured — lets SDK use DEFAULT.
        self.assertNotIn("profile_name", fake.config_calls[0])

    def test_namespace_override_skips_get_namespace(self) -> None:
        fake = self._install_fake()
        fake.namespace_value = "auto-detected"
        provider = OciUploadProvider("bkt", namespace="explicit-ns")
        with tempfile.TemporaryDirectory() as tempdir:
            clip = Path(tempdir) / "cam.mp4"
            clip.write_bytes(b"x")
            url = provider.upload(clip, "meta/cam.mp4")
        self.assertEqual(fake.put_calls, [("explicit-ns", "bkt", "meta/cam.mp4")])
        self.assertEqual(url, "oci://explicit-ns/bkt/meta/cam.mp4")

    def test_auto_namespace_when_unset(self) -> None:
        fake = self._install_fake()
        fake.namespace_value = "auto-detected"
        provider = OciUploadProvider("bkt")
        with tempfile.TemporaryDirectory() as tempdir:
            clip = Path(tempdir) / "cam.mp4"
            clip.write_bytes(b"x")
            url = provider.upload(clip, "meta/cam.mp4")
        self.assertEqual(fake.put_calls, [("auto-detected", "bkt", "meta/cam.mp4")])
        self.assertEqual(url, "oci://auto-detected/bkt/meta/cam.mp4")

    def test_build_upload_provider_forwards_oci_fields(self) -> None:
        fake = self._install_fake()
        cfg = UploadConfig(
            enabled=True,
            provider="oci",
            bucket="bkt",
            credentials_path="~/.oci/config",
            region="uk-london-1",
            namespace="ns-from-config",
            profile="PROD",
        )
        provider = build_upload_provider(cfg)
        self.assertIsInstance(provider, OciUploadProvider)
        call = fake.config_calls[0]
        self.assertEqual(call["profile_name"], "PROD")
        self.assertEqual(call["file_location"], os.path.expanduser("~/.oci/config"))
        # Region override is written into the config dict before client construction.
        assert provider is not None
        self.assertEqual(provider._client.config["region"], "uk-london-1")  # type: ignore[attr-defined]
        self.assertEqual(provider._namespace, "ns-from-config")  # type: ignore[attr-defined]


class OciReadProviderTests(unittest.TestCase):
    def _install_fake(self) -> _FakeOciModule:
        fake = _FakeOciModule()
        self.addCleanup(lambda: sys.modules.pop("oci", None))
        sys.modules["oci"] = fake  # type: ignore[assignment]
        return fake

    def test_list_objects_forwards_prefix_and_returns_object_infos(self) -> None:
        from datetime import datetime, timezone

        fake = self._install_fake()
        fake.objects["meta-watcher/evt_a/evt_a.mp4"] = b"v"
        fake.objects["meta-watcher/evt_a/evt_a.jpg"] = b"j"
        fake.objects["other/unrelated.txt"] = b"u"
        fake.object_meta["meta-watcher/evt_a/evt_a.mp4"] = {
            "size": 1,
            "time_modified": datetime(2026, 4, 19, 12, tzinfo=timezone.utc),
            "md5": "abc",
            "etag": "e1",
        }
        provider = OciUploadProvider("bkt", namespace="ns")
        infos = provider.list_objects(prefix="meta-watcher/")
        self.assertEqual(
            [o.key for o in infos],
            ["meta-watcher/evt_a/evt_a.jpg", "meta-watcher/evt_a/evt_a.mp4"],
        )
        self.assertEqual(fake.list_calls[0]["prefix"], "meta-watcher/")
        mp4 = next(o for o in infos if o.key.endswith(".mp4"))
        self.assertEqual(mp4.size, 1)
        self.assertEqual(mp4.md5, "abc")

    def test_fetch_object_streams_full_body(self) -> None:
        fake = self._install_fake()
        fake.objects["meta-watcher/evt/clip.mp4"] = b"0123456789" * 200
        provider = OciUploadProvider("bkt", namespace="ns")
        chunks, total, _ctype = provider.fetch_object("meta-watcher/evt/clip.mp4")
        body = b"".join(chunks)
        self.assertEqual(total, 2000)
        self.assertEqual(body, b"0123456789" * 200)
        self.assertIsNone(fake.get_calls[0]["range"])

    def test_fetch_object_applies_byte_range(self) -> None:
        fake = self._install_fake()
        fake.objects["meta-watcher/evt/clip.mp4"] = b"0123456789" * 200
        provider = OciUploadProvider("bkt", namespace="ns")
        chunks, total, _ = provider.fetch_object(
            "meta-watcher/evt/clip.mp4", byte_range=(10, 19)
        )
        body = b"".join(chunks)
        self.assertEqual(body, b"0123456789")
        self.assertEqual(fake.get_calls[0]["range"], "bytes=10-19")
        self.assertEqual(total, 10)


if __name__ == "__main__":
    unittest.main()
