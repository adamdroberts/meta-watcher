from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest
from typing import Iterator

from meta_watcher.storage_browser import ArtifactKind, StorageBrowser
from meta_watcher.upload import ObjectInfo, UploadProvider


class FakeReadProvider(UploadProvider):
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    @property
    def scheme(self) -> str:
        return "fake"

    def upload(self, local_path: Path, remote_key: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def list_objects(self, prefix: str = "", *, start=None, limit=1000):
        default_time = datetime(2026, 4, 19, 12, tzinfo=timezone.utc)
        return [
            ObjectInfo(key=k, size=len(v), time_modified=default_time)
            for k, v in sorted(self._objects.items())
            if k.startswith(prefix)
        ]

    def fetch_object(self, key, *, byte_range=None):
        body = self._objects[key]
        if byte_range:
            s, e = byte_range
            body = body[s : e + 1]

        def it() -> Iterator[bytes]:
            yield body

        return it(), len(body), "application/octet-stream"


class StorageBrowserTests(unittest.TestCase):
    def test_list_events_groups_artifacts_by_event_id(self) -> None:
        provider = FakeReadProvider({
            "meta-watcher/evt_a/evt_a.mp4": b"v",
            "meta-watcher/evt_a/evt_a.mp4.ots": b"t",
            "meta-watcher/evt_a/evt_a.jpg": b"j",
            "meta-watcher/evt_a/evt_a.json": b'{"mode":"event"}',
            "meta-watcher/evt_a/frames/evt_a_x.jpg": b"f",
            "meta-watcher/evt_b/evt_b.mp4": b"v2",
            "meta-watcher/evt_b/evt_b.json": b'{"mode":"event"}',
            "other/unrelated.txt": b"u",
        })
        browser = StorageBrowser(provider, prefix="meta-watcher/")
        events = browser.list_events()
        ids = [e.event_id for e in events]
        self.assertEqual(ids, ["evt_b", "evt_a"])  # newest-first
        evt_a = next(e for e in events if e.event_id == "evt_a")
        self.assertEqual(evt_a.snapshot_key, "meta-watcher/evt_a/evt_a.jpg")
        self.assertEqual(evt_a.clip_key, "meta-watcher/evt_a/evt_a.mp4")
        self.assertEqual(evt_a.metadata_key, "meta-watcher/evt_a/evt_a.json")
        self.assertEqual(evt_a.frame_count, 1)
        self.assertTrue(evt_a.has_timestamp("meta-watcher/evt_a/evt_a.mp4"))
        self.assertFalse(evt_a.has_timestamp("meta-watcher/evt_a/evt_a.json"))

    def test_event_detail_returns_artifacts_with_kinds(self) -> None:
        provider = FakeReadProvider({
            "meta-watcher/evt/evt.mp4": b"v",
            "meta-watcher/evt/evt.mp4.ots": b"t",
            "meta-watcher/evt/evt.jpg": b"j",
            "meta-watcher/evt/evt.json": b'{"mode":"event","frame_index":42}',
            "meta-watcher/evt/frames/evt_001.jpg": b"f1",
            "meta-watcher/evt/frames/evt_002.jpg": b"f2",
        })
        browser = StorageBrowser(provider, prefix="meta-watcher/")
        detail = browser.event_detail("evt")
        kinds = {a.kind for a in detail.artifacts}
        self.assertIn(ArtifactKind.CLIP, kinds)
        self.assertIn(ArtifactKind.SNAPSHOT, kinds)
        self.assertIn(ArtifactKind.METADATA, kinds)
        self.assertIn(ArtifactKind.FRAME, kinds)
        self.assertNotIn(ArtifactKind.TIMESTAMP_SIDECAR, kinds)
        clip = next(a for a in detail.artifacts if a.kind is ArtifactKind.CLIP)
        self.assertEqual(clip.sidecar_key, "meta-watcher/evt/evt.mp4.ots")
        frames = [a for a in detail.artifacts if a.kind is ArtifactKind.FRAME]
        self.assertEqual(
            [a.key for a in frames],
            [
                "meta-watcher/evt/frames/evt_001.jpg",
                "meta-watcher/evt/frames/evt_002.jpg",
            ],
        )
        self.assertEqual(detail.metadata["frame_index"], 42)

    def test_event_detail_raises_on_unknown_event(self) -> None:
        provider = FakeReadProvider({"meta-watcher/evt_a/evt_a.mp4": b"v"})
        browser = StorageBrowser(provider, prefix="meta-watcher/")
        with self.assertRaises(KeyError):
            browser.event_detail("missing")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
