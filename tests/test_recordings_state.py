from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import unittest
from typing import Iterator

from meta_watcher.config import UploadConfig, default_config
from meta_watcher.upload import ObjectInfo, UploadProvider
from meta_watcher.web.state import RuntimeState


class FakeProvider(UploadProvider):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.list_calls = 0

    @property
    def scheme(self) -> str:
        return "fake"

    def upload(self, local_path: Path, remote_key: str) -> str:  # pragma: no cover
        raise NotImplementedError

    def list_objects(self, prefix: str = "", *, start=None, limit=1000):
        self.list_calls += 1
        now = datetime(2026, 4, 19, 12, tzinfo=timezone.utc)
        return [
            ObjectInfo(key=k, size=len(v), time_modified=now)
            for k, v in sorted(self.objects.items())
            if k.startswith(prefix)
        ]

    def fetch_object(self, key, *, byte_range=None):
        body = self.objects[key]
        if byte_range:
            s, e = byte_range
            body = body[s : e + 1]

        def it() -> Iterator[bytes]:
            yield body

        return it(), len(body), "application/octet-stream"


def _state_with_provider(provider: UploadProvider) -> RuntimeState:
    cfg = default_config()
    cfg.upload = UploadConfig(
        enabled=True, provider="fake", bucket="bkt", prefix="meta-watcher/"
    )
    state = RuntimeState(cfg)
    state._storage_provider_override = provider  # type: ignore[attr-defined]
    return state


class ListRecordingsTests(unittest.TestCase):
    def test_list_recordings_returns_event_summaries(self) -> None:
        provider = FakeProvider()
        provider.objects = {
            "meta-watcher/evt_a/evt_a.mp4": b"v",
            "meta-watcher/evt_a/evt_a.jpg": b"j",
            "meta-watcher/evt_a/evt_a.json": b'{"mode":"event"}',
        }
        state = _state_with_provider(provider)
        payload = state.list_recordings()
        self.assertEqual(len(payload["events"]), 1)
        evt = payload["events"][0]
        self.assertEqual(evt["event_id"], "evt_a")
        self.assertEqual(evt["clip_key"], "meta-watcher/evt_a/evt_a.mp4")
        self.assertEqual(evt["snapshot_key"], "meta-watcher/evt_a/evt_a.jpg")

    def test_list_recordings_when_upload_disabled_returns_empty(self) -> None:
        cfg = default_config()
        cfg.upload = UploadConfig(enabled=False)
        state = RuntimeState(cfg)
        payload = state.list_recordings()
        self.assertEqual(payload["events"], [])
        self.assertFalse(payload["enabled"])

    def test_recording_detail_proxies_to_browser(self) -> None:
        provider = FakeProvider()
        provider.objects = {
            "meta-watcher/evt/evt.mp4": b"v",
            "meta-watcher/evt/evt.json": b'{"frame_index":7}',
        }
        state = _state_with_provider(provider)
        detail = state.recording_detail("evt")
        self.assertEqual(detail["event_id"], "evt")
        self.assertEqual(detail["metadata"]["frame_index"], 7)

    def test_recording_detail_raises_not_found(self) -> None:
        provider = FakeProvider()
        state = _state_with_provider(provider)
        with self.assertRaises(KeyError):
            state.recording_detail("ghost")

    def test_stream_artifact_forbids_keys_outside_prefix(self) -> None:
        provider = FakeProvider()
        provider.objects["other/secret.txt"] = b"no"
        state = _state_with_provider(provider)
        with self.assertRaises(PermissionError):
            state.stream_artifact("other/secret.txt")

    def test_storage_health_reports_provider_kind(self) -> None:
        provider = FakeProvider()
        state = _state_with_provider(provider)
        health = state.storage_health()
        self.assertEqual(health["provider"], "fake")
        self.assertTrue(health["ok"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
