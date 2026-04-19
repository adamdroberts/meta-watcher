from __future__ import annotations

from datetime import datetime, timezone
import unittest
from typing import Iterator

from fastapi.testclient import TestClient

from meta_watcher.config import UploadConfig, default_config
from meta_watcher.upload import ObjectInfo, UploadProvider
from meta_watcher.web.server import build_app
from meta_watcher.web.state import RuntimeState


class FakeProvider(UploadProvider):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    @property
    def scheme(self) -> str:
        return "fake"

    def upload(self, local_path, remote_key):  # pragma: no cover
        raise NotImplementedError

    def list_objects(self, prefix: str = "", *, start=None, limit=1000):
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

        return (
            it(),
            len(body),
            "video/mp4" if key.endswith(".mp4") else "application/octet-stream",
        )


def _client() -> tuple[TestClient, FakeProvider]:
    cfg = default_config()
    cfg.upload = UploadConfig(
        enabled=True, provider="fake", bucket="bkt", prefix="meta-watcher/"
    )
    state = RuntimeState(cfg)
    provider = FakeProvider()
    state._storage_provider_override = provider  # type: ignore[attr-defined]
    return TestClient(build_app(state)), provider


class RecordingsApiTests(unittest.TestCase):
    def test_get_recordings_returns_event_list(self) -> None:
        client, provider = _client()
        provider.objects = {
            "meta-watcher/evt_a/evt_a.mp4": b"v",
            "meta-watcher/evt_a/evt_a.jpg": b"j",
            "meta-watcher/evt_a/evt_a.json": b'{"mode":"event"}',
        }
        resp = client.get("/api/recordings")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["enabled"])
        self.assertEqual(len(data["events"]), 1)
        self.assertEqual(data["events"][0]["event_id"], "evt_a")

    def test_get_recording_detail_includes_metadata(self) -> None:
        client, provider = _client()
        provider.objects = {
            "meta-watcher/evt/evt.mp4": b"v",
            "meta-watcher/evt/evt.json": b'{"frame_index":42}',
        }
        resp = client.get("/api/recordings/evt")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["event_id"], "evt")
        self.assertEqual(data["metadata"]["frame_index"], 42)

    def test_get_recording_detail_404s_on_unknown(self) -> None:
        client, _ = _client()
        resp = client.get("/api/recordings/ghost")
        self.assertEqual(resp.status_code, 404)

    def test_get_artifact_proxies_bytes(self) -> None:
        client, provider = _client()
        provider.objects["meta-watcher/evt/evt.mp4"] = b"HELLOWORLD"
        resp = client.get(
            "/api/recordings/evt/artifact",
            params={"key": "meta-watcher/evt/evt.mp4"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"HELLOWORLD")
        self.assertEqual(resp.headers["content-type"], "video/mp4")

    def test_get_artifact_honours_range(self) -> None:
        client, provider = _client()
        provider.objects["meta-watcher/evt/evt.mp4"] = b"0123456789"
        resp = client.get(
            "/api/recordings/evt/artifact",
            params={"key": "meta-watcher/evt/evt.mp4"},
            headers={"Range": "bytes=2-5"},
        )
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(resp.content, b"2345")
        self.assertEqual(resp.headers["content-range"], "bytes 2-5/10")

    def test_get_artifact_rejects_key_outside_event(self) -> None:
        client, provider = _client()
        provider.objects["meta-watcher/evt_a/evt_a.mp4"] = b"v"
        resp = client.get(
            "/api/recordings/evt_b/artifact",
            params={"key": "meta-watcher/evt_a/evt_a.mp4"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_storage_health_reports_ok_when_reachable(self) -> None:
        client, _ = _client()
        resp = client.get("/api/storage/health")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
