"""Read-side browser over the configured upload bucket.

Takes raw ``ObjectInfo`` rows from the ``UploadProvider`` and groups them into
per-event bundles the dashboard UI can render as cards. Event ids come from
the second path component of the prefix layout used by ``EventUploader``:

    {prefix}{event_id}/{event_id}.mp4
    {prefix}{event_id}/{event_id}.jpg
    {prefix}{event_id}/{event_id}.json
    {prefix}{event_id}/frames/{frame_name}
    {prefix}{event_id}/<anything>.ots        # timestamp sidecar

``.ots`` sidecars are attached to their parent artifact rather than exposed
as first-class artifacts — the UI renders a verification badge per artifact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from typing import Iterator

from .upload import ObjectInfo, UploadProvider


class ArtifactKind(str, Enum):
    CLIP = "clip"
    OVERLAY_CLIP = "overlay_clip"
    SNAPSHOT = "snapshot"
    METADATA = "metadata"
    FRAME = "frame"
    TIMESTAMP_SIDECAR = "timestamp_sidecar"  # internal — not in artifacts lists
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class ArtifactInfo:
    key: str
    kind: ArtifactKind
    size: int
    time_modified: datetime | None
    sidecar_key: str | None = None  # sibling .ots if present


@dataclass(slots=True)
class EventSummary:
    event_id: str
    clip_key: str | None
    overlay_clip_key: str | None
    snapshot_key: str | None
    metadata_key: str | None
    frame_count: int
    total_size: int
    earliest_modified: datetime | None
    latest_modified: datetime | None
    _timestamped_keys: frozenset[str] = field(default_factory=frozenset)

    def has_timestamp(self, key: str) -> bool:
        return key in self._timestamped_keys


@dataclass(slots=True)
class EventDetail:
    event_id: str
    artifacts: list[ArtifactInfo]
    metadata: dict
    metadata_key: str | None


def classify(key: str, event_id: str) -> ArtifactKind:
    if key.endswith(".ots"):
        return ArtifactKind.TIMESTAMP_SIDECAR
    if f"/{event_id}/frames/" in key:
        return ArtifactKind.FRAME
    if key.endswith(f"/{event_id}.mp4"):
        return ArtifactKind.CLIP
    if key.endswith(f"/{event_id}_overlay.mp4"):
        return ArtifactKind.OVERLAY_CLIP
    if key.endswith(f"/{event_id}.jpg"):
        return ArtifactKind.SNAPSHOT
    if key.endswith(f"/{event_id}.json"):
        return ArtifactKind.METADATA
    return ArtifactKind.UNKNOWN


def _event_id_from_key(key: str, prefix: str) -> str | None:
    if prefix and not key.startswith(prefix):
        return None
    tail = key[len(prefix):] if prefix else key
    parts = tail.split("/", 1)
    if len(parts) < 2 or not parts[0]:
        return None
    return parts[0]


class StorageBrowser:
    def __init__(self, provider: UploadProvider, *, prefix: str = "") -> None:
        self._provider = provider
        self._prefix = prefix

    def list_events(self, *, limit: int = 500) -> list[EventSummary]:
        rows = self._provider.list_objects(prefix=self._prefix, limit=limit)
        buckets: dict[str, list[ObjectInfo]] = {}
        for row in rows:
            eid = _event_id_from_key(row.key, self._prefix)
            if eid is None:
                continue
            buckets.setdefault(eid, []).append(row)

        summaries: list[EventSummary] = []
        for event_id, items in buckets.items():
            clip = overlay = snapshot = metadata = None
            frame_count = 0
            total = 0
            earliest: datetime | None = None
            latest: datetime | None = None
            ots_parents = {i.key[:-4] for i in items if i.key.endswith(".ots")}
            timestamped: set[str] = set()
            for info in items:
                total += info.size
                if info.time_modified is not None:
                    if earliest is None or info.time_modified < earliest:
                        earliest = info.time_modified
                    if latest is None or info.time_modified > latest:
                        latest = info.time_modified
                if info.key in ots_parents:
                    timestamped.add(info.key)
                kind = classify(info.key, event_id)
                if kind is ArtifactKind.CLIP:
                    clip = info.key
                elif kind is ArtifactKind.OVERLAY_CLIP:
                    overlay = info.key
                elif kind is ArtifactKind.SNAPSHOT:
                    snapshot = info.key
                elif kind is ArtifactKind.METADATA:
                    metadata = info.key
                elif kind is ArtifactKind.FRAME:
                    frame_count += 1
            summaries.append(
                EventSummary(
                    event_id=event_id,
                    clip_key=clip,
                    overlay_clip_key=overlay,
                    snapshot_key=snapshot,
                    metadata_key=metadata,
                    frame_count=frame_count,
                    total_size=total,
                    earliest_modified=earliest,
                    latest_modified=latest,
                    _timestamped_keys=frozenset(timestamped),
                )
            )

        summaries.sort(key=lambda s: s.event_id, reverse=True)
        return summaries

    def event_detail(self, event_id: str) -> EventDetail:
        rows = self._provider.list_objects(
            prefix=f"{self._prefix}{event_id}/", limit=10_000
        )
        if not rows:
            raise KeyError(event_id)
        ots_parents = {r.key[:-4] for r in rows if r.key.endswith(".ots")}
        artifacts: list[ArtifactInfo] = []
        metadata_key: str | None = None
        for row in rows:
            kind = classify(row.key, event_id)
            if kind is ArtifactKind.TIMESTAMP_SIDECAR:
                continue
            sidecar = (row.key + ".ots") if row.key in ots_parents else None
            artifacts.append(
                ArtifactInfo(
                    key=row.key,
                    kind=kind,
                    size=row.size,
                    time_modified=row.time_modified,
                    sidecar_key=sidecar,
                )
            )
            if kind is ArtifactKind.METADATA:
                metadata_key = row.key

        kind_order = {
            ArtifactKind.CLIP: 0,
            ArtifactKind.OVERLAY_CLIP: 1,
            ArtifactKind.SNAPSHOT: 2,
            ArtifactKind.METADATA: 3,
            ArtifactKind.FRAME: 4,
            ArtifactKind.UNKNOWN: 5,
        }
        artifacts.sort(key=lambda a: (kind_order[a.kind], a.key))

        metadata_doc: dict = {}
        if metadata_key is not None:
            chunks, _t, _c = self._provider.fetch_object(metadata_key)
            raw = b"".join(chunks)
            try:
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(parsed, dict):
                    metadata_doc = parsed
            except (json.JSONDecodeError, UnicodeDecodeError):
                metadata_doc = {"__parse_error": True, "__raw_bytes": len(raw)}

        return EventDetail(
            event_id=event_id,
            artifacts=artifacts,
            metadata=metadata_doc,
            metadata_key=metadata_key,
        )

    def open_artifact(
        self, key: str, *, byte_range: tuple[int, int] | None = None
    ) -> tuple[Iterator[bytes], int, str]:
        return self._provider.fetch_object(key, byte_range=byte_range)
