from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import queue
import sys
import threading
from typing import Any, Iterator

from .config import TimestampConfig, UploadConfig
from .core import EventArtifact
from .timestamp import TimestampError, stamp_file


_CONTENT_TYPE_BY_EXT: dict[str, str] = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
    ".txt": "text/plain",
    ".ots": "application/vnd.opentimestamps.ots",
}


def _infer_content_type(key: str, *, fallback: str = "application/octet-stream") -> str:
    """Map a storage key's extension to a sensible browser-friendly MIME.

    OCI stores uploaded objects with whatever Content-Type the upload call
    advertised; Meta Watcher uploads as octet-stream, so when the dashboard
    streams them back, the browser has no hint for ``<video>`` / ``<img>``.
    This lets callers override by filename extension when the stored type
    is generic.
    """
    lowered = key.lower()
    dot = lowered.rfind(".")
    if dot == -1:
        return fallback
    return _CONTENT_TYPE_BY_EXT.get(lowered[dot:], fallback)


@dataclass(slots=True, frozen=True)
class ObjectInfo:
    """Metadata about a single object in the configured bucket.

    `time_modified` is tz-aware UTC. `size` is bytes. `md5`/`etag` are
    provider-reported (empty on providers that don't populate them)."""

    key: str
    size: int
    time_modified: datetime | None
    md5: str = ""
    etag: str = ""


class UploadProvider(ABC):
    @property
    @abstractmethod
    def scheme(self) -> str:
        """Short scheme label used in log messages, e.g. 'gs', 's3', 'oci'."""

    @abstractmethod
    def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload `local_path` to `remote_key` in the configured bucket.

        Returns a human-readable URL/key on success, raises on failure.
        """

    # --- read-side (dashboard browser) -----------------------------------
    def list_objects(
        self, prefix: str = "", *, start: str | None = None, limit: int = 1000
    ) -> list[ObjectInfo]:
        raise NotImplementedError(
            f"{type(self).__name__} does not yet support listing objects; "
            "only the 'oci' provider is supported by the analytics dashboard."
        )

    def fetch_object(
        self, key: str, *, byte_range: tuple[int, int] | None = None
    ) -> tuple[Iterator[bytes], int, str]:
        """Return ``(chunk_iterator, total_bytes, content_type)``.

        When ``byte_range=(start, end)`` is given, only that byte range
        (inclusive) is streamed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not yet support fetching objects."
        )


class GcpUploadProvider(UploadProvider):
    def __init__(self, bucket: str, *, credentials_path: str = "") -> None:
        from google.cloud import storage  # type: ignore[import-not-found]

        if credentials_path:
            client = storage.Client.from_service_account_json(credentials_path)
        else:
            client = storage.Client()
        self._bucket = client.bucket(bucket)
        self._bucket_name = bucket

    @property
    def scheme(self) -> str:
        return "gs"

    def upload(self, local_path: Path, remote_key: str) -> str:
        blob = self._bucket.blob(remote_key)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self._bucket_name}/{remote_key}"


class AwsUploadProvider(UploadProvider):
    def __init__(
        self,
        bucket: str,
        *,
        credentials_path: str = "",
        region: str = "",
    ) -> None:
        if credentials_path:
            os.environ.setdefault("AWS_SHARED_CREDENTIALS_FILE", credentials_path)
        import boto3  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        self._client = boto3.client("s3", **kwargs)
        self._bucket_name = bucket

    @property
    def scheme(self) -> str:
        return "s3"

    def upload(self, local_path: Path, remote_key: str) -> str:
        self._client.upload_file(str(local_path), self._bucket_name, remote_key)
        return f"s3://{self._bucket_name}/{remote_key}"


class OciUploadProvider(UploadProvider):
    def __init__(
        self,
        bucket: str,
        *,
        credentials_path: str = "",
        region: str = "",
        namespace: str = "",
        profile: str = "",
    ) -> None:
        import oci  # type: ignore[import-not-found]

        raw_path = credentials_path or "~/.oci/config"
        config_path = os.path.expanduser(raw_path)
        from_file_kwargs: dict[str, Any] = {"file_location": config_path}
        if profile:
            from_file_kwargs["profile_name"] = profile
        config = oci.config.from_file(**from_file_kwargs)
        if region:
            config["region"] = region
        self._client = oci.object_storage.ObjectStorageClient(config)
        self._namespace = namespace or self._client.get_namespace().data
        self._bucket_name = bucket

    @property
    def scheme(self) -> str:
        return "oci"

    def upload(self, local_path: Path, remote_key: str) -> str:
        with local_path.open("rb") as handle:
            self._client.put_object(
                self._namespace,
                self._bucket_name,
                remote_key,
                handle,
            )
        return f"oci://{self._namespace}/{self._bucket_name}/{remote_key}"

    def list_objects(
        self, prefix: str = "", *, start: str | None = None, limit: int = 1000
    ) -> list[ObjectInfo]:
        collected: list[ObjectInfo] = []
        cursor = start
        fields = "name,size,timeModified,md5,etag"
        while len(collected) < limit:
            page_size = min(limit - len(collected), 1000)
            resp = self._client.list_objects(
                self._namespace,
                self._bucket_name,
                prefix=prefix or None,
                start=cursor,
                limit=page_size,
                fields=fields,
            )
            data = resp.data
            for obj in getattr(data, "objects", []) or []:
                collected.append(
                    ObjectInfo(
                        key=obj.name,
                        size=int(getattr(obj, "size", 0) or 0),
                        time_modified=getattr(obj, "time_modified", None),
                        md5=getattr(obj, "md5", "") or "",
                        etag=getattr(obj, "etag", "") or "",
                    )
                )
            cursor = getattr(data, "next_start_with", None)
            if not cursor:
                break
        return collected

    def fetch_object(
        self, key: str, *, byte_range: tuple[int, int] | None = None
    ) -> tuple[Iterator[bytes], int, str]:
        range_header: str | None = None
        if byte_range is not None:
            start, end = byte_range
            range_header = f"bytes={start}-{end}"
        resp = self._client.get_object(
            self._namespace,
            self._bucket_name,
            key,
            range=range_header,
        )
        headers = getattr(resp, "headers", {}) or {}
        content_length = int(
            headers.get("Content-Length") or headers.get("content-length") or 0
        )
        reported_type = (
            headers.get("Content-Type")
            or headers.get("content-type")
            or "application/octet-stream"
        )
        # Meta Watcher uploads files without an explicit Content-Type, so OCI
        # stores them as application/octet-stream. That's a problem for the
        # dashboard — browsers need the real MIME to render <video>/<img>. If
        # OCI gave us a generic type, infer from the key's extension.
        content_type = reported_type
        if reported_type in ("application/octet-stream", "binary/octet-stream"):
            content_type = _infer_content_type(key, fallback=reported_type)
        stream = resp.data

        def _iter() -> Iterator[bytes]:
            # Real OCI SDK exposes `.raw` (urllib3) with `.stream()`; the fake
            # uses `.iter_content()`. Support both shapes.
            if hasattr(stream, "iter_content"):
                yield from stream.iter_content(chunk_size=65536)
            elif hasattr(stream, "raw"):
                yield from stream.raw.stream(amt=65536, decode_content=False)
            else:  # pragma: no cover
                data = stream.read() if hasattr(stream, "read") else bytes(stream)
                if data:
                    yield data

        return _iter(), content_length, content_type


def build_upload_provider(config: UploadConfig) -> UploadProvider | None:
    if not config.enabled or not config.bucket:
        return None
    kind = config.provider.lower()
    if kind == "gcp":
        return GcpUploadProvider(config.bucket, credentials_path=config.credentials_path)
    if kind == "aws":
        return AwsUploadProvider(
            config.bucket,
            credentials_path=config.credentials_path,
            region=config.region,
        )
    if kind == "oci":
        return OciUploadProvider(
            config.bucket,
            credentials_path=config.credentials_path,
            region=config.region,
            namespace=config.namespace,
            profile=config.profile,
        )
    raise ValueError(f"Unsupported upload provider: {config.provider!r}")


@dataclass(slots=True)
class _UploadJob:
    local_path: Path
    remote_key: str
    delete_after_upload: bool | None = None  # None = inherit config default
    timestamp: bool = False  # create + upload an OpenTimestamps .ots sidecar


class EventUploader:
    """Background uploader that accepts EventArtifacts and ships them off-device.

    A small `ThreadPoolExecutor` (size = `UploadConfig.upload_workers`) runs
    uploads in parallel; a single feeder thread drains a bounded drop-oldest
    queue into the pool. Upload failures are logged to stderr and silently
    dropped — there is no retry loop.

    When a TimestampConfig is provided and `timestamps.enabled` is true, each
    worker runs `ots stamp` on its own uploaded artifact (when tagged with
    `timestamp=True` at enqueue time), then uploads the resulting `.ots`
    sidecar. Keeping stamp+sidecar inside the same worker means a slow
    `ots stamp` only consumes 1/N of the pool, not the whole queue. Timestamp
    failures are non-fatal: the primary upload is already complete, and the
    parent file is still deleted if `delete_after_upload` was set.

    The bounded input queue is the real backpressure; a semaphore caps
    in-flight submissions so the pool's internal queue can't balloon past
    `2 × upload_workers`, keeping drop-oldest observable under load.
    """

    def __init__(
        self,
        provider: UploadProvider,
        config: UploadConfig,
        timestamps: TimestampConfig | None = None,
    ) -> None:
        self._provider = provider
        self._config = config
        self._timestamps = timestamps if timestamps is not None else TimestampConfig()
        self._queue: queue.Queue[_UploadJob] = queue.Queue(maxsize=max(1, config.queue_size))
        self._stop = threading.Event()
        self._max_workers = max(1, getattr(config, "upload_workers", 4))
        self._slots = threading.BoundedSemaphore(self._max_workers * 2)
        self._feeder: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None

    def start(self) -> None:
        if self._feeder is not None:
            return
        self._stop.clear()
        self._pool = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="mw-upload",
        )
        self._feeder = threading.Thread(
            target=self._run_feeder,
            name="meta-watcher-uploader-feeder",
            daemon=True,
        )
        self._feeder.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        feeder = self._feeder
        if feeder is not None:
            feeder.join(timeout=timeout)
        self._feeder = None
        pool = self._pool
        self._pool = None
        if pool is not None:
            # wait=True lets in-flight uploads finish cleanly; cancel_futures
            # drops anything queued-but-not-started so shutdown returns promptly.
            pool.shutdown(wait=True, cancel_futures=True)

    def enqueue_artifact(self, artifact: EventArtifact, *, skip_snapshot: bool = False) -> None:
        event_id = artifact.clip_path.stem
        ts = self._timestamps
        jobs: list[_UploadJob] = []
        if self._config.upload_videos and artifact.clip_path is not None:
            jobs.append(self._job_for(artifact.clip_path, event_id, stamp=ts.stamp_videos))
        if self._config.upload_videos and artifact.overlay_clip_path is not None:
            jobs.append(
                self._job_for(artifact.overlay_clip_path, event_id, stamp=ts.stamp_videos)
            )
        if (
            self._config.upload_snapshots
            and artifact.snapshot_path is not None
            and not skip_snapshot
        ):
            jobs.append(
                self._job_for(artifact.snapshot_path, event_id, stamp=ts.stamp_snapshots)
            )
        if self._config.upload_metadata and artifact.metadata_path is not None:
            jobs.append(
                self._job_for(artifact.metadata_path, event_id, stamp=ts.stamp_metadata)
            )
        for job in jobs:
            self._enqueue_drop_oldest(job)

    def enqueue_snapshot(self, snapshot_path: Path, event_id: str) -> None:
        """Fire the event snapshot off before the clip is finished recording.

        The same file is intentionally NOT re-uploaded by
        `enqueue_artifact(..., skip_snapshot=True)` at event close.
        """
        if not self._config.upload_snapshots:
            return
        self._enqueue_drop_oldest(
            self._job_for(snapshot_path, event_id, stamp=self._timestamps.stamp_snapshots)
        )

    def enqueue_frame(
        self,
        local_path: Path,
        event_id: str,
        *,
        delete_after_upload: bool | None = None,
    ) -> None:
        """Enqueue a live frame captured mid-event.

        Keyed under `{prefix}{event_id}/frames/{file.name}` so periodic frames
        group alongside the event's clip + snapshot. `delete_after_upload=None`
        (the default) inherits the global upload config — frames are kept on
        disk alongside the clip + snapshot by default so operators have a
        complete local record. Pass `True` to explicitly make frames ephemeral.
        """
        remote_key = f"{self._config.prefix}{event_id}/frames/{local_path.name}"
        self._enqueue_drop_oldest(
            _UploadJob(
                local_path=local_path,
                remote_key=remote_key,
                delete_after_upload=delete_after_upload,
                timestamp=self._timestamps.stamp_frames,
            )
        )

    def _job_for(self, path: Path, event_id: str, *, stamp: bool = False) -> _UploadJob:
        return _UploadJob(
            local_path=path,
            remote_key=f"{self._config.prefix}{event_id}/{path.name}",
            timestamp=stamp,
        )

    def _enqueue_drop_oldest(self, job: _UploadJob) -> None:
        try:
            self._queue.put_nowait(job)
            return
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
                print(
                    f"[meta-watcher] upload queue full; dropping {dropped.local_path.name}",
                    file=sys.stderr,
                    flush=True,
                )
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            pass

    def _maybe_stamp_and_upload(self, job: _UploadJob) -> Path | None:
        """Create an OpenTimestamps .ots sidecar for `job`'s local file and
        upload it to `{remote_key}.ots`. Returns the local sidecar path on
        success so the caller can clean it up alongside the parent.

        Returns None when timestamping is disabled, the job wasn't tagged, or
        anything along the stamp/upload chain fails. Failures are logged but
        never propagated — the primary upload is already safe.
        """
        if not job.timestamp or not self._timestamps.enabled:
            return None
        try:
            sidecar = stamp_file(
                job.local_path,
                ots_binary=self._timestamps.ots_binary,
                calendar_urls=list(self._timestamps.calendar_urls),
                timeout_seconds=self._timestamps.timeout_seconds,
            )
        except TimestampError as exc:
            print(
                f"[meta-watcher] timestamp failed for {job.local_path.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None
        sidecar_key = f"{job.remote_key}.ots"
        try:
            remote = self._provider.upload(sidecar, sidecar_key)
        except Exception as exc:  # noqa: BLE001 — SDKs throw anything
            print(
                f"[meta-watcher] timestamp upload failed for {sidecar.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return sidecar  # still return so caller can cleanup the local file
        print(
            f"[meta-watcher] timestamp upload ok: {remote}",
            file=sys.stderr,
            flush=True,
        )
        return sidecar

    def _run_feeder(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            # Block until a worker slot is free so the pool's internal queue
            # stays shallow and drop-oldest stays meaningful on the input queue.
            acquired = False
            while not self._stop.is_set():
                if self._slots.acquire(timeout=0.2):
                    acquired = True
                    break
            if not acquired:
                return
            pool = self._pool
            if pool is None:
                self._slots.release()
                return
            try:
                pool.submit(self._process_job_slot, job)
            except RuntimeError:
                # Pool was shut down while we were submitting; exit cleanly.
                self._slots.release()
                return

    def _process_job_slot(self, job: _UploadJob) -> None:
        try:
            self._process_job(job)
        finally:
            self._slots.release()

    def _process_job(self, job: _UploadJob) -> None:
        try:
            remote = self._provider.upload(job.local_path, job.remote_key)
            print(
                f"[meta-watcher] upload ok: {remote}",
                file=sys.stderr,
                flush=True,
            )
            delete_after = (
                job.delete_after_upload
                if job.delete_after_upload is not None
                else self._config.delete_after_upload
            )
            sidecar_path = self._maybe_stamp_and_upload(job)
            if delete_after:
                try:
                    job.local_path.unlink(missing_ok=True)
                except OSError as exc:
                    print(
                        f"[meta-watcher] delete-after-upload failed for {job.local_path}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                if sidecar_path is not None:
                    try:
                        sidecar_path.unlink(missing_ok=True)
                    except OSError as exc:
                        print(
                            f"[meta-watcher] delete-after-upload failed for {sidecar_path}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
        except Exception as exc:  # noqa: BLE001 — we want to swallow anything from SDKs.
            print(
                f"[meta-watcher] upload failed for {job.local_path.name}: {exc}",
                file=sys.stderr,
                flush=True,
            )
