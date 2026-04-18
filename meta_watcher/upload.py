from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import os
import queue
import sys
import threading
from typing import Any

from .config import TimestampConfig, UploadConfig
from .core import EventArtifact
from .timestamp import TimestampError, stamp_file


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

    One worker thread, bounded drop-oldest queue (same policy as
    `StreamRuntime._put_latest`). Upload failures are logged to stderr and
    silently dropped — there is no retry loop.

    When a TimestampConfig is provided and `timestamps.enabled` is true, the
    uploader will run `ots stamp` on each uploaded artifact that was tagged
    with `timestamp=True` at enqueue time, then upload the resulting `.ots`
    sidecar to `{remote_key}.ots`. Timestamp failures are non-fatal: the
    primary upload is already complete, and the parent file is still deleted
    if `delete_after_upload` was set (so the sidecar is best-effort).
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
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="meta-watcher-uploader",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        self._thread = None

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

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
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
