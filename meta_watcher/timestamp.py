"""OpenTimestamps integration.

Creates cryptographically verifiable timestamps of on-disk artifacts by
shelling out to the `ots` CLI (from the `opentimestamps-client` pip package).

`ots stamp <file>` writes `<file>.ots` next to the input — it's a sidecar
commit of the file's SHA-256 to one or more OpenTimestamps calendar servers.
The sidecar is immediately useful (calendar-notarized) and can be "upgraded"
later with `ots upgrade <file>.ots` to pin it to a Bitcoin block.

The helper here is intentionally a thin wrapper: we call the same command
operators already know, and we log the resulting sidecar path loudly so it's
obvious where the file went. (The CLI writes the sidecar silently, which
surprised operators during the first deployment.)
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


class TimestampError(RuntimeError):
    """Raised when `ots` fails to produce a usable sidecar."""


def stamp_file(
    path: Path,
    *,
    ots_binary: str = "ots",
    calendar_urls: list[str] | None = None,
    timeout_seconds: float = 30.0,
) -> Path:
    """Stamp `path` with OpenTimestamps, producing `path.ots` alongside it.

    Returns the absolute sidecar path. Raises TimestampError if the binary is
    missing, the subprocess fails, or the expected sidecar isn't produced.
    """
    target = Path(path)
    if not target.is_file():
        raise TimestampError(f"cannot stamp {target!s}: file does not exist")

    resolved_binary = shutil.which(ots_binary) or ots_binary
    cmd: list[str] = [resolved_binary, "stamp"]
    for url in calendar_urls or []:
        cmd += ["--calendar", url]
    cmd.append(str(target))

    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise TimestampError(
            f"ots binary {ots_binary!r} not found — install with "
            "`pip install opentimestamps-client`"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TimestampError(
            f"ots timed out after {timeout_seconds:.1f}s stamping {target.name}"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "no output"
        raise TimestampError(
            f"ots stamp failed (exit {result.returncode}) for {target.name}: {stderr}"
        )

    sidecar = target.with_suffix(target.suffix + ".ots")
    if not sidecar.is_file():
        raise TimestampError(
            f"ots claimed success but sidecar {sidecar.name} was not created"
        )

    print(
        f"[meta-watcher] timestamp created: {sidecar} "
        f"(ots wrote sidecar — upload will push this to the bucket as well)",
        file=sys.stderr,
        flush=True,
    )
    return sidecar
