"""OpenTimestamps verification wrapper.

Verifies a local file against its ``.ots`` sidecar by shelling out to
``ots verify`` (the same CLI ``timestamp.py`` already uses for stamping).

``ots verify`` reports one of three broad outcomes we care about:

* exit 0 with ``Success! Bitcoin block …``           → BITCOIN_CONFIRMED
* exit 1 with ``No attestation for now`` / pending   → PENDING (calendar only)
* exit 1 with ``File does not match original!``      → INVALID

We parse stdout/stderr for well-known phrases and fall back to UNKNOWN.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import shutil
import subprocess


class VerifyStatus(str, Enum):
    BITCOIN_CONFIRMED = "bitcoin_confirmed"
    PENDING = "pending"
    INVALID = "invalid"
    TOOL_MISSING = "tool_missing"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class VerifyResult:
    status: VerifyStatus
    message: str
    raw_stdout: str = ""
    raw_stderr: str = ""


def verify_file(
    path: Path,
    ots_path: Path,
    *,
    ots_binary: str = "ots",
    timeout_seconds: float = 60.0,
) -> VerifyResult:
    target = Path(path)
    sidecar = Path(ots_path)
    if not target.is_file():
        return VerifyResult(VerifyStatus.INVALID, f"target file missing: {target}")
    if not sidecar.is_file():
        return VerifyResult(VerifyStatus.INVALID, f"sidecar missing: {sidecar}")

    resolved = shutil.which(ots_binary) or ots_binary
    cmd = [resolved, "verify", "-f", str(target), str(sidecar)]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return VerifyResult(
            VerifyStatus.TOOL_MISSING,
            f"`{ots_binary}` not on PATH; install opentimestamps-client",
        )
    except subprocess.TimeoutExpired:
        return VerifyResult(
            VerifyStatus.UNKNOWN,
            f"ots verify timed out after {timeout_seconds:.0f}s",
        )

    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    lower = out.lower()
    if "does not match original" in lower or "bad signature" in lower:
        return VerifyResult(
            VerifyStatus.INVALID,
            "sidecar does not match file",
            proc.stdout or "",
            proc.stderr or "",
        )
    if proc.returncode == 0 and "bitcoin block" in lower:
        summary = next(
            (line.strip() for line in out.splitlines() if "bitcoin block" in line.lower()),
            "Bitcoin-confirmed",
        )
        return VerifyResult(
            VerifyStatus.BITCOIN_CONFIRMED,
            summary,
            proc.stdout or "",
            proc.stderr or "",
        )
    if "no attestation" in lower or "pending" in lower or "try again" in lower:
        return VerifyResult(
            VerifyStatus.PENDING,
            "timestamp accepted by calendar; Bitcoin upgrade pending",
            proc.stdout or "",
            proc.stderr or "",
        )
    return VerifyResult(
        VerifyStatus.UNKNOWN,
        (proc.stderr.strip() or proc.stdout.strip() or "unknown ots output")[:200],
        proc.stdout or "",
        proc.stderr or "",
    )
