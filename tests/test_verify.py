from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from meta_watcher.verify import VerifyStatus, verify_file


class VerifyFileTests(unittest.TestCase):
    def _write_pair(self) -> tuple[Path, Path]:
        tmp = Path(tempfile.mkdtemp())
        target = tmp / "clip.mp4"
        sidecar = tmp / "clip.mp4.ots"
        target.write_bytes(b"payload")
        sidecar.write_bytes(b"pseudo-ots-bytes")

        def _cleanup() -> None:
            target.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            try:
                tmp.rmdir()
            except OSError:
                pass

        self.addCleanup(_cleanup)
        return target, sidecar

    def test_verify_reports_bitcoin_confirmed(self) -> None:
        target, sidecar = self._write_pair()
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Success! Bitcoin block 830123 attests existence as of 2026-04-18 UTC",
            stderr="",
        )
        with mock.patch(
            "meta_watcher.verify.subprocess.run", return_value=completed
        ) as run:
            result = verify_file(target, sidecar, ots_binary="ots")
        self.assertEqual(result.status, VerifyStatus.BITCOIN_CONFIRMED)
        self.assertIn("2026-04-18", result.message)
        args = run.call_args.args[0]
        # args[0] may be the literal "ots" or an absolute path resolved by
        # shutil.which; either way it must end in the binary name.
        self.assertTrue(
            args[0] == "ots" or args[0].endswith("/ots"),
            f"unexpected binary path: {args[0]!r}",
        )
        self.assertEqual(args[1], "verify")
        self.assertIn(str(sidecar), args)

    def test_verify_reports_pending_when_no_bitcoin_yet(self) -> None:
        target, sidecar = self._write_pair()
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Failed! No attestation for now; try again in a few hours",
        )
        with mock.patch(
            "meta_watcher.verify.subprocess.run", return_value=completed
        ):
            result = verify_file(target, sidecar)
        self.assertEqual(result.status, VerifyStatus.PENDING)

    def test_verify_reports_invalid_on_tamper(self) -> None:
        target, sidecar = self._write_pair()
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="File does not match original!",
        )
        with mock.patch(
            "meta_watcher.verify.subprocess.run", return_value=completed
        ):
            result = verify_file(target, sidecar)
        self.assertEqual(result.status, VerifyStatus.INVALID)

    def test_missing_binary_reports_unavailable(self) -> None:
        target, sidecar = self._write_pair()
        with mock.patch(
            "meta_watcher.verify.subprocess.run", side_effect=FileNotFoundError("ots")
        ):
            result = verify_file(target, sidecar, ots_binary="ots")
        self.assertEqual(result.status, VerifyStatus.TOOL_MISSING)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
