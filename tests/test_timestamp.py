from __future__ import annotations

import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from meta_watcher.timestamp import TimestampError, stamp_file


def _write_fake_ots(
    tempdir: Path,
    *,
    behavior: str = "ok",
    extra: str = "",
) -> Path:
    """Drop an `ots` shim in `tempdir` and return its absolute path.

    `behavior`:
      - "ok"         → touch `<path>.ots`, exit 0
      - "no-sidecar" → print success, do NOT touch the sidecar, exit 0
      - "fail"       → exit 1 with stderr
    """
    script = tempdir / "ots"
    body = textwrap.dedent(
        f"""
        #!{sys.executable}
        import pathlib, sys
        args = sys.argv[1:]
        assert args and args[0] == "stamp", f"bad args: {{args!r}}"
        # everything after "stamp" may include --calendar flags; the last
        # positional is the file path.
        path = pathlib.Path(args[-1])
        behavior = {behavior!r}
        if behavior == "ok":
            (path.parent / (path.name + ".ots")).write_bytes(b"OTS-PROOF")
            sys.exit(0)
        if behavior == "no-sidecar":
            print("calendar submitted")
            sys.exit(0)
        if behavior == "fail":
            print("ots: network unreachable", file=sys.stderr)
            sys.exit(1)
        sys.exit(2)
        {extra}
        """
    ).strip() + "\n"
    script.write_text(body, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


class StampFileTests(unittest.TestCase):
    def test_writes_sidecar_and_returns_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            ots = _write_fake_ots(base, behavior="ok")
            target = base / "clip.mp4"
            target.write_bytes(b"binary")
            sidecar = stamp_file(target, ots_binary=str(ots))
            self.assertEqual(sidecar, target.with_suffix(".mp4.ots"))
            self.assertTrue(sidecar.is_file())
            self.assertEqual(sidecar.read_bytes(), b"OTS-PROOF")

    def test_raises_when_sidecar_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            ots = _write_fake_ots(base, behavior="no-sidecar")
            target = base / "clip.mp4"
            target.write_bytes(b"binary")
            with self.assertRaises(TimestampError) as ctx:
                stamp_file(target, ots_binary=str(ots))
            self.assertIn("sidecar", str(ctx.exception))

    def test_raises_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            ots = _write_fake_ots(base, behavior="fail")
            target = base / "clip.mp4"
            target.write_bytes(b"binary")
            with self.assertRaises(TimestampError) as ctx:
                stamp_file(target, ots_binary=str(ots))
            self.assertIn("network unreachable", str(ctx.exception))

    def test_raises_when_binary_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "clip.mp4"
            target.write_bytes(b"binary")
            with self.assertRaises(TimestampError) as ctx:
                stamp_file(
                    target,
                    ots_binary=str(Path(tempdir) / "does-not-exist"),
                )
            self.assertIn("not found", str(ctx.exception))

    def test_raises_when_target_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            missing = Path(tempdir) / "nope.mp4"
            with self.assertRaises(TimestampError):
                stamp_file(missing)

    def test_passes_calendar_urls_through(self) -> None:
        # The shim asserts presence of "stamp"; separately confirm the helper
        # accepts and forwards calendar URLs. Use an extra that reports argv.
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            argfile = base / "argv.txt"
            script = base / "ots"
            body = textwrap.dedent(
                f"""
                #!{sys.executable}
                import pathlib, sys
                pathlib.Path({str(argfile)!r}).write_text(" ".join(sys.argv[1:]))
                path = pathlib.Path(sys.argv[-1])
                (path.parent / (path.name + ".ots")).write_bytes(b"p")
                sys.exit(0)
                """
            ).strip() + "\n"
            script.write_text(body, encoding="utf-8")
            script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            target = base / "clip.mp4"
            target.write_bytes(b"binary")
            stamp_file(
                target,
                ots_binary=str(script),
                calendar_urls=["https://a.example", "https://b.example"],
            )
            argv = argfile.read_text(encoding="utf-8").split()
            self.assertIn("--calendar", argv)
            self.assertIn("https://a.example", argv)
            self.assertIn("https://b.example", argv)


if __name__ == "__main__":
    unittest.main()
