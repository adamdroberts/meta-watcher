from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from meta_watcher.config import (
    AppConfig,
    default_config,
    list_config_files,
    load_config,
    save_config,
)


class SaveConfigTests(unittest.TestCase):
    def test_json_round_trip_matches_defaults(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "out.json"
            written = save_config(target, cfg)
            self.assertEqual(written, target)
            # Trailing newline, pretty-printed (indent=2).
            text = target.read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"))
            self.assertIn("\n  ", text)
            reloaded = load_config(target)
        self.assertIsInstance(reloaded, AppConfig)
        self.assertEqual(reloaded.source.kind, cfg.source.kind)
        self.assertEqual(
            reloaded.thresholds.person_confidence, cfg.thresholds.person_confidence
        )
        self.assertEqual(reloaded.upload.queue_size, cfg.upload.queue_size)

    def test_yaml_path_writes_json_sibling(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tempdir:
            yaml_path = Path(tempdir) / "foo.yaml"
            yaml_path.write_text("source:\n  kind: webcam\n", encoding="utf-8")
            written = save_config(yaml_path, cfg)
            self.assertEqual(written.suffix, ".json")
            self.assertEqual(written.stem, "foo")
            self.assertTrue(written.exists())
            # Original YAML is untouched.
            self.assertEqual(
                yaml_path.read_text(encoding="utf-8"),
                "source:\n  kind: webcam\n",
            )

    def test_atomic_on_serialization_error(self) -> None:
        cfg = default_config()
        with tempfile.TemporaryDirectory() as tempdir:
            target = Path(tempdir) / "nope.json"
            target.write_text('{"existing": true}\n', encoding="utf-8")
            with mock.patch("meta_watcher.config.json.dump", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    save_config(target, cfg)
            # Original file unchanged.
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                '{"existing": true}\n',
            )
            # No .tmp file left behind.
            siblings = {p.name for p in Path(tempdir).iterdir()}
            self.assertEqual(siblings, {"nope.json"})


class ListConfigFilesTests(unittest.TestCase):
    def test_skips_hidden_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            base = Path(tempdir)
            (base / "a.json").write_text("{}", encoding="utf-8")
            (base / ".hidden.json").write_text("{}", encoding="utf-8")
            (base / "b.yaml").write_text("source: {}\n", encoding="utf-8")
            try:
                os.symlink(base / "a.json", base / "a_link.json")
                expect_link = True
            except OSError:
                expect_link = False

            results = list_config_files([base])
            names = [p.name for p in results]
            # Hidden files never appear.
            self.assertNotIn(".hidden.json", names)
            # The yaml is always listed.
            self.assertIn("b.yaml", names)
            # The real file or its symlink is listed exactly once.
            a_entries = [n for n in names if n in {"a.json", "a_link.json"}]
            if expect_link:
                self.assertEqual(len(a_entries), 1, f"expected dedup, got {a_entries}")
            else:
                self.assertEqual(a_entries, ["a.json"])

    def test_missing_dirs_are_skipped(self) -> None:
        self.assertEqual(list_config_files([Path("/does/not/exist")]), [])


if __name__ == "__main__":
    unittest.main()
