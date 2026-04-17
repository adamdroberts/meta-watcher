from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import sys

from .config import AppConfig, load_config
from .core import ClipRecorder, PipelineSnapshot
from .inference import build_label_proposer, build_provider
from .pipeline import StreamProcessor, StreamRuntime
from .sources import build_source


def main(argv: list[str] | None = None) -> int:
    mp.freeze_support()
    parser = argparse.ArgumentParser(description="Meta Watcher desktop application")
    parser.add_argument("--config", default=None, help="Optional path to a JSON or YAML config file")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    try:
        from PySide6.QtCore import QObject, Qt, Signal
        from PySide6.QtGui import QImage, QPixmap
        from PySide6.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QFileDialog,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QListWidget,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QDoubleSpinBox,
            QSpinBox,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit("PySide6 is required to run the desktop application.") from exc

    class Bridge(QObject):
        snapshot_ready = Signal(object)
        error_raised = Signal(str)

    class MainWindow(QMainWindow):
        def __init__(self, initial_config: AppConfig) -> None:
            super().__init__()
            self.setWindowTitle("Meta Watcher")
            self.resize(1440, 900)
            self.config = initial_config
            self.runtime: StreamRuntime | None = None
            self.bridge = Bridge()
            self.bridge.snapshot_ready.connect(self._apply_snapshot)
            self.bridge.error_raised.connect(self._show_error)

            central = QWidget(self)
            self.setCentralWidget(central)

            layout = QHBoxLayout(central)
            left = QVBoxLayout()
            right = QVBoxLayout()
            layout.addLayout(left, stretch=3)
            layout.addLayout(right, stretch=2)

            self.video_label = QLabel("No stream")
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setMinimumSize(960, 640)
            self.video_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")
            left.addWidget(self.video_label)

            controls = QWidget()
            form = QFormLayout(controls)

            self.source_kind = QComboBox()
            self.source_kind.addItems(["webcam", "rtsp", "file"])
            self.source_kind.setCurrentText(self.config.source.kind)
            form.addRow("Source kind", self.source_kind)

            self.source_value = QLineEdit(self.config.source.value)
            form.addRow("Source value", self.source_value)

            self.output_dir = QLineEdit(self.config.output.directory)
            choose_output = QPushButton("Choose…")
            choose_output.clicked.connect(self._choose_output_dir)
            output_row = QWidget()
            output_layout = QHBoxLayout(output_row)
            output_layout.setContentsMargins(0, 0, 0, 0)
            output_layout.addWidget(self.output_dir)
            output_layout.addWidget(choose_output)
            form.addRow("Output dir", output_row)

            self.auto_rescan = QCheckBox()
            self.auto_rescan.setChecked(self.config.inventory.auto_rescan)
            form.addRow("Auto rescan", self.auto_rescan)

            self.recording_enabled = QCheckBox()
            self.recording_enabled.setChecked(True)
            form.addRow("Record clips", self.recording_enabled)

            self.person_threshold = QDoubleSpinBox()
            self.person_threshold.setRange(0.0, 1.0)
            self.person_threshold.setSingleStep(0.05)
            self.person_threshold.setValue(self.config.thresholds.person_confidence)
            form.addRow("Person threshold", self.person_threshold)

            self.inventory_threshold = QDoubleSpinBox()
            self.inventory_threshold.setRange(0.0, 1.0)
            self.inventory_threshold.setSingleStep(0.05)
            self.inventory_threshold.setValue(self.config.thresholds.inventory_confidence)
            form.addRow("Inventory threshold", self.inventory_threshold)

            self.required_samples = QSpinBox()
            self.required_samples.setRange(1, 10)
            self.required_samples.setValue(self.config.inventory.required_samples)
            form.addRow("Inventory samples", self.required_samples)

            self.start_button = QPushButton("Start")
            self.stop_button = QPushButton("Stop")
            self.stop_button.setEnabled(False)
            self.rescan_button = QPushButton("Manual rescan")
            self.rescan_button.setEnabled(False)
            self.start_button.clicked.connect(self._start_runtime)
            self.stop_button.clicked.connect(self._stop_runtime)
            self.rescan_button.clicked.connect(self._request_rescan)

            buttons = QWidget()
            buttons_layout = QHBoxLayout(buttons)
            buttons_layout.setContentsMargins(0, 0, 0, 0)
            buttons_layout.addWidget(self.start_button)
            buttons_layout.addWidget(self.stop_button)
            buttons_layout.addWidget(self.rescan_button)
            form.addRow(buttons)

            right.addWidget(controls)

            self.mode_label = QLabel("Mode: idle")
            self.status_label = QLabel("Status: idle")
            self.recording_label = QLabel("Recording: idle")
            right.addWidget(self.mode_label)
            right.addWidget(self.status_label)
            right.addWidget(self.recording_label)

            right.addWidget(QLabel("Frozen scene inventory"))
            self.inventory_list = QListWidget()
            right.addWidget(self.inventory_list, stretch=1)

            right.addWidget(QLabel("Completed clips"))
            self.completed_clips = QTextEdit()
            self.completed_clips.setReadOnly(True)
            right.addWidget(self.completed_clips, stretch=1)

        def closeEvent(self, event) -> None:  # type: ignore[override]
            self._stop_runtime()
            super().closeEvent(event)

        def _choose_output_dir(self) -> None:
            directory = QFileDialog.getExistingDirectory(self, "Choose output directory", self.output_dir.text() or ".")
            if directory:
                self.output_dir.setText(directory)

        def _collect_config(self) -> AppConfig:
            config = load_config()
            config.source.kind = self.source_kind.currentText()
            config.source.value = self.source_value.text().strip()
            config.output.directory = self.output_dir.text().strip() or "recordings"
            config.inventory.auto_rescan = self.auto_rescan.isChecked()
            config.inventory.required_samples = int(self.required_samples.value())
            config.thresholds.person_confidence = float(self.person_threshold.value())
            config.thresholds.inventory_confidence = float(self.inventory_threshold.value())
            config.models = self.config.models
            config.timings = self.config.timings
            return config

        def _start_runtime(self) -> None:
            self._stop_runtime()
            self.config = self._collect_config()
            provider = build_provider(self.config.models)
            label_proposer = build_label_proposer(self.config.models)
            recorder = ClipRecorder(
                self.config.output.directory,
                pre_roll_seconds=self.config.timings.pre_roll_seconds,
                post_roll_seconds=self.config.timings.post_roll_seconds,
            )
            processor = StreamProcessor(self.config, provider, label_proposer, recorder)
            processor.set_recording_enabled(self.recording_enabled.isChecked())
            source = build_source(self.config.source)
            self.runtime = StreamRuntime(
                source,
                processor,
                on_snapshot=lambda snapshot: self.bridge.snapshot_ready.emit(snapshot),
                on_error=lambda message: self.bridge.error_raised.emit(message),
            )
            self.runtime.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.rescan_button.setEnabled(True)
            self.statusBar().showMessage("Starting stream…")

        def _stop_runtime(self) -> None:
            if self.runtime is not None:
                self.runtime.stop()
                self.runtime = None
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.rescan_button.setEnabled(False)

        def _request_rescan(self) -> None:
            if self.runtime is not None:
                self.runtime.request_manual_rescan()

        def _apply_snapshot(self, snapshot: PipelineSnapshot) -> None:
            self.mode_label.setText(f"Mode: {snapshot.mode}")
            self.status_label.setText(f"Status: {snapshot.status_text}")
            self.recording_label.setText(f"Recording: {'active' if snapshot.recording_active else 'idle'}")
            self.inventory_list.clear()
            for item in snapshot.inventory_items:
                self.inventory_list.addItem(f"{item.label} ({item.confidence:.2f})")
            for clip_path in snapshot.completed_clips:
                self.completed_clips.append(str(clip_path))
            self._set_video_frame(snapshot.overlay)
            self.statusBar().showMessage(f"{snapshot.source_id} frame {snapshot.frame_index}")

        def _set_video_frame(self, image_array) -> None:
            height, width, channels = image_array.shape
            bytes_per_line = channels * width
            image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(image)
            scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(scaled)

        def _show_error(self, message: str) -> None:
            self.statusBar().showMessage(message)
            QMessageBox.critical(self, "Meta Watcher error", message)
            self._stop_runtime()

    app = QApplication(list(sys.argv if argv is None else ["meta-watcher", *argv]))
    window = MainWindow(config)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
