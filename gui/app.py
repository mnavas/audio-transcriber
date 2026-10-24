"""Minimal PySide6 GUI on top of the transcription core (milestone M5).

Scaffold only in v0.0.0: a window that picks a file, runs the pipeline on a worker
thread, shows the speaker-labeled transcript, and exports it. PySide6 is imported
lazily so the rest of the project does not depend on it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def run() -> int:
    try:
        from PySide6.QtCore import QThread, Signal
        from PySide6.QtWidgets import (
            QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
            QMainWindow, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
        )
    except ImportError:
        print("PySide6 is not installed. Install it with: pip install PySide6",
              file=sys.stderr)
        return 1

    from transcriber.config import MODEL_SIZES, SUPPORTED_LANGUAGES, Config
    from transcriber.formats import FORMATS, render
    from transcriber.pipeline import transcribe_file

    class Worker(QThread):
        done = Signal(object)
        failed = Signal(str)
        progress = Signal(str)

        def __init__(self, path: str, config: Config):
            super().__init__()
            self.path, self.config = path, config

        def run(self):
            try:
                t = transcribe_file(self.path, self.config, progress=self.progress.emit)
                self.done.emit(t)
            except Exception as exc:  # noqa: BLE001
                self.failed.emit(str(exc))

    class Window(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Audio Transcriber")
            self.resize(760, 560)
            self.transcript = None
            self.worker = None

            central = QWidget()
            root = QVBoxLayout(central)

            row = QHBoxLayout()
            self.file_label = QLabel("No file selected")
            choose = QPushButton("Choose audio…")
            choose.clicked.connect(self.choose_file)
            self.lang = QComboBox(); self.lang.addItems(["auto", *SUPPORTED_LANGUAGES])
            self.model = QComboBox(); self.model.addItems(MODEL_SIZES)
            self.model.setCurrentText("medium")
            self.transcribe_btn = QPushButton("Transcribe")
            self.transcribe_btn.clicked.connect(self.start)
            self.transcribe_btn.setEnabled(False)
            for w in (choose, QLabel("Language:"), self.lang,
                      QLabel("Model:"), self.model, self.transcribe_btn):
                row.addWidget(w)
            root.addLayout(row)
            root.addWidget(self.file_label)

            self.output = QPlainTextEdit(); self.output.setReadOnly(True)
            root.addWidget(self.output, 1)

            exports = QHBoxLayout()
            exports.addWidget(QLabel("Export:"))
            for fmt in FORMATS:
                b = QPushButton(fmt.upper())
                b.clicked.connect(lambda _, f=fmt: self.export(f))
                exports.addWidget(b)
            root.addLayout(exports)

            self.setCentralWidget(central)
            self._path = None

        def choose_file(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose audio",
                filter="Audio/Video (*.wav *.mp3 *.m4a *.flac *.ogg *.opus "
                       "*.mp4 *.mkv *.webm);;All files (*)",
            )
            if path:
                self._path = path
                self.file_label.setText(path)
                self.transcribe_btn.setEnabled(True)

        def start(self):
            if not self._path:
                return
            cfg = Config(
                model=self.model.currentText(),
                language=None if self.lang.currentText() == "auto" else self.lang.currentText(),
            )
            self.transcribe_btn.setEnabled(False)
            self.output.setPlainText("")
            self.worker = Worker(self._path, cfg)
            self.worker.progress.connect(lambda m: self.statusBar().showMessage(m))
            self.worker.done.connect(self.on_done)
            self.worker.failed.connect(self.on_failed)
            self.worker.start()

        def on_done(self, transcript):
            self.transcript = transcript
            self.output.setPlainText(render(transcript, "txt"))
            self.statusBar().showMessage(
                f"Done — {transcript.language}, {len(transcript.speakers)} speaker(s)")
            self.transcribe_btn.setEnabled(True)

        def on_failed(self, msg):
            self.statusBar().showMessage(f"Failed: {msg}")
            self.transcribe_btn.setEnabled(True)

        def export(self, fmt):
            if not self.transcript:
                return
            suggested = str(Path(self._path).with_suffix(f".{fmt}"))
            dest, _ = QFileDialog.getSaveFileName(self, f"Export {fmt}", suggested)
            if dest:
                Path(dest).write_text(render(self.transcript, fmt), encoding="utf-8")
                self.statusBar().showMessage(f"Saved {dest}")

    app = QApplication(sys.argv)
    win = Window()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
