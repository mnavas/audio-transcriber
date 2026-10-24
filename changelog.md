# Changelog

All notable changes to Audio Transcriber are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project uses
semantic versioning.

## [Unreleased] — v0.0.0 in progress

### 2026-10-23 — Project kickoff + core implementation

**Planning (M0 groundwork)**
- Added planning docs under `docs/v0.0.0/`: `analysis.md` and `plan.md`.
- Locked the three founding decisions: **both interfaces** (CLI core + GUI on top),
  **speaker labels required** (diarization in scope for v0), **fully offline**.
- Probed the environment: Python 3.12, ffmpeg present, **no GPU → CPU-only baseline**.
  Defaults set accordingly (`medium` model, `int8` compute type).
- **Resolved analysis open-question Q1 (WhisperX vs. hand-rolled):** chose
  **hand-rolled `faster-whisper` + `pyannote`** over WhisperX — fewer fragile pinned
  deps on Python 3.12, and word-level alignment (WhisperX's main extra) is weak for
  Japanese anyway. Speaker assignment is done by segment/word↔turn temporal overlap.

**Core library (`transcriber/`) — milestones M1–M3**
- `model.py`: `Transcript` / `Segment` / `Word` dataclasses; `speakers`/`text`
  properties; JSON-serializable `to_dict()`.
- `config.py`: `Config` with hardware-aware `resolve()` (device auto-detect, compute
  type), supported languages (es/en/ja), model-size list.
- `audio.py`: ffmpeg ingest → 16 kHz mono WAV; `ensure_ffmpeg()`, `probe_duration()`.
- `backends/base.py` + `backends/faster_whisper_backend.py`: `TranscriptionBackend`
  interface and the offline faster-whisper implementation (lazy ML import, VAD filter,
  word timestamps).
- `diarization.py`: `PyannoteDiarizer` (lazy, HF-token aware), stable `Speaker N`
  relabeling, and the **pure** `assign_speakers()` overlap function.
- `formats.py`: render to `txt` (speaker-grouped), `srt`, `vtt`, `json` (canonical).
- `pipeline.py`: end-to-end `transcribe_file()` orchestration + folder iteration.

**Interfaces — milestones M4–M5**
- `cli/main.py`: stdlib-argparse CLI (`transcribe`), single-file and folder/batch,
  per-file failure tolerance, all output formats. Heavy imports deferred so
  `--help`/`--version` work without the ML stack.
- `gui/app.py`: minimal PySide6 window (choose file → threaded transcribe → view →
  export). Scaffold.

**Project scaffolding (M0)**
- `pyproject.toml` (with `transcribe` entry point, `gui`/`dev` extras),
  `requirements.txt`, `.gitignore`, `README.md` (incl. one-time pyannote/HF setup).
- `tests/`: `test_formats.py` and `test_diarization.py` — pure unit tests, no ML deps.

### Still to do for the v0.0.0 release
- Install the ML stack in a venv and run the first real es/en/ja transcriptions
  (M1–M2 checkpoints) on sample audio.
- Verify a 2-speaker conversation comes out labeled correctly end-to-end.
- Smoke-test the GUI on one file.
- Tag `v0.0.0`.
