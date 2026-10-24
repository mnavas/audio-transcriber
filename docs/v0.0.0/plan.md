# Audio Transcriber — Plan (v0.0.0)

**Status:** draft · **Date:** 2026-10-23 · **Author:** Mario

> Goal of v0.0.0: a working **offline** transcriber that takes a conversation audio
> file (Spanish, English, or Japanese), produces an accurate transcript **with speaker
> labels and timestamps**, and can export it to txt/srt/vtt/json — driven from a **CLI**
> first, with a **PySide6 GUI** on top of the same core. See [analysis.md](analysis.md)
> for the reasoning behind these choices.

## 1. Scope for v0.0.0

**In:**
- Offline ASR via `faster-whisper` (es / en / ja, auto language detection).
- Speaker diarization (pyannote via WhisperX) → speaker-labeled segments.
- Folder/batch processing for unattended jobs.
- CLI as the primary interface; reusable core library underneath.
- A minimal PySide6 GUI (drop file → transcribe → view → export).
- Output: `.txt`, `.srt`, `.vtt`, `.json`.

**Out** (see analysis §8): mixed-language files, live mic, cloud backends,
translation, full transcript editor.

## 2. Target architecture

```
audio-transcriber/
├── docs/
│   └── v0.0.0/{plan.md, analysis.md}
├── transcriber/                 # the reusable core package
│   ├── __init__.py
│   ├── audio.py                 # ffmpeg ingest → 16kHz mono wav
│   ├── backends/
│   │   ├── base.py              # TranscriptionBackend interface
│   │   └── whisperx_local.py    # faster-whisper + align + pyannote
│   ├── diarization.py           # speaker assignment helpers
│   ├── model.py                 # Transcript / Segment / Word / Speaker types
│   ├── formats.py               # render Transcript → txt/srt/vtt/json
│   ├── pipeline.py              # orchestrates ingest→asr→align→diarize→assign
│   └── config.py                # model size, device, compute type, cache dirs
├── cli/
│   └── main.py                  # Typer app: transcribe file / folder
├── gui/
│   └── app.py                   # PySide6 window on top of the core
├── tests/
├── pyproject.toml
├── requirements.txt / lock
└── README.md                    # install + one-time pyannote/HF setup
```

**Key principle:** CLI and GUI are thin shells over `transcriber.pipeline`. The pipeline
returns a `Transcript` object; `formats.py` renders every output from it. Engine sits
behind `backends/base.py` so a cloud engine can be added later without touching shells.

## 3. Milestones

### M0 — Project skeleton & environment
- [ ] `git init`, create repo (GitHub `mnavas/audio-transcriber`), `.gitignore`
      (ignore model cache, venv, audio samples, outputs).
- [ ] `pyproject.toml`, `venv`, pinned `requirements.txt`.
- [ ] Verify `ffmpeg` present; add a startup check with a clear error if missing.
- [ ] README stub with prerequisites.

### M1 — Core: ingest + transcription (no speakers yet)
- [ ] `audio.py`: decode any input → 16 kHz mono via ffmpeg.
- [ ] `model.py`: `Transcript`, `Segment`, `Word`, `Speaker` dataclasses.
- [ ] `backends/base.py` + `whisperx_local.py`: run faster-whisper, return segments
      with timestamps and detected language.
- [ ] `config.py`: auto-detect CUDA vs CPU; pick model size + compute type defaults.
- [ ] **Checkpoint:** transcribe one es, one en, one ja sample to plain segments.

### M2 — Alignment + diarization (who said what)
- [ ] Add word-level alignment (WhisperX) where the language supports it.
- [ ] Integrate pyannote diarization; document the one-time HF-token model download.
- [ ] `diarization.py`: assign each segment/word to a speaker turn.
- [ ] Optional `--num-speakers` / min/max hints.
- [ ] **Checkpoint:** a 2-speaker conversation comes out labeled Speaker 1 / Speaker 2.

### M3 — Output formats
- [ ] `formats.py`: render `Transcript` → `.json` (canonical), `.txt` (speaker-grouped),
      `.srt`, `.vtt`.
- [ ] **Checkpoint:** all four formats produced from one run and visually correct.

### M4 — CLI
- [ ] `cli/main.py` (Typer): `transcribe <path>` for a file or folder.
- [ ] Flags: `--language auto|es|en|ja`, `--model`, `--device`, `--diarize/--no-diarize`,
      `--num-speakers`, `--format`, `--output-dir`.
- [ ] Progress output for long files; batch over a folder.
- [ ] **Checkpoint:** `transcribe ./recordings --format srt` processes a folder.

### M5 — Minimal GUI (PySide6)
- [ ] Single window: choose/drop audio file, pick language + model, Transcribe button.
- [ ] Progress indicator; show the transcript (speaker-labeled) when done.
- [ ] Export buttons (txt/srt/vtt/json) → call the same `formats.py`.
- [ ] **Checkpoint:** transcribe + export entirely from the GUI.

### M6 — Polish & release v0.0.0
- [ ] README: install, ffmpeg, one-time pyannote/HF setup, model-size guidance,
      CPU vs GPU expectations.
- [ ] `tests/`: unit tests for `formats.py` and the `Transcript` model; a small
      end-to-end smoke test on a short clip per language.
- [ ] `changelog.md` entry for v0.0.0.
- [ ] Tag `v0.0.0`.

## 4. Dependencies to pin

- `faster-whisper`, `whisperx`, `pyannote.audio`, `torch` (+ matching CUDA build or
  CPU), `ctranslate2`, `typer`, `PySide6`, `ffmpeg` (system binary).
- Record the exact torch/CUDA combination that works — this is the most common source
  of breakage. Keep a lock file.

## 5. Definition of done (v0.0.0)

- From the CLI, transcribe a Spanish, an English, and a Japanese conversation file and
  get a speaker-labeled transcript with timestamps, exported in all four formats.
- The same end-to-end flow works from the PySide6 GUI for at least one file.
- Everything runs offline after the one-time model downloads.
- README lets a fresh machine reproduce the setup.

## 6. First actions (after this plan)

1. `git init` + skeleton (M0).
2. Stand up `audio.py` + a bare faster-whisper call on one sample (M1 checkpoint).
3. Decide WhisperX vs. hand-rolled before starting M2 (analysis §9, Q1).

## 7. Suggested next step

Once you've reviewed these docs, say the word and I'll scaffold **M0** (folder structure,
`pyproject.toml`, `requirements.txt`, `.gitignore`, README stub) so you have a runnable
skeleton to build M1 on.
