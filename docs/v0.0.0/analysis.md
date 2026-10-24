# Audio Transcriber — Analysis (v0.0.0)

**Status:** draft · **Date:** 2026-10-23 · **Author:** Mario

## 1. Problem statement

Take recordings of **conversations** and turn them into accurate, usable text.
The recordings are in **Spanish, English, or Japanese** (one language per file for
v0; mixed-language within a single file is out of scope for now). Because these are
conversations, the output should say **who said what** (speaker labels), not just a
wall of text.

Guiding constraints (from project decisions):

- **Both interfaces** — build a reusable transcription *core*, expose it as a **CLI**
  first, then add a **PySide6 GUI** on top of the same core.
- **Speaker labels required** — diarization ("Speaker 1 / Speaker 2") is in scope for v0.
- **Fully offline** — all processing runs locally on Mario's machine. No audio leaves
  the device, no per-minute cloud cost. Consistent with the offline-first / self-hosted
  pattern across his other projects.

## 2. Users & use cases

- **Primary user:** Mario, transcribing his own conversation recordings.
- **Representative jobs:**
  - "I recorded a 40-minute meeting in Spanish — give me a readable transcript with
    speakers and timestamps."
  - "Transcribe this Japanese interview and give me an SRT I can use for subtitles."
  - "Point the tool at a folder of .m4a files and transcribe all of them overnight."

## 3. The hard parts (what actually makes this non-trivial)

### 3.1 Multilingual ASR (es / en / ja)
- **Whisper** (OpenAI's model) handles all three languages well and auto-detects
  language. It is the de-facto open model for this.
- We run it **offline** via **`faster-whisper`** (a CTranslate2 reimplementation):
  4–5× faster than the reference implementation and far lower memory, with int8/float16
  quantization. Same weights, same accuracy tier.
- **Model size trade-off:** `large-v3` gives the best accuracy (important for Japanese
  and for noisy conversational audio) but is heavy. `medium` is a reasonable fallback on
  CPU-only machines. Model choice should be a config/CLI flag.

### 3.2 Japanese specifics
- Japanese text has **no spaces** between words; Whisper emits natural Japanese text,
  which is what we want — do **not** post-tokenize with spaces.
- **Word-level timestamp alignment** (see below) is weaker for Japanese than for
  space-delimited languages; segment-level timestamps are reliable, word-level is
  best-effort.
- Whisper can optionally transliterate/translate to English (`task=translate`). Keep
  the default as `transcribe` (same language in, same language out); translation is a
  later, optional feature.

### 3.3 Speaker diarization (who said what)
- **`pyannote.audio`** (`pyannote/speaker-diarization-3.1`) is the standard offline
  diarization pipeline. It produces "speaker A speaks 0:00–0:04, speaker B 0:04–0:09…".
- It is a **separate model** from Whisper. The pipeline is: diarize → transcribe →
  **assign each transcribed segment/word to the overlapping speaker turn**.
- **Gotcha:** pyannote models are gated on Hugging Face. You must accept the model terms
  once and download the weights with a HF token. After that, the weights are cached
  locally and **everything runs offline**. This is a one-time setup step, not a runtime
  cloud dependency — worth calling out clearly in the README so it doesn't look like it
  breaks the offline promise.
- Diarization does **not** know how many speakers there are unless told; we can let it
  estimate or accept an optional `--num-speakers` / min/max hint for better results.

### 3.4 Alignment (mapping words to the timeline and to speakers)
- Vanilla Whisper gives rough, segment-level timestamps. For clean speaker assignment
  and good subtitles we want tighter timing.
- **`whisperx`** packages the whole flow we need: faster-whisper transcription →
  forced alignment (wav2vec2) for word-level timestamps → pyannote diarization →
  speaker assignment. It supports es/en/ja alignment models.
- **Decision to make (see Open questions):** adopt **WhisperX** as the orchestrator, or
  wire `faster-whisper` + `pyannote` together ourselves. WhisperX is faster to get
  working and solves alignment+assignment; rolling our own gives more control and fewer
  dependencies. Leaning WhisperX for v0.

### 3.5 Audio ingest
- Inputs will be varied: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, maybe video containers.
- **`ffmpeg`** decodes anything and resamples to the **16 kHz mono PCM** that Whisper and
  pyannote expect. ffmpeg is an external binary dependency (not pip-installable as the
  real codec set) — document it as a prerequisite.

### 3.6 Performance / hardware
- Offline large-model inference is compute-heavy. **GPU (CUDA)** transforms the
  experience; **CPU-only** works but is slow (roughly real-time to several× slower on
  `large`). The tool must:
  - auto-detect GPU vs CPU and pick sane defaults (compute type, model size),
  - show progress for long files,
  - support **batch/folder** processing so long jobs can run unattended.

## 4. Technology choices (proposed stack)

| Concern | Choice | Why |
|---|---|---|
| ASR engine | **faster-whisper** (CTranslate2) | Offline, fast, low memory, es/en/ja |
| Orchestration | **WhisperX** (align + diarize + assign) | Bundles the hard timing/speaker glue |
| Diarization | **pyannote.audio 3.1** (via WhisperX) | Standard offline diarization |
| Audio decode | **ffmpeg** (external binary) | Universal format support, resampling |
| Core language | **Python 3.11+** | Matches the ML ecosystem and Mario's stack |
| CLI | **Typer** (or argparse) | Clean, typed CLI with minimal boilerplate |
| GUI | **PySide6** | Consistent with his other desktop apps |
| Packaging | pip + `pyproject.toml`; `venv` | Reproducible local install |

## 5. Backend abstraction (future-proofing)

Even though v0 is offline-only, define a small **`TranscriptionBackend` interface** so a
cloud engine could be dropped in later without touching the CLI/GUI. The core returns a
normalized **`Transcript`** object (segments, words, speakers, timestamps, language),
and every output format (txt/srt/vtt/json) is rendered from that one object. This keeps
engine choice, interface, and output format independent.

## 6. Output formats

- **`.txt`** — readable transcript, optionally grouped by speaker.
- **`.srt` / `.vtt`** — subtitles with timestamps.
- **`.json`** — full structured result (segments, words, speaker IDs, confidence,
  language) — the canonical artifact the other formats are derived from.

## 7. Risks & unknowns

| Risk | Impact | Mitigation |
|---|---|---|
| pyannote HF gating confuses "offline" story | Setup friction | Clear one-time setup doc; cache weights locally |
| CPU-only machine too slow on `large-v3` | Poor UX | Default to `medium` on CPU; expose model flag; batch mode |
| Japanese word-level alignment is weak | Imperfect subtitle timing | Fall back to segment timestamps for ja |
| Diarization mislabels with overlapping speech | Wrong speaker tags | Allow speaker-count hints; let user correct in GUI later |
| WhisperX dependency churn / version pinning | Build breakage | Pin exact versions; lock file; document CUDA/torch combo |
| Model downloads are large (GBs) | First-run friction | Document disk needs; cache dir; pre-download step |

## 8. Out of scope for v0.0.0

- Mixed-language-within-one-file handling.
- Real-time / live microphone transcription.
- Cloud backends (interface designed for it, not implemented).
- Transcript editing UI beyond basic viewing (GUI starts read-only/export).
- Automatic translation between languages.

## 9. Open questions

1. **WhisperX vs. hand-rolled faster-whisper + pyannote?** (Leaning WhisperX for v0.)
2. **Target hardware** — is there a CUDA GPU available, or must CPU-only be the baseline?
   Drives default model size and realistic performance expectations.
3. **Distribution** — run from source in a venv, or eventually a packaged app
   (PyInstaller) like the other desktop tools?
4. **GUI scope for v0** — just drop-file + transcribe + export, or also an inline
   transcript viewer with speaker colors?
