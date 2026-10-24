# Audio Transcriber

Offline, multilingual transcription of **conversations** — Spanish, English and
Japanese — with **speaker labels** and timestamps. Everything runs locally; audio
never leaves your machine.

- **ASR:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (offline Whisper)
- **Speakers:** [pyannote.audio](https://github.com/pyannote/pyannote-audio) diarization
- **Interfaces:** a CLI (`transcribe`) and a PySide6 GUI, both over one reusable core
- **Outputs:** `.txt`, `.srt`, `.vtt`, `.json`

See [docs/v0.0.0/](docs/v0.0.0/) for the plan and analysis, and
[changelog.md](changelog.md) for progress.

## Prerequisites

- **Python 3.11+**
- **ffmpeg** on your PATH (`sudo apt install ffmpeg` / `brew install ffmpeg`)
- A **Hugging Face token** for the one-time diarization model download (see below)

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # core (CPU wheels by default)
pip install -e .                       # installs the `transcribe` command
# optional:
pip install PySide6                     # GUI
pip install pytest                      # tests
```

> **GPU:** this project was bootstrapped on a CPU-only machine. For CUDA, install the
> matching `torch` build from <https://pytorch.org> *before* `requirements.txt`.

### One-time diarization setup (keeps it offline afterwards)

pyannote's diarization model is gated on Hugging Face. Once:

1. Accept the terms for **`pyannote/speaker-diarization-3.1`** (and the models it
   depends on) on huggingface.co while logged in.
2. Create a token and export it:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   ```

The weights download on first run and are cached; after that diarization works offline.
(You can skip all of this and run with `--no-diarize` for a plain transcript.)

## Usage (CLI)

```bash
transcribe recording.m4a                      # auto language, speakers, .txt
transcribe ./recordings --format srt          # whole folder -> subtitles
transcribe interview.wav --language ja --num-speakers 2 --model large-v3
transcribe meeting.mp3 --no-diarize --format all --output-dir ./out
```

Key flags: `--language auto|es|en|ja`, `--model {tiny,base,small,medium,large-v3}`,
`--device auto|cpu|cuda`, `--diarize/--no-diarize`, `--num-speakers N`,
`--format {txt,srt,vtt,json,all}`, `--output-dir DIR`.

## Usage (GUI)

```bash
python -m gui.app
```

Choose a file, pick language/model, Transcribe, then export.

## Model-size guidance (CPU)

`large-v3` is most accurate (best for Japanese and noisy audio) but slow on CPU.
Start with `medium`; drop to `small` for speed, raise to `large-v3` for accuracy.

## Development

```bash
pytest        # format + speaker-assignment unit tests (no ML stack required)
```
