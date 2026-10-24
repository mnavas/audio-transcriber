"""Orchestration: ingest -> transcribe -> diarize -> assign speakers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from . import audio
from .backends.faster_whisper_backend import FasterWhisperBackend
from .config import Config
from .diarization import PyannoteDiarizer, assign_speakers
from .model import Transcript

ProgressHook = Callable[[str], None]


def transcribe_file(
    input_path: str | Path,
    config: Optional[Config] = None,
    progress: Optional[ProgressHook] = None,
) -> Transcript:
    """Transcribe one audio file end-to-end and return a :class:`Transcript`."""
    config = (config or Config()).resolve()

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    input_path = Path(input_path)
    log(f"Decoding {input_path.name} -> 16 kHz mono wav")
    wav = audio.decode_to_wav(input_path)
    try:
        log(f"Transcribing with faster-whisper ({config.model}, {config.device})")
        backend = FasterWhisperBackend(config)
        transcript = backend.transcribe(str(wav), language=config.language)
        log(f"Detected language: {transcript.language}, {len(transcript.segments)} segments")

        if config.diarize:
            log("Diarizing (pyannote) and assigning speakers")
            turns = PyannoteDiarizer(config).diarize(str(wav))
            assign_speakers(transcript.segments, turns)
            log(f"Found {len(transcript.speakers)} speaker(s)")

        if transcript.duration is None:
            transcript.duration = audio.probe_duration(input_path)
        return transcript
    finally:
        try:
            wav.unlink(missing_ok=True)
        except OSError:
            pass


def iter_audio_files(path: str | Path) -> list[Path]:
    """Return the audio file(s) at ``path`` (single file, or a folder's contents)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in audio.AUDIO_EXTENSIONS
        )
    raise FileNotFoundError(path)
