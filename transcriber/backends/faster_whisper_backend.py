"""Offline Whisper transcription via faster-whisper (CTranslate2).

The ``faster_whisper`` import is deferred to :meth:`_load` so importing this module
does not require the ML stack to be installed.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config
from ..model import Segment, Transcript, Word
from .base import TranscriptionBackend


class FasterWhisperBackend(TranscriptionBackend):
    def __init__(self, config: Config):
        self.config = config.resolve()
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "faster-whisper is not installed. Install the ML extras:\n"
                    "  pip install -r requirements.txt"
                ) from exc
            self._model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        return self._model

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> Transcript:
        model = self._load()
        lang = language or self.config.language  # None => auto-detect
        segments_iter, info = model.transcribe(
            wav_path,
            language=lang,
            word_timestamps=True,
            vad_filter=True,  # drop long silences; helps conversational audio
        )

        segments: list[Segment] = []
        for seg in segments_iter:
            words = [
                Word(start=w.start, end=w.end, text=w.word, probability=w.probability)
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(start=seg.start, end=seg.end, text=seg.text.strip(), words=words)
            )

        return Transcript(
            language=info.language,
            segments=segments,
            duration=getattr(info, "duration", None),
            model=self.config.model,
        )
