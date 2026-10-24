"""The transcription backend interface.

v0 ships a single offline backend (faster-whisper). This abstraction exists so a
cloud backend can be added later without touching the pipeline, CLI or GUI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..model import Transcript


class TranscriptionBackend(ABC):
    @abstractmethod
    def transcribe(self, wav_path: str, language: Optional[str] = None) -> Transcript:
        """Transcribe a 16 kHz mono WAV and return a normalized Transcript."""
        raise NotImplementedError
