"""Normalized transcript data model.

Every backend produces a :class:`Transcript`; every output format is rendered from
one. This keeps engine choice, interface and output format independent of each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """A single recognized word with its timing."""

    start: float
    end: float
    text: str
    probability: Optional[float] = None
    speaker: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "probability": self.probability,
            "speaker": self.speaker,
        }


@dataclass
class Segment:
    """A contiguous span of speech (one Whisper segment)."""

    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker": self.speaker,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class Transcript:
    """The canonical result of transcribing one audio file."""

    language: str
    segments: list[Segment] = field(default_factory=list)
    duration: Optional[float] = None
    model: Optional[str] = None

    @property
    def speakers(self) -> list[str]:
        """Speaker labels in order of first appearance."""
        seen: list[str] = []
        for seg in self.segments:
            if seg.speaker and seg.speaker not in seen:
                seen.append(seg.speaker)
        return seen

    @property
    def text(self) -> str:
        """The full transcript as plain text, one segment per line."""
        return "\n".join(seg.text.strip() for seg in self.segments if seg.text.strip())

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": round(self.duration, 3) if self.duration is not None else None,
            "model": self.model,
            "speakers": self.speakers,
            "segments": [s.to_dict() for s in self.segments],
        }
