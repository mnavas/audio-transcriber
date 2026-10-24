"""Speaker diarization (who said what) and speaker assignment.

Two parts:

* :class:`PyannoteDiarizer` wraps pyannote.audio (lazy import; needs a one-time HF
  model download, after which it runs offline). It returns a list of ``Turn`` objects.
* :func:`assign_speakers` is a *pure* function that maps transcribed segments/words to
  speaker turns by temporal overlap. It has no ML dependency, so it is fully testable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .config import Config
from .model import Segment


@dataclass
class Turn:
    """A span during which one speaker is talking."""

    start: float
    end: float
    speaker: str  # normalized label, e.g. "Speaker 1"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the temporal overlap between two intervals (0 if disjoint)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_turn(start: float, end: float, turns: list[Turn]) -> Optional[str]:
    """Label of the turn with the most overlap; nearest turn as fallback."""
    if not turns:
        return None
    best, best_ov = None, 0.0
    for t in turns:
        ov = _overlap(start, end, t.start, t.end)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    if best is not None:
        return best
    # No overlap (e.g. a word in a gap): snap to the temporally nearest turn.
    mid = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)))
    return nearest.speaker


def assign_speakers(segments: list[Segment], turns: list[Turn]) -> list[Segment]:
    """Attach a speaker label to each segment (and word), in place. Returns segments.

    When a segment has word-level timing, each word is assigned by its own overlap and
    the segment takes the majority speaker among its words; otherwise the segment is
    assigned directly by its own span.
    """
    for seg in segments:
        if seg.words:
            for w in seg.words:
                w.speaker = _best_turn(w.start, w.end, turns)
            counts: dict[str, int] = {}
            for w in seg.words:
                if w.speaker:
                    counts[w.speaker] = counts.get(w.speaker, 0) + 1
            seg.speaker = (
                max(counts, key=counts.get) if counts
                else _best_turn(seg.start, seg.end, turns)
            )
        else:
            seg.speaker = _best_turn(seg.start, seg.end, turns)
    return segments


class PyannoteDiarizer:
    """Offline diarization via pyannote/speaker-diarization-3.1.

    Requires a Hugging Face token the first time (to download the gated model):
    set ``HF_TOKEN`` in the environment. After the download the weights are cached
    and no network is needed.
    """

    PIPELINE = "pyannote/speaker-diarization-3.1"

    def __init__(self, config: Config):
        self.config = config
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "pyannote.audio is not installed. Install the ML extras:\n"
                    "  pip install -r requirements.txt"
                ) from exc
            self._pipeline = Pipeline.from_pretrained(self.PIPELINE, use_auth_token=token)
            if self.config.device == "cuda":
                try:
                    import torch
                    self._pipeline.to(torch.device("cuda"))
                except Exception:
                    pass
        return self._pipeline

    def diarize(self, wav_path: str) -> list[Turn]:
        pipeline = self._load()
        kwargs: dict = {}
        if self.config.num_speakers:
            kwargs["num_speakers"] = self.config.num_speakers
        else:
            if self.config.min_speakers:
                kwargs["min_speakers"] = self.config.min_speakers
            if self.config.max_speakers:
                kwargs["max_speakers"] = self.config.max_speakers

        annotation = pipeline(wav_path, **kwargs)
        return normalize_turns(
            (seg.start, seg.end, label)
            for seg, _, label in annotation.itertracks(yield_label=True)
        )


def normalize_turns(raw_turns) -> list[Turn]:
    """Convert raw (start, end, label) tuples into Turns with 'Speaker N' labels.

    Speakers are numbered by order of first appearance along the timeline, so output
    is stable and human-friendly regardless of pyannote's internal label names.
    """
    raw = sorted(((float(s), float(e), str(lbl)) for s, e, lbl in raw_turns),
                 key=lambda t: t[0])
    mapping: dict[str, str] = {}
    turns: list[Turn] = []
    for start, end, label in raw:
        if label not in mapping:
            mapping[label] = f"Speaker {len(mapping) + 1}"
        turns.append(Turn(start=start, end=end, speaker=mapping[label]))
    return turns
