"""Render a :class:`~transcriber.model.Transcript` to output formats.

``json`` is the canonical artifact; ``txt``/``srt``/``vtt`` are derived views.
All functions are pure (string in, string out) and have no ML dependency.
"""

from __future__ import annotations

import json

from .model import Transcript

FORMATS = ["txt", "srt", "vtt", "json"]


def _clock(seconds: float, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS<sep>mmm (sep ',' for SRT, '.' for VTT)."""
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_json(transcript: Transcript) -> str:
    return json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2)


def to_txt(transcript: Transcript, group_by_speaker: bool = True) -> str:
    """Readable transcript. When speakers are present, consecutive segments by the
    same speaker are merged into one labeled paragraph."""
    segs = [s for s in transcript.segments if s.text.strip()]
    if not segs:
        return ""

    has_speakers = any(s.speaker for s in segs)
    if not (group_by_speaker and has_speakers):
        return "\n".join(s.text.strip() for s in segs) + "\n"

    lines: list[str] = []
    current_speaker = object()  # sentinel distinct from any label
    buffer: list[str] = []

    def flush():
        if buffer:
            label = current_speaker if isinstance(current_speaker, str) else "Unknown"
            lines.append(f"[{label}] {' '.join(buffer)}")

    for s in segs:
        spk = s.speaker or "Unknown"
        if spk != current_speaker:
            flush()
            buffer = []
            current_speaker = spk
        buffer.append(s.text.strip())
    flush()
    return "\n\n".join(lines) + "\n"


def _cue_text(seg) -> str:
    text = seg.text.strip()
    return f"{seg.speaker}: {text}" if seg.speaker else text


def to_srt(transcript: Transcript) -> str:
    blocks = []
    idx = 1
    for seg in transcript.segments:
        if not seg.text.strip():
            continue
        blocks.append(
            f"{idx}\n{_clock(seg.start, ',')} --> {_clock(seg.end, ',')}\n"
            f"{_cue_text(seg)}\n"
        )
        idx += 1
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    out = ["WEBVTT", ""]
    for seg in transcript.segments:
        if not seg.text.strip():
            continue
        out.append(f"{_clock(seg.start, '.')} --> {_clock(seg.end, '.')}")
        out.append(_cue_text(seg))
        out.append("")
    return "\n".join(out)


def render(transcript: Transcript, fmt: str) -> str:
    fmt = fmt.lower()
    if fmt == "txt":
        return to_txt(transcript)
    if fmt == "srt":
        return to_srt(transcript)
    if fmt == "vtt":
        return to_vtt(transcript)
    if fmt == "json":
        return to_json(transcript)
    raise ValueError(f"Unknown format: {fmt!r}. Choose from {FORMATS}.")
