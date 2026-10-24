"""Unit tests for output rendering and the data model (no ML deps needed)."""

import json

from transcriber.formats import _clock, render, to_srt, to_txt, to_vtt
from transcriber.model import Segment, Transcript, Word


def _sample() -> Transcript:
    return Transcript(
        language="es",
        model="medium",
        duration=9.0,
        segments=[
            Segment(0.0, 3.0, "Hola, ¿qué tal?", speaker="Speaker 1"),
            Segment(3.0, 5.5, "Muy bien, gracias.", speaker="Speaker 2"),
            Segment(5.5, 9.0, "Me alegro mucho.", speaker="Speaker 1"),
        ],
    )


def test_clock_formats():
    assert _clock(0) == "00:00:00,000"
    assert _clock(3661.5) == "01:01:01,500"
    assert _clock(5.5, ".") == "00:00:05.500"
    assert _clock(-1) == "00:00:00,000"


def test_json_roundtrip_and_speakers():
    data = json.loads(render(_sample(), "json"))
    assert data["language"] == "es"
    assert data["speakers"] == ["Speaker 1", "Speaker 2"]
    assert len(data["segments"]) == 3


def test_srt_numbering_and_speaker_prefix():
    srt = to_srt(_sample())
    assert srt.startswith("1\n")
    assert "00:00:00,000 --> 00:00:03,000" in srt
    assert "Speaker 1: Hola" in srt
    assert "\n3\n" in srt  # three cues


def test_vtt_header_and_timing():
    vtt = to_vtt(_sample())
    assert vtt.startswith("WEBVTT")
    assert "00:00:03.000 --> 00:00:05.500" in vtt


def test_txt_groups_consecutive_same_speaker():
    txt = to_txt(_sample())
    # Speaker 1's two non-adjacent turns stay separate (Speaker 2 is between them).
    assert txt.count("[Speaker 1]") == 2
    assert txt.count("[Speaker 2]") == 1


def test_txt_plain_when_no_speakers():
    t = Transcript(language="en", segments=[Segment(0, 1, "hello"), Segment(1, 2, "world")])
    assert to_txt(t) == "hello\nworld\n"


def test_word_speaker_serialized():
    t = Transcript(language="en", segments=[
        Segment(0, 1, "hi", speaker="Speaker 1",
                words=[Word(0, 1, "hi", probability=0.9, speaker="Speaker 1")]),
    ])
    data = json.loads(render(t, "json"))
    assert data["segments"][0]["words"][0]["speaker"] == "Speaker 1"
