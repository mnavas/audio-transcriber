"""Unit tests for speaker assignment (pure, no pyannote needed)."""

from transcriber.diarization import Turn, assign_speakers, normalize_turns
from transcriber.model import Segment, Word


def test_normalize_turns_numbers_by_first_appearance():
    turns = normalize_turns([(3.0, 4.0, "SPEAKER_01"), (0.0, 2.0, "SPEAKER_00")])
    # Sorted by start time; the earliest speaker becomes "Speaker 1".
    assert turns[0].speaker == "Speaker 1"
    assert turns[1].speaker == "Speaker 2"


def test_assign_by_segment_overlap():
    turns = [Turn(0, 3, "Speaker 1"), Turn(3, 6, "Speaker 2")]
    segs = [Segment(0.1, 2.5, "a"), Segment(3.2, 5.0, "b")]
    assign_speakers(segs, turns)
    assert segs[0].speaker == "Speaker 1"
    assert segs[1].speaker == "Speaker 2"


def test_assign_by_word_majority():
    turns = [Turn(0, 2, "Speaker 1"), Turn(2, 4, "Speaker 2")]
    seg = Segment(0, 4, "a b c", words=[
        Word(0.0, 0.9, "a"), Word(1.0, 1.9, "b"), Word(2.1, 3.0, "c"),
    ])
    assign_speakers([seg], turns)
    assert seg.words[0].speaker == "Speaker 1"
    assert seg.words[2].speaker == "Speaker 2"
    assert seg.speaker == "Speaker 1"  # majority (2 of 3)


def test_assign_snaps_to_nearest_when_no_overlap():
    turns = [Turn(0, 2, "Speaker 1"), Turn(10, 12, "Speaker 2")]
    segs = [Segment(2.4, 2.6, "gap")]  # in a silence, closest to Speaker 1
    assign_speakers(segs, turns)
    assert segs[0].speaker == "Speaker 1"


def test_assign_empty_turns_leaves_none():
    segs = [Segment(0, 1, "x")]
    assign_speakers(segs, [])
    assert segs[0].speaker is None
