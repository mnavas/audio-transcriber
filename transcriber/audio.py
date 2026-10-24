"""Audio ingest: decode any input to the 16 kHz mono WAV that Whisper/pyannote expect.

ffmpeg is an external binary dependency (not pip-installable with the full codec set).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

TARGET_SAMPLE_RATE = 16_000

# Input container/extensions we advertise support for (ffmpeg decodes far more).
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac",
    ".wma", ".mp4", ".mov", ".mkv", ".webm", ".m4b", ".aiff", ".aif",
}


class FFmpegNotFound(RuntimeError):
    pass


def ensure_ffmpeg() -> str:
    """Return the ffmpeg path, or raise a clear error telling the user to install it."""
    path = shutil.which("ffmpeg")
    if not path:
        raise FFmpegNotFound(
            "ffmpeg was not found on PATH. Install it first:\n"
            "  Debian/Ubuntu: sudo apt install ffmpeg\n"
            "  macOS:         brew install ffmpeg"
        )
    return path


def probe_duration(input_path: str | Path) -> float | None:
    """Best-effort media duration in seconds via ffprobe; None if unavailable."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def decode_to_wav(input_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Decode ``input_path`` to 16 kHz mono 16-bit PCM WAV.

    If ``out_path`` is None a temp file is created; the caller is responsible for
    deleting it (see :func:`pipeline.transcribe_file`).
    """
    ffmpeg = ensure_ffmpeg()
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if out_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="transcriber_")
        import os
        os.close(fd)
        out_path = Path(tmp)
    out_path = Path(out_path)

    subprocess.run(
        [ffmpeg, "-y", "-i", str(input_path),
         "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-vn",
         "-c:a", "pcm_s16le", str(out_path)],
        capture_output=True, check=True,
    )
    return out_path
