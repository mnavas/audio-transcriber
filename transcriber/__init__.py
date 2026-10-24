"""Audio Transcriber — offline, multilingual (es/en/ja) conversation transcription.

The ML-heavy backends import their dependencies lazily, so this package, the data
model, the output formats and the CLI can be imported and tested without the
faster-whisper / pyannote / torch stack installed.
"""

from .model import Transcript, Segment, Word

__version__ = "0.0.0"

__all__ = ["Transcript", "Segment", "Word", "__version__"]
