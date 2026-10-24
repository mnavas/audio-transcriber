"""Runtime configuration and hardware-aware defaults.

Device detection imports torch lazily; if torch is not installed we assume CPU so the
rest of the package stays importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Whisper model sizes, smallest/fastest first. large-v3 is most accurate (and the
# best choice for Japanese and noisy conversational audio) but heavy on CPU.
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3"]

SUPPORTED_LANGUAGES = {"es": "Spanish", "en": "English", "ja": "Japanese"}


def detect_device() -> str:
    """Return 'cuda' if a GPU is usable, else 'cpu'. Never raises."""
    try:
        import torch  # noqa: PLC0415 (lazy on purpose)

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class Config:
    """Everything the pipeline needs to know about *how* to run."""

    model: str = "medium"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    compute_type: Optional[str] = None  # resolved from device if None
    language: Optional[str] = None  # None => auto-detect
    diarize: bool = True
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    def resolve(self) -> "Config":
        """Fill in device/compute_type based on available hardware."""
        if self.device == "auto":
            self.device = detect_device()
        if self.compute_type is None:
            # int8 keeps CPU usable; float16 is the GPU sweet spot.
            self.compute_type = "int8" if self.device == "cpu" else "float16"
        return self
