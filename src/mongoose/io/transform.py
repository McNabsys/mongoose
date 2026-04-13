"""Parser for Nabsys _transForm.txt files containing per-channel T2D parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChannelTransform:
    """Per-channel T2D transform parameters."""

    channel: str
    mult_const: float
    addit_const: float
    alpha: float


def load_transforms(path: str | Path) -> dict[str, ChannelTransform]:
    """Load per-channel T2D parameters from a _transForm.txt file.

    File format:
        Lines starting with // are comments.
        Header line: Channel  TdmsFileName  MultiplicativeConstant  AdditiveConstant  Alpha
        Data lines: Ch002  file.tdb  5101  -1200  0.56

    Args:
        path: Path to the _transForm.txt file.

    Returns:
        Dict keyed by channel string (e.g. "Ch002") to ChannelTransform.
    """
    transforms: dict[str, ChannelTransform] = {}
    header_seen = False

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            if not header_seen:
                # First non-comment line is the header
                header_seen = True
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            channel = parts[0]
            # parts[1] is TdmsFileName -- skip it
            mult_const = float(parts[2])
            addit_const = float(parts[3])
            alpha = float(parts[4])
            transforms[channel] = ChannelTransform(
                channel=channel,
                mult_const=mult_const,
                addit_const=addit_const,
                alpha=alpha,
            )

    return transforms
