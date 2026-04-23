"""Parser for Nabsys `M1_probeWidths.txt` files.

Each run has its own histogram of probe widths, stratified by
"velocity groups" (groups of 10,000 velocity units starting at 75,000).
File structure::

    //M1 Probe Widths
    Group: 0 bins: 35 min: 75000 max: 85000 width: 10000
    <N comma-separated float counts, length == bins>
    Group: 1 bins: 39 min: 85000 max: 95000 width: 10000
    <counts>
    ...

The distribution within a single group is **bimodal**: a narrow-peak
(free tag / misdetection) mode at small widths and a wide-peak
(bound-tag) mode at larger widths. The classifier's
``expected_min_probe_width_factor=0.45`` threshold compares a probe's
duration against the *wide-peak* mode.

Known open question: the file does not declare the probe-width axis
(bin-index → ms conversion). We parse and expose the raw bin counts
and the wide-peak argmax bin index; converting bin index to absolute
duration requires a separate calibration step. The ETL writes the
bin index into ``expected_width_at_velocity_bin`` and leaves
``expected_width_at_velocity_ms`` null by default until the calibration
is resolved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_GROUP_RE = re.compile(
    r"^Group:\s*(\d+)\s+bins:\s*(\d+)\s+min:\s*(\d+)\s+max:\s*(\d+)\s+width:\s*(\d+)"
)


@dataclass(frozen=True)
class VelocityGroup:
    group: int
    n_bins: int
    velocity_min: int
    velocity_max: int
    velocity_width: int
    counts: np.ndarray  # float64, length == n_bins


@dataclass(frozen=True)
class ProbeWidths:
    path: Path
    groups: list[VelocityGroup]

    def group_for_velocity(self, velocity: float) -> int | None:
        """Return the 0-based group index for a velocity, or None if out of range.

        Velocity units match the file header (e.g., 75000..85000 for group 0).
        Exclusive upper bound per group, inclusive lower.
        """
        for g in self.groups:
            if g.velocity_min <= velocity < g.velocity_max:
                return g.group
        return None


def load_probe_widths(path: Path | str) -> ProbeWidths:
    path = Path(path)
    groups: list[VelocityGroup] = []

    with open(path, encoding="latin-1") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("//"):
            i += 1
            continue
        m = _GROUP_RE.match(line)
        if m is None:
            i += 1
            continue
        group_id = int(m.group(1))
        n_bins = int(m.group(2))
        vmin = int(m.group(3))
        vmax = int(m.group(4))
        vwidth = int(m.group(5))

        if i + 1 >= len(lines):
            raise ValueError(f"{path}: missing counts line after group header {line!r}")
        counts_line = lines[i + 1].strip()
        parts = [p for p in counts_line.split(",") if p.strip()]
        if len(parts) != n_bins:
            raise ValueError(
                f"{path}: group {group_id} header declared {n_bins} bins but "
                f"counts line has {len(parts)} values"
            )
        counts = np.array([float(p) for p in parts], dtype=np.float64)
        groups.append(
            VelocityGroup(
                group=group_id,
                n_bins=n_bins,
                velocity_min=vmin,
                velocity_max=vmax,
                velocity_width=vwidth,
                counts=counts,
            )
        )
        i += 2

    if not groups:
        raise ValueError(f"{path}: no velocity groups parsed")

    return ProbeWidths(path=path, groups=groups)


def wide_peak_bin(counts: np.ndarray) -> int:
    """Return the bin index of the wider (upper-half) peak of a bimodal histogram.

    Heuristic: split the array at the midpoint and return the argmax of
    the upper half. Robust enough for the bimodal shape observed in
    Nabsys M1 probe-width histograms. Callers who need a more careful
    bimodal detector can do it after seeing the raw counts.
    """
    if counts.size == 0:
        raise ValueError("wide_peak_bin: empty counts array")
    split = counts.size // 2
    upper = counts[split:]
    return int(split + np.argmax(upper))
