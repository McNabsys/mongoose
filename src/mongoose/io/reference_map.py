"""Parser for Nabsys _referenceMap.txt files."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceMap:
    genome_name: str
    genome_length: int
    probe_positions: np.ndarray  # int64, sorted ascending


def load_reference_map(path: Path) -> ReferenceMap:
    genome_name = None
    genome_length = None
    probe_positions = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("//DNA Sequence:"):
                genome_name = line.split(":", 1)[1]
            elif line.startswith("//Total Basepair Length:"):
                genome_length = int(line.split(":")[1])
            elif line.startswith("//") or not line:
                continue
            elif probe_positions is None and genome_length is not None:
                probe_positions = np.array(
                    [int(x) for x in line.split("\t") if x.strip()],
                    dtype=np.int64,
                )
                break

    if probe_positions is None or genome_length is None:
        raise ValueError(f"Could not parse reference map from {path}")

    return ReferenceMap(
        genome_name=genome_name,
        genome_length=genome_length,
        probe_positions=np.sort(probe_positions),
    )
