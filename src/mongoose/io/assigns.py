"""Parser for Nabsys _probeassignment.assigns files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MoleculeAssignment:
    """A single molecule's probe-to-reference assignment."""

    ref_index: int  # -1 = unmapped, 0 = E. coli reference
    fragment_uid: int
    direction: int  # 1 = forward, -1 = reverse
    alignment_score: int
    second_best_score: int
    stretch_factor: float
    stretch_offset: float
    probe_indices: tuple[int, ...]  # reference probe indices (1-based), 0 = unmatched


def load_assigns(path: str | Path) -> list[MoleculeAssignment]:
    """Parse a _probeassignment.assigns file.

    Args:
        path: Path to the assigns file.

    Returns:
        List of MoleculeAssignment ordered by fragment UID.
    """
    assignments: list[MoleculeAssignment] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("RefIndex"):
                continue
            parts = line.split("\t")
            ref_index = int(parts[0])
            fragment_uid = int(parts[1])
            direction = int(parts[2])
            alignment_score = int(parts[3])
            second_best = int(parts[4])
            stretch_factor = float(parts[5])
            stretch_offset = float(parts[6])
            # parts[7] is Weight -- skip it
            probe_indices = tuple(int(p) for p in parts[8:] if p.strip())
            assignments.append(MoleculeAssignment(
                ref_index=ref_index,
                fragment_uid=fragment_uid,
                direction=direction,
                alignment_score=alignment_score,
                second_best_score=second_best,
                stretch_factor=stretch_factor,
                stretch_offset=stretch_offset,
                probe_indices=probe_indices,
            ))
    return assignments
