"""Parser for Nabsys _referenceMap.txt files.

File layout (header comments then three data rows):

    //File Type:Reference Chromosome Maps
    //Strands:0 - Top, 1 - Bottom
    //EnzymeUsed:0 - Nb.BssSI
    //Line 1:Probe Location
    //Line 2:Nicked Strand
    //Line 3:Enzyme Index
    //DNA Sequence:<name>
    //Total Basepair Length:<N>
    <tab-sep positions>
    <tab-sep strand values (0|1)>
    <tab-sep enzyme indices>
    //Completed:<date>

The three data rows are **parallel and in file order** — a ``_probeassignment.assigns``
row's ``ProbeK`` value is a 1-based index into this file-order array. We must not
re-sort silently; `.assigns` would then index into the wrong slot. The current
invariant is that positions arrive already strictly ascending in the source file;
we assert that explicitly rather than defensively sorting.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferenceMap:
    """Parallel arrays indexed 0..N-1 in source file order.

    ``.assigns`` uses 1-based indices into these arrays; call
    :meth:`lookup` to retrieve (position, strand) for a given probe index.
    """

    genome_name: str
    genome_length: int
    probe_positions: np.ndarray  # int64, strictly ascending
    strands: np.ndarray          # int8, 0 = top, 1 = bottom
    enzyme_indices: np.ndarray   # int8, enzyme lookup index (0 for single-enzyme runs)

    def lookup(self, ref_idx_1_based: int) -> tuple[int, int]:
        """Map a 1-based ``.assigns`` ProbeK value to (position_bp, strand).

        Raises ``IndexError`` for non-positive or out-of-range indices so
        callers can choose to surface the miss as an explicit null rather
        than silently returning position 0.
        """
        if ref_idx_1_based < 1 or ref_idx_1_based > len(self.probe_positions):
            raise IndexError(
                f"ref_idx {ref_idx_1_based} out of range "
                f"[1, {len(self.probe_positions)}]"
            )
        i = ref_idx_1_based - 1
        return int(self.probe_positions[i]), int(self.strands[i])


def _parse_int_row(line: str) -> np.ndarray:
    return np.array(
        [int(x) for x in line.split("\t") if x.strip()],
        dtype=np.int64,
    )


def load_reference_map(path: Path) -> ReferenceMap:
    genome_name: str | None = None
    genome_length: int | None = None
    data_rows: list[np.ndarray] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("//DNA Sequence:"):
                genome_name = line.split(":", 1)[1]
            elif line.startswith("//Total Basepair Length:"):
                genome_length = int(line.split(":")[1])
            elif line.startswith("//") or not line:
                continue
            else:
                data_rows.append(_parse_int_row(line))
                if len(data_rows) == 3:
                    break

    if genome_length is None:
        raise ValueError(f"Could not parse //Total Basepair Length from {path}")
    if len(data_rows) < 3:
        raise ValueError(
            f"Expected 3 data rows (positions, strand, enzyme) in {path}, "
            f"got {len(data_rows)}"
        )

    positions, strands, enzyme_indices = data_rows[0], data_rows[1], data_rows[2]

    if not (positions.size == strands.size == enzyme_indices.size):
        raise ValueError(
            f"Reference-map row lengths disagree in {path}: "
            f"positions={positions.size}, strands={strands.size}, "
            f"enzyme={enzyme_indices.size}"
        )

    # Assert strictly-ascending file order. The invariant is load-bearing:
    # .assigns ProbeK values are 1-based indices into this array, so if the
    # source file were ever unsorted a silent sort here would corrupt every
    # downstream genomic-position lookup.
    if not np.all(positions[:-1] < positions[1:]):
        raise ValueError(
            f"Reference-map positions are not strictly ascending in {path}; "
            "ProbeK lookups from .assigns would be corrupted."
        )

    return ReferenceMap(
        genome_name=genome_name or "",
        genome_length=genome_length,
        probe_positions=positions.astype(np.int64),
        strands=strands.astype(np.int8),
        enzyme_indices=enzyme_indices.astype(np.int8),
    )
