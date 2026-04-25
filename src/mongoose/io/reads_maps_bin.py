"""Binary parser for Nabsys ``_reads_maps.bin`` files (format V5, little-endian).

A ``_reads_maps.bin`` is the post-remap output of Nabsys's HDM Analysis
pipeline. Each molecule has its detected probes encoded in **bp space**
(reference-genome coordinates) -- distinct from ``probes.bin`` which has
probes in time-domain (ms) on the raw waveform.

For Direction C (production-output residual learning) we have two of
these per run, same length per molecule, same probe arrays except for the
bp-position values:

* ``*_uncorrected_reads_maps.bin``: TVC + head-dive corrections REMOVED
* ``*_reads_maps.bin``: TVC + head-dive corrections APPLIED

Subtracting probe positions across the pair gives the per-probe residual
the V4 residual model needs to learn.

Format reference: ``support/FileFormat_maps.bin.V5.2.pdf``.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


NABS_MAGIC = 0x5342414E  # 'N' 'A' 'B' 'S'
FILE_TYPE_MAPS = 0x00010001
FILE_VERSION_5 = 5

HEADER_SIZE = 20
MOLECULE_FIXED_SIZE = 44  # 11 int32 fields per spec
STRUCTURE_TRIPLET_SIZE = 12  # int32 + int32 + uint32

# Probe attribute bit field positions (per the V5.2 spec):
PROBE_ATTR_IN_CLEAN_REGION = 1 << 0
PROBE_ATTR_IN_FOLDED_END = 1 << 1
PROBE_ATTR_IN_FOLDED_START = 1 << 2
PROBE_ATTR_IN_STRUCTURE = 1 << 3
PROBE_ATTR_EXCLUDED_AMPLITUDE = 1 << 4
PROBE_ATTR_EXCLUDED_WIDTH_SP = 1 << 5  # signal processing
PROBE_ATTR_EXCLUDED_WIDTH_RM = 1 << 6  # remapping
PROBE_ATTR_ACCEPTED = 1 << 7
PROBE_ATTR_OUTSIDE_PARTIAL = 1 << 8

# Structure attribute bit field positions:
STRUCT_ATTR_POTENTIALLY_RECOVERABLE = 1 << 0
STRUCT_ATTR_AMPLITUDE_TOO_HIGH = 1 << 1
STRUCT_ATTR_RECOVERED = 1 << 2


@dataclass
class MapProbe:
    """A single probe record (bp-space) from a ``_reads_maps.bin`` molecule."""

    position_bp: int  # relative to molecule start
    attribute: int  # bit field; see PROBE_ATTR_* constants
    width_bp: int  # post-T2D-optimization probe width

    @property
    def accepted(self) -> bool:
        """True iff probe is accepted for use (bit 7)."""
        return bool(self.attribute & PROBE_ATTR_ACCEPTED)

    @property
    def in_clean_region(self) -> bool:
        return bool(self.attribute & PROBE_ATTR_IN_CLEAN_REGION)

    @property
    def in_structure(self) -> bool:
        return bool(self.attribute & PROBE_ATTR_IN_STRUCTURE)

    @property
    def in_folded_start(self) -> bool:
        return bool(self.attribute & PROBE_ATTR_IN_FOLDED_START)

    @property
    def in_folded_end(self) -> bool:
        return bool(self.attribute & PROBE_ATTR_IN_FOLDED_END)

    @property
    def excluded_by_pf(self) -> bool:
        """True iff PF (probe-width filter) excluded this probe at either stage."""
        mask = PROBE_ATTR_EXCLUDED_WIDTH_SP | PROBE_ATTR_EXCLUDED_WIDTH_RM
        return bool(self.attribute & mask)


@dataclass
class MapStructure:
    """A structure record (bp-space) on a molecule."""

    start_bp: int  # relative to molecule start
    end_bp: int
    attribute: int  # bit field; see STRUCT_ATTR_* constants

    @property
    def recovered(self) -> bool:
        return bool(self.attribute & STRUCT_ATTR_RECOVERED)


@dataclass
class MapMolecule:
    """A single molecule record from a ``_reads_maps.bin`` file."""

    uid: int
    file_name_index: int
    channel: int  # 1-based
    molecule_id: int
    molecule_length_bp: int
    use_partial_length_bp: int  # 0 for clean molecules
    folded_start_end_bp: int  # 0 if no folded start
    folded_end_start_bp: int  # 0 if no folded end
    num_probes: int
    num_structures: int
    num_recovered_structures: int
    probes: list[MapProbe] = field(default_factory=list)
    structures: list[MapStructure] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Per the spec: clean = !(FoldedStart || FoldedEnd || NumStructures > 0)."""
        return (
            self.folded_start_end_bp == 0
            and self.folded_end_start_bp == 0
            and self.num_structures == 0
        )


@dataclass
class ReadsMapsBinFile:
    """Parsed contents of a ``_reads_maps.bin`` V5 file."""

    file_version: int
    num_maps: int
    molecules: list[MapMolecule] = field(default_factory=list)


def load_reads_maps_bin(
    path: str | Path,
    *,
    max_molecules: int | None = None,
) -> ReadsMapsBinFile:
    """Parse a ``_reads_maps.bin`` V5 file end-to-end.

    Args:
        path: Path to the ``_reads_maps.bin`` file.
        max_molecules: If set, stop after parsing this many molecules.
            Useful for partial-read sanity checks.

    Returns:
        A populated :class:`ReadsMapsBinFile`. The header magic / file
        type / version are validated; format mismatches raise
        :class:`ValueError`.
    """
    path = Path(path)
    with open(path, "rb") as f:
        buf = f.read()

    if len(buf) < HEADER_SIZE:
        raise ValueError(
            f"{path}: file shorter than V5 header ({HEADER_SIZE} bytes)."
        )

    header_size, magic, file_type, file_version, num_maps = struct.unpack_from(
        "<iIiii", buf, 0
    )
    if header_size != HEADER_SIZE:
        raise ValueError(
            f"{path}: unexpected header_size {header_size!r}; expected {HEADER_SIZE}."
        )
    if magic != NABS_MAGIC:
        raise ValueError(
            f"{path}: bad magic {magic:#010x}; expected {NABS_MAGIC:#010x} ('NABS')."
        )
    if file_type != FILE_TYPE_MAPS:
        raise ValueError(
            f"{path}: unexpected file_type {file_type:#010x}; "
            f"expected {FILE_TYPE_MAPS:#010x}."
        )
    if file_version != FILE_VERSION_5:
        raise ValueError(
            f"{path}: unsupported file_version {file_version}; this parser handles V5 only."
        )

    molecules: list[MapMolecule] = []
    cursor = HEADER_SIZE
    target_mols = num_maps if max_molecules is None else min(num_maps, max_molecules)

    for _ in range(target_mols):
        if cursor + MOLECULE_FIXED_SIZE > len(buf):
            raise ValueError(
                f"{path}: truncated molecule header at offset {cursor}."
            )

        (
            uid,
            file_name_index,
            channel,
            mol_id,
            mol_len_bp,
            use_partial_len_bp,
            folded_start_end_bp,
            folded_end_start_bp,
            num_probes,
            num_structures,
            num_recovered,
        ) = struct.unpack_from("<IIiIiiiiiiI", buf, cursor)
        cursor += MOLECULE_FIXED_SIZE

        # Three parallel arrays of length N: positions (int32), attributes
        # (uint32), widths (uint32) in that order.
        probe_arr_size = 4 * num_probes
        if cursor + 3 * probe_arr_size > len(buf):
            raise ValueError(
                f"{path}: truncated probe arrays for molecule uid={uid} "
                f"at offset {cursor}."
            )
        positions = struct.unpack_from(f"<{num_probes}i", buf, cursor)
        cursor += probe_arr_size
        attributes = struct.unpack_from(f"<{num_probes}I", buf, cursor)
        cursor += probe_arr_size
        widths = struct.unpack_from(f"<{num_probes}I", buf, cursor)
        cursor += probe_arr_size

        probes = [
            MapProbe(position_bp=p, attribute=a, width_bp=w)
            for p, a, w in zip(positions, attributes, widths, strict=True)
        ]

        # Structure triplets: [start_bp, end_bp, attribute] * num_structures.
        struct_size_total = STRUCTURE_TRIPLET_SIZE * num_structures
        if cursor + struct_size_total > len(buf):
            raise ValueError(
                f"{path}: truncated structure array for molecule uid={uid} "
                f"at offset {cursor}."
            )
        structures: list[MapStructure] = []
        for _ in range(num_structures):
            s, e, a = struct.unpack_from("<iiI", buf, cursor)
            cursor += STRUCTURE_TRIPLET_SIZE
            structures.append(MapStructure(start_bp=s, end_bp=e, attribute=a))

        molecules.append(
            MapMolecule(
                uid=uid,
                file_name_index=file_name_index,
                channel=channel,
                molecule_id=mol_id,
                molecule_length_bp=mol_len_bp,
                use_partial_length_bp=use_partial_len_bp,
                folded_start_end_bp=folded_start_end_bp,
                folded_end_start_bp=folded_end_start_bp,
                num_probes=num_probes,
                num_structures=num_structures,
                num_recovered_structures=num_recovered,
                probes=probes,
                structures=structures,
            )
        )

    return ReadsMapsBinFile(
        file_version=file_version,
        num_maps=num_maps,
        molecules=molecules,
    )
