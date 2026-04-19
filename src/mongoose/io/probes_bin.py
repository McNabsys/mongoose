"""Binary parser for Nabsys probes.bin files (format V5, little-endian)."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


NABS_MAGIC = 0x5342414E
FILE_TYPE_PROBES = 1
FILE_VERSION_5 = 5

# Fixed sizes
HEADER_SIZE = 20
MISC_SIZE = 16
MOLECULE_FIXED_SIZE = 93
PROBE_SIZE = 24
STRUCTURE_SIZE = 12


@dataclass
class Probe:
    """A single probe event detected on a molecule."""

    start_ms: float
    duration_ms: float
    center_ms: float
    area: float
    max_amplitude: float
    attribute: int

    @property
    def accepted(self) -> bool:
        """Bit 7: probe accepted."""
        return bool(self.attribute & 0x80)

    @property
    def in_clean_region(self) -> bool:
        """Bit 0: probe is in a clean region."""
        return bool(self.attribute & 0x01)


@dataclass
class Structure:
    """A structure detected on a molecule."""

    start_time: float
    end_time: float
    attribute: int


@dataclass
class Molecule:
    """A single molecule record from a probes.bin file."""

    file_name_index: int
    channel: int
    molecule_id: int
    uid: int
    start_ms: float
    start_within_tdb_ms: float
    transloc_time_ms: float
    use_partial_time_ms: float
    mean_lvl1: float
    rise_t10: float
    rise_t50: float
    rise_t90: float
    fall_t90: float
    fall_t50: float
    fall_t10: float
    folded_start_end: float
    folded_end_start: float
    why_structured: int
    num_probes: int
    num_structures: int
    structured: bool
    use_partial: bool
    folded_start: bool
    folded_end: bool
    num_recovered_structures: int
    do_not_use: bool
    probes: list[Probe] = field(default_factory=list)
    structures: list[Structure] = field(default_factory=list)


@dataclass
class ProbesBinFile:
    """Parsed contents of a probes.bin file."""

    file_version: int
    num_molecules: int
    max_probes: int
    last_sample_time: float
    molecules: list[Molecule] = field(default_factory=list)


def load_probes_bin(path: str | Path, *, max_molecules: int | None = None) -> ProbesBinFile:
    """Parse a probes.bin V5 file.

    Args:
        path: Path to the probes.bin file.
        max_molecules: Maximum number of molecules to read. None reads all,
            0 reads only the header/misc data.

    Returns:
        A ProbesBinFile containing the parsed data.
    """
    path = Path(path)

    with open(path, "rb") as f:
        # -- Header (20 bytes) --
        header_size = struct.unpack("<I", f.read(4))[0]
        assert header_size == HEADER_SIZE, f"Unexpected header_size: {header_size}"

        magic = struct.unpack("<I", f.read(4))[0]
        assert magic == NABS_MAGIC, f"Bad magic: {magic:#010x}"

        file_type = struct.unpack("<I", f.read(4))[0]
        assert file_type == FILE_TYPE_PROBES, f"Not a probes file: file_type={file_type}"

        file_version = struct.unpack("<I", f.read(4))[0]
        assert file_version == FILE_VERSION_5, f"Unsupported version: {file_version}"

        data_type = struct.unpack("<I", f.read(4))[0]

        # -- Miscellaneous data (16 bytes) --
        num_molecules = struct.unpack("<I", f.read(4))[0]
        max_probes = struct.unpack("<I", f.read(4))[0]
        last_sample_time = struct.unpack("<d", f.read(8))[0]

        result = ProbesBinFile(
            file_version=file_version,
            num_molecules=num_molecules,
            max_probes=max_probes,
            last_sample_time=last_sample_time,
        )

        # Determine how many molecules to read
        if max_molecules is None:
            count = num_molecules
        else:
            count = min(max_molecules, num_molecules)

        # -- Per-molecule blocks --
        for _ in range(count):
            # Fixed fields (93 bytes)
            file_name_index = struct.unpack("<I", f.read(4))[0]
            channel = struct.unpack("<i", f.read(4))[0]
            molecule_id = struct.unpack("<I", f.read(4))[0]
            uid = struct.unpack("<I", f.read(4))[0]
            start_ms = struct.unpack("<d", f.read(8))[0]
            start_within_tdb_ms = struct.unpack("<f", f.read(4))[0]
            transloc_time_ms = struct.unpack("<f", f.read(4))[0]
            use_partial_time_ms = struct.unpack("<f", f.read(4))[0]
            mean_lvl1 = struct.unpack("<f", f.read(4))[0]
            rise_t10 = struct.unpack("<f", f.read(4))[0]
            rise_t50 = struct.unpack("<f", f.read(4))[0]
            rise_t90 = struct.unpack("<f", f.read(4))[0]
            fall_t90 = struct.unpack("<f", f.read(4))[0]
            fall_t50 = struct.unpack("<f", f.read(4))[0]
            fall_t10 = struct.unpack("<f", f.read(4))[0]
            folded_start_end = struct.unpack("<f", f.read(4))[0]
            folded_end_start = struct.unpack("<f", f.read(4))[0]
            why_structured = struct.unpack("<I", f.read(4))[0]
            num_probes = struct.unpack("<I", f.read(4))[0]
            num_structures = struct.unpack("<I", f.read(4))[0]
            structured = bool(struct.unpack("<B", f.read(1))[0])
            use_partial = bool(struct.unpack("<B", f.read(1))[0])
            folded_start = bool(struct.unpack("<B", f.read(1))[0])
            folded_end = bool(struct.unpack("<B", f.read(1))[0])
            num_recovered_structures = struct.unpack("<I", f.read(4))[0]
            do_not_use = bool(struct.unpack("<B", f.read(1))[0])

            # Per-probe data (24 bytes each)
            probes: list[Probe] = []
            for _ in range(num_probes):
                p_start_ms = struct.unpack("<f", f.read(4))[0]
                p_duration_ms = struct.unpack("<f", f.read(4))[0]
                p_center_ms = struct.unpack("<f", f.read(4))[0]
                p_area = struct.unpack("<f", f.read(4))[0]
                p_max_amplitude = struct.unpack("<f", f.read(4))[0]
                p_attribute = struct.unpack("<I", f.read(4))[0]
                probes.append(Probe(
                    start_ms=p_start_ms,
                    duration_ms=p_duration_ms,
                    center_ms=p_center_ms,
                    area=p_area,
                    max_amplitude=p_max_amplitude,
                    attribute=p_attribute,
                ))

            # Per-structure data (12 bytes each)
            structures: list[Structure] = []
            for _ in range(num_structures):
                s_start_time = struct.unpack("<f", f.read(4))[0]
                s_end_time = struct.unpack("<f", f.read(4))[0]
                s_attribute = struct.unpack("<I", f.read(4))[0]
                structures.append(Structure(
                    start_time=s_start_time,
                    end_time=s_end_time,
                    attribute=s_attribute,
                ))

            mol = Molecule(
                file_name_index=file_name_index,
                channel=channel,
                molecule_id=molecule_id,
                uid=uid,
                start_ms=start_ms,
                start_within_tdb_ms=start_within_tdb_ms,
                transloc_time_ms=transloc_time_ms,
                use_partial_time_ms=use_partial_time_ms,
                mean_lvl1=mean_lvl1,
                rise_t10=rise_t10,
                rise_t50=rise_t50,
                rise_t90=rise_t90,
                fall_t90=fall_t90,
                fall_t50=fall_t50,
                fall_t10=fall_t10,
                folded_start_end=folded_start_end,
                folded_end_start=folded_end_start,
                why_structured=why_structured,
                num_probes=num_probes,
                num_structures=num_structures,
                structured=structured,
                use_partial=use_partial,
                folded_start=folded_start,
                folded_end=folded_end,
                num_recovered_structures=num_recovered_structures,
                do_not_use=do_not_use,
                probes=probes,
                structures=structures,
            )
            result.molecules.append(mol)

    return result
