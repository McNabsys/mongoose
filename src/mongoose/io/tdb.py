"""Binary parser for Nabsys TDB (Time Domain Block) files (V4 format, little-endian)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NABS_MAGIC = 0x5342414E
FILE_TYPE_TDB = 71000
FILE_TYPE_INDEX = 71001
FILE_VERSION = 4

# Number of variable-length string fields in the header.
_NUM_STRING_FIELDS = 23


@dataclass
class TdbHeader:
    """Parsed TDB file header."""

    channel_count: int
    file_number: int
    module_number: int
    total_acq_samples: int
    sample_rate: int
    amplitude_scale_factors: list[float]  # uV per LSB, one per channel
    mean_rms: list[float]
    channel_ids: list[int]
    header_byte_length: int


@dataclass
class TdbMolecule:
    """A single molecule block from a TDB file."""

    channel_source: int
    molecule_id: int
    data_start_index: int
    rise_conv_max_index: int
    fall_conv_min_index: int
    rise_conv_end_index: int
    fall_conv_end_index: int
    structured: bool
    fall_conv_min_value: int
    waveform: np.ndarray  # int16


def _read_length_prefixed_string(f) -> str:
    """Read a uint32-length-prefixed UTF-8 string."""
    (length,) = struct.unpack("<I", f.read(4))
    return f.read(length).decode("utf-8")


def load_tdb_header(path: str | Path) -> TdbHeader:
    """Parse the TDB file header.

    Args:
        path: Path to the .tdb file.

    Returns:
        A TdbHeader with parsed metadata.
    """
    path = Path(path)

    with open(path, "rb") as f:
        # Fixed header fields
        (magic,) = struct.unpack("<I", f.read(4))
        assert magic == NABS_MAGIC, f"Bad magic: {magic:#010x}"

        (file_type,) = struct.unpack("<I", f.read(4))
        assert file_type == FILE_TYPE_TDB, f"Not a TDB file: file_type={file_type}"

        (file_version,) = struct.unpack("<I", f.read(4))
        assert file_version == FILE_VERSION, f"Unsupported version: {file_version}"

        (channel_count,) = struct.unpack("<I", f.read(4))
        (file_number,) = struct.unpack("<I", f.read(4))
        (module_number,) = struct.unpack("<I", f.read(4))
        (total_acq_samples,) = struct.unpack("<I", f.read(4))

        # Run start time (16 bytes, skip)
        f.read(16)

        # Mean RMS per channel (float32)
        mean_rms = list(struct.unpack(f"<{channel_count}f", f.read(4 * channel_count)))

        # Free probe rate per channel (float32, skip)
        f.read(4 * channel_count)

        # 23 variable-length strings (skip all)
        for _ in range(_NUM_STRING_FIELDS):
            _read_length_prefixed_string(f)

        # Sample rate (uint32)
        (sample_rate,) = struct.unpack("<I", f.read(4))

        # Amplitude scale per channel (float64)
        amplitude_scale_factors = list(
            struct.unpack(f"<{channel_count}d", f.read(8 * channel_count))
        )

        # Low pass filter per channel (float64, skip)
        f.read(8 * channel_count)

        # High pass filter per channel (float64, skip)
        f.read(8 * channel_count)

        # Channel IDs per channel (uint32)
        channel_ids = list(
            struct.unpack(f"<{channel_count}I", f.read(4 * channel_count))
        )

        # Sample count time 0 (16 bytes, skip)
        f.read(16)

        # 14 detection parameters (float64)
        f.read(8 * 14)

        # RiseDeNoise, FallDeNoise (uint32)
        f.read(4 * 2)

        # 5 more detection parameters (float64)
        f.read(8 * 5)

        header_byte_length = f.tell()

    return TdbHeader(
        channel_count=channel_count,
        file_number=file_number,
        module_number=module_number,
        total_acq_samples=total_acq_samples,
        sample_rate=sample_rate,
        amplitude_scale_factors=amplitude_scale_factors,
        mean_rms=mean_rms,
        channel_ids=channel_ids,
        header_byte_length=header_byte_length,
    )


def _read_molecule_block(f) -> TdbMolecule:
    """Read a single molecule block at the current file position."""
    (channel_source,) = struct.unpack("<I", f.read(4))
    (molecule_id,) = struct.unpack("<I", f.read(4))
    (data_start_index,) = struct.unpack("<Q", f.read(8))
    (rise_conv_max_index,) = struct.unpack("<I", f.read(4))
    (fall_conv_min_index,) = struct.unpack("<I", f.read(4))
    (rise_conv_end_index,) = struct.unpack("<I", f.read(4))
    (fall_conv_end_index,) = struct.unpack("<I", f.read(4))
    (structured_byte,) = struct.unpack("<B", f.read(1))
    (rise_conv_thresh,) = struct.unpack("<I", f.read(4))
    (fall_conv_thresh,) = struct.unpack("<I", f.read(4))
    (fall_conv_min_value,) = struct.unpack("<i", f.read(4))
    (total_data_count,) = struct.unpack("<I", f.read(4))

    # Molecule samples (int16)
    waveform = np.frombuffer(f.read(total_data_count * 2), dtype=np.int16).copy()

    # MorphOpen
    (morph_count,) = struct.unpack("<I", f.read(4))
    if morph_count > 0:
        f.read(morph_count * 2)  # int16

    # RiseConv
    (rise_count,) = struct.unpack("<I", f.read(4))
    if rise_count > 0:
        f.read(rise_count * 4)  # int32

    # FallConv
    (fall_count,) = struct.unpack("<I", f.read(4))
    if fall_count > 0:
        f.read(fall_count * 4)  # int32

    return TdbMolecule(
        channel_source=channel_source,
        molecule_id=molecule_id,
        data_start_index=data_start_index,
        rise_conv_max_index=rise_conv_max_index,
        fall_conv_min_index=fall_conv_min_index,
        rise_conv_end_index=rise_conv_end_index,
        fall_conv_end_index=fall_conv_end_index,
        structured=bool(structured_byte),
        fall_conv_min_value=fall_conv_min_value,
        waveform=waveform,
    )


def _skip_molecule_block(f) -> None:
    """Skip a molecule block at the current file position without allocating arrays."""
    # Skip fixed fields up to total_data_count:
    # channel(4) + mid(4) + start_idx(8) + rise_max(4) + fall_min(4) +
    # rise_end(4) + fall_end(4) + structured(1) + rise_thresh(4) +
    # fall_thresh(4) + fall_min_val(4) = 45 bytes
    f.read(45)  # everything before total_data_count
    (total_data_count,) = struct.unpack("<I", f.read(4))

    # Skip waveform (int16)
    f.read(total_data_count * 2)

    # MorphOpen
    (morph_count,) = struct.unpack("<I", f.read(4))
    if morph_count > 0:
        f.read(morph_count * 2)

    # RiseConv
    (rise_count,) = struct.unpack("<I", f.read(4))
    if rise_count > 0:
        f.read(rise_count * 4)

    # FallConv
    (fall_count,) = struct.unpack("<I", f.read(4))
    if fall_count > 0:
        f.read(fall_count * 4)


def _get_molecule_offset_from_index(index_path: Path, molecule_index: int) -> int:
    """Look up the byte offset for a molecule from an index file.

    Args:
        index_path: Path to the _index file.
        molecule_index: Zero-based index of the molecule.

    Returns:
        Byte offset in the TDB file where the molecule block starts.
    """
    index_path = Path(index_path)

    with open(index_path, "rb") as f:
        # Index header: magic(4) + file_type(4) + version(4) = 12 bytes
        (magic,) = struct.unpack("<I", f.read(4))
        assert magic == NABS_MAGIC, f"Bad index magic: {magic:#010x}"

        (file_type,) = struct.unpack("<I", f.read(4))
        assert file_type == FILE_TYPE_INDEX, f"Not an index file: {file_type}"

        (version,) = struct.unpack("<I", f.read(4))

        # Each record: channel(4) + uid(4) + offset(8) = 16 bytes
        record_size = 16
        f.seek(12 + molecule_index * record_size)
        data = f.read(record_size)
        if len(data) < record_size:
            raise IndexError(
                f"molecule_index {molecule_index} out of range in index file"
            )

        _channel, _uid, byte_offset = struct.unpack("<IIQ", data)
        return byte_offset


def load_tdb_index(index_path: str | Path) -> dict[tuple[int, int], int]:
    """Load a TDB index file into a (channel, MID) -> byte_offset dict.

    Args:
        index_path: Path to the _index file.

    Returns:
        Dict mapping (channel_source, MID) to the molecule's byte offset
        in its TDB file. MID is the per-channel sequential molecule ID.
    """
    index_path = Path(index_path)
    result: dict[tuple[int, int], int] = {}

    with open(index_path, "rb") as f:
        (magic,) = struct.unpack("<I", f.read(4))
        assert magic == NABS_MAGIC, f"Bad index magic: {magic:#010x}"

        (file_type,) = struct.unpack("<I", f.read(4))
        assert file_type == FILE_TYPE_INDEX, f"Not an index file: {file_type}"

        # Version: spec says "must match base file version" but separately
        # documents the index as "File version 1". We don't enforce here;
        # format of the record itself hasn't changed.
        f.read(4)

        # Records: (channel:uint32, mid:uint32, offset:uint64) = 16 bytes each.
        while True:
            data = f.read(16)
            if len(data) == 0:
                break
            if len(data) < 16:
                raise ValueError(
                    f"Truncated index record at {f.tell() - len(data)}: "
                    f"got {len(data)} bytes, expected 16"
                )
            channel, mid, offset = struct.unpack("<IIQ", data)
            result[(channel, mid)] = offset

    return result


def load_tdb_molecule(
    path: str | Path,
    header: TdbHeader,
    molecule_index: int,
    *,
    index_path: str | Path | None = None,
) -> TdbMolecule:
    """Load a single molecule from a TDB file.

    If *index_path* is provided, uses random access via the index file.
    Otherwise, reads sequentially from the header end, skipping preceding
    molecule blocks.

    Args:
        path: Path to the .tdb file.
        header: Previously parsed TdbHeader.
        molecule_index: Zero-based molecule index.
        index_path: Optional path to the _index file for random access.

    Returns:
        A TdbMolecule with waveform data.
    """
    path = Path(path)

    with open(path, "rb") as f:
        if index_path is not None:
            offset = _get_molecule_offset_from_index(
                Path(index_path), molecule_index
            )
            f.seek(offset)
        else:
            # Sequential: seek past header, then skip preceding molecules
            f.seek(header.header_byte_length)
            for _ in range(molecule_index):
                _skip_molecule_block(f)

        return _read_molecule_block(f)
