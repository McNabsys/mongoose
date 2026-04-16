"""Tests for TDB binary parser using synthetic TDB files."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from mongoose.io.tdb import (
    TdbHeader,
    TdbMolecule,
    load_tdb_header,
    load_tdb_molecule,
)

# ---------------------------------------------------------------------------
# Synthetic TDB writer
# ---------------------------------------------------------------------------

NABS_MAGIC = 0x5342414E
FILE_TYPE_TDB = 71000
FILE_TYPE_INDEX = 71001
FILE_VERSION = 4

# 23 variable-length string field names (content is irrelevant for parsing)
_STRING_FIELDS = [
    "sys_id", "software", "sw_version", "sc_sw_version", "operator",
    "run_id", "protocol_name", "protocol_version", "settings_group",
    "settings_version", "module_model", "module_serial", "detector_type",
    "detector_lot", "detector_wafer", "detector_die", "reagent_lot",
    "sample_id", "scientist", "sample_type", "sample_desc", "sample_method",
    "tag_size",
]


def _write_length_prefixed_string(f, s: str) -> None:
    encoded = s.encode("utf-8")
    f.write(struct.pack("<I", len(encoded)))
    f.write(encoded)


def _write_synthetic_tdb(
    path: Path,
    *,
    channel_count: int = 1,
    sample_rate: int = 40000,
    amplitude_scale: float = 1.0,
    molecules: list[dict] | None = None,
) -> None:
    """Write a minimal valid TDB V4 file.

    Each molecule dict should have:
        channel_source: int
        molecule_id: int
        data_start_index: int
        waveform: np.ndarray (int16)
    Optional per-molecule fields default to zero.
    """
    if molecules is None:
        molecules = []

    with open(path, "wb") as f:
        # -- File header fixed fields --
        f.write(struct.pack("<I", NABS_MAGIC))
        f.write(struct.pack("<I", FILE_TYPE_TDB))
        f.write(struct.pack("<I", FILE_VERSION))
        f.write(struct.pack("<I", channel_count))
        f.write(struct.pack("<I", 0))  # file_number
        f.write(struct.pack("<I", 0))  # module_number
        f.write(struct.pack("<I", 0))  # total_acq_samples
        f.write(b"\x00" * 16)          # run start time (16 bytes)

        # mean_rms per channel
        for _ in range(channel_count):
            f.write(struct.pack("<f", 0.0))

        # free_probe_rate per channel
        for _ in range(channel_count):
            f.write(struct.pack("<f", 0.0))

        # 23 variable-length strings
        for name in _STRING_FIELDS:
            _write_length_prefixed_string(f, name)

        # sample_rate
        f.write(struct.pack("<I", sample_rate))

        # amplitude_scale per channel (float64)
        for _ in range(channel_count):
            f.write(struct.pack("<d", amplitude_scale))

        # low_pass_filter per channel (float64)
        for _ in range(channel_count):
            f.write(struct.pack("<d", 0.0))

        # high_pass_filter per channel (float64)
        for _ in range(channel_count):
            f.write(struct.pack("<d", 0.0))

        # channel_ids per channel (uint32)
        for i in range(channel_count):
            f.write(struct.pack("<I", i))

        # sample_count_time_0 (16 bytes)
        f.write(b"\x00" * 16)

        # 21 detection parameters
        detection_params = [
            0.0,  # MinLevel1 (float64)
            0.0,  # InitialLevel1
            0.0,  # FallConvMinToLevel1
            0.0,  # MorphOpenWindow
            0.0,  # MorphCloseWindow
            0.0,  # RiseConvPorch
            0.0,  # RiseConvHole
            0.0,  # RiseConvThreshFrac
            0.0,  # FallConvPorch
            0.0,  # FallConvHole
            0.0,  # FallConvThreshFrac
            0.0,  # MaxMoleculeLength
            0.0,  # DataBeforeRise
            0.0,  # DataAfterFall
        ]
        for val in detection_params:
            f.write(struct.pack("<d", val))

        # RiseDeNoise, FallDeNoise (uint32)
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

        # Remaining float64 detection params
        remaining_params = [
            0.0,  # Level1MaxStepFactor
            0.0,  # FreetagPorch
            0.0,  # FreetagHole
            0.0,  # FreetagMinThresh
            0.0,  # FreetagMaxThresh
        ]
        for val in remaining_params:
            f.write(struct.pack("<d", val))

        # -- Molecule blocks --
        for mol in molecules:
            waveform = np.asarray(mol["waveform"], dtype=np.int16)
            f.write(struct.pack("<I", mol.get("channel_source", 0)))
            f.write(struct.pack("<I", mol.get("molecule_id", 0)))
            f.write(struct.pack("<Q", mol.get("data_start_index", 0)))
            f.write(struct.pack("<I", mol.get("rise_conv_max_index", 0)))
            f.write(struct.pack("<I", mol.get("fall_conv_min_index", 0)))
            f.write(struct.pack("<I", mol.get("rise_conv_end_index", 0)))
            f.write(struct.pack("<I", mol.get("fall_conv_end_index", 0)))
            f.write(struct.pack("<B", int(mol.get("structured", False))))
            f.write(struct.pack("<I", mol.get("rise_conv_thresh", 0)))
            f.write(struct.pack("<I", mol.get("fall_conv_thresh", 0)))
            f.write(struct.pack("<i", mol.get("fall_conv_min_value", 0)))
            f.write(struct.pack("<I", len(waveform)))

            # Molecule samples (int16)
            f.write(waveform.tobytes())

            # MorphOpen: count=0 (omitted)
            f.write(struct.pack("<I", 0))
            # RiseConv: count=0
            f.write(struct.pack("<I", 0))
            # FallConv: count=0
            f.write(struct.pack("<I", 0))


def _write_synthetic_index(
    path: Path,
    molecules: list[dict],
) -> None:
    """Write a TDB index file.

    Each molecule dict must have: channel_source, molecule_id, byte_offset.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<I", NABS_MAGIC))
        f.write(struct.pack("<I", FILE_TYPE_INDEX))
        f.write(struct.pack("<I", FILE_VERSION))
        for mol in molecules:
            f.write(struct.pack("<I", mol["channel_source"]))
            f.write(struct.pack("<I", mol["molecule_id"]))
            f.write(struct.pack("<Q", mol["byte_offset"]))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bump_waveform(n_samples: int = 100, bump_index: int = 50) -> np.ndarray:
    """Create a recognizable waveform: baseline 1000 with a dip at bump_index."""
    waveform = np.full(n_samples, 1000, dtype=np.int16)
    waveform[bump_index] = 500
    return waveform


@pytest.fixture
def synthetic_tdb(tmp_path: Path) -> Path:
    """Create a synthetic TDB with 3 molecules and return its path."""
    mol_defs = []
    for i in range(3):
        wf = _make_bump_waveform(n_samples=100, bump_index=50)
        # Offset the bump value per molecule so we can distinguish them
        wf[50] = np.int16(500 - i * 100)
        mol_defs.append({
            "channel_source": 0,
            "molecule_id": i,
            "data_start_index": i * 10000,
            "rise_conv_max_index": 55,
            "fall_conv_min_index": 60,
            "waveform": wf,
        })

    tdb_path = tmp_path / "test.tdb"
    _write_synthetic_tdb(tdb_path, molecules=mol_defs)
    return tdb_path


@pytest.fixture
def synthetic_tdb_with_index(tmp_path: Path) -> tuple[Path, Path]:
    """Create a synthetic TDB + index file. Return (tdb_path, index_path)."""
    tdb_path = tmp_path / "test.tdb"

    # We need to compute byte offsets as we go. First write the TDB, then
    # read back the file to discover molecule offsets, then write the index.
    mol_defs = []
    for i in range(3):
        wf = _make_bump_waveform(n_samples=100, bump_index=50)
        wf[50] = np.int16(500 - i * 100)
        mol_defs.append({
            "channel_source": 0,
            "molecule_id": i,
            "data_start_index": i * 10000,
            "waveform": wf,
        })

    _write_synthetic_tdb(tdb_path, molecules=mol_defs)

    # Now parse header to find where molecules begin, then compute offsets
    header = load_tdb_header(tdb_path)
    header_end = header.header_byte_length

    # Each molecule block size: fixed fields + waveform + 3 count fields
    # fixed = 4+4+8+4+4+4+4+1+4+4+4+4 = 49 bytes
    # waveform = 100 * 2 = 200 bytes
    # 3 x uint32 counts (all zero) = 12 bytes
    # total per molecule = 49 + 200 + 12 = 261 bytes
    mol_block_size = 49 + 100 * 2 + 12

    index_records = []
    for i, mol in enumerate(mol_defs):
        index_records.append({
            "channel_source": mol["channel_source"],
            "molecule_id": mol["molecule_id"],
            "byte_offset": header_end + i * mol_block_size,
        })

    index_path = tmp_path / "test.tdb_index"
    _write_synthetic_index(index_path, index_records)
    return tdb_path, index_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadTdbHeader:
    def test_sample_rate(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        assert header.sample_rate == 40000

    def test_channel_count(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        assert header.channel_count == 1

    def test_amplitude_scale_factors(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        assert len(header.amplitude_scale_factors) == 1
        assert header.amplitude_scale_factors[0] == 1.0

    def test_channel_ids(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        assert header.channel_ids == [0]

    def test_mean_rms(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        assert len(header.mean_rms) == 1

    def test_header_byte_length_positive(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        assert header.header_byte_length > 0

    def test_multichannel_header(self, tmp_path: Path):
        path = tmp_path / "multi.tdb"
        _write_synthetic_tdb(path, channel_count=4, amplitude_scale=2.5)
        header = load_tdb_header(path)
        assert header.channel_count == 4
        assert len(header.amplitude_scale_factors) == 4
        assert all(s == 2.5 for s in header.amplitude_scale_factors)
        assert header.channel_ids == [0, 1, 2, 3]


class TestLoadTdbMolecule:
    def test_single_molecule_waveform_dtype(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        mol = load_tdb_molecule(synthetic_tdb, header, molecule_index=0)
        assert mol.waveform.dtype == np.int16

    def test_single_molecule_waveform_length(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        mol = load_tdb_molecule(synthetic_tdb, header, molecule_index=0)
        assert len(mol.waveform) == 100

    def test_single_molecule_waveform_values(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        mol = load_tdb_molecule(synthetic_tdb, header, molecule_index=0)
        # Baseline should be 1000
        assert mol.waveform[0] == 1000
        assert mol.waveform[99] == 1000
        # Bump at index 50 should be 500
        assert mol.waveform[50] == 500

    def test_molecule_metadata(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        mol = load_tdb_molecule(synthetic_tdb, header, molecule_index=0)
        assert mol.channel_source == 0
        assert mol.molecule_id == 0
        assert mol.data_start_index == 0
        assert mol.rise_conv_max_index == 55
        assert mol.fall_conv_min_index == 60

    def test_second_molecule(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        mol = load_tdb_molecule(synthetic_tdb, header, molecule_index=1)
        assert mol.molecule_id == 1
        assert mol.data_start_index == 10000
        # Bump for second molecule: 500 - 1*100 = 400
        assert mol.waveform[50] == 400

    def test_third_molecule(self, synthetic_tdb: Path):
        header = load_tdb_header(synthetic_tdb)
        mol = load_tdb_molecule(synthetic_tdb, header, molecule_index=2)
        assert mol.molecule_id == 2
        assert mol.waveform[50] == 300

    def test_structured_flag(self, tmp_path: Path):
        path = tmp_path / "structured.tdb"
        wf = _make_bump_waveform()
        _write_synthetic_tdb(path, molecules=[{
            "channel_source": 0,
            "molecule_id": 0,
            "data_start_index": 0,
            "structured": True,
            "waveform": wf,
        }])
        header = load_tdb_header(path)
        mol = load_tdb_molecule(path, header, molecule_index=0)
        assert mol.structured is True


class TestIndexFileAccess:
    def test_random_access_molecule_0(
        self, synthetic_tdb_with_index: tuple[Path, Path]
    ):
        tdb_path, index_path = synthetic_tdb_with_index
        header = load_tdb_header(tdb_path)
        mol = load_tdb_molecule(
            tdb_path, header, molecule_index=0, index_path=index_path
        )
        assert mol.molecule_id == 0
        assert mol.waveform[50] == 500

    def test_random_access_molecule_2(
        self, synthetic_tdb_with_index: tuple[Path, Path]
    ):
        tdb_path, index_path = synthetic_tdb_with_index
        header = load_tdb_header(tdb_path)
        mol = load_tdb_molecule(
            tdb_path, header, molecule_index=2, index_path=index_path
        )
        assert mol.molecule_id == 2
        assert mol.waveform[50] == 300

    def test_random_access_matches_sequential(
        self, synthetic_tdb_with_index: tuple[Path, Path]
    ):
        """Index-based access should return the same data as sequential."""
        tdb_path, index_path = synthetic_tdb_with_index
        header = load_tdb_header(tdb_path)

        for i in range(3):
            seq_mol = load_tdb_molecule(tdb_path, header, molecule_index=i)
            idx_mol = load_tdb_molecule(
                tdb_path, header, molecule_index=i, index_path=index_path
            )
            assert seq_mol.molecule_id == idx_mol.molecule_id
            np.testing.assert_array_equal(seq_mol.waveform, idx_mol.waveform)


def test_load_header_accepts_version_4(tmp_path):
    """Synthetic TDB written as version=4 must parse cleanly."""
    tdb_path = tmp_path / "v4.tdb"
    _write_synthetic_tdb(tdb_path, molecules=[])
    header = load_tdb_header(tdb_path)
    assert header.channel_count == 1


def test_load_tdb_index_returns_channel_mid_dict(tmp_path):
    """load_tdb_index returns dict keyed by (channel, MID) mapping to byte offset."""
    from mongoose.io.tdb import load_tdb_index

    index_path = tmp_path / "test.tdb_index"
    # Three molecules: channel 5 mid 0 @ offset 1000; channel 5 mid 1 @ 2500;
    # channel 9 mid 0 @ 4000. Note (channel, mid) is unique but mid repeats across channels.
    _write_synthetic_index(
        index_path,
        [
            {"channel_source": 5, "molecule_id": 0, "byte_offset": 1000},
            {"channel_source": 5, "molecule_id": 1, "byte_offset": 2500},
            {"channel_source": 9, "molecule_id": 0, "byte_offset": 4000},
        ],
    )

    idx = load_tdb_index(index_path)

    assert idx[(5, 0)] == 1000
    assert idx[(5, 1)] == 2500
    assert idx[(9, 0)] == 4000
    assert len(idx) == 3


def test_load_tdb_index_rejects_bad_magic(tmp_path):
    from mongoose.io.tdb import load_tdb_index

    bad = tmp_path / "bad.tdb_index"
    bad.write_bytes(b"\x00\x00\x00\x00" + b"\x00" * 100)
    with pytest.raises(AssertionError, match="magic"):
        load_tdb_index(bad)
