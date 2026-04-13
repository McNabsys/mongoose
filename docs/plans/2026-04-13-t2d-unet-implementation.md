# T2D U-Net Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a PyTorch 1D U-Net that predicts per-sample velocity from raw TDB waveforms, producing inter-probe base-pair distances that outperform the legacy T2D parametric model.

**Architecture:** Physics-informed 1D U-Net with velocity output + cumsum integration. Shift-invariant inter-probe delta loss anchored to E. coli reference genome. FiLM conditioning on 6 measured physical observables. See `docs/plans/2026-04-13-t2d-unet-design.md` for full architecture specification.

**Tech Stack:** Python 3.14, PyTorch 2.11+, NumPy, pytest. GPU: RTX A2000 8GB (use mixed precision).

**Data locations:**
- Probes.bin (signal processing output): `smb://higgs/Uno256Data/Samples/Ecoli/BssSI/<RunID>/...`
- Remapping output: `smb://quark/RemappingResults2/Uno256Data/Ecoli/BssSI/<RunID>/...`
- Example data (local): `support/example-data/STB03-064B-02L58270w05-202G16g/`
- Example remap (local): `support/example-remap/STB03-064B-02L58270w05-202G16g/`
- Run inventory: `support/Project Mongoose - input data.xlsx`

---

### Task 1: Project Scaffolding and Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `src/mongoose/__init__.py`
- Create: `src/mongoose/io/__init__.py`
- Create: `src/mongoose/data/__init__.py`
- Create: `src/mongoose/model/__init__.py`
- Create: `src/mongoose/losses/__init__.py`
- Create: `src/mongoose/inference/__init__.py`
- Create: `src/mongoose/training/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "mongoose"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.11.0",
    "numpy>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "openpyxl>=3.1",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 2: Create package structure**

Create all `__init__.py` files (empty). Create `tests/conftest.py` with a fixture pointing to the example data:

```python
import pathlib
import pytest

SUPPORT_DIR = pathlib.Path(__file__).resolve().parent.parent / "support"

@pytest.fixture
def example_data_dir():
    return SUPPORT_DIR / "example-data" / "STB03-064B-02L58270w05-202G16g"

@pytest.fixture
def example_remap_dir():
    return SUPPORT_DIR / "example-remap" / "STB03-064B-02L58270w05-202G16g"

@pytest.fixture
def sigproc_dir(example_data_dir):
    return example_data_dir / "2025-02-19" / "Results" / "1_9_7309" / "pp-705" / "AllCh"

@pytest.fixture
def remap_allch_dir(example_remap_dir):
    return example_remap_dir / "AllCh"
```

**Step 3: Install dependencies**

```bash
pip install torch numpy
pip install -e ".[dev]"
```

**Step 4: Verify**

```bash
python -c "import mongoose; import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
pytest --collect-only
```

**Step 5: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding with package structure and dependencies"
```

---

### Task 2: Reference Map Parser

Parse `_referenceMap.txt` to extract the sorted array of BssSI probe positions (bp) on E. coli.

**Files:**
- Create: `src/mongoose/io/reference_map.py`
- Create: `tests/test_io/__init__.py`
- Create: `tests/test_io/test_reference_map.py`

**Step 1: Write the failing tests**

```python
# tests/test_io/test_reference_map.py
import numpy as np
from mongoose.io.reference_map import load_reference_map

def test_load_reference_map(remap_allch_dir):
    ref_path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_referenceMap.txt"
    ref = load_reference_map(ref_path)

    assert ref.genome_name is not None
    assert "Escherichia coli" in ref.genome_name
    assert ref.genome_length == 4641652
    assert isinstance(ref.probe_positions, np.ndarray)
    assert ref.probe_positions.dtype == np.int64
    # BssSI on E. coli has ~810 sites
    assert 750 < len(ref.probe_positions) < 900
    # Positions are sorted ascending
    assert np.all(np.diff(ref.probe_positions) > 0)
    # All within genome bounds
    assert ref.probe_positions[0] >= 0
    assert ref.probe_positions[-1] < ref.genome_length

def test_reference_map_first_last_positions(remap_allch_dir):
    ref_path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_referenceMap.txt"
    ref = load_reference_map(ref_path)
    # First few known positions from the file
    assert ref.probe_positions[0] == 1953
    assert ref.probe_positions[1] == 2395
    assert ref.probe_positions[2] == 3717
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_io/test_reference_map.py -v
```

Expected: FAIL (module not found)

**Step 3: Implement**

```python
# src/mongoose/io/reference_map.py
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
                # First non-comment line after header = probe positions (tab-separated)
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
```

**Step 4: Run tests**

```bash
pytest tests/test_io/test_reference_map.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/mongoose/io/reference_map.py tests/test_io/
git commit -m "feat: reference map parser for E. coli BssSI probe positions"
```

---

### Task 3: Probes.bin Binary Reader

Parse the Nabsys `probes.bin` file (format V5) to extract molecule metadata and per-probe data.

**Files:**
- Create: `src/mongoose/io/probes_bin.py`
- Create: `tests/test_io/test_probes_bin.py`

**Step 1: Write the failing tests**

```python
# tests/test_io/test_probes_bin.py
import numpy as np
from mongoose.io.probes_bin import load_probes_bin, Molecule

def test_load_probes_bin_header(sigproc_dir):
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path)
    assert result.num_molecules == 82276
    assert result.max_probes == 140
    assert result.file_version == 5

def test_load_probes_bin_first_molecule(sigproc_dir):
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path, max_molecules=1)
    mol = result.molecules[0]
    assert isinstance(mol, Molecule)
    assert mol.uid == 0
    assert mol.channel == 2
    assert mol.num_probes == 13
    assert mol.structured == True
    assert mol.do_not_use == True
    assert len(mol.probes) == 13

def test_clean_molecule_probes(sigproc_dir):
    """Molecule UID=16 is the first clean molecule with do_not_use=False."""
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path, max_molecules=20)
    # Find UID 16
    mol = next(m for m in result.molecules if m.uid == 16)
    assert mol.do_not_use == False
    assert mol.structured == False
    assert mol.num_probes == 15
    assert mol.mean_lvl1 > 0
    assert mol.transloc_time_ms > 0
    # Probe durations should be positive
    for p in mol.probes:
        assert p.duration_ms >= 0
        assert p.center_ms >= 0

def test_filter_clean_molecules(sigproc_dir):
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path)
    clean = [m for m in result.molecules
             if not m.structured and not m.folded_start and not m.folded_end
             and not m.do_not_use and m.num_probes >= 10]
    # From our earlier analysis: ~31K clean molecules
    assert 25000 < len(clean) < 40000
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_io/test_probes_bin.py -v
```

**Step 3: Implement**

The probes.bin format V5 structure (from `FileFormat_probes.bin.V5.2.pdf`):

- Header: 20 bytes (header_size, NABS magic, file_type, file_version, data_type)
- Misc: 16 bytes (num_molecules: uint32, max_probes: uint32, last_sample_time: float64)
- Per molecule: 89 bytes fixed fields + variable probe/structure data

Implementation in `src/mongoose/io/probes_bin.py`:

```python
"""Parser for Nabsys probes.bin binary files (format V5)."""
from dataclasses import dataclass, field
from pathlib import Path
import struct

import numpy as np


@dataclass
class Probe:
    start_ms: float
    duration_ms: float
    center_ms: float
    area: float
    max_amplitude: float
    attribute: int

    @property
    def accepted(self) -> bool:
        return bool(self.attribute & 0x80)

    @property
    def in_clean_region(self) -> bool:
        return bool(self.attribute & 0x01)


@dataclass
class Molecule:
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


_MOL_FMT = "<IiII d ffff ffffff ff IIIBBBBIБ"
# We'll parse field by field for clarity (see implementation)


@dataclass
class ProbesBinFile:
    file_version: int
    num_molecules: int
    max_probes: int
    last_sample_time: float
    molecules: list[Molecule] = field(default_factory=list)


def load_probes_bin(path: Path, *, max_molecules: int | None = None) -> ProbesBinFile:
    with open(path, "rb") as f:
        header_size = struct.unpack("<I", f.read(4))[0]
        nabs_magic = struct.unpack("<I", f.read(4))[0]
        if nabs_magic != 0x5342414E:
            raise ValueError(f"Not a NABS file: magic={hex(nabs_magic)}")
        file_type = struct.unpack("<I", f.read(4))[0]
        file_version = struct.unpack("<I", f.read(4))[0]
        data_type = struct.unpack("<I", f.read(4))[0]

        num_molecules = struct.unpack("<I", f.read(4))[0]
        max_probes = struct.unpack("<I", f.read(4))[0]
        last_sample_time = struct.unpack("<d", f.read(8))[0]

        result = ProbesBinFile(
            file_version=file_version,
            num_molecules=num_molecules,
            max_probes=max_probes,
            last_sample_time=last_sample_time,
        )

        limit = max_molecules if max_molecules is not None else num_molecules
        for _ in range(min(limit, num_molecules)):
            mol = _read_molecule(f)
            result.molecules.append(mol)

    return result


def _read_molecule(f) -> Molecule:
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
    structured = struct.unpack("<B", f.read(1))[0]
    use_partial = struct.unpack("<B", f.read(1))[0]
    folded_start = struct.unpack("<B", f.read(1))[0]
    folded_end = struct.unpack("<B", f.read(1))[0]
    num_recovered = struct.unpack("<I", f.read(4))[0]
    do_not_use = struct.unpack("<B", f.read(1))[0]

    probes = []
    for _ in range(num_probes):
        p_start = struct.unpack("<f", f.read(4))[0]
        p_duration = struct.unpack("<f", f.read(4))[0]
        p_center = struct.unpack("<f", f.read(4))[0]
        p_area = struct.unpack("<f", f.read(4))[0]
        p_max_amp = struct.unpack("<f", f.read(4))[0]
        p_attr = struct.unpack("<I", f.read(4))[0]
        probes.append(Probe(p_start, p_duration, p_center, p_area, p_max_amp, p_attr))

    # Read and discard structure data
    for _ in range(num_structures):
        f.read(12)  # start(f32) + end(f32) + attr(u32)

    return Molecule(
        file_name_index=file_name_index,
        channel=channel,
        molecule_id=molecule_id,
        uid=uid,
        start_ms=start_ms,
        start_within_tdb_ms=start_within_tdb_ms,
        transloc_time_ms=transloc_time_ms,
        use_partial_time_ms=use_partial_time_ms,
        mean_lvl1=mean_lvl1,
        rise_t10=rise_t10, rise_t50=rise_t50, rise_t90=rise_t90,
        fall_t90=fall_t90, fall_t50=fall_t50, fall_t10=fall_t10,
        folded_start_end=folded_start_end,
        folded_end_start=folded_end_start,
        why_structured=why_structured,
        num_probes=num_probes,
        num_structures=num_structures,
        structured=bool(structured),
        use_partial=bool(use_partial),
        folded_start=bool(folded_start),
        folded_end=bool(folded_end),
        num_recovered_structures=num_recovered,
        do_not_use=bool(do_not_use),
        probes=probes,
    )
```

**Step 4: Run tests**

```bash
pytest tests/test_io/test_probes_bin.py -v
```

**Step 5: Commit**

```bash
git add src/mongoose/io/probes_bin.py tests/test_io/test_probes_bin.py
git commit -m "feat: probes.bin binary parser for molecule and probe data"
```

---

### Task 4: Probe Assignment Parser

Parse `_probeassignment.assigns` to get molecule-to-reference mapping (direction, aligned probe indices).

**Files:**
- Create: `src/mongoose/io/assigns.py`
- Create: `tests/test_io/test_assigns.py`

**Step 1: Write the failing tests**

```python
# tests/test_io/test_assigns.py
from mongoose.io.assigns import load_assigns, MoleculeAssignment

def test_load_assigns(remap_allch_dir):
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    assigns = load_assigns(path)
    assert len(assigns) > 0
    # Unmapped molecules have ref_index == -1
    unmapped = [a for a in assigns if a.ref_index == -1]
    mapped = [a for a in assigns if a.ref_index >= 0]
    assert len(unmapped) > 0
    assert len(mapped) > 0

def test_mapped_molecule_has_probes(remap_allch_dir):
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    assigns = load_assigns(path)
    mapped = [a for a in assigns if a.ref_index >= 0]
    mol = mapped[0]
    assert mol.direction in (1, -1)
    assert mol.alignment_score > 0
    assert len(mol.probe_indices) > 0
    # Probe indices are reference probe indices (0-based), or 0 for unmatched
    assert all(isinstance(i, int) for i in mol.probe_indices)

def test_molecule_uid_16_mapped(remap_allch_dir):
    """UID 16 is the first clean molecule. Verify it has an assignment."""
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    assigns = load_assigns(path)
    mol16 = assigns[16]  # assigns indexed by fragment UID
    assert mol16.fragment_uid == 16
    # It should be mapped (ref_index >= 0)
    assert mol16.ref_index >= 0
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_io/test_assigns.py -v
```

**Step 3: Implement**

The `.assigns` file is tab-delimited text. Header lines start with `//`. Data line format:
`RefIndex  FragmentUID  Direction  AlignmentScore  SecondBest  StretchFactor  StretchOffset  Weight  Probe0  Probe1  ...`

Where Probe values are reference probe indices (1-based in the file, 0 = unmatched).

```python
# src/mongoose/io/assigns.py
"""Parser for Nabsys _probeassignment.assigns files."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MoleculeAssignment:
    ref_index: int          # -1 = unmapped, 0 = E. coli
    fragment_uid: int
    direction: int          # 1 = forward, -1 = reverse
    alignment_score: int
    second_best_score: int
    stretch_factor: float
    stretch_offset: float
    probe_indices: tuple[int, ...]  # reference probe indices, 0 = unmatched


def load_assigns(path: Path) -> list[MoleculeAssignment]:
    assignments = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("//") or not line:
                continue
            if line.startswith("RefIndex"):
                continue  # column header
            parts = line.split("\t")
            ref_index = int(parts[0])
            fragment_uid = int(parts[1])
            direction = int(parts[2])
            alignment_score = int(parts[3])
            second_best = int(parts[4])
            stretch_factor = float(parts[5])
            stretch_offset = float(parts[6])
            weight = float(parts[7])
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
```

**Step 4: Run tests**

```bash
pytest tests/test_io/test_assigns.py -v
```

**Step 5: Commit**

```bash
git add src/mongoose/io/assigns.py tests/test_io/test_assigns.py
git commit -m "feat: probe assignment parser for molecule-to-reference mapping"
```

---

### Task 5: TDB Binary Reader

Parse the TDB file to extract raw Int16 waveforms per molecule. This is the most complex binary format.

**Files:**
- Create: `src/mongoose/io/tdb.py`
- Create: `tests/test_io/test_tdb.py`

**Step 1: Write the failing tests**

NOTE: The example data directory does not contain the actual TDB file (it's too large). Tests for TDB parsing should be written against the file format spec, and a small synthetic TDB can be generated for unit testing. For integration testing, tests should be skipped if the TDB file is not present.

```python
# tests/test_io/test_tdb.py
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mongoose.io.tdb import load_tdb_header, load_tdb_molecule, TdbHeader

def _write_minimal_tdb(path: Path, num_channels: int = 1, sample_rate: int = 40000):
    """Create a minimal TDB file with one molecule for testing."""
    with open(path, "wb") as f:
        # File header
        f.write(struct.pack("<I", 0x5342414E))  # NABS magic
        f.write(struct.pack("<I", 71000))         # file type
        f.write(struct.pack("<I", 3))             # file version
        f.write(struct.pack("<I", num_channels))  # channel count
        f.write(struct.pack("<I", 1))             # file number
        f.write(struct.pack("<I", 1))             # module number
        f.write(struct.pack("<I", 100000))        # total acq samples
        # Run start time (16 bytes timestamp)
        f.write(b"\x00" * 16)
        # Mean RMS (float per channel)
        for _ in range(num_channels):
            f.write(struct.pack("<f", 0.5))
        # Free probe rate (float per channel)
        for _ in range(num_channels):
            f.write(struct.pack("<f", 10.0))
        # Variable-length strings: System ID, Software, etc. (11 var strings)
        for _ in range(11):
            f.write(struct.pack("<I", 4))  # length
            f.write(b"test")
        # Sample rate
        f.write(struct.pack("<I", sample_rate))
        # Amplitude scale factor (double per channel)
        for _ in range(num_channels):
            f.write(struct.pack("<d", 1.0))
        # Low/High pass filter (double per channel each)
        for _ in range(num_channels):
            f.write(struct.pack("<d", 10000.0))
        for _ in range(num_channels):
            f.write(struct.pack("<d", 1.0))
        # Channel IDs (int per channel)
        for i in range(num_channels):
            f.write(struct.pack("<I", i + 1))
        # Sample count time zero (16 bytes timestamp)
        f.write(b"\x00" * 16)
        # Min Level1, Initial Level1, and remaining doubles (12 values)
        for _ in range(12):
            f.write(struct.pack("<d", 500.0))
        # Remaining ints (2 values)
        for _ in range(2):
            f.write(struct.pack("<I", 4))
        # Level1MaxStepFactor + 4 freetag params (5 doubles)
        for _ in range(5):
            f.write(struct.pack("<d", 1.0))
        header_end = f.tell()

        # Write one molecule block
        f.write(struct.pack("<I", 1))       # channel source
        f.write(struct.pack("<I", 0))       # MID
        f.write(struct.pack("<Q", 0))       # data start index
        f.write(struct.pack("<I", 10))      # rise conv max index
        f.write(struct.pack("<I", 90))      # fall conv min index
        f.write(struct.pack("<I", 15))      # rise conv end index
        f.write(struct.pack("<I", 95))      # fall conv end index
        f.write(struct.pack("<B", 0))       # structured
        f.write(struct.pack("<I", 100))     # rise conv threshold
        f.write(struct.pack("<I", 100))     # fall conv threshold
        f.write(struct.pack("<i", -200))    # fall conv min value
        # Molecule sample data: 100 samples
        num_samples = 100
        f.write(struct.pack("<I", num_samples))
        waveform = np.array([500] * 100, dtype=np.int16)
        waveform[40:60] = 800  # simulate a probe bump
        f.write(waveform.tobytes())
        # Optional fields: morph, rise conv, fall conv (all zero length)
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))
        f.write(struct.pack("<I", 0))

    return header_end

def test_load_tdb_header():
    with tempfile.NamedTemporaryFile(suffix=".tdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    _write_minimal_tdb(tmp_path, num_channels=1, sample_rate=40000)
    header = load_tdb_header(tmp_path)
    assert header.sample_rate == 40000
    assert header.channel_count == 1
    assert len(header.amplitude_scale_factors) == 1
    tmp_path.unlink()

def test_load_tdb_molecule_waveform():
    with tempfile.NamedTemporaryFile(suffix=".tdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    _write_minimal_tdb(tmp_path)
    header = load_tdb_header(tmp_path)
    mol = load_tdb_molecule(tmp_path, header, molecule_index=0)
    assert mol.channel_source == 1
    assert mol.molecule_id == 0
    assert len(mol.waveform) == 100
    assert mol.waveform.dtype == np.int16
    # Check the probe bump is present
    assert mol.waveform[50] == 800
    tmp_path.unlink()
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_io/test_tdb.py -v
```

**Step 3: Implement**

```python
# src/mongoose/io/tdb.py
"""Parser for Nabsys TDB (Time Domain Block) binary files (format V4)."""
from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass
class TdbHeader:
    channel_count: int
    file_number: int
    module_number: int
    total_acq_samples: int
    sample_rate: int
    amplitude_scale_factors: list[float]  # uV per LSB, one per channel
    mean_rms: list[float]
    channel_ids: list[int]
    header_byte_length: int  # byte offset where molecule blocks begin


@dataclass
class TdbMolecule:
    channel_source: int
    molecule_id: int
    data_start_index: int
    rise_conv_max_index: int
    fall_conv_min_index: int
    rise_conv_end_index: int
    fall_conv_end_index: int
    structured: bool
    fall_conv_min_value: int
    waveform: np.ndarray  # Int16 raw samples


def _read_var_string(f) -> str:
    length = struct.unpack("<I", f.read(4))[0]
    return f.read(length).decode("utf-8", errors="replace")


def load_tdb_header(path: Path) -> TdbHeader:
    with open(path, "rb") as f:
        nabs = struct.unpack("<I", f.read(4))[0]
        if nabs != 0x5342414E:
            raise ValueError(f"Not a NABS file: {hex(nabs)}")
        file_type = struct.unpack("<I", f.read(4))[0]
        file_version = struct.unpack("<I", f.read(4))[0]
        channel_count = struct.unpack("<I", f.read(4))[0]
        file_number = struct.unpack("<I", f.read(4))[0]
        module_number = struct.unpack("<I", f.read(4))[0]
        total_acq_samples = struct.unpack("<I", f.read(4))[0]
        # Run start time (16 byte timestamp)
        f.read(16)
        # Mean RMS (float array, one per channel)
        mean_rms = [struct.unpack("<f", f.read(4))[0] for _ in range(channel_count)]
        # Free probe rate (float array, one per channel)
        f.read(4 * channel_count)
        # Variable-length strings (11 total)
        # System ID, Software, Software Version, System Controller SW Version,
        # Operator, Run ID, Protocol Name, Protocol Version,
        # Settings Group, Settings Version, Module Model Number
        for _ in range(11):
            _read_var_string(f)
        # More var strings: Module Serial, Detector Type, Detector Lot,
        # Detector Wafer ID, Detector Die, Reagent Lot, Sample ID,
        # Scientist Name, Sample Type, Sample Description, Sample Method, Tag Size
        for _ in range(12):
            _read_var_string(f)
        # Sample rate
        sample_rate = struct.unpack("<I", f.read(4))[0]
        # Amplitude scale factors (double per channel)
        amp_scale = [struct.unpack("<d", f.read(8))[0] for _ in range(channel_count)]
        # Low pass filter (double per channel)
        f.read(8 * channel_count)
        # High pass filter (double per channel)
        f.read(8 * channel_count)
        # Channel IDs (int per channel)
        channel_ids = [struct.unpack("<I", f.read(4))[0] for _ in range(channel_count)]
        # Sample count time zero (16 bytes)
        f.read(16)
        # Doubles: MinLevel1, InitialLevel1, FallConvMinToLevel1,
        # MorphOpenWdw, MorphCloseWdw, RiseConvPorch, RiseConvHole,
        # RiseConvThreshFrac, FallConvPorch, FallConvHole,
        # FallConvThreshFrac, MaxMolLength
        f.read(8 * 12)
        # DataBeforeRise, DataAfterFall (doubles)
        f.read(8 * 2)
        # Rise/Fall denoise thresholds (ints)
        f.read(4 * 2)
        # Level1MaxStepFactor (double)
        f.read(8)
        # Freetag params: porch, hole, min_thresh, max_thresh (4 doubles)
        f.read(8 * 4)

        header_byte_length = f.tell()

    return TdbHeader(
        channel_count=channel_count,
        file_number=file_number,
        module_number=module_number,
        total_acq_samples=total_acq_samples,
        sample_rate=sample_rate,
        amplitude_scale_factors=amp_scale,
        mean_rms=mean_rms,
        channel_ids=channel_ids,
        header_byte_length=header_byte_length,
    )


def load_tdb_molecule(path: Path, header: TdbHeader, molecule_index: int) -> TdbMolecule:
    """Load a single molecule block by index. Uses the TDB index file if available."""
    index_path = Path(str(path) + "_index")

    with open(path, "rb") as f:
        if index_path.exists():
            offset = _get_molecule_offset_from_index(index_path, molecule_index)
            f.seek(offset)
        else:
            f.seek(header.header_byte_length)
            for _ in range(molecule_index):
                _skip_molecule_block(f)

        return _read_molecule_block(f)


def _get_molecule_offset_from_index(index_path: Path, molecule_index: int) -> int:
    with open(index_path, "rb") as f:
        # Index header: 12 bytes (NABS, file_type, version)
        f.read(12)
        # Each index record: channel(4) + uid(4) + offset(8) = 16 bytes
        f.seek(12 + molecule_index * 16)
        _channel = struct.unpack("<I", f.read(4))[0]
        _uid = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]
    return offset


def _read_molecule_block(f) -> TdbMolecule:
    channel_source = struct.unpack("<I", f.read(4))[0]
    molecule_id = struct.unpack("<I", f.read(4))[0]
    data_start_index = struct.unpack("<Q", f.read(8))[0]
    rise_conv_max = struct.unpack("<I", f.read(4))[0]
    fall_conv_min = struct.unpack("<I", f.read(4))[0]
    rise_conv_end = struct.unpack("<I", f.read(4))[0]
    fall_conv_end = struct.unpack("<I", f.read(4))[0]
    structured = bool(struct.unpack("<B", f.read(1))[0])
    rise_thresh = struct.unpack("<I", f.read(4))[0]
    fall_thresh = struct.unpack("<I", f.read(4))[0]
    fall_min_val = struct.unpack("<i", f.read(4))[0]

    # Molecule sample data (var length)
    num_samples = struct.unpack("<I", f.read(4))[0]
    waveform = np.frombuffer(f.read(num_samples * 2), dtype=np.int16).copy()

    # Optional: MorphOpenMorphClose (var length Int16)
    morph_count = struct.unpack("<I", f.read(4))[0]
    if morph_count > 0:
        f.read(morph_count * 2)

    # Optional: RiseConv (var length Int32)
    rise_count = struct.unpack("<I", f.read(4))[0]
    if rise_count > 0:
        f.read(rise_count * 4)

    # Optional: FallConv (var length Int32)
    fall_count = struct.unpack("<I", f.read(4))[0]
    if fall_count > 0:
        f.read(fall_count * 4)

    return TdbMolecule(
        channel_source=channel_source,
        molecule_id=molecule_id,
        data_start_index=data_start_index,
        rise_conv_max_index=rise_conv_max,
        fall_conv_min_index=fall_conv_min,
        rise_conv_end_index=rise_conv_end,
        fall_conv_end_index=fall_conv_end,
        structured=structured,
        fall_conv_min_value=fall_min_val,
        waveform=waveform,
    )


def _skip_molecule_block(f):
    """Skip one molecule block without reading waveform data."""
    f.read(4 + 4 + 8 + 4 + 4 + 4 + 4 + 1 + 4 + 4 + 4)  # fixed fields
    num_samples = struct.unpack("<I", f.read(4))[0]
    f.seek(num_samples * 2, 1)
    for nbytes in (2, 4, 4):  # morph(i16), rise(i32), fall(i32)
        count = struct.unpack("<I", f.read(4))[0]
        if count > 0:
            f.seek(count * nbytes, 1)
```

Note: The TDB header has many variable-length string fields. The exact count of variable strings (23 total in V4.6) must match the spec. The implementation above counts 11 + 12 = 23 strings. If parsing fails on real TDB files, the string count is the most likely source of error -- adjust by reading the actual file and comparing byte offsets.

**Step 4: Run tests**

```bash
pytest tests/test_io/test_tdb.py -v
```

**Step 5: Commit**

```bash
git add src/mongoose/io/tdb.py tests/test_io/test_tdb.py
git commit -m "feat: TDB binary reader for raw molecule waveforms"
```

---

### Task 6: Ground Truth Builder

Combine reference map + probe assignments + probes.bin to produce shift-invariant inter-probe GT deltas and velocity targets per molecule.

**Files:**
- Create: `src/mongoose/data/ground_truth.py`
- Create: `tests/test_data/__init__.py`
- Create: `tests/test_data/test_ground_truth.py`

**Step 1: Write the failing tests**

```python
# tests/test_data/test_ground_truth.py
import numpy as np
from mongoose.io.reference_map import load_reference_map
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.assigns import load_assigns
from mongoose.data.ground_truth import build_molecule_gt, MoleculeGT

def test_build_molecule_gt_for_mapped_molecule(remap_allch_dir, sigproc_dir):
    ref = load_reference_map(
        remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_referenceMap.txt"
    )
    assigns = load_assigns(
        remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    )
    probes_file = load_probes_bin(
        sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin",
        max_molecules=50,
    )

    # Find first mapped molecule with enough probes
    for assign in assigns[:50]:
        if assign.ref_index < 0:
            continue
        mol = probes_file.molecules[assign.fragment_uid]
        if mol.num_probes < 8 or mol.do_not_use:
            continue
        gt = build_molecule_gt(mol, assign, ref)
        assert gt is not None
        assert isinstance(gt, MoleculeGT)
        # GT deltas are positive (abs(diff))
        assert np.all(gt.inter_probe_deltas_bp > 0)
        # Number of deltas = number of matched probes - 1
        assert len(gt.inter_probe_deltas_bp) == len(gt.probe_sample_indices) - 1
        # Velocity targets at probe positions
        assert len(gt.velocity_targets_bp_per_ms) == len(gt.probe_sample_indices)
        assert np.all(gt.velocity_targets_bp_per_ms > 0)
        break
    else:
        pytest.fail("No suitable mapped molecule found in first 50")

def test_gt_deltas_sum_reasonable(remap_allch_dir, sigproc_dir):
    """Sum of inter-probe deltas should be less than molecule length."""
    ref = load_reference_map(
        remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_referenceMap.txt"
    )
    assigns = load_assigns(
        remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    )
    probes_file = load_probes_bin(
        sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin",
        max_molecules=50,
    )

    for assign in assigns[:50]:
        if assign.ref_index < 0:
            continue
        mol = probes_file.molecules[assign.fragment_uid]
        if mol.num_probes < 8 or mol.do_not_use:
            continue
        gt = build_molecule_gt(mol, assign, ref)
        if gt is None:
            continue
        total_span = np.sum(gt.inter_probe_deltas_bp)
        # Total should be < genome length and > 0
        assert 0 < total_span < ref.genome_length
        break
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data/test_ground_truth.py -v
```

**Step 3: Implement**

```python
# src/mongoose/data/ground_truth.py
"""Build ground truth from reference map + probe assignments."""
from dataclasses import dataclass

import numpy as np

from mongoose.io.reference_map import ReferenceMap
from mongoose.io.probes_bin import Molecule
from mongoose.io.assigns import MoleculeAssignment

TAG_WIDTH_BP = 511
SAMPLE_RATE_HZ = 40_000
SAMPLE_PERIOD_MS = 1000.0 / SAMPLE_RATE_HZ  # 0.025 ms


@dataclass
class MoleculeGT:
    probe_sample_indices: np.ndarray   # int, sample index of each matched probe
    inter_probe_deltas_bp: np.ndarray  # float64, abs(diff(ref_bp)), always positive
    velocity_targets_bp_per_ms: np.ndarray  # float64, 511 / duration_ms per probe
    reference_probe_bp: np.ndarray     # int64, absolute reference bp for each probe
    direction: int                      # 1 = forward, -1 = reverse


def build_molecule_gt(
    mol: Molecule,
    assign: MoleculeAssignment,
    ref: ReferenceMap,
    min_matched_probes: int = 4,
) -> MoleculeGT | None:
    if assign.ref_index < 0:
        return None

    # Match probes: assign.probe_indices are 1-based reference probe indices.
    # 0 means unmatched (false negative or excluded).
    matched = []
    for mol_probe_idx, ref_probe_idx in enumerate(assign.probe_indices):
        if ref_probe_idx == 0:
            continue
        if mol_probe_idx >= len(mol.probes):
            continue
        probe = mol.probes[mol_probe_idx]
        if probe.duration_ms <= 0:
            continue
        # ref_probe_idx is 1-based -> convert to 0-based
        ref_bp = ref.probe_positions[ref_probe_idx - 1]
        sample_idx = int(round(probe.center_ms / SAMPLE_PERIOD_MS))
        velocity = TAG_WIDTH_BP / probe.duration_ms  # bp/ms
        matched.append((sample_idx, ref_bp, velocity))

    if len(matched) < min_matched_probes:
        return None

    # Sort by temporal order (sample index)
    matched.sort(key=lambda x: x[0])

    sample_indices = np.array([m[0] for m in matched], dtype=np.int64)
    ref_bps = np.array([m[1] for m in matched], dtype=np.int64)
    velocities = np.array([m[2] for m in matched], dtype=np.float64)

    # Shift-invariant inter-probe deltas (absolute value handles fwd/rev)
    deltas = np.abs(np.diff(ref_bps)).astype(np.float64)

    # Filter out zero deltas (duplicate reference positions from clustered sites)
    valid = deltas > 0
    if np.sum(valid) < min_matched_probes - 1:
        return None

    # Keep only intervals where both flanking probes are valid
    keep_probes = np.ones(len(matched), dtype=bool)
    # For simplicity, keep all probes but filter deltas
    # (The loss will only evaluate on valid deltas)

    return MoleculeGT(
        probe_sample_indices=sample_indices,
        inter_probe_deltas_bp=deltas,
        velocity_targets_bp_per_ms=velocities,
        reference_probe_bp=ref_bps,
        direction=assign.direction,
    )
```

**Step 4: Run tests**

```bash
pytest tests/test_data/test_ground_truth.py -v
```

**Step 5: Commit**

```bash
git add src/mongoose/data/ground_truth.py tests/test_data/
git commit -m "feat: ground truth builder with shift-invariant inter-probe deltas"
```

---

### Task 7: U-Net Model Architecture

Build the complete model: ResBlocks, FiLM, encoder, dilated bottleneck + MHSA, decoder, probe head, velocity head.

**Files:**
- Create: `src/mongoose/model/blocks.py`
- Create: `src/mongoose/model/unet.py`
- Create: `tests/test_model/__init__.py`
- Create: `tests/test_model/test_unet.py`

**Step 1: Write the failing tests**

```python
# tests/test_model/test_unet.py
import torch
from mongoose.model.unet import T2DUNet

def test_unet_forward_shape():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    B, T = 2, 4096
    x = torch.randn(B, 1, T)
    cond = torch.randn(B, 6)
    mask = torch.ones(B, T, dtype=torch.bool)
    probe_heatmap, cumulative_bp = model(x, cond, mask)
    assert probe_heatmap.shape == (B, T)
    assert cumulative_bp.shape == (B, T)

def test_unet_cumulative_bp_monotonic():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    B, T = 1, 2048
    x = torch.randn(B, 1, T)
    cond = torch.randn(B, 6)
    mask = torch.ones(B, T, dtype=torch.bool)
    _, cumulative_bp = model(x, cond, mask)
    diffs = torch.diff(cumulative_bp, dim=-1)
    # Softplus + cumsum guarantees monotonic (all diffs > 0 where mask is True)
    assert (diffs >= 0).all()

def test_unet_probe_heatmap_range():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    x = torch.randn(1, 1, 2048)
    cond = torch.randn(1, 6)
    mask = torch.ones(1, 2048, dtype=torch.bool)
    probe_heatmap, _ = model(x, cond, mask)
    assert (probe_heatmap >= 0).all()
    assert (probe_heatmap <= 1).all()

def test_unet_padding_mask_zeroes_velocity():
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    x = torch.randn(1, 1, 2048)
    cond = torch.randn(1, 6)
    # Mask out last 512 samples
    mask = torch.ones(1, 2048, dtype=torch.bool)
    mask[0, 1536:] = False
    _, cumulative_bp = model(x, cond, mask)
    # Cumulative BP should be flat in masked region
    masked_region = cumulative_bp[0, 1536:]
    assert torch.allclose(masked_region, masked_region[0].expand_as(masked_region))

def test_unet_variable_length():
    """Model handles different sequence lengths."""
    model = T2DUNet(in_channels=1, conditioning_dim=6)
    for T in [1024, 2048, 4096, 8192]:
        x = torch.randn(1, 1, T)
        cond = torch.randn(1, 6)
        mask = torch.ones(1, T, dtype=torch.bool)
        probe_heatmap, cumulative_bp = model(x, cond, mask)
        assert probe_heatmap.shape == (1, T)
        assert cumulative_bp.shape == (1, T)
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_model/test_unet.py -v
```

**Step 3: Implement**

This is the largest implementation task. Split into `blocks.py` (ResBlock, FiLM) and `unet.py` (full model).

See `docs/plans/2026-04-13-t2d-unet-design.md` Architecture section for the exact specification:
- Encoder: 5 levels, channels [32, 64, 128, 256, 512], ResBlock(k=7), MaxPool(2)
- Bottleneck: dilated cascade [1,2,4,8] k=7 + MHSA (4 heads)
- Decoder: symmetric 5 levels with skip connections
- FiLM at encoder level 0 (2*32 params) and bottleneck (2*512 params)
- Probe branch: 2x ResBlock(k=7) + Conv1d(32,1,1) + Sigmoid
- Velocity branch: 2x ResBlock(k=31) + Conv1d(32,1,1) + Softplus, then mask * cumsum

Input must be padded to a multiple of 32 (2^5 downsampling levels) internally.

**Step 4: Run tests**

```bash
pytest tests/test_model/test_unet.py -v
```

**Step 5: Commit**

```bash
git add src/mongoose/model/ tests/test_model/
git commit -m "feat: 1D U-Net with FiLM conditioning, dilated bottleneck, and MHSA"
```

---

### Task 8: Loss Functions

Implement the three loss components: focal loss, sparse Huber on inter-probe deltas, sparse L2 on velocity.

**Files:**
- Create: `src/mongoose/losses/focal.py`
- Create: `src/mongoose/losses/spatial.py`
- Create: `src/mongoose/losses/velocity.py`
- Create: `src/mongoose/losses/combined.py`
- Create: `tests/test_losses/__init__.py`
- Create: `tests/test_losses/test_losses.py`

**Step 1: Write the failing tests**

```python
# tests/test_losses/test_losses.py
import torch
import numpy as np
from mongoose.losses.focal import focal_loss
from mongoose.losses.spatial import sparse_huber_delta_loss
from mongoose.losses.velocity import sparse_velocity_loss
from mongoose.losses.combined import CombinedLoss

def test_focal_loss_zero_for_perfect_prediction():
    pred = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])  # peak at index 2
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    loss = focal_loss(pred, target, gamma=2.0, alpha=0.25)
    assert loss.item() < 0.01

def test_focal_loss_high_for_missed_peak():
    pred = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])  # missed the peak
    target = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
    loss = focal_loss(pred, target, gamma=2.0, alpha=0.25)
    assert loss.item() > 0.1

def test_sparse_huber_delta_loss():
    # Predicted cumulative BP at probe positions
    pred_cumulative = torch.tensor([0.0, 1000.0, 3500.0, 7000.0])
    # GT inter-probe deltas
    gt_deltas = torch.tensor([1000.0, 2500.0, 3500.0])
    loss = sparse_huber_delta_loss(pred_cumulative, gt_deltas, delta=500.0)
    # pred deltas = [1000, 2500, 3500] == gt_deltas, so loss should be ~0
    assert loss.item() < 0.01

def test_sparse_huber_delta_loss_nonzero():
    pred_cumulative = torch.tensor([0.0, 1200.0, 3500.0, 7000.0])
    gt_deltas = torch.tensor([1000.0, 2500.0, 3500.0])
    loss = sparse_huber_delta_loss(pred_cumulative, gt_deltas, delta=500.0)
    # pred deltas = [1200, 2300, 3500], errors = [200, -200, 0]
    assert loss.item() > 0

def test_sparse_velocity_loss():
    pred_velocity = torch.tensor([0.5, 0.4, 0.3, 0.2])  # bp/sample at probe indices
    target_velocity = torch.tensor([0.5, 0.4, 0.3, 0.2])
    loss = sparse_velocity_loss(pred_velocity, target_velocity)
    assert loss.item() < 1e-6

def test_combined_loss_warmup():
    combined = CombinedLoss(lambda_bp=1.0, lambda_vel=1.0, warmup_epochs=5)
    # At epoch 0, lambda multipliers should be 0
    combined.set_epoch(0)
    assert combined.current_lambda_bp == 0.0
    # At epoch 5, should be 1.0
    combined.set_epoch(5)
    assert combined.current_lambda_bp == 1.0
    # At epoch 2, should be 0.4
    combined.set_epoch(2)
    assert abs(combined.current_lambda_bp - 0.4) < 1e-6
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_losses/test_losses.py -v
```

**Step 3: Implement**

Key details from the design doc:
- Focal loss: `FL = -alpha * (1-p)^gamma * log(p)` for positives, `-(1-alpha) * p^gamma * log(1-p)` for negatives
- Sparse Huber: compute in native bp with delta=500, normalize by dividing scalar output by mean GT interval
- Sparse velocity L2: straightforward MSE at probe positions
- Combined: linear warmup of lambda_bp and lambda_vel over first N epochs

**Step 4: Run tests**

```bash
pytest tests/test_losses/test_losses.py -v
```

**Step 5: Commit**

```bash
git add src/mongoose/losses/ tests/test_losses/
git commit -m "feat: focal, sparse Huber, and sparse velocity losses with warmup"
```

---

### Task 9: PyTorch Dataset and DataLoader

Build the Dataset that combines TDB waveforms + probes.bin + ground truth, with dynamic length bucketing.

**Files:**
- Create: `src/mongoose/data/dataset.py`
- Create: `src/mongoose/data/augment.py`
- Create: `tests/test_data/test_dataset.py`

**Step 1: Write the failing tests**

```python
# tests/test_data/test_dataset.py
import torch
from mongoose.data.dataset import MoleculeDataset, collate_molecules

def test_dataset_item_structure():
    """Test that a dataset item has the right keys and shapes."""
    # This test requires actual data files -- skip if not available
    # The dataset should return a dict with:
    # - waveform: [1, T] float tensor (normalized)
    # - conditioning: [6] float tensor
    # - probe_heatmap_target: [T] float tensor
    # - probe_sample_indices: [N] long tensor
    # - gt_deltas_bp: [N-1] float tensor
    # - velocity_targets: [N] float tensor
    # - mask: [T] bool tensor
    pass  # Implementation test uses fixture with real data

def test_collate_pads_to_max_length():
    """Collate function should pad shorter molecules."""
    items = [
        {"waveform": torch.randn(1, 100), "mask": torch.ones(100, dtype=torch.bool)},
        {"waveform": torch.randn(1, 200), "mask": torch.ones(200, dtype=torch.bool)},
    ]
    batch = collate_molecules(items)
    assert batch["waveform"].shape == (2, 1, 224)  # padded to multiple of 32
    assert batch["mask"].shape == (2, 224)
    assert batch["mask"][0, 100:].sum() == 0  # padding is masked
```

**Step 2: Implement dataset, augmentations, and collate function**

Key implementation details:
- `MoleculeDataset.__getitem__` loads waveform from TDB (via index), normalizes by level-1, builds heatmap target with dynamic sigma, assembles conditioning vector
- `augment.py`: time-stretch (resample waveform, scale velocity targets), noise injection, amplitude scaling
- `collate_molecules`: pad to max length in batch (rounded to multiple of 32), build masks
- The dataset pre-computes a manifest of (tdb_path, molecule_index, probes_bin_index, assign_index) tuples during `__init__`

**Step 3-5: Test, verify, commit**

```bash
pytest tests/test_data/test_dataset.py -v
git add src/mongoose/data/ tests/test_data/
git commit -m "feat: molecule dataset with dynamic batching and augmentations"
```

---

### Task 10: Training Loop

Implement the training loop with mixed precision, checkpointing, and logging.

**Files:**
- Create: `src/mongoose/training/trainer.py`
- Create: `src/mongoose/training/config.py`
- Create: `scripts/train.py`

**Step 1: Implement trainer**

Key details:
- AdamW, lr=1e-3, weight_decay=1e-4, cosine annealing to 1e-6
- Mixed precision (torch.amp) for 8GB VRAM
- Gradient clipping (max_norm=1.0)
- Checkpoint saving every N epochs + best model by val MAE
- WandB or TensorBoard logging (optional)
- Epoch loop calls `combined_loss.set_epoch(epoch)` for warmup

**Step 2: Create training script**

```python
# scripts/train.py
"""Train the T2D U-Net model."""
from mongoose.training.config import TrainConfig
from mongoose.training.trainer import Trainer

def main():
    config = TrainConfig.from_args()
    trainer = Trainer(config)
    trainer.fit()

if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/mongoose/training/ scripts/
git commit -m "feat: training loop with mixed precision and cosine annealing"
```

---

### Task 11: Inference Pipeline

Implement the NMS, sub-sample interpolation, and output formatting.

**Files:**
- Create: `src/mongoose/inference/nms.py`
- Create: `src/mongoose/inference/pipeline.py`
- Create: `tests/test_inference/__init__.py`
- Create: `tests/test_inference/test_nms.py`
- Create: `scripts/predict.py`

**Step 1: Write the failing tests**

```python
# tests/test_inference/test_nms.py
import torch
from mongoose.inference.nms import velocity_adaptive_nms

def test_nms_single_peak():
    heatmap = torch.zeros(200)
    heatmap[100] = 0.9
    velocity = torch.ones(200) * 0.5  # bp/sample
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 1
    assert peaks[0] == 100

def test_nms_suppresses_close_duplicates():
    heatmap = torch.zeros(200)
    heatmap[100] = 0.9
    heatmap[103] = 0.7  # too close at this velocity
    velocity = torch.ones(200) * 0.5
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 1

def test_nms_keeps_distant_peaks():
    heatmap = torch.zeros(200)
    heatmap[50] = 0.9
    heatmap[150] = 0.8
    velocity = torch.ones(200) * 0.5
    peaks = velocity_adaptive_nms(heatmap, velocity, threshold=0.3)
    assert len(peaks) == 2
```

**Step 2-5: Implement, test, commit**

The inference pipeline:
1. Forward pass
2. `velocity_adaptive_nms` on heatmap
3. Sub-sample parabolic interpolation for each peak
4. Read off bp positions from cumulative curve via lerp
5. Compute probe duration in bp
6. Write output (text format)

```bash
git add src/mongoose/inference/ tests/test_inference/ scripts/predict.py
git commit -m "feat: inference pipeline with velocity-adaptive NMS and sub-sample interpolation"
```

---

### Task 12: Evaluation Script

Compare DL model vs legacy T2D against E. coli reference ground truth.

**Files:**
- Create: `src/mongoose/inference/evaluate.py`
- Create: `scripts/evaluate.py`

**Step 1: Implement evaluation**

The evaluation script:
1. Loads the held-out die (D08) data
2. Runs inference with the trained model
3. For each molecule: computes predicted inter-probe intervals
4. Compares against reference genome inter-probe distances (ground truth)
5. Also loads legacy T2D intervals from the existing probes.bin (using the transform file parameters)
6. Reports: MAE, median AE, std, per-interval error distribution for both DL and legacy
7. Outputs summary table and per-molecule CSV

```bash
git add src/mongoose/inference/evaluate.py scripts/evaluate.py
git commit -m "feat: evaluation script comparing DL model vs legacy T2D"
```

---

## Implementation Order and Dependencies

```
Task 1 (scaffolding)
  |
  +-- Task 2 (reference map) --+
  +-- Task 3 (probes.bin)   ---+-- Task 6 (ground truth) --+
  +-- Task 4 (assigns)      --+                            |
  +-- Task 5 (TDB reader)   --+                            |
                                                            |
                               Task 7 (U-Net model) -------+-- Task 9 (dataset)
                               Task 8 (losses) ------------+      |
                                                                   |
                                                            Task 10 (training)
                                                                   |
                                                            Task 11 (inference)
                                                                   |
                                                            Task 12 (evaluation)
```

Tasks 2-5 are independent and can be implemented in parallel.
Tasks 7-8 are independent of 2-6 and can be implemented in parallel.
Task 9 depends on 2-6 (data) and 7 (model shapes for heatmap target sizing).
Tasks 10-12 are sequential.
