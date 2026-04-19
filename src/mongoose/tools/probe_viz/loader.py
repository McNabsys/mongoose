"""Loader for the probe-viz tool: discovery, iteration state, molecule fetch."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mongoose.io.probes_bin import Molecule as ProbesMolecule
from mongoose.io.probes_bin import ProbesBinFile, load_probes_bin
from mongoose.io.probes_bin_files import parse_probes_bin_files
from mongoose.io.tdb import (
    TdbHeader,
    TdbMolecule,
    load_tdb_header,
    load_tdb_index,
    load_tdb_molecule_at_offset,
)


@dataclass
class ViewData:
    """Everything the viewer needs to render one molecule."""

    probe_molecule: ProbesMolecule
    tdb_molecule: TdbMolecule
    sample_rate: int
    scale_uv_per_lsb: float
    tdb_basename: str
    iter_index: int
    iter_total: int


def discover_probes_bin(sample_dir: Path) -> tuple[Path, Path]:
    """Locate the probes.bin and its .files sidecar in a sample-run directory.

    Searches (in order):
        1. <sample_dir>/Remapped/AllCh/*_probes.bin
        2. <sample_dir>/Remapping/AllCh/*_probes.bin

    Args:
        sample_dir: Path to a sample-run directory (e.g. a 'YYYY-MM-DD' folder).

    Returns:
        (probes_bin_path, probes_bin_files_path)

    Raises:
        FileNotFoundError: If no probes.bin is found under the expected paths
            or if the .files sidecar is missing next to the .bin.
    """
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")

    candidates: list[Path] = []
    for subdir in ("Remapped/AllCh", "Remapping/AllCh"):
        d = sample_dir / subdir
        if d.is_dir():
            candidates.extend(sorted(d.glob("*_probes.bin")))
        if candidates:
            break

    if not candidates:
        raise FileNotFoundError(
            f"No _probes.bin found under {sample_dir}/Remapped/AllCh "
            f"or {sample_dir}/Remapping/AllCh"
        )

    probes_bin = candidates[0]
    probes_bin_files = probes_bin.with_suffix(".bin.files")
    if not probes_bin_files.exists():
        raise FileNotFoundError(
            f"Found {probes_bin.name} but no .files sidecar at {probes_bin_files}"
        )
    return probes_bin, probes_bin_files


class ProbeVizLoader:
    """Iteration-state manager over molecules in a sample run's probes.bin.

    Molecules flagged ``do_not_use`` are excluded by default. Call
    ``toggle_do_not_use()`` at runtime to include them.
    """

    def __init__(self, sample_dir: Path, include_do_not_use: bool = False) -> None:
        self._sample_dir = Path(sample_dir)
        self._include_do_not_use = include_do_not_use
        probes_bin_path, probes_bin_files_path = discover_probes_bin(self._sample_dir)
        self._probes: ProbesBinFile = load_probes_bin(probes_bin_path)
        self._tdb_basenames: list[str] = parse_probes_bin_files(probes_bin_files_path)
        self._iter_list: list[int] = []
        self._current_position: int = 0
        self._rebuild_iter_list()
        self._tdb_cache: dict[str, tuple[TdbHeader, dict[tuple[int, int], int]]] = {}

    def _rebuild_iter_list(self) -> None:
        if self._include_do_not_use:
            self._iter_list = list(range(len(self._probes.molecules)))
        else:
            self._iter_list = [
                i for i, m in enumerate(self._probes.molecules) if not m.do_not_use
            ]
        if not self._iter_list:
            raise ValueError(
                f"No molecules to display in {self._sample_dir} "
                f"(include_do_not_use={self._include_do_not_use})"
            )
        self._current_position = 0

    @property
    def total(self) -> int:
        return len(self._iter_list)

    @property
    def current_index(self) -> int:
        return self._current_position

    def advance(self, delta: int) -> None:
        new = self._current_position + delta
        if new < 0:
            new = 0
        elif new >= self.total:
            new = self.total - 1
        self._current_position = new

    def _get_tdb(self, basename: str) -> tuple[TdbHeader, dict[tuple[int, int], int]]:
        cached = self._tdb_cache.get(basename)
        if cached is not None:
            return cached
        tdb_path = self._sample_dir / basename
        if not tdb_path.exists():
            raise FileNotFoundError(
                f"probes.bin references {basename!r} but it is not present at {tdb_path}"
            )
        index_path = tdb_path.with_suffix(".tdb_index")
        if not index_path.exists():
            raise FileNotFoundError(
                f"TDB index missing for {basename}: {index_path}"
            )
        header = load_tdb_header(tdb_path)
        index = load_tdb_index(index_path)
        self._tdb_cache[basename] = (header, index)
        return header, index

    def current_view(self) -> ViewData:
        mol_idx = self._iter_list[self._current_position]
        probe_mol = self._probes.molecules[mol_idx]
        basename = self._tdb_basenames[probe_mol.file_name_index]
        header, index = self._get_tdb(basename)

        offset = index[(probe_mol.channel, probe_mol.molecule_id)]
        tdb_mol = load_tdb_molecule_at_offset(self._sample_dir / basename, offset)

        try:
            ch_pos = header.channel_ids.index(probe_mol.channel)
        except ValueError:
            ch_pos = probe_mol.channel - 1
        scale = header.amplitude_scale_factors[ch_pos]

        return ViewData(
            probe_molecule=probe_mol,
            tdb_molecule=tdb_mol,
            sample_rate=header.sample_rate,
            scale_uv_per_lsb=scale,
            tdb_basename=basename,
            iter_index=self._current_position,
            iter_total=self.total,
        )
