"""Loader for the probe-viz tool: discovery, iteration state, molecule fetch."""

from __future__ import annotations

from pathlib import Path


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
