"""Resolve filesystem inputs for a single E. coli run.

Layout on disk (Nabsys-standard):

    <root>/<run_id>/<YYYY-MM-DD>/
        <run_id>-*.tdb              (one or more, basenames listed in .files)
        <run_id>-*.tdb_index        (sidecar per TDB)
        Remapped/AllCh/
            <run_id>_probes.bin
            <run_id>_probes.bin.files
            <run_id>_probes.txt_probeassignment.assigns
            <run_id>_probes.txt_referenceMap.txt

resolve_run_inputs() is the one-stop helper scripts call to get all the
paths preprocess_run() needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mongoose.io.probes_bin_files import parse_probes_bin_files


@dataclass(frozen=True)
class RunInputs:
    """Resolved input paths for a single run."""

    run_id: str
    probes_bin_path: Path
    assigns_path: Path
    reference_map_path: Path
    tdb_paths: list[Path]        # ordered by file_name_index
    tdb_index_paths: list[Path]  # parallel to tdb_paths


def resolve_run_inputs(date_dir: Path, run_id: str) -> RunInputs:
    """Resolve all filesystem inputs for a single run.

    Args:
        date_dir: The `<YYYY-MM-DD>` directory containing TDBs + Remapped/AllCh/.
        run_id: Run identifier used as the prefix for probes.bin et al.

    Returns:
        Resolved paths (absolute, verified to exist).

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    date_dir = Path(date_dir)
    # The remap output dir has two observed spellings across runs —
    # "Remapped" on some batches, "Remapping" on others. Contents are
    # identical. Prefer "Remapped" if both exist; fall back to "Remapping".
    for candidate in ("Remapped", "Remapping"):
        candidate_dir = date_dir / candidate / "AllCh"
        if candidate_dir.is_dir():
            remapped_dir = candidate_dir
            break
    else:
        remapped_dir = date_dir / "Remapped" / "AllCh"  # original path for error msg

    probes_bin = remapped_dir / f"{run_id}_probes.bin"
    assigns = remapped_dir / f"{run_id}_probes.txt_probeassignment.assigns"
    reference_map = remapped_dir / f"{run_id}_probes.txt_referenceMap.txt"
    files_sidecar = remapped_dir / f"{run_id}_probes.bin.files"

    for p, label in [
        (probes_bin, "probes.bin"),
        (assigns, "probeassignment.assigns"),
        (reference_map, "referenceMap.txt"),
        (files_sidecar, "probes.bin.files"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} for {run_id}: {p}")

    tdb_names = parse_probes_bin_files(files_sidecar)
    tdb_paths: list[Path] = []
    tdb_index_paths: list[Path] = []
    for name in tdb_names:
        tdb = date_dir / name
        idx = date_dir / (name + "_index")
        if not tdb.exists():
            raise FileNotFoundError(
                f"TDB referenced by probes.bin.files not found: {tdb}"
            )
        if not idx.exists():
            raise FileNotFoundError(
                f"TDB index sidecar not found: {idx} "
                f"(TDB itself exists: {tdb.exists()})"
            )
        tdb_paths.append(tdb)
        tdb_index_paths.append(idx)

    return RunInputs(
        run_id=run_id,
        probes_bin_path=probes_bin,
        assigns_path=assigns,
        reference_map_path=reference_map,
        tdb_paths=tdb_paths,
        tdb_index_paths=tdb_index_paths,
    )
