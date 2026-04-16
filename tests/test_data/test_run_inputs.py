"""Tests for run-input path resolution."""

from pathlib import Path

import pytest


def _mk_run(tmp_path: Path, run_id: str, tdb_names: list[str]) -> Path:
    """Build a synthetic V:-style run layout. Returns the date directory."""
    date_dir = tmp_path / run_id / "2025-02-19"
    remapped = date_dir / "Remapped" / "AllCh"
    remapped.mkdir(parents=True)

    # Standard fixtures in Remapped/AllCh
    (remapped / f"{run_id}_probes.bin").write_bytes(b"stub")
    (remapped / f"{run_id}_probes.txt_probeassignment.assigns").write_bytes(b"stub")
    (remapped / f"{run_id}_probes.txt_referenceMap.txt").write_bytes(b"stub")

    # Create TDBs + index files in date_dir
    for name in tdb_names:
        (date_dir / name).write_bytes(b"stub")
        (date_dir / (name + "_index")).write_bytes(b"stub")

    # probes.bin.files referencing TDBs in same order
    lines = []
    for idx, name in enumerate(tdb_names):
        lines.append(f"{idx:06d}D:\\SharedData\\Samples\\{run_id}\\2025-02-19\\{name}")
    (remapped / f"{run_id}_probes.bin.files").write_bytes(
        ("\r\n".join(lines) + "\r\n").encode("latin-1")
    )

    return date_dir


def test_resolve_single_tdb(tmp_path):
    from mongoose.data.run_inputs import resolve_run_inputs

    run_id = "RUN-A"
    date_dir = _mk_run(tmp_path, run_id, ["RUN-A-only.tdb"])

    inputs = resolve_run_inputs(date_dir, run_id)

    assert inputs.run_id == run_id
    assert inputs.probes_bin_path.name == f"{run_id}_probes.bin"
    assert inputs.assigns_path.name == f"{run_id}_probes.txt_probeassignment.assigns"
    assert inputs.reference_map_path.name == f"{run_id}_probes.txt_referenceMap.txt"
    assert len(inputs.tdb_paths) == 1
    assert inputs.tdb_paths[0].name == "RUN-A-only.tdb"
    assert inputs.tdb_index_paths[0].name == "RUN-A-only.tdb_index"


def test_resolve_multi_tdb_preserves_index_order(tmp_path):
    from mongoose.data.run_inputs import resolve_run_inputs

    run_id = "RUN-B"
    date_dir = _mk_run(tmp_path, run_id, ["first.tdb", "second.tdb", "third.tdb"])

    inputs = resolve_run_inputs(date_dir, run_id)

    assert [p.name for p in inputs.tdb_paths] == ["first.tdb", "second.tdb", "third.tdb"]
    assert [p.name for p in inputs.tdb_index_paths] == [
        "first.tdb_index", "second.tdb_index", "third.tdb_index"
    ]


def test_resolve_raises_on_missing_tdb(tmp_path):
    """probes.bin.files references a TDB not actually on disk."""
    from mongoose.data.run_inputs import resolve_run_inputs

    run_id = "RUN-C"
    date_dir = _mk_run(tmp_path, run_id, ["only.tdb"])
    # Delete the TDB but leave the .files reference
    (date_dir / "only.tdb").unlink()
    (date_dir / "only.tdb_index").unlink()

    with pytest.raises(FileNotFoundError, match="only.tdb"):
        resolve_run_inputs(date_dir, run_id)


def test_resolve_raises_on_missing_probes_bin(tmp_path):
    from mongoose.data.run_inputs import resolve_run_inputs

    run_id = "RUN-D"
    date_dir = _mk_run(tmp_path, run_id, ["only.tdb"])
    (date_dir / "Remapped" / "AllCh" / f"{run_id}_probes.bin").unlink()

    with pytest.raises(FileNotFoundError, match="probes.bin"):
        resolve_run_inputs(date_dir, run_id)


def test_resolve_raises_on_missing_index_sidecar(tmp_path):
    from mongoose.data.run_inputs import resolve_run_inputs

    run_id = "RUN-E"
    date_dir = _mk_run(tmp_path, run_id, ["only.tdb"])
    (date_dir / "only.tdb_index").unlink()

    with pytest.raises(FileNotFoundError, match="tdb_index"):
        resolve_run_inputs(date_dir, run_id)
