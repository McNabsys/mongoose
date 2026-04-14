"""Tests for the preprocessing sanity-check logic."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

# Import the check functions from the script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from sanity_check_preprocess import (
    check_cache_structure,
    check_manifest_counts,
    check_data_integrity,
    check_dataset_loads,
)


def _build_minimal_cache(cache_dir: Path, num_molecules: int = 3):
    """Build a minimal valid cache directory for testing."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Molecule sample lengths
    lengths = [100 + i * 50 for i in range(num_molecules)]

    # Waveforms.bin
    with open(cache_dir / "waveforms.bin", "wb") as f:
        offsets = []
        current = 0
        for length in lengths:
            wfm = np.ones(length, dtype=np.int16) * 500
            f.write(wfm.tobytes())
            offsets.append([current, length])
            current += length * 2  # int16 = 2 bytes

    # Offsets
    np.save(cache_dir / "offsets.npy", np.array(offsets, dtype=np.int64))

    # Conditioning
    conditioning = np.random.default_rng(42).normal(size=(num_molecules, 6)).astype(np.float32)
    np.save(cache_dir / "conditioning.npy", conditioning)

    # Ground truth (must have probe_sample_indices in ascending order)
    gt_list = []
    for i, length in enumerate(lengths):
        num_probes = 4
        indices = np.linspace(10, length - 10, num_probes, dtype=np.int64)
        ref_bp = np.array([1000 + j * 1000 for j in range(num_probes)], dtype=np.int64)
        gt_list.append({
            "probe_sample_indices": indices,
            "inter_probe_deltas_bp": np.abs(np.diff(ref_bp)).astype(np.float64),
            "velocity_targets_bp_per_ms": np.array([400.0, 380.0, 360.0, 340.0]),
            "reference_probe_bp": ref_bp,
            "direction": 1 if i % 2 == 0 else -1,
        })

    with open(cache_dir / "molecules.pkl", "wb") as f:
        pickle.dump(gt_list, f)

    # Manifest
    manifest = {
        "run_id": "test",
        "stats": {
            "run_id": "test",
            "total_molecules": 100000,
            "clean_molecules": 30000,
            "remapped_molecules": 20000,
            "cached_molecules": num_molecules,
            "total_waveform_bytes": current,
        },
        "molecules": [
            {
                "uid": i,
                "channel": 2,
                "num_samples": length,
                "num_probes": 4,
                "num_matched_probes": 4,
                "transloc_time_ms": length * 0.025,
                "mean_lvl1": 0.5 + i * 0.1,
                "direction": 1 if i % 2 == 0 else -1,
                "amplitude_scale": 1.0,
            }
            for i, length in enumerate(lengths)
        ],
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f)


def test_check_cache_structure_passes(tmp_path):
    cache_dir = tmp_path / "run1"
    _build_minimal_cache(cache_dir)
    results = check_cache_structure(cache_dir)
    assert all(r.passed for r in results)


def test_check_cache_structure_detects_missing_file(tmp_path):
    cache_dir = tmp_path / "run1"
    _build_minimal_cache(cache_dir)
    (cache_dir / "waveforms.bin").unlink()
    results = check_cache_structure(cache_dir)
    assert not all(r.passed for r in results)


def test_check_manifest_counts_respects_bounds(tmp_path):
    cache_dir = tmp_path / "run1"
    _build_minimal_cache(cache_dir, num_molecules=3)
    results = check_manifest_counts(cache_dir, min_cached=2, max_cached=10)
    # Some checks require larger counts (total_molecules > 1000) -- should still pass because we set 100000
    # The cached_molecules check: 2 <= 3 <= 10 -> pass
    cached_checks = [r for r in results if "Cached" in r.name]
    assert all(r.passed for r in cached_checks)


def test_check_data_integrity_passes(tmp_path):
    cache_dir = tmp_path / "run1"
    _build_minimal_cache(cache_dir)
    results = check_data_integrity(cache_dir)
    assert all(r.passed for r in results), f"Failed: {[r.name for r in results if not r.passed]}"


def test_check_dataset_loads_passes(tmp_path):
    cache_dir = tmp_path / "run1"
    _build_minimal_cache(cache_dir)
    results = check_dataset_loads(cache_dir)
    assert all(r.passed for r in results), f"Failed: {[r.name for r in results if not r.passed]}"
