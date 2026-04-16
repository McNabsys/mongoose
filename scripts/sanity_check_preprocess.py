"""Sanity-check preprocessing on a single run before batch processing.

Validates:
- Cache files exist with expected formats
- Molecule counts are in expected ranges
- Waveform data integrity (dtype, length matches manifest)
- Ground truth arrays have consistent shapes
- CachedMoleculeDataset loads and returns well-formed items
- Conditioning vectors are finite and reasonable

Exit codes:
  0 = all checks passed
  1 = cache creation failed
  2 = cache structure failed
  3 = molecule count out of expected range
  4 = data integrity failed
  5 = dataset loading failed

Run with:
    python scripts/sanity_check_preprocess.py \
      --tdb /path/to/run.tdb \
      --probes-bin /path/to/run_probes.bin \
      --assigns /path/to/run_probeassignment.assigns \
      --reference-map /path/to/run_referenceMap.txt \
      --run-id STB03-064B-02L58270w05-202G16g

To re-validate an existing cache without reprocessing:
    python scripts/sanity_check_preprocess.py --skip-preprocess ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from mongoose.data.cached_dataset import CachedMoleculeDataset
from mongoose.data.preprocess import preprocess_run


class CheckResult:
    """A single check's result with name, pass/fail status, and optional detail."""

    def __init__(self, name: str, passed: bool, detail: str = ""):
        self.name = name
        self.passed = passed
        self.detail = detail

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name}" + (f": {self.detail}" if self.detail else "")


def check_cache_structure(cache_dir: Path) -> list[CheckResult]:
    """Verify all expected cache files exist and have correct formats."""
    results = []
    expected_files = [
        "manifest.json",
        "waveforms.bin",
        "offsets.npy",
        "conditioning.npy",
        "molecules.pkl",
    ]
    for fname in expected_files:
        p = cache_dir / fname
        results.append(
            CheckResult(
                f"File exists: {fname}",
                p.exists() and p.stat().st_size > 0,
                f"size={p.stat().st_size if p.exists() else 0} bytes",
            )
        )
    return results


def check_manifest_counts(
    cache_dir: Path, min_cached: int = 5000, max_cached: int = 30000
) -> list[CheckResult]:
    """Verify molecule counts are in expected range."""
    results = []
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)
    stats = manifest["stats"]

    results.append(
        CheckResult(
            "Total molecules in probes.bin",
            stats["total_molecules"] > 1000,
            f"total={stats['total_molecules']}",
        )
    )
    results.append(
        CheckResult(
            "Clean molecules within expected range",
            20000 <= stats["clean_molecules"] <= 50000,
            f"clean={stats['clean_molecules']}",
        )
    )
    results.append(
        CheckResult(
            f"Cached molecules within expected range ({min_cached}-{max_cached})",
            min_cached <= stats["cached_molecules"] <= max_cached,
            f"cached={stats['cached_molecules']}",
        )
    )
    results.append(
        CheckResult(
            "Manifest molecule list matches cached count",
            len(manifest["molecules"]) == stats["cached_molecules"],
            f"manifest_entries={len(manifest['molecules'])}",
        )
    )
    return results


def check_data_integrity(cache_dir: Path) -> list[CheckResult]:
    """Verify data arrays have consistent shapes and dtypes."""
    results = []
    offsets = np.load(cache_dir / "offsets.npy")
    conditioning = np.load(cache_dir / "conditioning.npy")
    with open(cache_dir / "molecules.pkl", "rb") as f:
        gt_list = pickle.load(f)
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    n = len(manifest["molecules"])

    results.append(
        CheckResult(
            "Offsets shape matches molecule count",
            offsets.shape == (n, 2),
            f"offsets.shape={offsets.shape}, n={n}",
        )
    )
    results.append(
        CheckResult(
            "Offsets dtype is int64",
            offsets.dtype == np.int64,
            f"dtype={offsets.dtype}",
        )
    )
    results.append(
        CheckResult(
            "Conditioning shape matches molecule count",
            conditioning.shape == (n, 6),
            f"conditioning.shape={conditioning.shape}",
        )
    )
    results.append(
        CheckResult(
            "Conditioning is float32",
            conditioning.dtype == np.float32,
            f"dtype={conditioning.dtype}",
        )
    )
    results.append(
        CheckResult(
            "Conditioning values are finite",
            bool(np.all(np.isfinite(conditioning))),
            f"has_nan={bool(np.any(np.isnan(conditioning)))}, has_inf={bool(np.any(np.isinf(conditioning)))}",
        )
    )
    results.append(
        CheckResult(
            "GT list length matches molecule count",
            len(gt_list) == n,
            f"gt_list={len(gt_list)}, n={n}",
        )
    )

    # Spot-check first few GT entries (V1 rearchitecture schema).
    for i in range(min(3, n)):
        gt = gt_list[i]
        ref_bp = np.asarray(gt["reference_bp_positions"])
        direction = int(gt["direction"])
        n_ref = int(gt["n_ref_probes"])

        results.append(
            CheckResult(
                f"GT[{i}] n_ref_probes matches reference_bp_positions length",
                n_ref == len(ref_bp),
                f"n_ref={n_ref}, len={len(ref_bp)}",
            )
        )
        # For direction=1, temporal order == ascending bp; for direction=-1
        # it's descending. Either way, monotonicity is required.
        if direction == 1:
            monotonic = bool(len(ref_bp) <= 1 or np.all(np.diff(ref_bp) > 0))
        else:
            monotonic = bool(len(ref_bp) <= 1 or np.all(np.diff(ref_bp) < 0))
        results.append(
            CheckResult(
                f"GT[{i}] reference_bp_positions monotonic (dir={direction})",
                monotonic,
                f"n_ref={n_ref}",
            )
        )

        centers = gt.get("warmstart_probe_centers_samples")
        durations = gt.get("warmstart_probe_durations_samples")
        if centers is not None and durations is not None:
            results.append(
                CheckResult(
                    f"GT[{i}] warmstart arrays same length",
                    len(centers) == len(durations),
                    f"centers={len(centers)}, durations={len(durations)}",
                )
            )
            results.append(
                CheckResult(
                    f"GT[{i}] warmstart durations positive",
                    bool(np.all(np.asarray(durations) > 0)),
                    "",
                )
            )

    # Check waveform byte offsets don't overlap
    waveforms_bin_size = (cache_dir / "waveforms.bin").stat().st_size
    if n > 0:
        last_offset, last_samples = offsets[-1]
        expected_end = int(last_offset) + int(last_samples) * 2  # int16 = 2 bytes
        results.append(
            CheckResult(
                "Waveforms.bin size matches last offset + last length",
                waveforms_bin_size == expected_end,
                f"file_size={waveforms_bin_size}, expected={expected_end}",
            )
        )

    return results


def check_dataset_loads(cache_dir: Path) -> list[CheckResult]:
    """Verify CachedMoleculeDataset can load and return valid items."""
    results = []
    try:
        ds = CachedMoleculeDataset([cache_dir])
        results.append(CheckResult("Dataset initializes", True, f"len={len(ds)}"))
    except Exception as e:
        results.append(CheckResult("Dataset initializes", False, str(e)))
        return results

    if len(ds) == 0:
        return results

    # Load a few items and verify shapes/types
    for i in sorted(set([0, len(ds) // 2, len(ds) - 1])):
        try:
            item = ds[i]
            T = item["waveform"].shape[-1]
            results.append(
                CheckResult(
                    f"Item[{i}] waveform shape is (1, T)",
                    item["waveform"].shape == (1, T),
                    f"T={T}",
                )
            )
            results.append(
                CheckResult(
                    f"Item[{i}] mask shape is (T,)",
                    item["mask"].shape == (T,),
                    "",
                )
            )
            results.append(
                CheckResult(
                    f"Item[{i}] conditioning shape is (6,)",
                    item["conditioning"].shape == (6,),
                    "",
                )
            )
            results.append(
                CheckResult(
                    f"Item[{i}] waveform values are finite",
                    bool(torch.all(torch.isfinite(item["waveform"]))),
                    "",
                )
            )
            results.append(
                CheckResult(
                    f"Item[{i}] reference_bp_positions length matches n_ref_probes",
                    item["reference_bp_positions"].numel()
                    == int(item["n_ref_probes"].item()),
                    f"len={item['reference_bp_positions'].numel()}, n_ref={int(item['n_ref_probes'].item())}",
                )
            )
            if item["warmstart_heatmap"] is not None:
                results.append(
                    CheckResult(
                        f"Item[{i}] warmstart_heatmap shape is (T,)",
                        item["warmstart_heatmap"].shape == (T,),
                        "",
                    )
                )
                results.append(
                    CheckResult(
                        f"Item[{i}] warmstart_heatmap values in [0, 1]",
                        bool(
                            torch.all(item["warmstart_heatmap"] >= 0)
                            and torch.all(item["warmstart_heatmap"] <= 1.0 + 1e-6)
                        ),
                        f"max={item['warmstart_heatmap'].max().item():.3f}",
                    )
                )
        except Exception as e:
            results.append(CheckResult(f"Item[{i}] loads", False, str(e)))

    return results


def summary_statistics(cache_dir: Path) -> None:
    """Print summary statistics for visual inspection."""
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    mols = manifest["molecules"]
    if not mols:
        print("  No molecules cached.")
        return

    transloc_times = np.array([m["transloc_time_ms"] for m in mols])
    lvl1s = np.array([m["mean_lvl1_from_tdb"] for m in mols])
    num_samples = np.array([m["num_samples"] for m in mols])
    num_probes = np.array([m["num_probes"] for m in mols])
    num_matched = np.array([m["num_matched_probes"] for m in mols])
    directions = np.array([m["direction"] for m in mols])

    print()
    print("Summary statistics:")
    print(f"  Molecules cached: {len(mols)}")
    print(
        f"  Translocation time (ms):  min={transloc_times.min():.1f}  "
        f"median={np.median(transloc_times):.1f}  max={transloc_times.max():.1f}"
    )
    print(
        f"  Mean level-1 (mV):        min={lvl1s.min():.2f}  "
        f"median={np.median(lvl1s):.2f}  max={lvl1s.max():.2f}"
    )
    print(
        f"  Samples per molecule:     min={num_samples.min()}  "
        f"median={int(np.median(num_samples))}  max={num_samples.max()}"
    )
    print(
        f"  Total probes per mol:     min={num_probes.min()}  "
        f"median={int(np.median(num_probes))}  max={num_probes.max()}"
    )
    print(
        f"  Matched probes per mol:   min={num_matched.min()}  "
        f"median={int(np.median(num_matched))}  max={num_matched.max()}"
    )
    print(
        f"  Direction distribution:   forward={int(np.sum(directions == 1))}  "
        f"reverse={int(np.sum(directions == -1))}"
    )
    print(
        f"  Cache size: {(cache_dir / 'waveforms.bin').stat().st_size / 1e6:.1f} MB waveforms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity-check preprocessing on a single run"
    )
    parser.add_argument("--tdb", type=Path, required=False)
    parser.add_argument("--probes-bin", type=Path, required=False)
    parser.add_argument("--assigns", type=Path, required=False)
    parser.add_argument("--reference-map", type=Path, required=False)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output", type=Path, default=Path("cache"))
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing, only validate existing cache",
    )
    parser.add_argument(
        "--min-cached", type=int, default=5000, help="Minimum expected cached molecules"
    )
    parser.add_argument(
        "--max-cached",
        type=int,
        default=30000,
        help="Maximum expected cached molecules",
    )
    args = parser.parse_args()

    cache_dir = args.output / args.run_id

    # Step 1: Preprocess (unless skipped)
    if not args.skip_preprocess:
        required = {
            "--tdb": args.tdb,
            "--probes-bin": args.probes_bin,
            "--assigns": args.assigns,
            "--reference-map": args.reference_map,
        }
        missing = [name for name, val in required.items() if val is None]
        if missing:
            print(f"FAIL: Missing required arguments for preprocessing: {missing}")
            return 1

        print(f"Preprocessing run: {args.run_id}")
        try:
            stats = preprocess_run(
                run_id=args.run_id,
                tdb_path=args.tdb,
                probes_bin_path=args.probes_bin,
                assigns_path=args.assigns,
                reference_map_path=args.reference_map,
                output_dir=args.output,
            )
            print(
                f"  Preprocessing complete: {stats.cached_molecules} molecules cached"
            )
        except Exception as e:
            print(f"FAIL: Preprocessing raised exception: {e}")
            return 1
    else:
        print(f"Skipping preprocessing, validating existing cache: {cache_dir}")
        if not cache_dir.exists():
            print(f"FAIL: Cache directory does not exist: {cache_dir}")
            return 1

    # Step 2: Check cache structure
    print("\nCache structure:")
    structure_results = check_cache_structure(cache_dir)
    for r in structure_results:
        print(r)
    if not all(r.passed for r in structure_results):
        return 2

    # Step 3: Check molecule counts
    print("\nMolecule counts:")
    count_results = check_manifest_counts(cache_dir, args.min_cached, args.max_cached)
    for r in count_results:
        print(r)
    if not all(r.passed for r in count_results):
        return 3

    # Step 4: Check data integrity
    print("\nData integrity:")
    integrity_results = check_data_integrity(cache_dir)
    for r in integrity_results:
        print(r)
    if not all(r.passed for r in integrity_results):
        return 4

    # Step 5: Check dataset loading
    print("\nDataset loading:")
    loading_results = check_dataset_loads(cache_dir)
    for r in loading_results:
        print(r)
    if not all(r.passed for r in loading_results):
        return 5

    # Summary
    summary_statistics(cache_dir)

    all_results = (
        structure_results + count_results + integrity_results + loading_results
    )
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    print(f"\nAll checks: {passed}/{total} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
