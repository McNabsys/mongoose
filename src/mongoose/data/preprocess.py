"""Preprocess TDB files into compact training cache.

Converts raw TDB waveform data (~900 GB) into a compact format (~5 GB)
containing only clean, remapped molecules with ground truth annotations.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mongoose.data.ground_truth import build_molecule_gt
from mongoose.io.assigns import load_assigns
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.reference_map import load_reference_map
from mongoose.io.tdb import load_tdb_header, load_tdb_molecule

logger = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    """Statistics from preprocessing a single run."""

    run_id: str
    total_molecules: int
    clean_molecules: int
    remapped_molecules: int
    cached_molecules: int
    total_waveform_bytes: int


def preprocess_run(
    run_id: str,
    tdb_path: Path,
    probes_bin_path: Path,
    assigns_path: Path,
    reference_map_path: Path,
    output_dir: Path,
    min_probes: int = 8,
    min_transloc_ms: float = 30.0,
) -> PreprocessStats:
    """Preprocess one run into compact training cache.

    Reads a run's TDB file, probes.bin, assigns, and reference map.
    Filters to clean, remapped molecules with enough probes.
    Writes a compact cache directory:
        <output_dir>/<run_id>/
            manifest.json
            waveforms.bin
            offsets.npy
            conditioning.npy
            molecules.pkl

    Args:
        run_id: Identifier for this run.
        tdb_path: Path to the .tdb file.
        probes_bin_path: Path to the _probes.bin file.
        assigns_path: Path to the _probeassignment.assigns file.
        reference_map_path: Path to the _referenceMap.txt file.
        output_dir: Where to write the cache directory.
        min_probes: Minimum number of probes required per molecule.
        min_transloc_ms: Minimum translocation time in milliseconds.

    Returns:
        PreprocessStats with counts from this run.
    """
    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all metadata
    tdb_header = load_tdb_header(tdb_path)
    probes_file = load_probes_bin(probes_bin_path)
    assigns = load_assigns(assigns_path)
    ref = load_reference_map(reference_map_path)

    # Build channel -> amplitude scale factor mapping
    channel_to_scale: dict[int, float] = {}
    for ch_id, scale in zip(tdb_header.channel_ids, tdb_header.amplitude_scale_factors):
        channel_to_scale[ch_id] = scale

    manifest_entries: list[dict] = []
    offsets: list[tuple[int, int]] = []  # (byte_offset, num_samples)
    conditioning_rows: list[np.ndarray] = []
    gt_list: list[dict] = []

    current_offset = 0

    stats = PreprocessStats(
        run_id=run_id,
        total_molecules=probes_file.num_molecules,
        clean_molecules=0,
        remapped_molecules=0,
        cached_molecules=0,
        total_waveform_bytes=0,
    )

    waveform_file = open(output_dir / "waveforms.bin", "wb")

    try:
        for mol_idx, mol in enumerate(probes_file.molecules):
            # Quality filter
            if mol.structured or mol.folded_start or mol.folded_end or mol.do_not_use:
                continue
            if mol.num_probes < min_probes or mol.transloc_time_ms < min_transloc_ms:
                continue
            stats.clean_molecules += 1

            # Check if mapped
            if mol.uid >= len(assigns):
                continue
            assign = assigns[mol.uid]
            if assign.ref_index < 0:
                continue
            stats.remapped_molecules += 1

            # Build ground truth
            gt = build_molecule_gt(mol, assign, ref, min_matched_probes=min_probes // 2)
            if gt is None:
                continue

            # Load waveform from TDB
            try:
                tdb_mol = load_tdb_molecule(tdb_path, tdb_header, mol_idx)
            except Exception:
                logger.warning("Failed to load TDB molecule %d, skipping", mol_idx)
                continue

            waveform = tdb_mol.waveform  # int16
            num_samples = len(waveform)

            # Write waveform
            waveform_bytes = waveform.tobytes()
            waveform_file.write(waveform_bytes)
            offsets.append((current_offset, num_samples))
            current_offset += len(waveform_bytes)

            # Build conditioning vector [6]:
            # 0: Absolute pre-event baseline (mean_lvl1 in mV)
            # 1: log(molecule duration in samples)
            # 2: log(inter-event interval + 1) -- placeholder
            # 3: Time-in-run (fraction)
            # 4: Applied bias voltage (placeholder)
            # 5: Applied pressure (placeholder)
            duration_samples = num_samples
            inter_event_ms = 0.0  # placeholder
            time_in_run_frac = (
                mol.start_ms / (probes_file.last_sample_time * 1000)
                if probes_file.last_sample_time > 0
                else 0.0
            )

            cond = np.array(
                [
                    mol.mean_lvl1,
                    np.log(max(1, duration_samples)),
                    np.log(inter_event_ms + 1),
                    time_in_run_frac,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            conditioning_rows.append(cond)

            # Store GT as a plain dict with numpy arrays
            gt_dict = {
                "probe_sample_indices": gt.probe_sample_indices,
                "inter_probe_deltas_bp": gt.inter_probe_deltas_bp,
                "velocity_targets_bp_per_ms": gt.velocity_targets_bp_per_ms,
                "reference_probe_bp": gt.reference_probe_bp,
                "direction": gt.direction,
            }
            gt_list.append(gt_dict)

            # Manifest entry
            manifest_entries.append(
                {
                    "uid": int(mol.uid),
                    "channel": int(mol.channel),
                    "num_samples": num_samples,
                    "num_probes": int(mol.num_probes),
                    "num_matched_probes": len(gt.probe_sample_indices),
                    "transloc_time_ms": float(mol.transloc_time_ms),
                    "mean_lvl1": float(mol.mean_lvl1),
                    "direction": gt.direction,
                    "amplitude_scale": float(channel_to_scale.get(mol.channel, 1.0)),
                }
            )

            stats.cached_molecules += 1
    finally:
        waveform_file.close()

    stats.total_waveform_bytes = current_offset

    # Write remaining files
    np.save(output_dir / "offsets.npy", np.array(offsets, dtype=np.int64))
    np.save(
        output_dir / "conditioning.npy",
        np.stack(conditioning_rows) if conditioning_rows else np.empty((0, 6), dtype=np.float32),
    )

    with open(output_dir / "molecules.pkl", "wb") as f:
        pickle.dump(gt_list, f)

    manifest = {
        "run_id": run_id,
        "stats": asdict(stats),
        "molecules": manifest_entries,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Preprocessed %s: %d/%d molecules cached (%.2f GB waveform data)",
        run_id,
        stats.cached_molecules,
        stats.total_molecules,
        stats.total_waveform_bytes / 1e9,
    )

    return stats
