"""Preprocess TDB files into compact training cache.

Converts raw TDB waveform data (~900 GB) into a compact format (~5 GB)
containing only clean, remapped molecules with ground truth annotations.

V1 rearchitecture (R4):

- Ground truth now primary-keys on ``reference_bp_positions`` (see
  ``mongoose.data.ground_truth``). The deprecated ``probe_sample_indices``,
  ``inter_probe_deltas_bp`` and ``velocity_targets_bp_per_ms`` fields are
  no longer produced here -- those are derived at training time.
- Mean level-1 backbone is now computed directly from the raw TDB waveform
  via ``mongoose.data.level1.estimate_level1`` using the TDB rise/fall
  edge indices, and stored as ``mean_lvl1_from_tdb``. This replaces the
  wfmproc-provided ``mol.mean_lvl1`` in the conditioning vector (position 0)
  and in the manifest.
- ``molecules.pkl`` now contains dicts with keys:
    reference_bp_positions, n_ref_probes, direction,
    warmstart_probe_centers_samples, warmstart_probe_durations_samples.

Cache format has CHANGED with this revision; any caches produced by the
previous V1 preprocessing are INCOMPATIBLE and must be regenerated.
"""

from __future__ import annotations

import json
import logging
import pickle
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mongoose.data.ground_truth import build_molecule_gt
from mongoose.data.level1 import estimate_level1
from mongoose.io.assigns import load_assigns
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.reference_map import load_reference_map
from mongoose.io.tdb import load_tdb_header, load_tdb_index, load_tdb_molecule_at_offset

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
    tdb_paths: list[Path],
    tdb_index_paths: list[Path],
    probes_bin_path: Path,
    assigns_path: Path,
    reference_map_path: Path,
    output_dir: Path,
    min_probes: int = 8,
    min_transloc_ms: float = 30.0,
) -> PreprocessStats:
    """Preprocess one run into a compact training cache.

    Supports runs composed of one or more TDB files. Each probes.bin molecule
    carries a `file_name_index` selecting which TDB it came from, plus
    `(channel, molecule_id)` that identifies its block in that TDB's index
    (molecule_id is the TDB's per-channel sequential MID).

    Args:
        run_id: Identifier for this run.
        tdb_paths: TDB files indexed by `file_name_index`
            (as resolved from probes.bin.files).
        tdb_index_paths: Parallel list of .tdb_index sidecar paths.
        probes_bin_path: Path to the _probes.bin file.
        assigns_path: Path to the _probeassignment.assigns file.
        reference_map_path: Path to the _referenceMap.txt file.
        output_dir: Where to write the cache directory.
        min_probes: Minimum number of probes per molecule.
        min_transloc_ms: Minimum translocation time in milliseconds.

    Returns:
        PreprocessStats with counts from this run.
    """
    if len(tdb_paths) != len(tdb_index_paths):
        raise ValueError(
            f"tdb_paths ({len(tdb_paths)}) and tdb_index_paths "
            f"({len(tdb_index_paths)}) must be parallel"
        )

    output_dir = output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load per-TDB metadata once.
    tdb_headers = [load_tdb_header(p) for p in tdb_paths]
    tdb_indexes = [load_tdb_index(p) for p in tdb_index_paths]

    # Merge channel -> amplitude_scale across all TDBs. In practice all TDBs
    # in one run share the same amplitude_scale per channel; if they disagree
    # we accept the first TDB's value and log.
    channel_to_scale: dict[int, float] = {}
    for hdr in tdb_headers:
        for ch_id, scale in zip(hdr.channel_ids, hdr.amplitude_scale_factors):
            if ch_id in channel_to_scale and channel_to_scale[ch_id] != scale:
                logger.warning(
                    "Channel %d amplitude_scale disagreement across TDBs: "
                    "%g vs %g; keeping first",
                    ch_id, channel_to_scale[ch_id], scale,
                )
                continue
            channel_to_scale[ch_id] = scale

    probes_file = load_probes_bin(probes_bin_path)
    assigns = load_assigns(assigns_path)
    ref = load_reference_map(reference_map_path)

    manifest_entries: list[dict] = []
    offsets: list[tuple[int, int]] = []
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

    skipped_identity = 0
    waveform_file = open(output_dir / "waveforms.bin", "wb")
    try:
        for mol in probes_file.molecules:
            if mol.structured or mol.folded_start or mol.folded_end or mol.do_not_use:
                continue
            if mol.num_probes < min_probes or mol.transloc_time_ms < min_transloc_ms:
                continue
            stats.clean_molecules += 1

            if mol.uid >= len(assigns):
                continue
            assign = assigns[mol.uid]
            if assign.ref_index < 0:
                continue
            stats.remapped_molecules += 1

            gt = build_molecule_gt(
                mol,
                assign,
                ref,
                min_matched_probes=min_probes // 2,
                include_warmstart=True,
            )
            if gt is None:
                continue

            # Route to the correct TDB via file_name_index + (channel, MID).
            fni = int(mol.file_name_index)
            if fni < 0 or fni >= len(tdb_paths):
                logger.warning(
                    "Molecule uid=%d has file_name_index=%d, out of range for "
                    "%d TDBs - skipping",
                    mol.uid, fni, len(tdb_paths),
                )
                skipped_identity += 1
                continue

            key = (int(mol.channel), int(mol.molecule_id))
            offset = tdb_indexes[fni].get(key)
            if offset is None:
                logger.warning(
                    "Molecule uid=%d (ch=%d, mid=%d) not found in TDB %s index "
                    "- skipping",
                    mol.uid, key[0], key[1], tdb_paths[fni].name,
                )
                skipped_identity += 1
                continue

            try:
                tdb_mol = load_tdb_molecule_at_offset(tdb_paths[fni], offset)
            except (OSError, struct.error, ValueError):
                logger.warning(
                    "Failed to read TDB molecule at %s offset %d - skipping",
                    tdb_paths[fni].name, offset,
                    exc_info=True,
                )
                skipped_identity += 1
                continue

            waveform = tdb_mol.waveform
            num_samples = len(waveform)

            mean_lvl1_from_tdb = estimate_level1(
                waveform,
                rise_end_idx=int(tdb_mol.rise_conv_end_index),
                fall_min_idx=int(tdb_mol.fall_conv_min_index),
            )

            waveform_bytes = waveform.tobytes()
            waveform_file.write(waveform_bytes)
            offsets.append((current_offset, num_samples))
            current_offset += len(waveform_bytes)

            duration_samples = num_samples
            inter_event_ms = 0.0
            time_in_run_frac = (
                mol.start_ms / (probes_file.last_sample_time * 1000)
                if probes_file.last_sample_time > 0
                else 0.0
            )

            cond = np.array(
                [
                    mean_lvl1_from_tdb,
                    np.log(max(1, duration_samples)),
                    np.log(inter_event_ms + 1),
                    time_in_run_frac,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            )
            conditioning_rows.append(cond)

            gt_dict = {
                "reference_bp_positions": gt.reference_bp_positions,
                "n_ref_probes": gt.n_ref_probes,
                "direction": gt.direction,
                "warmstart_probe_centers_samples": gt.warmstart_probe_centers_samples,
                "warmstart_probe_durations_samples": gt.warmstart_probe_durations_samples,
            }
            gt_list.append(gt_dict)

            num_matched_probes = (
                len(gt.warmstart_probe_centers_samples)
                if gt.warmstart_probe_centers_samples is not None
                else 0
            )
            manifest_entries.append(
                {
                    "uid": int(mol.uid),
                    "channel": int(mol.channel),
                    "molecule_id": int(mol.molecule_id),
                    "file_name_index": fni,
                    "num_samples": num_samples,
                    "num_probes": int(mol.num_probes),
                    "n_ref_probes": int(gt.n_ref_probes),
                    "num_matched_probes": int(num_matched_probes),
                    "transloc_time_ms": float(mol.transloc_time_ms),
                    "mean_lvl1_from_tdb": float(mean_lvl1_from_tdb),
                    "direction": gt.direction,
                    "amplitude_scale": float(channel_to_scale.get(mol.channel, 1.0)),
                }
            )

            stats.cached_molecules += 1
    finally:
        waveform_file.close()

    stats.total_waveform_bytes = current_offset

    if skipped_identity > 0:
        logger.warning(
            "Preprocessing %s: skipped %d molecules due to TDB identity lookup failures",
            run_id, skipped_identity,
        )

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
        "tdb_files": [p.name for p in tdb_paths],
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
