"""Tests for the V1 rearchitecture ground truth builder."""

import numpy as np

from mongoose.io.reference_map import load_reference_map
from mongoose.io.probes_bin import load_probes_bin, Molecule, Probe
from mongoose.io.assigns import MoleculeAssignment, load_assigns
from mongoose.io.reference_map import ReferenceMap
from mongoose.data.ground_truth import build_molecule_gt, MoleculeGT


# Default sample rate used by the Nabsys TDB fixture data (32 kHz, per the
# TDB header). Tests that exercise the real fixture data pass this value.
_FIXTURE_SAMPLE_RATE_HZ = 32_000


def test_warmstart_center_uses_start_within_tdb_and_sample_rate():
    """Verify probe.center_ms -> sample index uses both start_within_tdb_ms and
    the TDB-provided sample_rate. Regression test for a silent mis-alignment
    bug where the sample rate was hardcoded to 40 kHz and start_within_tdb_ms
    was ignored, producing labels that landed on baseline rather than peaks.

    Numbers pinned to a real molecule (uid=13577 from the E. coli dataset):
        start_within_tdb_ms = 13.409842491149902
        sample_rate         = 32_000 Hz (TDB header)
        probe.center_ms     = 7.4339075 -> expected sample 667
        probe.center_ms     = 38.4339   -> expected sample 1659
    """
    # Build a synthetic probe list covering first and last known probes for
    # this molecule. Only the fields used by build_molecule_gt matter.
    p0 = Probe(start_ms=0.0, duration_ms=1.2812, center_ms=7.4339075,
               area=0.0, max_amplitude=0.0, attribute=0x81)
    p9 = Probe(start_ms=0.0, duration_ms=0.3125, center_ms=38.4339,
               area=0.0, max_amplitude=0.0, attribute=0x81)
    mol = Molecule(
        file_name_index=0, channel=24, molecule_id=83, uid=13577,
        start_ms=0.0, start_within_tdb_ms=13.409842491149902,
        transloc_time_ms=39.398765563964844, use_partial_time_ms=0.0,
        mean_lvl1=0.0, rise_t10=0.0, rise_t50=0.0, rise_t90=0.0,
        fall_t90=0.0, fall_t50=0.0, fall_t10=0.0,
        folded_start_end=0.0, folded_end_start=0.0, why_structured=0,
        num_probes=2, num_structures=0, structured=False, use_partial=False,
        folded_start=False, folded_end=False, num_recovered_structures=0,
        do_not_use=False, probes=[p0, p9], structures=[],
    )
    # Fake assignment pointing p0 -> ref_idx 1, p9 -> ref_idx 2.
    assign = MoleculeAssignment(
        ref_index=0, fragment_uid=13577, direction=1,
        alignment_score=0, second_best_score=0,
        stretch_factor=1.0, stretch_offset=0.0, weight=1.0,
        probe_indices=(1, 2),
    )
    ref = ReferenceMap(
        genome_name="synthetic",
        genome_length=10_000_000,
        probe_positions=np.array([0, 100_000, 200_000], dtype=np.int64),
        strands=np.array([0, 0, 0], dtype=np.int8),
        enzyme_indices=np.array([0, 0, 0], dtype=np.int8),
    )

    gt = build_molecule_gt(
        mol, assign, ref,
        sample_rate_hz=32_000,
        min_matched_probes=2,
        include_warmstart=True,
    )
    assert gt is not None
    centers = gt.warmstart_probe_centers_samples
    assert centers is not None
    assert len(centers) == 2
    # start_within_tdb_ms = 13.409842... -> 429.11 samples at 32 kHz
    # + probe 0 center_ms 7.4339... -> 237.88 samples
    # = 667.0 samples (rounded)
    assert centers[0] == 667, (
        f"probe 0 sample index: expected 667, got {centers[0]} "
        "(regression: either start_within_tdb_ms ignored or sample_rate wrong)"
    )
    # probe 9 center_ms 38.4339 -> 1229.88 samples, + 429.11 offset = 1659.0
    assert centers[1] == 1659, (
        f"probe 9 sample index: expected 1659, got {centers[1]}"
    )

    # Durations are in the same sample-period units.
    durations = gt.warmstart_probe_durations_samples
    assert durations is not None
    # 1.2812 ms * 32 kHz / 1000 = 41.00 samples
    np.testing.assert_allclose(durations[0], 41.00, atol=0.1)
    # 0.3125 ms * 32 kHz / 1000 = 10.00 samples
    np.testing.assert_allclose(durations[1], 10.00, atol=0.1)


def _first_mapped_clean(probes_file, assigns):
    for assign in assigns[:50]:
        if assign.ref_index < 0:
            continue
        mol = probes_file.molecules[assign.fragment_uid]
        if mol.num_probes < 8 or mol.do_not_use:
            continue
        return mol, assign
    return None, None


def test_build_gt_reference_positions_populated(remap_allch_dir, sigproc_dir):
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
    mol, assign = _first_mapped_clean(probes_file, assigns)
    assert mol is not None

    gt = build_molecule_gt(mol, assign, ref, sample_rate_hz=_FIXTURE_SAMPLE_RATE_HZ)
    assert gt is not None
    assert isinstance(gt, MoleculeGT)
    assert gt.reference_bp_positions.dtype == np.int64
    assert gt.n_ref_probes == len(gt.reference_bp_positions)
    assert gt.n_ref_probes >= 4
    # All values are within genome bounds
    assert np.all(gt.reference_bp_positions >= 0)
    assert np.all(gt.reference_bp_positions < ref.genome_length)


def test_build_gt_reference_positions_ordered_by_direction(remap_allch_dir, sigproc_dir):
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
    mol, assign = _first_mapped_clean(probes_file, assigns)
    gt = build_molecule_gt(mol, assign, ref, sample_rate_hz=_FIXTURE_SAMPLE_RATE_HZ)
    assert gt is not None
    if gt.direction == 1:
        # Forward: ascending bp in temporal order
        assert np.all(np.diff(gt.reference_bp_positions) > 0)
    else:
        # Reverse: descending bp in temporal order (trailing-end first)
        assert np.all(np.diff(gt.reference_bp_positions) < 0)


def test_build_gt_unmapped_returns_none(remap_allch_dir, sigproc_dir):
    ref = load_reference_map(
        remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_referenceMap.txt"
    )
    assigns = load_assigns(
        remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    )
    probes_file = load_probes_bin(
        sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin",
        max_molecules=20,
    )
    unmapped = next(a for a in assigns[:20] if a.ref_index == -1)
    mol = probes_file.molecules[unmapped.fragment_uid]
    gt = build_molecule_gt(mol, unmapped, ref, sample_rate_hz=_FIXTURE_SAMPLE_RATE_HZ)
    assert gt is None


def test_build_gt_warmstart_fields_present(remap_allch_dir, sigproc_dir):
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
    mol, assign = _first_mapped_clean(probes_file, assigns)
    gt = build_molecule_gt(mol, assign, ref, sample_rate_hz=_FIXTURE_SAMPLE_RATE_HZ, include_warmstart=True)
    assert gt is not None
    assert gt.warmstart_probe_centers_samples is not None
    assert gt.warmstart_probe_durations_samples is not None
    assert len(gt.warmstart_probe_centers_samples) == len(gt.warmstart_probe_durations_samples)
    assert len(gt.warmstart_probe_centers_samples) > 0
    # V4 schema invariant: warmstart arrays are paired 1:1 with
    # reference_bp_positions, with -1 in centers (and 0 in durations) for
    # probes that ground_truth.py dropped (duration_ms <= 0).
    assert len(gt.warmstart_probe_centers_samples) == len(gt.reference_bp_positions)


def test_build_gt_warmstart_uses_sentinels_for_dropped_probes():
    """Probes with duration_ms <= 0 must show as -1 in centers (and 0 in
    durations), preserving the 1:1 pairing with reference_bp_positions
    that the noise-model NLL relies on."""
    p_valid = Probe(start_ms=0.0, duration_ms=1.0, center_ms=10.0,
                    area=0.0, max_amplitude=0.0, attribute=0x81)
    p_invalid = Probe(start_ms=0.0, duration_ms=0.0, center_ms=20.0,
                      area=0.0, max_amplitude=0.0, attribute=0x81)
    p_also_valid = Probe(start_ms=0.0, duration_ms=1.0, center_ms=30.0,
                         area=0.0, max_amplitude=0.0, attribute=0x81)
    mol = Molecule(
        file_name_index=0, channel=1, molecule_id=0, uid=42,
        start_ms=0.0, start_within_tdb_ms=0.0,
        transloc_time_ms=40.0, use_partial_time_ms=0.0,
        mean_lvl1=0.0, rise_t10=0.0, rise_t50=0.0, rise_t90=0.0,
        fall_t90=0.0, fall_t50=0.0, fall_t10=0.0,
        folded_start_end=0.0, folded_end_start=0.0, why_structured=0,
        num_probes=3, num_structures=0, structured=False, use_partial=False,
        folded_start=False, folded_end=False, num_recovered_structures=0,
        do_not_use=False, probes=[p_valid, p_invalid, p_also_valid], structures=[],
    )
    assign = MoleculeAssignment(
        ref_index=0, fragment_uid=42, direction=1,
        alignment_score=0, second_best_score=0,
        stretch_factor=1.0, stretch_offset=0.0, weight=1.0,
        probe_indices=(1, 2, 3),
    )
    ref = ReferenceMap(
        genome_name="synthetic",
        genome_length=10_000_000,
        probe_positions=np.array([1000, 2000, 3000], dtype=np.int64),
        strands=np.array([0, 0, 0], dtype=np.int8),
        enzyme_indices=np.array([0, 0, 0], dtype=np.int8),
    )

    gt = build_molecule_gt(
        mol, assign, ref,
        sample_rate_hz=32_000,
        min_matched_probes=1,
        include_warmstart=True,
    )
    assert gt is not None
    centers = gt.warmstart_probe_centers_samples
    durations = gt.warmstart_probe_durations_samples
    assert centers is not None and durations is not None
    # Pair-length invariant.
    assert len(centers) == len(gt.reference_bp_positions) == 3
    # Middle probe was invalid -> sentinel.
    assert int(centers[1]) == -1
    assert float(durations[1]) == 0.0
    # The other two probes have valid centers.
    assert int(centers[0]) > 0
    assert int(centers[2]) > 0
    assert float(durations[0]) > 0.0
    assert float(durations[2]) > 0.0


def test_build_gt_warmstart_excluded(remap_allch_dir, sigproc_dir):
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
    mol, assign = _first_mapped_clean(probes_file, assigns)
    gt = build_molecule_gt(mol, assign, ref, sample_rate_hz=_FIXTURE_SAMPLE_RATE_HZ, include_warmstart=False)
    assert gt is not None
    assert gt.warmstart_probe_centers_samples is None
    assert gt.warmstart_probe_durations_samples is None
