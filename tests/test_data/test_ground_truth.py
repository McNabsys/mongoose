"""Tests for the V1 rearchitecture ground truth builder."""

import numpy as np

from mongoose.io.reference_map import load_reference_map
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.assigns import load_assigns
from mongoose.data.ground_truth import build_molecule_gt, MoleculeGT


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

    gt = build_molecule_gt(mol, assign, ref)
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
    gt = build_molecule_gt(mol, assign, ref)
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
    gt = build_molecule_gt(mol, unmapped, ref)
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
    gt = build_molecule_gt(mol, assign, ref, include_warmstart=True)
    assert gt is not None
    assert gt.warmstart_probe_centers_samples is not None
    assert gt.warmstart_probe_durations_samples is not None
    assert len(gt.warmstart_probe_centers_samples) == len(gt.warmstart_probe_durations_samples)
    assert len(gt.warmstart_probe_centers_samples) > 0


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
    gt = build_molecule_gt(mol, assign, ref, include_warmstart=False)
    assert gt is not None
    assert gt.warmstart_probe_centers_samples is None
    assert gt.warmstart_probe_durations_samples is None
