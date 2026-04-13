"""Tests for the ground truth builder."""

import numpy as np
import pytest

from mongoose.io.reference_map import load_reference_map
from mongoose.io.probes_bin import load_probes_bin
from mongoose.io.assigns import load_assigns
from mongoose.data.ground_truth import build_molecule_gt, MoleculeGT


def test_build_gt_for_mapped_molecule(remap_allch_dir, sigproc_dir):
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

    # Find first mapped, clean molecule with enough probes
    for assign in assigns[:50]:
        if assign.ref_index < 0:
            continue
        mol = probes_file.molecules[assign.fragment_uid]
        if mol.num_probes < 8 or mol.do_not_use:
            continue
        gt = build_molecule_gt(mol, assign, ref)
        assert gt is not None
        assert isinstance(gt, MoleculeGT)
        assert np.all(gt.inter_probe_deltas_bp > 0)
        assert len(gt.inter_probe_deltas_bp) == len(gt.probe_sample_indices) - 1
        assert len(gt.velocity_targets_bp_per_ms) == len(gt.probe_sample_indices)
        assert np.all(gt.velocity_targets_bp_per_ms > 0)
        assert gt.direction in (1, -1)
        break
    else:
        pytest.fail("No suitable mapped molecule found in first 50")


def test_gt_deltas_sum_reasonable(remap_allch_dir, sigproc_dir):
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
    for assign in assigns[:50]:
        if assign.ref_index < 0:
            continue
        mol = probes_file.molecules[assign.fragment_uid]
        if mol.num_probes < 8 or mol.do_not_use:
            continue
        gt = build_molecule_gt(mol, assign, ref)
        if gt is None:
            continue
        total_span = np.sum(gt.inter_probe_deltas_bp)
        assert 0 < total_span < ref.genome_length
        break


def test_unmapped_molecule_returns_none(remap_allch_dir, sigproc_dir):
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


def test_sample_indices_are_sorted(remap_allch_dir, sigproc_dir):
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
    for assign in assigns[:50]:
        if assign.ref_index < 0:
            continue
        mol = probes_file.molecules[assign.fragment_uid]
        if mol.num_probes < 8 or mol.do_not_use:
            continue
        gt = build_molecule_gt(mol, assign, ref)
        if gt is None:
            continue
        # Sample indices must be in ascending temporal order
        assert np.all(np.diff(gt.probe_sample_indices) > 0)
        break
