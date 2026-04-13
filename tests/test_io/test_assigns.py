from mongoose.io.assigns import load_assigns, MoleculeAssignment


def test_load_assigns_has_both_mapped_and_unmapped(remap_allch_dir):
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    assigns = load_assigns(path)
    assert len(assigns) > 0
    unmapped = [a for a in assigns if a.ref_index == -1]
    mapped = [a for a in assigns if a.ref_index >= 0]
    assert len(unmapped) > 0
    assert len(mapped) > 0


def test_mapped_molecule_has_valid_fields(remap_allch_dir):
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    assigns = load_assigns(path)
    mapped = [a for a in assigns if a.ref_index >= 0]
    mol = mapped[0]
    assert isinstance(mol, MoleculeAssignment)
    assert mol.direction in (1, -1)
    assert mol.alignment_score > 0
    assert len(mol.probe_indices) > 0
    assert all(isinstance(i, int) for i in mol.probe_indices)


def test_assigns_indexed_by_fragment_uid(remap_allch_dir):
    """Assigns list should be ordered by fragment UID (line order in file)."""
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    assigns = load_assigns(path)
    # Fragment UIDs should be sequential (0, 1, 2, ...)
    for i, a in enumerate(assigns[:100]):
        assert a.fragment_uid == i


def test_molecule_uid_16_is_mapped(remap_allch_dir):
    """UID 16 is the first clean molecule -- should be mapped."""
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_probeassignment.assigns"
    assigns = load_assigns(path)
    mol16 = assigns[16]
    assert mol16.fragment_uid == 16
    assert mol16.ref_index >= 0
    assert mol16.direction in (1, -1)
    # Should have probe indices with some non-zero (matched) values
    matched = [i for i in mol16.probe_indices if i > 0]
    assert len(matched) >= 5
