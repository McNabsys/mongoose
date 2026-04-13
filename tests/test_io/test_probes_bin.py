from mongoose.io.probes_bin import load_probes_bin, Molecule


def test_load_probes_bin_header(sigproc_dir):
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path, max_molecules=0)
    assert result.num_molecules == 82276
    assert result.max_probes == 140
    assert result.file_version == 5


def test_load_first_molecule(sigproc_dir):
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path, max_molecules=1)
    mol = result.molecules[0]
    assert isinstance(mol, Molecule)
    assert mol.uid == 0
    assert mol.channel == 2
    assert mol.num_probes == 13
    assert mol.structured == True
    assert mol.do_not_use == True
    assert len(mol.probes) == 13


def test_clean_molecule_uid_16(sigproc_dir):
    """UID 16 is the first clean, usable molecule."""
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path, max_molecules=20)
    mol = next(m for m in result.molecules if m.uid == 16)
    assert mol.do_not_use == False
    assert mol.structured == False
    assert mol.num_probes == 15
    assert mol.mean_lvl1 > 0
    assert mol.transloc_time_ms > 50  # ~97ms
    for p in mol.probes:
        assert p.center_ms >= 0


def test_filter_clean_molecules(sigproc_dir):
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path)
    clean = [m for m in result.molecules
             if not m.structured and not m.folded_start and not m.folded_end
             and not m.do_not_use and m.num_probes >= 10]
    assert 25000 < len(clean) < 40000


def test_probe_attributes(sigproc_dir):
    path = sigproc_dir / "STB03-064B-02L58270w05-202G16g_probes.bin"
    result = load_probes_bin(path, max_molecules=20)
    mol = next(m for m in result.molecules if m.uid == 16)
    # All probes on this clean molecule should be accepted
    for p in mol.probes:
        assert p.accepted
