import numpy as np
from mongoose.io.reference_map import load_reference_map

def test_load_reference_map(remap_allch_dir):
    ref_path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_referenceMap.txt"
    ref = load_reference_map(ref_path)

    assert ref.genome_name is not None
    assert "Escherichia coli" in ref.genome_name
    assert ref.genome_length == 4641652
    assert isinstance(ref.probe_positions, np.ndarray)
    assert ref.probe_positions.dtype == np.int64
    assert 750 < len(ref.probe_positions) < 900
    assert np.all(np.diff(ref.probe_positions) > 0)
    assert ref.probe_positions[0] >= 0
    assert ref.probe_positions[-1] < ref.genome_length

def test_reference_map_first_last_positions(remap_allch_dir):
    ref_path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_probes.txt_referenceMap.txt"
    ref = load_reference_map(ref_path)
    assert ref.probe_positions[0] == 1953
    assert ref.probe_positions[1] == 2395
    assert ref.probe_positions[2] == 3717
