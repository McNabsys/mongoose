"""Tests for the _transForm.txt parser."""

from mongoose.io.transform import ChannelTransform, load_transforms


def test_load_transforms(remap_allch_dir):
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_transForm.txt"
    transforms = load_transforms(path)
    assert len(transforms) > 0
    # Check a known channel
    ch = transforms.get("Ch002")
    assert ch is not None
    assert ch.mult_const == 5101.0
    assert ch.addit_const == -1200.0
    assert ch.alpha == 0.56


def test_load_transforms_channel_with_decimals(remap_allch_dir):
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_transForm.txt"
    transforms = load_transforms(path)
    ch = transforms.get("Ch010")
    assert ch is not None
    assert abs(ch.mult_const - 5417.41652089406) < 0.01
    assert ch.addit_const == -1200.0
    assert abs(ch.alpha - 0.577414763828979) < 1e-6


def test_load_transforms_all_channels(remap_allch_dir):
    path = remap_allch_dir / "STB03-064B-02L58270w05-202G16g_transForm.txt"
    transforms = load_transforms(path)
    # All channels should have alpha in (0, 1) and positive mult_const
    for name, ch in transforms.items():
        assert ch.mult_const > 0, f"{name}: mult_const should be positive"
        assert 0 < ch.alpha < 1, f"{name}: alpha should be in (0, 1)"
        assert ch.channel == name
