import pathlib
import pytest

SUPPORT_DIR = pathlib.Path(__file__).resolve().parent.parent / "support"

@pytest.fixture
def example_data_dir():
    return SUPPORT_DIR / "example-data" / "STB03-064B-02L58270w05-202G16g"

@pytest.fixture
def example_remap_dir():
    return SUPPORT_DIR / "example-remap" / "STB03-064B-02L58270w05-202G16g"

@pytest.fixture
def sigproc_dir(example_data_dir):
    return example_data_dir / "2025-02-19" / "Results" / "1_9_7309" / "pp-705" / "AllCh"

@pytest.fixture
def remap_allch_dir(example_remap_dir):
    return example_remap_dir / "AllCh"
