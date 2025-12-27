import pytest
from pathlib import Path

@pytest.fixture
def sample_data():
    return Path(__file__).parent / "fixtures" / "sample_data.json"