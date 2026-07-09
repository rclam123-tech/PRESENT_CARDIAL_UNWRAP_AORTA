import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aortic_unwrap import phantoms  # noqa: E402


@pytest.fixture(scope="session")
def built_phantoms():
    """Build all four phantoms once for the whole test session."""
    return phantoms.build_all()
