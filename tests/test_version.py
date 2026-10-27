"""The package version has one source, pyproject.toml; ccp.__version__ reads it from the installed metadata."""
import re
from pathlib import Path

import ccp


def test_version_matches_pyproject():
    txt = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    want = re.search(r'^version = "([^"]+)"', txt, re.M).group(1)
    assert ccp.__version__ == want, f"ccp.__version__ {ccp.__version__} != pyproject {want}: run `pip install -e . --no-deps`"
