import re
from pathlib import Path

import flyres

REPO = Path(__file__).resolve().parents[1]


def test_version_is_the_same_everywhere():
    """pyproject.toml, flyres.__version__ and CITATION.cff (GitHub's "Cite this repository") agree."""
    pyproject = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    citation = re.search(r"^version: (\S+)", (REPO / "CITATION.cff").read_text(encoding="utf-8"), re.M)
    assert pyproject and citation
    assert pyproject.group(1) == flyres.__version__ == citation.group(1)
