"""Root pytest configuration.

Its presence puts the repository root on ``sys.path``, which is what lets tests import
the ``scripts`` package (see ``scripts/__init__.py``). The library itself is imported
from the installed ``src/`` layout, not from here.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the golden-trajectory regeneration flag (see tests/dynamics)."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite committed golden artifacts instead of asserting against them.",
    )
