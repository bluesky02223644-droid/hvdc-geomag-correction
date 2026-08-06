"""Shared paths for the figure scripts.

Importing this module also puts the figure directory on ``sys.path`` so that
``palette_standard`` resolves whether a script is run from the repository root
or from inside ``figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIG_DIR.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs" / "figures"

if str(FIG_DIR) not in sys.path:
    sys.path.insert(0, str(FIG_DIR))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
