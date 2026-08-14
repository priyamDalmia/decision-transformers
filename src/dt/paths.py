"""Project paths.

Every path in the project is derived from PROJECT_ROOT so that scripts work from
any working directory. Set DT_PROJECT_ROOT to relocate the tree (e.g. when data
and checkpoints live on shared compute rather than next to the source).
"""

import os
from pathlib import Path

# src/dt/paths.py -> parents[2] is the repo root.
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.environ.get("DT_PROJECT_ROOT", _DEFAULT_ROOT))

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "output"

LOGGING_CONFIG = CONFIG_DIR / "logging.yaml"


def resolve_path(path: str | Path) -> Path:
    """Resolve a path from config against PROJECT_ROOT.

    Config files spell paths relative to the project root ("data/foo.pkl") so
    they stay readable. Absolute paths are returned unchanged.
    """
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
