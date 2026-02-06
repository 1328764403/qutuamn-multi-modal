"""
Path utilities to guarantee we only read local datasets/models
from inside the `quantum_multimodal_comparison/` folder.
"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return absolute path to `quantum_multimodal_comparison/`."""
    # utils/path_guard.py -> utils/ -> quantum_multimodal_comparison/
    return Path(__file__).resolve().parent.parent


def resolve_inside_project(path: str | Path) -> Path:
    """
    Resolve a path and ensure it is inside the project root.

    This is a hard guarantee that training/validation will not read datasets
    outside `quantum_multimodal_comparison/`.
    """
    root = get_project_root()
    p = Path(path)
    resolved = p.resolve() if p.is_absolute() else (root / p).resolve()

    if resolved != root and root not in resolved.parents:
        raise ValueError(
            "For this task, dataset/model paths must stay inside "
            f"`{root}`.\n"
            f"Got: {resolved}"
        )
    return resolved

