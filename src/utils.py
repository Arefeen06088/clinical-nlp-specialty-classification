"""Shared reproducibility and plotting helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int = 42) -> None:
    """Set common random seeds used by the project."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_plots() -> None:
    """Apply the repository-wide Seaborn theme."""
    import seaborn as sns

    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette="colorblind",
        font_scale=1.05,
        rc={"figure.dpi": 120, "savefig.dpi": 300},
    )


def ensure_output_dirs() -> None:
    """Create generated-artifact directories when absent."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(figure, filename: str) -> Path:
    """Save a figure consistently and return its path."""
    ensure_output_dirs()
    output_path = FIGURES_DIR / filename
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    return output_path

