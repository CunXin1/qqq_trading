"""Matplotlib setup and common chart helpers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RCPARAMS = {
    "figure.figsize": (12, 6),
    "figure.dpi": 120,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def setup_matplotlib(figsize: tuple[int, int] | None = None) -> None:
    """Apply default plot styling."""
    params = DEFAULT_RCPARAMS.copy()
    if figsize:
        params["figure.figsize"] = figsize
    plt.rcParams.update(params)


def save_chart(fig, name: str, chart_dir=None) -> None:
    """Save figure to chart directory."""
    if chart_dir is None:
        from utils.paths import CHART_DIR
        chart_dir = CHART_DIR
    fig.tight_layout()
    fig.savefig(chart_dir / name, bbox_inches="tight")
    plt.close(fig)
