# computation/plotting.py
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_field(
    u: np.ndarray,
    out_path: Optional[str | Path] = None,
    title: str = "Concentration field",
) -> None:
    """
    Plot a 2D field using imshow.

    Parameters
    ----------
    u : ndarray, shape (ny, nx)
        Field to plot.
    out_path : str or Path, optional
        If given, save the figure to this path instead of showing it.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(u, origin="lower", cmap="viridis")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Concentration")

    if out_path is None:
        plt.show()
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
