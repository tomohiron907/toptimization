"""2D topology optimization result visualization using matplotlib."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def save_density_png(
    rho: np.ndarray,
    path: str | Path,
    title: str = "Topology",
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
) -> None:
    """Save a density field image as PNG.

    Parameters
    ----------
    rho : ndarray, shape (nelx, nely)
        Density field. Values in [0, 1] where 1=solid, 0=void.
    path : str or Path
    title : str
    figsize : tuple, optional
    dpi : int
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    nelx, nely = rho.shape
    if figsize is None:
        aspect = nely / nelx
        figsize = (8.0, 8.0 * aspect + 0.6)

    fig, ax = plt.subplots(figsize=figsize)

    # Transpose so that x is horizontal, y is vertical (origin at bottom-left)
    img = rho.T  # shape (nely, nelx)

    # Black = solid (rho=1), white = void (rho=0)
    ax.imshow(
        1.0 - img,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_convergence_png(
    compliance_hist: list[float],
    volume_hist: list[float],
    path: str | Path,
    dpi: int = 100,
) -> None:
    """Save compliance and volume convergence history as PNG."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    iters = list(range(1, len(compliance_hist) + 1))

    ax1.semilogy(iters, compliance_hist, linewidth=1.5, color="steelblue")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Compliance")
    ax1.set_title("Compliance history")
    ax1.grid(True, alpha=0.3)

    ax2.plot(iters, volume_hist, linewidth=1.5, color="darkorange")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Volume fraction")
    ax2.set_title("Volume fraction history")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def show_density(
    rho: np.ndarray,
    title: str = "Topology",
    block: bool = True,
) -> None:
    """Display density field interactively."""
    import matplotlib.pyplot as plt

    nelx, nely = rho.shape
    aspect = nely / nelx
    fig, ax = plt.subplots(figsize=(8.0, 8.0 * aspect))
    ax.imshow(
        1.0 - rho.T,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    plt.show(block=block)
