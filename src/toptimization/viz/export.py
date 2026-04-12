"""Export topology optimization results to various formats."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def save_npy(rho: np.ndarray, path: str | Path) -> None:
    """Save density field as NumPy .npy file."""
    np.save(str(path), rho)
    print(f"[export] Saved density to {path}")


def save_vtk_2d(
    rho: np.ndarray,
    path: str | Path,
    element_size: float = 1.0,
) -> None:
    """Save 2D density field as VTK rectilinear grid (legacy ASCII format).

    The output can be opened in ParaView.

    Parameters
    ----------
    rho : ndarray, shape (nelx, nely)
    path : str or Path
    element_size : float
    """
    nelx, nely = rho.shape
    path = Path(path)

    # Node coordinates
    x_coords = np.arange(nelx + 1, dtype=float) * element_size
    y_coords = np.arange(nely + 1, dtype=float) * element_size
    z_coords = np.array([0.0])

    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Topology optimization density\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nelx+1} {nely+1} 1\n")

        f.write(f"X_COORDINATES {nelx+1} float\n")
        f.write(" ".join(f"{v:.6f}" for v in x_coords) + "\n")

        f.write(f"Y_COORDINATES {nely+1} float\n")
        f.write(" ".join(f"{v:.6f}" for v in y_coords) + "\n")

        f.write(f"Z_COORDINATES 1 float\n")
        f.write("0.000000\n")

        n_cells = nelx * nely
        f.write(f"CELL_DATA {n_cells}\n")
        f.write("SCALARS density float 1\n")
        f.write("LOOKUP_TABLE default\n")
        # VTK cell ordering: x varies fastest for rectilinear grids
        for jy in range(nely):
            for jx in range(nelx):
                e = jx * nely + jy
                f.write(f"{rho.flat[e]:.6f}\n")

    print(f"[export] Saved VTK to {path}")


def save_vtk_3d(
    rho: np.ndarray,
    path: str | Path,
    element_size: float = 1.0,
) -> None:
    """Save 3D density field as VTK rectilinear grid.

    Parameters
    ----------
    rho : ndarray, shape (nelx, nely, nelz)
    path : str or Path
    element_size : float
    """
    nelx, nely, nelz = rho.shape
    path = Path(path)

    x_coords = np.arange(nelx + 1, dtype=float) * element_size
    y_coords = np.arange(nely + 1, dtype=float) * element_size
    z_coords = np.arange(nelz + 1, dtype=float) * element_size

    with path.open("w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Topology optimization density 3D\n")
        f.write("ASCII\n")
        f.write("DATASET RECTILINEAR_GRID\n")
        f.write(f"DIMENSIONS {nelx+1} {nely+1} {nelz+1}\n")

        f.write(f"X_COORDINATES {nelx+1} float\n")
        f.write(" ".join(f"{v:.6f}" for v in x_coords) + "\n")
        f.write(f"Y_COORDINATES {nely+1} float\n")
        f.write(" ".join(f"{v:.6f}" for v in y_coords) + "\n")
        f.write(f"Z_COORDINATES {nelz+1} float\n")
        f.write(" ".join(f"{v:.6f}" for v in z_coords) + "\n")

        n_cells = nelx * nely * nelz
        f.write(f"CELL_DATA {n_cells}\n")
        f.write("SCALARS density float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for ez in range(nelz):
            for ey in range(nely):
                for ex in range(nelx):
                    e = ex * nely * nelz + ey * nelz + ez
                    f.write(f"{rho.flat[e]:.6f}\n")

    print(f"[export] Saved 3D VTK to {path}")
