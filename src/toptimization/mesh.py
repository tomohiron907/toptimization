"""Structured mesh: node numbering, element-to-DOF connectivity (edof).

Node numbering convention (2D):
  Node (ix, iy) -> global index = ix * (nely+1) + iy
  ix in [0, nelx], iy in [0, nely]

  y (iy)
  ^
  |  (0,2)-(1,2)-(2,2)
  |  |  e2 |  e5 |
  |  (0,1)-(1,1)-(2,1)
  |  |  e1 |  e4 |
  |  (0,0)-(1,0)-(2,0)
  +-------------------> x (ix)
  (nelx=2, nely=2, element numbering: e = ex*nely + ey)

For element (ex, ey), the 4 corner nodes (counter-clockwise from bottom-left):
  n1 = (ex,   ey  )   bottom-left
  n2 = (ex+1, ey  )   bottom-right
  n3 = (ex+1, ey+1)   top-right
  n4 = (ex,   ey+1)   top-left

DOF ordering per element (8 DOFs in 2D, Q4):
  [ux_n1, uy_n1, ux_n2, uy_n2, ux_n3, uy_n3, ux_n4, uy_n4]
"""

from __future__ import annotations

import numpy as np
from .problem import Problem


def build_edof_2d(problem: Problem) -> np.ndarray:
    """Build element-to-DOF connectivity for a 2D structured mesh.

    Returns
    -------
    edof : ndarray, shape (n_elem, 8), dtype int32
        edof[e, i] = global DOF index for local DOF i of element e.
    """
    nelx, nely = problem.nelx, problem.nely
    n_elem = nelx * nely

    edof = np.zeros((n_elem, 8), dtype=np.int32)

    for ex in range(nelx):
        for ey in range(nely):
            e = ex * nely + ey

            # 4 corner node global indices
            n1 = ex * (nely + 1) + ey          # bottom-left
            n2 = (ex + 1) * (nely + 1) + ey    # bottom-right
            n3 = (ex + 1) * (nely + 1) + ey + 1  # top-right
            n4 = ex * (nely + 1) + ey + 1      # top-left

            edof[e, 0] = 2 * n1
            edof[e, 1] = 2 * n1 + 1
            edof[e, 2] = 2 * n2
            edof[e, 3] = 2 * n2 + 1
            edof[e, 4] = 2 * n3
            edof[e, 5] = 2 * n3 + 1
            edof[e, 6] = 2 * n4
            edof[e, 7] = 2 * n4 + 1

    return edof


def build_edof_3d(problem: Problem) -> np.ndarray:
    """Build element-to-DOF connectivity for a 3D structured mesh.

    Node (ix, iy, iz) -> global index = ix*(nely+1)*(nelz+1) + iy*(nelz+1) + iz

    For H8 hexahedral element (ex, ey, ez), 8 nodes (24 DOFs per element):
    Nodes are ordered: 4 bottom-face (z=ez) + 4 top-face (z=ez+1),
    each face in CCW order viewed from outside.

    Returns
    -------
    edof : ndarray, shape (n_elem, 24), dtype int32
    """
    nelx, nely, nelz = problem.nelx, problem.nely, problem.nelz
    n_elem = nelx * nely * nelz
    nely1 = nely + 1
    nelz1 = nelz + 1

    def node_idx(ix: int, iy: int, iz: int) -> int:
        return ix * nely1 * nelz1 + iy * nelz1 + iz

    edof = np.zeros((n_elem, 24), dtype=np.int32)

    for ex in range(nelx):
        for ey in range(nely):
            for ez in range(nelz):
                e = ex * nely * nelz + ey * nelz + ez

                # 8 corner nodes of the hexahedron
                nodes = [
                    node_idx(ex,     ey,     ez),
                    node_idx(ex + 1, ey,     ez),
                    node_idx(ex + 1, ey + 1, ez),
                    node_idx(ex,     ey + 1, ez),
                    node_idx(ex,     ey,     ez + 1),
                    node_idx(ex + 1, ey,     ez + 1),
                    node_idx(ex + 1, ey + 1, ez + 1),
                    node_idx(ex,     ey + 1, ez + 1),
                ]
                for k, ni in enumerate(nodes):
                    edof[e, 3 * k]     = 3 * ni
                    edof[e, 3 * k + 1] = 3 * ni + 1
                    edof[e, 3 * k + 2] = 3 * ni + 2

    return edof


def build_edof(problem: Problem) -> np.ndarray:
    """Build element-to-DOF connectivity (dispatch on problem dimension)."""
    if problem.dim == 2:
        return build_edof_2d(problem)
    else:
        return build_edof_3d(problem)


def element_centers_2d(problem: Problem) -> np.ndarray:
    """Element center coordinates for 2D mesh.

    Returns
    -------
    centers : ndarray, shape (n_elem, 2)
        centers[e] = [cx, cy] in element-size units.
    """
    nelx, nely = problem.nelx, problem.nely
    ex_idx = np.arange(nelx * nely) // nely  # ex for each element
    ey_idx = np.arange(nelx * nely) % nely   # ey for each element
    cx = ex_idx + 0.5
    cy = ey_idx + 0.5
    return np.stack([cx, cy], axis=1).astype(np.float32)


def element_centers_3d(problem: Problem) -> np.ndarray:
    """Element center coordinates for 3D mesh.

    Returns
    -------
    centers : ndarray, shape (n_elem, 3)
    """
    nelx, nely, nelz = problem.nelx, problem.nely, problem.nelz
    n = nelx * nely * nelz
    centers = np.zeros((n, 3), dtype=np.float32)
    for ex in range(nelx):
        for ey in range(nely):
            for ez in range(nelz):
                e = ex * nely * nelz + ey * nelz + ez
                centers[e] = [ex + 0.5, ey + 0.5, ez + 0.5]
    return centers


def element_centers(problem: Problem) -> np.ndarray:
    """Element centers dispatched on dimension."""
    if problem.dim == 2:
        return element_centers_2d(problem)
    else:
        return element_centers_3d(problem)
