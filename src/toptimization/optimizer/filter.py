"""Density and sensitivity filtering for mesh-independence.

The filter uses a cone-shaped weight function:
  H[e, j] = max(0, rmin - dist(center_e, center_j))

Filter neighbors and weights are precomputed once on CPU (NumPy) and
stored in Taichi fields for GPU-parallel application.

Two filter modes:
  "density"     : filter the density field before FEA solve
  "sensitivity" : filter the sensitivity dc after computing it
"""

import math
import numpy as np
import taichi as ti

from ..fem import fields as F
from ..problem import Problem


def precompute_filter(problem: Problem, centers: np.ndarray) -> int:
    """Precompute filter neighbor lists and weights.

    Populates F.filt_neighbors, F.filt_weights, F.filt_n_nb.

    Parameters
    ----------
    problem : Problem
    centers : ndarray, shape (n_elem, dim)
        Element center coordinates.

    Returns
    -------
    max_neighbors : int
        Actual maximum number of neighbors (for info only).
    """
    n_elem = problem.n_elem
    rmin = problem.rmin
    dim = problem.dim
    nelx = problem.nelx
    nely = problem.nely
    nelz = problem.nelz

    # Maximum possible neighbors (conservative bound)
    r_ceil = math.ceil(rmin)
    if dim == 2:
        max_nb = int((2 * r_ceil + 1) ** 2)
    else:
        max_nb = int((2 * r_ceil + 1) ** 3)

    neighbors_np = np.full((n_elem, max_nb), -1, dtype=np.int32)
    weights_np   = np.zeros((n_elem, max_nb), dtype=np.float32)
    n_nb_np      = np.zeros(n_elem, dtype=np.int32)

    actual_max = 0

    if dim == 2:
        for ex in range(nelx):
            for ey in range(nely):
                e = ex * nely + ey
                cx, cy = centers[e]
                count = 0
                for jx in range(max(0, ex - r_ceil), min(nelx, ex + r_ceil + 1)):
                    for jy in range(max(0, ey - r_ceil), min(nely, ey + r_ceil + 1)):
                        j = jx * nely + jy
                        dist = math.sqrt((cx - (jx + 0.5))**2 + (cy - (jy + 0.5))**2)
                        w = max(0.0, rmin - dist)
                        if w > 0.0:
                            neighbors_np[e, count] = j
                            weights_np[e, count] = w
                            count += 1
                n_nb_np[e] = count
                actual_max = max(actual_max, count)
    else:
        for ex in range(nelx):
            for ey in range(nely):
                for ez in range(nelz):
                    e = ex * nely * nelz + ey * nelz + ez
                    cx, cy, cz = centers[e]
                    count = 0
                    for jx in range(max(0, ex - r_ceil), min(nelx, ex + r_ceil + 1)):
                        for jy in range(max(0, ey - r_ceil), min(nely, ey + r_ceil + 1)):
                            for jz in range(max(0, ez - r_ceil), min(nelz, ez + r_ceil + 1)):
                                j = jx * nely * nelz + jy * nelz + jz
                                dist = math.sqrt(
                                    (cx - (jx + 0.5))**2 +
                                    (cy - (jy + 0.5))**2 +
                                    (cz - (jz + 0.5))**2
                                )
                                w = max(0.0, rmin - dist)
                                if w > 0.0:
                                    neighbors_np[e, count] = j
                                    weights_np[e, count] = w
                                    count += 1
                    n_nb_np[e] = count
                    actual_max = max(actual_max, count)

    F.filt_neighbors.from_numpy(neighbors_np)
    F.filt_weights.from_numpy(weights_np)
    F.filt_n_nb.from_numpy(n_nb_np)

    return actual_max


@ti.kernel
def apply_density_filter(
    rho: ti.template(),
    rho_filt: ti.template(),
    neighbors: ti.template(),
    weights: ti.template(),
    n_nb: ti.template(),
    n_elem: int,
):
    """rho_filt[e] = sum_j(H[e,j]*rho[j]) / sum_j(H[e,j])"""
    for e in range(n_elem):
        num = 0.0
        den = 0.0
        for k in range(n_nb[e]):
            j = neighbors[e, k]
            w = weights[e, k]
            num += w * rho[j]
            den += w
        rho_filt[e] = num / (den + 1e-20)


@ti.kernel
def apply_dc_average_filter(
    dc_in: ti.template(),
    dc_out: ti.template(),
    neighbors: ti.template(),
    weights: ti.template(),
    n_nb: ti.template(),
    n_elem: int,
):
    """Chain-rule sensitivity projection for density filter.

    dc_out[e] = sum_j(H[e,j]*dc_in[j]) / sum_j(H[e,j])
    Same formula as density filter but applied to sensitivities.
    """
    for e in range(n_elem):
        num = 0.0
        den = 0.0
        for k in range(n_nb[e]):
            j = neighbors[e, k]
            w = weights[e, k]
            num += w * dc_in[j]
            den += w
        dc_out[e] = num / (den + 1e-20)


@ti.kernel
def apply_sensitivity_filter(
    rho: ti.template(),
    dc: ti.template(),
    dc_filt: ti.template(),
    neighbors: ti.template(),
    weights: ti.template(),
    n_nb: ti.template(),
    n_elem: int,
):
    """dc_filt[e] = sum_j(H[e,j]*rho[j]*dc[j]) / (rho[e] * sum_j(H[e,j]))"""
    for e in range(n_elem):
        num = 0.0
        den = 0.0
        for k in range(n_nb[e]):
            j = neighbors[e, k]
            w = weights[e, k]
            num += w * rho[j] * dc[j]
            den += w
        dc_filt[e] = num / (rho[e] * den + 1e-20)


def apply_dc_filter_density_mode(n_elem: int) -> None:
    """For density filter: project dc_phys -> dc_design via chain rule."""
    apply_dc_average_filter(
        F.dc, F.dc_filt,
        F.filt_neighbors, F.filt_weights, F.filt_n_nb,
        n_elem,
    )


def apply_dc_filter_sensitivity_mode(n_elem: int) -> None:
    """For sensitivity filter: Sigmund's weighted sensitivity filter."""
    apply_sensitivity_filter(
        F.rho, F.dc, F.dc_filt,
        F.filt_neighbors, F.filt_weights, F.filt_n_nb,
        n_elem,
    )
