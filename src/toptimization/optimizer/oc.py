"""Optimality Criteria (OC) density update.

Finds the Lagrange multiplier lambda via bisection such that the
volume constraint is satisfied, then updates densities.

Update rule per element:
  rho_candidate = rho[e] * sqrt(-dc[e] / (lambda * dv[e]))
  rho_new[e] = clip(rho_candidate, rho[e]-move, rho[e]+move, 0.001, 1.0)

Bisection bounds: lambda_lo=0, lambda_hi=1e9.
"""

import taichi as ti

from ..fem import fields as F
from ..fem import kernels as K
from ..problem import Problem


@ti.kernel
def _oc_update_kernel(
    rho: ti.template(),
    rho_new: ti.template(),
    dc: ti.template(),
    dv: ti.template(),
    lam: float,
    move: float,
    n_elem: int,
) -> ti.f32:
    """Apply OC update for a given lambda. Returns max density change."""
    max_change = 0.0
    for e in range(n_elem):
        be = ti.max(-dc[e] / (lam * dv[e] + 1e-20), 0.0)
        rho_candidate = rho[e] * ti.sqrt(be)
        rho_lo = ti.max(0.001, rho[e] - move)
        rho_hi = ti.min(1.0,   rho[e] + move)
        rho_new[e] = ti.max(rho_lo, ti.min(rho_hi, rho_candidate))
        ti.atomic_max(max_change, ti.abs(rho_new[e] - rho[e]))
    return max_change


def oc_update(problem: Problem) -> float:
    """Perform OC density update with bisection on lambda.

    Uses F.dc_filt (for density filter) or F.dc (for sensitivity filter)
    as the sensitivity. Updates F.rho in-place.

    Returns
    -------
    max_change : float
        Maximum density change this iteration (for convergence check).
    """
    n_elem = problem.n_elem
    move = problem.move_limit
    volfrac = problem.volfrac

    # Always use filtered sensitivity (dc_filt is populated by the filter step)
    dc_field = F.dc_filt

    lam_lo, lam_hi = 0.0, 1e9
    max_change = 0.0

    for _ in range(50):  # bisection iterations
        lam_mid = 0.5 * (lam_lo + lam_hi)
        max_change = float(_oc_update_kernel(
            F.rho, F.rho_new, dc_field, F.dv,
            lam_mid, move, n_elem,
        ))
        vol = float(K.compute_volume(F.rho_new, n_elem))
        if vol > volfrac:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

        if (lam_hi - lam_lo) / (lam_hi + lam_lo + 1e-30) < 1e-4:
            break

    # Copy rho_new -> rho
    K.copy_field(F.rho_new, F.rho)
    return max_change
