"""Optimality Criteria (OC) density update — Phase 2 optimization.

Finds the Lagrange multiplier lambda via bisection such that the
volume constraint is satisfied, then updates densities.

Update rule per element:
  rho_candidate = rho[e] * sqrt(-dc[e] / (lambda * dv[e]))
  rho_new[e] = clip(rho_candidate, rho[e]-move, rho[e]+move, 0.001, 1.0)

Phase 2: Fused kernel computes OC update AND volume in one pass,
reducing GPU-CPU syncs from 2 → ~1 per bisection iteration.
(~80 syncs → ~40 syncs total for 40 bisection iterations)
"""

import taichi as ti

from ..fem import fields as F
from ..problem import Problem


@ti.kernel
def _oc_update_and_vol_kernel(
    rho: ti.template(),
    rho_new: ti.template(),
    dc: ti.template(),
    dv: ti.template(),
    max_result: ti.template(),
    lam: float,
    move: float,
    n_elem: int,
) -> ti.f32:
    """Apply OC update for a given lambda; returns volume fraction.

    Also writes max density change to max_result[None].
    Fuses the former _oc_update_kernel + compute_volume into one kernel,
    halving the number of GPU-CPU syncs per bisection step.
    """
    max_change = 0.0
    volume = 0.0
    for e in range(n_elem):
        be = ti.max(-dc[e] / (lam * dv[e] + 1e-20), 0.0)
        rho_candidate = rho[e] * ti.sqrt(be)
        rho_lo = ti.max(0.001, rho[e] - move)
        rho_hi = ti.min(1.0,   rho[e] + move)
        rho_new[e] = ti.max(rho_lo, ti.min(rho_hi, rho_candidate))
        ti.atomic_max(max_change, ti.abs(rho_new[e] - rho[e]))
        volume += rho_new[e]
    max_result[None] = max_change
    return volume / n_elem


def oc_update(problem: Problem) -> float:
    """Perform OC density update with bisection on lambda.

    Uses F.dc_filt as the sensitivity. Updates F.rho in-place.

    Returns
    -------
    max_change : float
        Maximum density change this iteration (for convergence check).
    """
    n_elem = problem.n_elem
    move = problem.move_limit
    volfrac = problem.volfrac

    dc_field = F.dc_filt

    lam_lo, lam_hi = 0.0, 1e9
    max_change = 0.0

    for _ in range(50):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        # Fused kernel: OC update + volume (1 GPU-CPU sync instead of 2)
        vol = float(_oc_update_and_vol_kernel(
            F.rho, F.rho_new, dc_field, F.dv, F._max_result,
            lam_mid, move, n_elem,
        ))
        # max_change already written to F._max_result[None] by the kernel;
        # reading it here is a cheap field access after the sync above.
        max_change = float(F._max_result[None])

        if vol > volfrac:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

        if (lam_hi - lam_lo) / (lam_hi + lam_lo + 1e-30) < 1e-4:
            break

    # Copy rho_new -> rho
    from ..fem import kernels as K
    K.copy_field(F.rho_new, F.rho)
    return max_change
