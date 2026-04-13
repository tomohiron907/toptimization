"""Preconditioned Conjugate Gradient (PCG) solver for FEA.

Implements matrix-free PCG:  solve K(rho) * u = f
Result stored in-place in F.u.

Phase 1 optimization (fused kernels):
  Reduces kernel launches 8 → 3-4 and GPU-CPU syncs 3 → 2 per CG iteration.

Phase 3 optimization (block Jacobi preconditioner):
  Reduces CG iteration count by 3-5x versus diagonal Jacobi.

Supported preconditioners:
  "jacobi"       — diagonal scaling (default, fast setup, moderate convergence)
  "block_jacobi" — element-block pseudoinverse (slower apply, much fewer iters)
  "none"         — no preconditioning
"""

import math
import taichi as ti

from . import fields as F
from . import kernels as K
from . import fused_kernels as FK
from . import preconditioner as PC


def solve(
    E_min: float,
    penalty: float,
    dim: int,
    max_iter: int = 2000,
    tol: float = 1e-8,
    preconditioner: str = "jacobi",
    warm_start: bool = True,
) -> int:
    """Solve K(rho)*u = f in-place (result stored in F.u).

    Parameters
    ----------
    E_min : float
    penalty : float
    dim : int
    max_iter : int
    tol : float
        Convergence tolerance on relative residual: ‖r‖ / ‖f‖.
    preconditioner : {"jacobi", "block_jacobi", "none"}
    warm_start : bool
        Initialize u from F.u_prev.

    Returns
    -------
    n_iter : int
        Number of CG iterations performed.
    """
    n_dofs = F.N_DOFS
    n_elem = F.N_ELEM

    # Reference norm ‖f‖
    f_norm_sq = K.l2_norm_sq(F.f)
    f_norm = math.sqrt(float(f_norm_sq))
    if f_norm < 1e-30:
        K.fill_scalar(F.u, 0.0)
        return 0

    tol_sq = (tol * f_norm) ** 2

    # Setup preconditioner
    use_jacobi = (preconditioner == "jacobi")
    use_block  = (preconditioner == "block_jacobi")

    if use_jacobi:
        K.compute_diagonal(
            F.diag, F.rho, F.Ke, F.edof, F.is_fixed,
            E_min, penalty, n_elem, dim,
        )

    # Initialize u
    if warm_start:
        K.copy_field(F.u_prev, F.u)
    else:
        K.fill_scalar(F.u, 0.0)

    _zero_fixed_dofs(F.u, F.is_fixed, n_dofs)

    # r = f - K*u
    K.compute_matvec(F.u, F.Ap, F.rho, F.Ke, F.edof, F.is_fixed,
                     E_min, penalty, n_elem, dim)
    K.axpy(-1.0, F.Ap, F.f, F.r)   # r = f - Ap

    # z = M^{-1} r  (initial)
    if use_jacobi:
        K.apply_diag_precond(F.diag, F.r, F.z)
    elif use_block:
        PC.apply_and_rz(
            F.r, F.z, F.rho, F.Ke_inv, F.edof, F.diag_count, F.is_fixed, F._dot_result,
            E_min, penalty, F.KE_ALPHA_REG, n_elem, dim,
        )
    else:
        K.copy_field(F.r, F.z)

    K.copy_field(F.z, F.p)
    rz_old = float(K.dot_product(F.r, F.z))

    # ------------------------------------------------------------------ #
    # CG main loop (fused kernels)
    # ------------------------------------------------------------------ #
    for it in range(max_iter):

        # --- Kernel 1: Ap = K*p, returns p⋅Ap  (1 GPU-CPU sync) ---
        pAp = float(FK.matvec_and_dot(
            F.p, F.Ap, F.rho, F.Ke, F.edof, F.is_fixed,
            E_min, penalty, n_elem, dim,
        ))
        if abs(pAp) < 1e-30:
            break

        alpha = rz_old / pAp

        if use_jacobi:
            # --- Kernel 2 (Jacobi): u+=αp, r-=αAp, z=r/diag;
            #     returns r⋅z; writes ‖r‖² to _dot_result  (1 sync) ---
            rz_new = float(FK.update_and_precond_jacobi(
                F.u, F.r, F.z, F.p, F.Ap, F.diag, F._dot_result, alpha,
            ))
            r_norm_sq = float(F._dot_result[None])

        elif use_block:
            # --- Kernel 2 (block): u+=αp, r-=αAp  (0 sync) ---
            FK.update_ur(F.u, F.r, F.p, F.Ap, alpha)
            # --- Kernel 3 (block): z=Ke_inv*r (averaged), returns r⋅z,
            #     writes ‖r‖² to _dot_result  (1 sync) ---
            rz_new = float(PC.apply_and_rz(
                F.r, F.z, F.rho, F.Ke_inv, F.edof, F.diag_count, F.is_fixed, F._dot_result,
                E_min, penalty, F.KE_ALPHA_REG, n_elem, dim,
            ))
            r_norm_sq = float(F._dot_result[None])

        else:
            # --- Kernel 2 (none): u+=αp, r-=αAp, z=r;
            #     returns ‖r‖²; writes ‖r‖² to _dot_result  (1 sync) ---
            rz_new = float(FK.update_and_precond_none(
                F.u, F.r, F.z, F.p, F.Ap, F._dot_result, alpha,
            ))
            r_norm_sq = float(F._dot_result[None])

        # Convergence check (‖r‖² < tol² ‖f‖²)
        if r_norm_sq < tol_sq:
            K.copy_field(F.u, F.u_prev)
            return it + 1

        beta = rz_new / (rz_old + 1e-60)

        # --- Kernel last: p = z + β*p  (0 sync) ---
        FK.update_p(F.z, F.p, beta)

        rz_old = rz_new

    K.copy_field(F.u, F.u_prev)
    return max_iter


@ti.kernel
def _zero_fixed_dofs(u: ti.template(), is_fixed: ti.template(), n_dofs: int):
    """Set fixed DOF displacements to zero."""
    for i in range(n_dofs):
        if is_fixed[i] == 1:
            u[i] = 0.0
