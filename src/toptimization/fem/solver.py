"""Preconditioned Conjugate Gradient (PCG) solver for FEA.

Implements matrix-free PCG:
  solve K(rho) * u = f
using the kernels in kernels.py.

Preconditioning: Jacobi (diagonal scaling).
Warm start: uses the previous displacement field u_prev as initial guess.
"""

import math
import taichi as ti

from . import fields as F
from . import kernels as K


def solve(
    E_min: float,
    penalty: float,
    dim: int,
    max_iter: int = 2000,
    tol: float = 1e-8,
    use_jacobi: bool = True,
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
        Convergence tolerance on relative residual: ||r|| / ||f||.
    use_jacobi : bool
        Whether to apply Jacobi preconditioning.
    warm_start : bool
        Initialize u from F.u_prev.

    Returns
    -------
    n_iter : int
        Number of CG iterations performed.
    """
    n_dofs = F.N_DOFS
    n_elem = F.N_ELEM

    # Compute reference norm ||f||
    f_norm_sq = K.l2_norm_sq(F.f)
    f_norm = math.sqrt(float(f_norm_sq))
    if f_norm < 1e-30:
        K.fill_scalar(F.u, 0.0)
        return 0

    # Build Jacobi preconditioner
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

    # Zero out fixed DOFs in initial u
    _zero_fixed_dofs(F.u, F.is_fixed, n_dofs)

    # r = f - K*u
    K.compute_matvec(F.u, F.Ap, F.rho, F.Ke, F.edof, F.is_fixed,
                     E_min, penalty, n_elem, dim)
    # r = f - Ap
    K.axpy(-1.0, F.Ap, F.f, F.r)   # r = -1*Ap + f

    # z = M^{-1} r
    if use_jacobi:
        K.apply_diag_precond(F.diag, F.r, F.z)
    else:
        K.copy_field(F.r, F.z)

    # p = z
    K.copy_field(F.z, F.p)

    rz_old = float(K.dot_product(F.r, F.z))

    for it in range(max_iter):
        # Ap = K * p
        K.compute_matvec(F.p, F.Ap, F.rho, F.Ke, F.edof, F.is_fixed,
                         E_min, penalty, n_elem, dim)

        pAp = float(K.dot_product(F.p, F.Ap))
        if abs(pAp) < 1e-30:
            break

        alpha = rz_old / pAp

        # u = u + alpha * p
        K.axpy(alpha, F.p, F.u, F.u)
        # r = r - alpha * Ap
        K.axpy(-alpha, F.Ap, F.r, F.r)

        r_norm_sq = float(K.l2_norm_sq(F.r))
        if math.sqrt(r_norm_sq) < tol * f_norm:
            # Save solution for warm start
            K.copy_field(F.u, F.u_prev)
            return it + 1

        # z = M^{-1} r
        if use_jacobi:
            K.apply_diag_precond(F.diag, F.r, F.z)
        else:
            K.copy_field(F.r, F.z)

        rz_new = float(K.dot_product(F.r, F.z))
        beta = rz_new / (rz_old + 1e-60)

        # p = z + beta * p
        K.axpy(beta, F.p, F.z, F.p)

        rz_old = rz_new

    # Save solution for warm start even if not converged
    K.copy_field(F.u, F.u_prev)
    return max_iter


@ti.kernel
def _zero_fixed_dofs(u: ti.template(), is_fixed: ti.template(), n_dofs: int):
    """Set fixed DOF displacements to zero."""
    for i in range(n_dofs):
        if is_fixed[i] == 1:
            u[i] = 0.0
