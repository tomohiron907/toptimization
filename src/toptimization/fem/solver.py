"""Preconditioned Conjugate Gradient (PCG) solver for FEA.

Supports two solver backends:
  "scipy"  — Assemble sparse K, solve with scipy.sparse.linalg (CPU).
             Best for problems <~200k DOFs. No GPU-CPU sync overhead.
  "taichi" — Matrix-free PCG using Taichi GPU kernels (fused).
             Best for very large problems (>200k DOFs).

The default is "scipy" which is dramatically faster for typical problem
sizes because it avoids the ~10ms GPU-CPU sync per CG iteration that
plagues the Taichi Metal backend.
"""

import math
import time
import taichi as ti
import numpy as np

from . import fields as F
from . import kernels as K
from . import fused_kernels as FK


def solve(
    E_min: float,
    penalty: float,
    dim: int,
    max_iter: int = 2000,
    tol: float = 1e-8,
    preconditioner: str = "jacobi",
    warm_start: bool = True,
    solver_mode: str = "scipy",
) -> int:
    """Solve K(rho)*u = f in-place (result stored in F.u).

    Parameters
    ----------
    E_min, penalty, dim : FEA parameters
    max_iter : int — max CG iterations (taichi) or max solver iterations (scipy)
    tol : float — convergence tolerance
    preconditioner : {"jacobi", "block_jacobi", "none"} — for taichi mode
    warm_start : bool — use F.u_prev as initial guess
    solver_mode : {"scipy", "taichi"} — which solver backend to use

    Returns
    -------
    n_iter : int — number of iterations (1 for direct solver)
    """
    if solver_mode == "scipy":
        return _solve_scipy(E_min, penalty, dim, tol, max_iter, warm_start)
    else:
        return _solve_taichi(E_min, penalty, dim, max_iter, tol,
                             preconditioner, warm_start)


def _solve_scipy(
    E_min: float,
    penalty: float,
    dim: int,
    tol: float,
    max_iter: int,
    warm_start: bool,
) -> int:
    """Solve using scipy sparse direct solver (spsolve)."""
    from .sparse_solve import solve_direct

    # Read current state from Taichi fields to numpy
    rho_np = F.rho.to_numpy()
    f_np = F.f.to_numpy()
    edof_np = F.edof.to_numpy()
    Ke_np = F.Ke.to_numpy().astype(np.float64)

    # Get fixed dofs from is_fixed field
    is_fixed_np = F.is_fixed.to_numpy()
    fixed_dofs = np.where(is_fixed_np == 1)[0]

    # Warm start
    u_prev = F.u_prev.to_numpy() if warm_start else None

    # Solve
    u_result, n_iter = solve_direct(
        rho_np=rho_np,
        Ke_np=Ke_np,
        edof_np=edof_np,
        f_np=f_np.astype(np.float64),
        fixed_dofs=fixed_dofs,
        n_dofs=F.N_DOFS,
        E_min=E_min,
        penalty=penalty,
        u_prev=u_prev,
    )

    # Write result back to Taichi fields
    F.u.from_numpy(u_result)
    K.copy_field(F.u, F.u_prev)

    return n_iter


def _solve_taichi(
    E_min: float,
    penalty: float,
    dim: int,
    max_iter: int,
    tol: float,
    preconditioner: str,
    warm_start: bool,
) -> int:
    """Solve using Taichi GPU matrix-free PCG (fused kernels).

    This is the original solver with Phase 1 fused kernel optimizations.
    Best for very large problems where GPU parallelism outweighs
    the sync overhead.
    """
    from . import preconditioner as PC

    n_dofs = F.N_DOFS
    n_elem = F.N_ELEM

    # Reference norm ||f||
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
    K.axpy(-1.0, F.Ap, F.f, F.r)

    # z = M^{-1} r
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

    # CG main loop (fused kernels)
    for it in range(max_iter):
        # Kernel 1: Ap = K*p, returns p·Ap
        pAp = float(FK.matvec_and_dot(
            F.p, F.Ap, F.rho, F.Ke, F.edof, F.is_fixed,
            E_min, penalty, n_elem, dim,
        ))
        if abs(pAp) < 1e-30:
            break

        alpha = rz_old / pAp

        if use_jacobi:
            rz_new = float(FK.update_and_precond_jacobi(
                F.u, F.r, F.z, F.p, F.Ap, F.diag, F._dot_result, alpha,
            ))
            r_norm_sq = float(F._dot_result[None])
        elif use_block:
            FK.update_ur(F.u, F.r, F.p, F.Ap, alpha)
            rz_new = float(PC.apply_and_rz(
                F.r, F.z, F.rho, F.Ke_inv, F.edof, F.diag_count, F.is_fixed, F._dot_result,
                E_min, penalty, F.KE_ALPHA_REG, n_elem, dim,
            ))
            r_norm_sq = float(F._dot_result[None])
        else:
            rz_new = float(FK.update_and_precond_none(
                F.u, F.r, F.z, F.p, F.Ap, F._dot_result, alpha,
            ))
            r_norm_sq = float(F._dot_result[None])

        if r_norm_sq < tol_sq:
            K.copy_field(F.u, F.u_prev)
            return it + 1

        beta = rz_new / (rz_old + 1e-60)
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
