"""Sparse stiffness matrix assembly and scipy-based solver.

Assembles K(rho) as a scipy CSR sparse matrix and solves Ku=f using
scipy.sparse.linalg.spsolve (direct) or cg (iterative).

This avoids the GPU-CPU sync overhead that dominates the Taichi-based
PCG solver for small-to-medium problems (~<200k DOFs).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def assemble_K(
    rho_np: np.ndarray,
    Ke_np: np.ndarray,
    edof_np: np.ndarray,
    fixed_dofs: np.ndarray,
    n_dofs: int,
    E_min: float,
    penalty: float,
) -> sparse.csc_matrix:
    """Assemble global stiffness matrix K(rho) as a sparse CSC matrix.

    Uses vectorised COO construction — no Python loops over elements.

    Parameters
    ----------
    rho_np : (n_elem,) float32/64
    Ke_np  : (dpe, dpe) float64 — reference element stiffness
    edof_np: (n_elem, dpe) int32 — element-to-DOF connectivity
    fixed_dofs : (n_fixed,) int — DOF indices with Dirichlet BC
    n_dofs : int — total degrees of freedom
    E_min  : float
    penalty: float

    Returns
    -------
    K : scipy.sparse.csc_matrix, shape (n_dofs, n_dofs)
    """
    n_elem, dpe = edof_np.shape

    # Element scaling: scale_e = E_min + rho_e^p * (1 - E_min)
    scale = E_min + np.power(rho_np, penalty) * (1.0 - E_min)  # (n_elem,)

    # Build COO triplets (vectorised over all elements)
    # For each element e, we have dpe*dpe entries: K[edof[e,i], edof[e,j]] += scale_e * Ke[i,j]
    # Row indices: edof[e, i] repeated dpe times for each j
    # Col indices: edof[e, j] repeated dpe times for each i
    row_local, col_local = np.meshgrid(np.arange(dpe), np.arange(dpe), indexing="ij")
    row_local_flat = row_local.ravel()  # (dpe*dpe,)
    col_local_flat = col_local.ravel()  # (dpe*dpe,)

    # Global row/col indices for all elements: (n_elem, dpe*dpe)
    rows = edof_np[:, row_local_flat]  # (n_elem, dpe*dpe)
    cols = edof_np[:, col_local_flat]  # (n_elem, dpe*dpe)

    # Values: scale_e * Ke[i,j] for all elements
    Ke_flat = Ke_np.ravel()  # (dpe*dpe,)
    vals = scale[:, None] * Ke_flat[None, :]  # (n_elem, dpe*dpe)

    # Flatten and assemble
    K = sparse.coo_matrix(
        (vals.ravel(), (rows.ravel(), cols.ravel())),
        shape=(n_dofs, n_dofs),
    ).tocsc()

    # Apply Dirichlet BCs: zero rows/cols of fixed DOFs, set diagonal to 1
    # This is the standard penalty/zeroing approach for sparse matrices
    if len(fixed_dofs) > 0:
        # Zero out rows and columns for fixed DOFs
        # Convert to lil for efficient row/col zeroing, then back to csc
        K = K.tolil()
        for dof in fixed_dofs:
            K[dof, :] = 0
            K[:, dof] = 0
            K[dof, dof] = 1.0
        K = K.tocsc()

    return K


def assemble_K_fast(
    rho_np: np.ndarray,
    Ke_np: np.ndarray,
    edof_np: np.ndarray,
    fixed_dofs: np.ndarray,
    n_dofs: int,
    E_min: float,
    penalty: float,
) -> sparse.csc_matrix:
    """Fast assembly using precomputed sparsity pattern.

    Same as assemble_K but uses a more efficient BC application
    via diagonal penalty method (avoids lil conversion).

    Parameters & Returns: same as assemble_K.
    """
    n_elem, dpe = edof_np.shape

    scale = E_min + np.power(rho_np, penalty) * (1.0 - E_min)

    row_local, col_local = np.meshgrid(np.arange(dpe), np.arange(dpe), indexing="ij")
    row_flat = row_local.ravel()
    col_flat = col_local.ravel()

    rows = edof_np[:, row_flat].ravel()
    cols = edof_np[:, col_flat].ravel()

    Ke_flat = Ke_np.ravel()
    vals = (scale[:, None] * Ke_flat[None, :]).ravel()

    K = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_dofs, n_dofs),
    )

    # Apply Dirichlet BCs via penalty method:
    # Zero rows and cols of fixed DOFs, set diagonal to a large value.
    # This is much faster than lil conversion.
    if len(fixed_dofs) > 0:
        # Get the maximum diagonal value for scaling
        diag_max = abs(K.diagonal()).max()
        penalty_val = diag_max * 1e6 if diag_max > 0 else 1.0

        # Zero rows
        for dof in fixed_dofs:
            K.data[K.indptr[dof]:K.indptr[dof+1]] = 0.0

        # Zero columns: work on transpose
        K = K.tocsc()
        for dof in fixed_dofs:
            K.data[K.indptr[dof]:K.indptr[dof+1]] = 0.0

        # Set diagonal
        K = K.tocsr()
        K[fixed_dofs, fixed_dofs] = penalty_val

    return K.tocsc()


def solve_direct(
    rho_np: np.ndarray,
    Ke_np: np.ndarray,
    edof_np: np.ndarray,
    f_np: np.ndarray,
    fixed_dofs: np.ndarray,
    n_dofs: int,
    E_min: float,
    penalty: float,
    u_prev: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Solve K(rho)*u = f using scipy sparse direct solver (Cholesky/LU).

    Parameters
    ----------
    rho_np : (n_elem,) density field
    Ke_np : (dpe, dpe) reference element stiffness
    edof_np : (n_elem, dpe) element-to-DOF connectivity
    f_np : (n_dofs,) force vector
    fixed_dofs : (n_fixed,) fixed DOF indices
    n_dofs : int
    E_min, penalty : SIMP parameters
    u_prev : (n_dofs,) optional warm-start (unused for direct solver)

    Returns
    -------
    u : (n_dofs,) solution
    n_iter : int — always 1 for direct solver
    """
    K = assemble_K_fast(rho_np, Ke_np, edof_np, fixed_dofs, n_dofs, E_min, penalty)

    # Zero force at fixed DOFs
    f_solve = f_np.copy()
    f_solve[fixed_dofs] = 0.0

    u = spla.spsolve(K, f_solve)

    # Ensure fixed DOFs are exactly zero
    u[fixed_dofs] = 0.0

    return u.astype(np.float32), 1


def solve_iterative(
    rho_np: np.ndarray,
    Ke_np: np.ndarray,
    edof_np: np.ndarray,
    f_np: np.ndarray,
    fixed_dofs: np.ndarray,
    n_dofs: int,
    E_min: float,
    penalty: float,
    u_prev: np.ndarray | None = None,
    tol: float = 1e-8,
    max_iter: int = 2000,
) -> tuple[np.ndarray, int]:
    """Solve K(rho)*u = f using scipy sparse CG with ILU preconditioning.

    Parameters
    ----------
    Same as solve_direct, plus:
    tol : float — relative tolerance
    max_iter : int — maximum CG iterations

    Returns
    -------
    u : (n_dofs,) solution
    n_iter : int — number of CG iterations
    """
    K = assemble_K_fast(rho_np, Ke_np, edof_np, fixed_dofs, n_dofs, E_min, penalty)

    f_solve = f_np.copy()
    f_solve[fixed_dofs] = 0.0

    x0 = u_prev if u_prev is not None else None

    # ILU preconditioning for faster convergence
    try:
        ilu = spla.spilu(K.tocsc(), drop_tol=1e-4)
        M = spla.LinearOperator(K.shape, matvec=ilu.solve)
    except Exception:
        M = None

    # Track iteration count
    iter_count = [0]
    def callback(xk):
        iter_count[0] += 1

    u, info = spla.cg(K, f_solve, x0=x0, tol=tol, maxiter=max_iter,
                      M=M, callback=callback)

    u[fixed_dofs] = 0.0

    return u.astype(np.float32), iter_count[0]
