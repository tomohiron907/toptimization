"""Taichi field allocation for FEA and optimization state.

All Taichi fields are allocated here as module-level globals so they are
shared across kernel calls without being re-created each iteration.
Fields are initialized by `allocate()` which must be called once after
`ti.init()`.
"""

from __future__ import annotations

import taichi as ti
import numpy as np

# Module-level field handles (set by allocate())
u: ti.Field = None         # displacement solution   (n_dofs,)
f: ti.Field = None         # external force vector   (n_dofs,)
r: ti.Field = None         # CG residual             (n_dofs,)
p: ti.Field = None         # CG search direction     (n_dofs,)
Ap: ti.Field = None        # K * p                   (n_dofs,)
z: ti.Field = None         # preconditioned residual (n_dofs,)
diag: ti.Field = None      # diagonal preconditioner (n_dofs,)
u_prev: ti.Field = None    # warm-start displacement (n_dofs,)

rho: ti.Field = None       # element density         (n_elem,)
rho_new: ti.Field = None   # updated density         (n_elem,)
dc: ti.Field = None        # compliance sensitivity  (n_elem,)
dc_filt: ti.Field = None   # filtered sensitivity    (n_elem,)
dv: ti.Field = None        # volume sensitivity      (n_elem,)
rho_filt: ti.Field = None  # filtered density        (n_elem,)

Ke: ti.Field = None        # element stiffness       (dpe, dpe)
Ke_inv: ti.Field = None    # element stiffness pinv  (dpe, dpe)  — block Jacobi
edof: ti.Field = None      # element connectivity    (n_elem, dpe)
is_fixed: ti.Field = None  # BC mask                 (n_dofs,)  1=fixed
diag_count: ti.Field = None  # elements per DOF      (n_dofs,)  — block Jacobi

# Filter precomputation fields
filt_neighbors: ti.Field = None   # (n_elem, max_nb)
filt_weights: ti.Field = None     # (n_elem, max_nb)
filt_n_nb: ti.Field = None        # (n_elem,)

# Scalar reduction fields
_dot_result: ti.Field = None
_max_result: ti.Field = None

# Dimensions (set by allocate)
N_DOFS: int = 0
N_ELEM: int = 0
DPE: int = 0    # DOFs per element (8 for 2D, 24 for 3D)
KE_ALPHA_REG: float = 0.0  # regularization diagonal used for Ke_inv


def allocate(
    n_dofs: int,
    n_elem: int,
    dpe: int,
    max_neighbors: int,
    Ke_np: np.ndarray,
    edof_np: np.ndarray,
    fixed_dofs: np.ndarray,
    force_dofs: np.ndarray,
    force_values: np.ndarray,
) -> None:
    """Allocate all Taichi fields and populate static data.

    Must be called once after ``ti.init()``.

    Parameters
    ----------
    n_dofs : int
    n_elem : int
    dpe : int
        DOFs per element.
    max_neighbors : int
        Maximum number of filter neighbors per element.
    Ke_np : ndarray, shape (dpe, dpe)
    edof_np : ndarray, shape (n_elem, dpe), int32
    fixed_dofs : ndarray, int
    force_dofs : ndarray, int
    force_values : ndarray, float
    """
    global u, f, r, p, Ap, z, diag, u_prev
    global rho, rho_new, dc, dc_filt, dv, rho_filt
    global Ke, Ke_inv, edof, is_fixed, diag_count
    global filt_neighbors, filt_weights, filt_n_nb
    global _dot_result, _max_result
    global N_DOFS, N_ELEM, DPE, KE_ALPHA_REG

    N_DOFS = n_dofs
    N_ELEM = n_elem
    DPE = dpe

    # FEA / solver fields
    u       = ti.field(dtype=ti.f32, shape=n_dofs)
    f       = ti.field(dtype=ti.f32, shape=n_dofs)
    r       = ti.field(dtype=ti.f32, shape=n_dofs)
    p       = ti.field(dtype=ti.f32, shape=n_dofs)
    Ap      = ti.field(dtype=ti.f32, shape=n_dofs)
    z       = ti.field(dtype=ti.f32, shape=n_dofs)
    diag    = ti.field(dtype=ti.f32, shape=n_dofs)
    u_prev  = ti.field(dtype=ti.f32, shape=n_dofs)

    # Optimization fields
    rho      = ti.field(dtype=ti.f32, shape=n_elem)
    rho_new  = ti.field(dtype=ti.f32, shape=n_elem)
    dc       = ti.field(dtype=ti.f32, shape=n_elem)
    dc_filt  = ti.field(dtype=ti.f32, shape=n_elem)
    dv       = ti.field(dtype=ti.f32, shape=n_elem)
    rho_filt = ti.field(dtype=ti.f32, shape=n_elem)

    # Element stiffness, pseudoinverse, and connectivity
    Ke       = ti.field(dtype=ti.f32, shape=(dpe, dpe))
    Ke_inv   = ti.field(dtype=ti.f32, shape=(dpe, dpe))
    edof     = ti.field(dtype=ti.i32, shape=(n_elem, dpe))

    # Boundary condition mask and block-Jacobi DOF count
    is_fixed   = ti.field(dtype=ti.i32, shape=n_dofs)
    diag_count = ti.field(dtype=ti.i32, shape=n_dofs)

    # Filter fields
    filt_neighbors = ti.field(dtype=ti.i32, shape=(n_elem, max_neighbors))
    filt_weights   = ti.field(dtype=ti.f32, shape=(n_elem, max_neighbors))
    filt_n_nb      = ti.field(dtype=ti.i32, shape=n_elem)

    # Reduction scalars
    _dot_result = ti.field(dtype=ti.f32, shape=())
    _max_result = ti.field(dtype=ti.f32, shape=())

    # Populate static data
    Ke.from_numpy(Ke_np.astype(np.float32))
    edof.from_numpy(edof_np.astype(np.int32))

    # Compute regularized element stiffness inverse.
    # Ke is rank-deficient (3 rigid body modes for Q4, 6 for H8).
    # Pseudoinverse maps those modes to zero, making the assembled block-Jacobi
    # preconditioner near-singular (condition number → ∞).
    # Adding alpha * diag_avg * I regularizes the null modes while preserving
    # the conditioning of deformation modes (alpha=0.1 gives ~3x cond improvement).
    Ke_np_f64 = Ke_np.astype(np.float64)
    alpha_reg = 0.1 * float(np.mean(np.abs(np.diag(Ke_np_f64))))
    KE_ALPHA_REG = alpha_reg
    Ke_inv_np = np.linalg.inv(Ke_np_f64 + alpha_reg * np.eye(dpe)).astype(np.float32)
    Ke_inv.from_numpy(Ke_inv_np)

    # Build is_fixed mask
    is_fixed_np = np.zeros(n_dofs, dtype=np.int32)
    is_fixed_np[fixed_dofs] = 1
    is_fixed.from_numpy(is_fixed_np)

    # Build force vector
    f_np = np.zeros(n_dofs, dtype=np.float32)
    for dof_idx, fval in zip(force_dofs, force_values):
        f_np[dof_idx] += fval
    # Zero out fixed DOFs in force vector
    f_np[fixed_dofs] = 0.0
    f.from_numpy(f_np)
