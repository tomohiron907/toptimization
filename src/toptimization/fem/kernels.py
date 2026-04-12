"""Taichi GPU kernels for matrix-free FEA.

All kernels operate on the global fields declared in fields.py.
They are compiled by Taichi's JIT on first call.

Key design:
  - compute_matvec: element-by-element K*p without assembling K
  - atomic_add for scatter (DOF sharing between elements)
  - ti.static(range(DPE)) unrolls inner loops at compile time
  - Boundary conditions: fixed DOFs get identity rows (Ap[i] = p[i])
"""

import taichi as ti
import numpy as np

from . import fields as F


@ti.kernel
def fill_scalar(field: ti.template(), val: float):
    """Fill every element of a 1D field with val."""
    for i in field:
        field[i] = val


@ti.kernel
def copy_field(src: ti.template(), dst: ti.template()):
    """dst[i] = src[i]"""
    for i in src:
        dst[i] = src[i]


@ti.kernel
def axpy(a: float, x: ti.template(), y: ti.template(), result: ti.template()):
    """result[i] = a * x[i] + y[i]"""
    for i in result:
        result[i] = a * x[i] + y[i]


@ti.kernel
def scale_field(a: float, x: ti.template(), result: ti.template()):
    """result[i] = a * x[i]"""
    for i in result:
        result[i] = a * x[i]


@ti.kernel
def dot_product(a: ti.template(), b: ti.template()) -> ti.f32:
    """Parallel dot product sum(a[i]*b[i])."""
    result = 0.0
    for i in a:
        result += a[i] * b[i]
    return result


@ti.kernel
def l2_norm_sq(a: ti.template()) -> ti.f32:
    """Squared L2 norm: sum(a[i]^2)."""
    result = 0.0
    for i in a:
        result += a[i] * a[i]
    return result


@ti.kernel
def apply_diag_precond(
    diag: ti.template(),
    r: ti.template(),
    z: ti.template(),
):
    """z[i] = r[i] / diag[i]  (Jacobi preconditioner application)."""
    for i in z:
        z[i] = r[i] / diag[i]


@ti.kernel
def compute_diagonal_2d(
    diag: ti.template(),
    rho: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    is_fixed: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
):
    """Compute the diagonal of K(rho) for 2D mesh (8 DOFs per element)."""
    for i in diag:
        diag[i] = 0.0

    for e in range(n_elem):
        scale = E_min + ti.pow(rho[e], penalty) * (1.0 - E_min)
        for i in ti.static(range(8)):
            dof_i = edof[e, i]
            ti.atomic_add(diag[dof_i], scale * Ke[i, i])

    for i in diag:
        if is_fixed[i] == 1:
            diag[i] = 1.0
        elif diag[i] < 1e-20:
            diag[i] = 1e-20


@ti.kernel
def compute_diagonal_3d(
    diag: ti.template(),
    rho: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    is_fixed: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
):
    """Compute the diagonal of K(rho) for 3D mesh (24 DOFs per element)."""
    for i in diag:
        diag[i] = 0.0

    for e in range(n_elem):
        scale = E_min + ti.pow(rho[e], penalty) * (1.0 - E_min)
        for i in ti.static(range(24)):
            dof_i = edof[e, i]
            ti.atomic_add(diag[dof_i], scale * Ke[i, i])

    for i in diag:
        if is_fixed[i] == 1:
            diag[i] = 1.0
        elif diag[i] < 1e-20:
            diag[i] = 1e-20


def compute_diagonal(
    diag, rho, Ke, edof, is_fixed,
    E_min: float, penalty: float, n_elem: int, dim: int,
) -> None:
    """Dispatch diagonal computation on dimension."""
    if dim == 2:
        compute_diagonal_2d(diag, rho, Ke, edof, is_fixed, E_min, penalty, n_elem)
    else:
        compute_diagonal_3d(diag, rho, Ke, edof, is_fixed, E_min, penalty, n_elem)


@ti.kernel
def compute_matvec_2d(
    p_in: ti.template(),
    Ap_out: ti.template(),
    rho: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    is_fixed: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
):
    """Matrix-free K(rho)*p for 2D mesh (8 DOFs per element)."""
    for i in Ap_out:
        Ap_out[i] = 0.0

    for e in range(n_elem):
        scale = E_min + ti.pow(rho[e], penalty) * (1.0 - E_min)

        p_local = ti.Matrix.zero(ti.f32, 8, 1)
        for i in ti.static(range(8)):
            p_local[i, 0] = p_in[edof[e, i]]

        Kp_local = ti.Matrix.zero(ti.f32, 8, 1)
        for i in ti.static(range(8)):
            for j in ti.static(range(8)):
                Kp_local[i, 0] += Ke[i, j] * p_local[j, 0]

        for i in ti.static(range(8)):
            ti.atomic_add(Ap_out[edof[e, i]], scale * Kp_local[i, 0])

    for i in Ap_out:
        if is_fixed[i] == 1:
            Ap_out[i] = p_in[i]


@ti.kernel
def compute_matvec_3d(
    p_in: ti.template(),
    Ap_out: ti.template(),
    rho: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    is_fixed: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
):
    """Matrix-free K(rho)*p for 3D mesh (24 DOFs per element)."""
    for i in Ap_out:
        Ap_out[i] = 0.0

    for e in range(n_elem):
        scale = E_min + ti.pow(rho[e], penalty) * (1.0 - E_min)

        p_local = ti.Matrix.zero(ti.f32, 24, 1)
        for i in ti.static(range(24)):
            p_local[i, 0] = p_in[edof[e, i]]

        Kp_local = ti.Matrix.zero(ti.f32, 24, 1)
        for i in ti.static(range(24)):
            for j in ti.static(range(24)):
                Kp_local[i, 0] += Ke[i, j] * p_local[j, 0]

        for i in ti.static(range(24)):
            ti.atomic_add(Ap_out[edof[e, i]], scale * Kp_local[i, 0])

    for i in Ap_out:
        if is_fixed[i] == 1:
            Ap_out[i] = p_in[i]


def compute_matvec(
    p_in: "ti.Field",
    Ap_out: "ti.Field",
    rho: "ti.Field",
    Ke: "ti.Field",
    edof: "ti.Field",
    is_fixed: "ti.Field",
    E_min: float,
    penalty: float,
    n_elem: int,
    dim: int,
) -> None:
    """Dispatch matvec kernel on dimension."""
    if dim == 2:
        compute_matvec_2d(p_in, Ap_out, rho, Ke, edof, is_fixed, E_min, penalty, n_elem)
    else:
        compute_matvec_3d(p_in, Ap_out, rho, Ke, edof, is_fixed, E_min, penalty, n_elem)


@ti.kernel
def compute_sensitivity_2d(
    u: ti.template(),
    rho: ti.template(),
    dc: ti.template(),
    dv: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
) -> ti.f32:
    """Compute compliance and sensitivities for 2D.

    dc[e] = -p * rho[e]^(p-1) * u_e^T * Ke * u_e
    dv[e] = 1.0
    Returns total compliance.
    """
    compliance = 0.0
    for e in range(n_elem):
        u_local = ti.Matrix.zero(ti.f32, 8, 1)
        for i in ti.static(range(8)):
            u_local[i, 0] = u[edof[e, i]]

        Ku_local = ti.Matrix.zero(ti.f32, 8, 1)
        for i in ti.static(range(8)):
            for j in ti.static(range(8)):
                Ku_local[i, 0] += Ke[i, j] * u_local[j, 0]

        uKu = 0.0
        for i in ti.static(range(8)):
            uKu += u_local[i, 0] * Ku_local[i, 0]

        rho_e = rho[e]
        scale = E_min + ti.pow(rho_e, penalty) * (1.0 - E_min)
        dc[e] = -penalty * ti.pow(rho_e, penalty - 1.0) * (1.0 - E_min) * uKu
        dv[e] = 1.0
        compliance += scale * uKu
    return compliance


@ti.kernel
def compute_sensitivity_3d(
    u: ti.template(),
    rho: ti.template(),
    dc: ti.template(),
    dv: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
) -> ti.f32:
    """Compute compliance and sensitivities for 3D."""
    compliance = 0.0
    for e in range(n_elem):
        u_local = ti.Matrix.zero(ti.f32, 24, 1)
        for i in ti.static(range(24)):
            u_local[i, 0] = u[edof[e, i]]

        Ku_local = ti.Matrix.zero(ti.f32, 24, 1)
        for i in ti.static(range(24)):
            for j in ti.static(range(24)):
                Ku_local[i, 0] += Ke[i, j] * u_local[j, 0]

        uKu = 0.0
        for i in ti.static(range(24)):
            uKu += u_local[i, 0] * Ku_local[i, 0]

        rho_e = rho[e]
        scale = E_min + ti.pow(rho_e, penalty) * (1.0 - E_min)
        dc[e] = -penalty * ti.pow(rho_e, penalty - 1.0) * (1.0 - E_min) * uKu
        dv[e] = 1.0
        compliance += scale * uKu
    return compliance


def compute_sensitivity(
    u: "ti.Field",
    rho: "ti.Field",
    dc: "ti.Field",
    dv: "ti.Field",
    Ke: "ti.Field",
    edof: "ti.Field",
    E_min: float,
    penalty: float,
    n_elem: int,
    dim: int,
) -> float:
    """Dispatch sensitivity kernel on dimension."""
    if dim == 2:
        return float(compute_sensitivity_2d(u, rho, dc, dv, Ke, edof, E_min, penalty, n_elem))
    else:
        return float(compute_sensitivity_3d(u, rho, dc, dv, Ke, edof, E_min, penalty, n_elem))


@ti.kernel
def compute_volume(rho: ti.template(), n_elem: int) -> ti.f32:
    """Compute mean volume fraction: sum(rho) / n_elem."""
    total = 0.0
    for e in range(n_elem):
        total += rho[e]
    return total / n_elem


@ti.kernel
def max_density_change(rho: ti.template(), rho_new: ti.template(), n_elem: int) -> ti.f32:
    """max |rho_new[e] - rho[e]|."""
    result = 0.0
    for e in range(n_elem):
        ti.atomic_max(result, ti.abs(rho_new[e] - rho[e]))
    return result
