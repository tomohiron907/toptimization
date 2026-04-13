"""Fused GPU kernels for PCG solver — Phase 1 optimization.

Reduces kernel launches and GPU-CPU syncs per CG iteration:
  Before: 8 kernels, 3 GPU-CPU syncs
  After:  3 kernels, 2 GPU-CPU syncs  (Jacobi / none paths)
          4 kernels, 2 GPU-CPU syncs  (block_jacobi path)

Kernel roles per CG iteration:
  matvec_and_dot_*:         Ap = K*p, returns p⋅Ap          (1 kernel, 1 sync)
  update_and_precond_jacobi: u+=αp, r-=αAp, z=r/diag,
                             returns r⋅z, writes ‖r‖² to
                             dot_result[None]                 (1 kernel, 1 sync)
  update_and_precond_none:  Same but z = r (no precond.)     (1 kernel, 1 sync)
  update_ur:                u+=αp, r-=αAp (block_jacobi path)(1 kernel, 0 sync)
  update_p:                 p = z + β*p                      (1 kernel, 0 sync)
"""

import taichi as ti


@ti.kernel
def matvec_and_dot_2d(
    p_in: ti.template(),
    Ap_out: ti.template(),
    rho: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    is_fixed: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
) -> ti.f32:
    """Ap = K(rho)*p; returns p⋅Ap. 2D mesh, 8 DOFs/element."""
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

    pAp = 0.0
    for i in Ap_out:
        pAp += p_in[i] * Ap_out[i]
    return pAp


@ti.kernel
def matvec_and_dot_3d(
    p_in: ti.template(),
    Ap_out: ti.template(),
    rho: ti.template(),
    Ke: ti.template(),
    edof: ti.template(),
    is_fixed: ti.template(),
    E_min: float,
    penalty: float,
    n_elem: int,
) -> ti.f32:
    """Ap = K(rho)*p; returns p⋅Ap. 3D mesh, 24 DOFs/element."""
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

    pAp = 0.0
    for i in Ap_out:
        pAp += p_in[i] * Ap_out[i]
    return pAp


def matvec_and_dot(
    p_in, Ap_out, rho, Ke, edof, is_fixed,
    E_min: float, penalty: float, n_elem: int, dim: int,
):
    """Dispatch matvec_and_dot on dimension. Returns p⋅Ap (scalar, triggers sync)."""
    if dim == 2:
        return matvec_and_dot_2d(
            p_in, Ap_out, rho, Ke, edof, is_fixed, E_min, penalty, n_elem)
    else:
        return matvec_and_dot_3d(
            p_in, Ap_out, rho, Ke, edof, is_fixed, E_min, penalty, n_elem)


@ti.kernel
def update_and_precond_jacobi(
    u: ti.template(),
    r: ti.template(),
    z: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
    diag: ti.template(),
    dot_result: ti.template(),
    alpha: float,
) -> ti.f32:
    """u+=α*p, r-=α*Ap, z=r/diag; returns r⋅z; writes ‖r‖² to dot_result[None].

    Fuses 4 operations (axpy×2 + apply_precond + dot) into 1 kernel.
    dot_result[None] can be read from Python after this call for ‖r‖².
    """
    for i in u:
        u[i] = u[i] + alpha * p[i]
        r[i] = r[i] - alpha * Ap[i]

    rz_new = 0.0
    r_norm_sq = 0.0
    for i in r:
        zi = r[i] / diag[i]
        z[i] = zi
        rz_new += r[i] * zi
        r_norm_sq += r[i] * r[i]

    dot_result[None] = r_norm_sq
    return rz_new


@ti.kernel
def update_and_precond_none(
    u: ti.template(),
    r: ti.template(),
    z: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
    dot_result: ti.template(),
    alpha: float,
) -> ti.f32:
    """u+=α*p, r-=α*Ap, z=r; returns r⋅z (=‖r‖²); writes ‖r‖² to dot_result[None]."""
    for i in u:
        u[i] = u[i] + alpha * p[i]
        r[i] = r[i] - alpha * Ap[i]

    rz_new = 0.0
    r_norm_sq = 0.0
    for i in r:
        z[i] = r[i]
        val = r[i] * r[i]
        rz_new += val
        r_norm_sq += val

    dot_result[None] = r_norm_sq
    return rz_new


@ti.kernel
def update_ur(
    u: ti.template(),
    r: ti.template(),
    p: ti.template(),
    Ap: ti.template(),
    alpha: float,
):
    """u[i]+=alpha*p[i], r[i]-=alpha*Ap[i]. No GPU-CPU sync. Used by block_jacobi path."""
    for i in u:
        u[i] = u[i] + alpha * p[i]
        r[i] = r[i] - alpha * Ap[i]


@ti.kernel
def update_p(z: ti.template(), p: ti.template(), beta: float):
    """p[i] = z[i] + beta*p[i]. No GPU-CPU sync."""
    for i in p:
        p[i] = z[i] + beta * p[i]
