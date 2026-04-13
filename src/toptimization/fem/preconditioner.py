"""Block Jacobi (element-wise) preconditioner for PCG — Phase 3 optimization.

For a mesh of uniform elements with element stiffness Ke_e = scale_e * Ke_ref:
  (Ke_e)^+ = (1/scale_e) * Ke_ref^+

**Symmetric formulation** (required for PCG convergence):
  z[i] = (1/sqrt(count_i)) * sum_{e∋i} (1/scale_e) * (Ke_pinv * (r/sqrt(count))_e)[local_i]

The factor 1/sqrt(count_i) ensures the assembled M^{-1} is symmetric and PSD,
making PCG converge properly (the asymmetric version with simple 1/count averaging
breaks the M-inner-product needed by PCG).

The fused apply kernel also returns r⋅z and writes ‖r‖² to dot_result[None].
"""

import taichi as ti


@ti.kernel
def _compute_diag_count_2d(
    edof: ti.template(),
    diag_count: ti.template(),
    n_elem: int,
):
    """Count elements sharing each DOF. 2D mesh, 8 DOFs/element."""
    for i in diag_count:
        diag_count[i] = 0
    for e in range(n_elem):
        for i in ti.static(range(8)):
            ti.atomic_add(diag_count[edof[e, i]], 1)


@ti.kernel
def _compute_diag_count_3d(
    edof: ti.template(),
    diag_count: ti.template(),
    n_elem: int,
):
    """Count elements sharing each DOF. 3D mesh, 24 DOFs/element."""
    for i in diag_count:
        diag_count[i] = 0
    for e in range(n_elem):
        for i in ti.static(range(24)):
            ti.atomic_add(diag_count[edof[e, i]], 1)


@ti.kernel
def apply_and_rz_2d(
    r: ti.template(),
    z: ti.template(),
    rho: ti.template(),
    Ke_inv: ti.template(),
    edof: ti.template(),
    diag_count: ti.template(),
    is_fixed: ti.template(),
    dot_result: ti.template(),
    E_min: float,
    penalty: float,
    alpha_reg: float,
    n_elem: int,
) -> ti.f32:
    """Apply symmetric block Jacobi; returns r⋅z; writes ‖r‖² to dot_result[None].
    2D mesh, 8 DOFs/element.

    Symmetric formulation (sqrt-count scaling):
      1. Zero z
      2. Scatter: z[i] += (1/scale_clamped_e) * (Ke_inv * r_scaled_e)[local_i]
                  where r_scaled[j] = r[j] / sqrt(count_j)
                  and scale_clamped = max(scale_e, alpha_reg) to prevent void-element
                  dominance (inv_scale would be ~1e9 for void elements without clamping)
      3. Post-scale: z[i] /= sqrt(count_i); zero fixed DOFs
    """
    for i in z:
        z[i] = 0.0

    for e in range(n_elem):
        scale = E_min + ti.pow(rho[e], penalty) * (1.0 - E_min)
        # inv_scale = scale / max(scale, alpha_reg)^2
        #   solid (scale >= alpha_reg): = scale/scale^2 = 1/scale  (correct block-Jacobi weight)
        #   void  (scale << alpha_reg): ≈ scale/alpha_reg^2 → 0   (void elements don't dominate)
        # Without this, void elements (1/E_min ≈ 1e9) overwhelm solid (1/0.5 ≈ 2).
        s_c = ti.max(scale, alpha_reg)
        inv_scale = scale / (s_c * s_c)
        # r_scaled = r / sqrt(count)  (pre-scale for symmetry)
        r_local = ti.Matrix.zero(ti.f32, 8, 1)
        for i in ti.static(range(8)):
            dof = edof[e, i]
            r_local[i, 0] = r[dof] / ti.sqrt(ti.cast(diag_count[dof], ti.f32))
        z_local = ti.Matrix.zero(ti.f32, 8, 1)
        for i in ti.static(range(8)):
            for j in ti.static(range(8)):
                z_local[i, 0] += Ke_inv[i, j] * r_local[j, 0]
        for i in ti.static(range(8)):
            ti.atomic_add(z[edof[e, i]], inv_scale * z_local[i, 0])

    rz_new = 0.0
    r_norm_sq = 0.0
    for i in z:
        # Post-scale: z[i] /= sqrt(count_i)
        z[i] /= ti.sqrt(ti.cast(diag_count[i], ti.f32))
        # Fixed DOFs must stay zero to maintain p[fixed]=0 invariant
        if is_fixed[i] == 1:
            z[i] = 0.0
        rz_new += r[i] * z[i]
        r_norm_sq += r[i] * r[i]

    dot_result[None] = r_norm_sq
    return rz_new


@ti.kernel
def apply_and_rz_3d(
    r: ti.template(),
    z: ti.template(),
    rho: ti.template(),
    Ke_inv: ti.template(),
    edof: ti.template(),
    diag_count: ti.template(),
    is_fixed: ti.template(),
    dot_result: ti.template(),
    E_min: float,
    penalty: float,
    alpha_reg: float,
    n_elem: int,
) -> ti.f32:
    """Apply symmetric block Jacobi; returns r⋅z; writes ‖r‖² to dot_result[None].
    3D mesh, 24 DOFs/element."""
    for i in z:
        z[i] = 0.0

    for e in range(n_elem):
        scale = E_min + ti.pow(rho[e], penalty) * (1.0 - E_min)
        s_c = ti.max(scale, alpha_reg)
        inv_scale = scale / (s_c * s_c)
        r_local = ti.Matrix.zero(ti.f32, 24, 1)
        for i in ti.static(range(24)):
            dof = edof[e, i]
            r_local[i, 0] = r[dof] / ti.sqrt(ti.cast(diag_count[dof], ti.f32))
        z_local = ti.Matrix.zero(ti.f32, 24, 1)
        for i in ti.static(range(24)):
            for j in ti.static(range(24)):
                z_local[i, 0] += Ke_inv[i, j] * r_local[j, 0]
        for i in ti.static(range(24)):
            ti.atomic_add(z[edof[e, i]], inv_scale * z_local[i, 0])

    rz_new = 0.0
    r_norm_sq = 0.0
    for i in z:
        z[i] /= ti.sqrt(ti.cast(diag_count[i], ti.f32))
        if is_fixed[i] == 1:
            z[i] = 0.0
        rz_new += r[i] * z[i]
        r_norm_sq += r[i] * r[i]

    dot_result[None] = r_norm_sq
    return rz_new


def setup(edof, diag_count, n_elem: int, dim: int) -> None:
    """Compute diag_count (elements per DOF). Call once after field allocation."""
    if dim == 2:
        _compute_diag_count_2d(edof, diag_count, n_elem)
    else:
        _compute_diag_count_3d(edof, diag_count, n_elem)


def apply_and_rz(
    r, z, rho, Ke_inv, edof, diag_count, is_fixed, dot_result,
    E_min: float, penalty: float, alpha_reg: float, n_elem: int, dim: int,
):
    """Apply symmetric block Jacobi. Returns r⋅z; writes ‖r‖² to dot_result[None]."""
    if dim == 2:
        return apply_and_rz_2d(
            r, z, rho, Ke_inv, edof, diag_count, is_fixed, dot_result,
            E_min, penalty, alpha_reg, n_elem,
        )
    else:
        return apply_and_rz_3d(
            r, z, rho, Ke_inv, edof, diag_count, is_fixed, dot_result,
            E_min, penalty, alpha_reg, n_elem,
        )
