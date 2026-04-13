"""SIMP topology optimization main loop.

NOTE: All Taichi-dependent imports (kernels, fields, filter, oc, solver)
are performed inside run() AFTER ti.init() is called, because
@ti.kernel decorators require the Taichi runtime to be initialized.
"""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np


def run(
    problem,
    backend: str = "auto",
    live_viz: bool = True,
    viz_callback: Callable | None = None,
) -> np.ndarray:
    """Run SIMP topology optimization.

    Parameters
    ----------
    problem : Problem
    backend : str
        Taichi backend: "metal", "cpu", "cuda", "vulkan", or "auto".
    live_viz : bool
        Whether to show live matplotlib visualization.
    viz_callback : callable, optional
        Called each iteration with (rho_np, iteration, compliance, volume).

    Returns
    -------
    rho_np : ndarray, shape (nelx, nely) for 2D or (nelx, nely, nelz) for 3D
        Final density field.
    """
    import taichi as ti

    # ------------------------------------------------------------------ #
    # 1. Initialize Taichi FIRST (required before @ti.kernel decorations)
    # ------------------------------------------------------------------ #
    arch = _select_backend(backend)
    ti.init(arch=arch, default_fp=ti.f32, default_ip=ti.i32)
    print(f"[toptimization] Backend: {arch}")

    # ------------------------------------------------------------------ #
    # 2. Import Taichi-dependent modules AFTER ti.init()
    # ------------------------------------------------------------------ #
    from toptimization.fem import fields as F
    from toptimization.fem import kernels as K
    from toptimization.fem import solver as pcg
    from toptimization.fem import preconditioner as PC
    from toptimization.optimizer import filter as filt
    from toptimization.optimizer import oc as oc_mod
    from toptimization.mesh import build_edof, element_centers
    from toptimization.material import compute_Ke

    # ------------------------------------------------------------------ #
    # 3. Mesh and element stiffness
    # ------------------------------------------------------------------ #
    print(f"[toptimization] Building mesh ({problem.nelx}x{problem.nely})...")
    edof_np = build_edof(problem)
    centers = element_centers(problem)
    Ke_np = compute_Ke(problem.E, problem.nu, problem.dim)
    dpe = Ke_np.shape[0]  # 8 for 2D, 24 for 3D

    r_ceil = math.ceil(problem.rmin)
    if problem.dim == 2:
        max_nb = int((2 * r_ceil + 1) ** 2)
    else:
        max_nb = int((2 * r_ceil + 1) ** 3)

    # ------------------------------------------------------------------ #
    # 4. Allocate Taichi fields
    # ------------------------------------------------------------------ #
    print(f"[toptimization] Allocating fields "
          f"(n_dofs={problem.n_dofs:,}, n_elem={problem.n_elem:,})...")
    F.allocate(
        n_dofs=problem.n_dofs,
        n_elem=problem.n_elem,
        dpe=dpe,
        max_neighbors=max_nb,
        Ke_np=Ke_np,
        edof_np=edof_np,
        fixed_dofs=problem.fixed_dofs,
        force_dofs=problem.force_dofs,
        force_values=problem.force_values,
    )

    # ------------------------------------------------------------------ #
    # 5. Precompute filter and block Jacobi DOF count (if needed)
    # ------------------------------------------------------------------ #
    print(f"[toptimization] Precomputing filter (rmin={problem.rmin})...")
    filt.precompute_filter(problem, centers)

    if problem.preconditioner == "block_jacobi":
        print(f"[toptimization] Setting up block Jacobi preconditioner...")
        PC.setup(F.edof, F.diag_count, problem.n_elem, problem.dim)

    # ------------------------------------------------------------------ #
    # 6. Initialize density and warm-start displacement
    # ------------------------------------------------------------------ #
    F.rho.fill(problem.volfrac)
    K.fill_scalar(F.u_prev, 0.0)

    # ------------------------------------------------------------------ #
    # 7. Setup output directory
    # ------------------------------------------------------------------ #
    problem.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 8. Setup live visualization
    # ------------------------------------------------------------------ #
    axes = None
    if live_viz:
        try:
            import matplotlib
            matplotlib.use("TkAgg" if platform.system() != "Darwin" else "MacOSX")
            import matplotlib.pyplot as plt
            plt.ion()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.tight_layout()
        except Exception:
            live_viz = False

    # ------------------------------------------------------------------ #
    # 9. Main optimization loop
    # ------------------------------------------------------------------ #
    print(f"\n[toptimization] Starting: {problem.name}")
    print(f"{'Iter':>5} {'Compliance':>14} {'Volume':>8} {'Change':>9} {'CG':>6} {'s/iter':>7}")
    print("-" * 55)

    compliance_history: list[float] = []
    volume_history: list[float] = []

    for it in range(1, problem.max_iter + 1):
        t0 = time.perf_counter()

        if problem.filter_type == "density":
            # Density filter workflow:
            # 1. Filter density: rho_filt = filter(rho)
            # 2. FEA with rho_filt (physical density)
            # 3. Compute dc with rho_filt
            # 4. Project dc back: dc_filt[e] = sum_j H[e,j]*dc[j] / sum_j H[e,j]
            filt.apply_density_filter(
                F.rho, F.rho_filt,
                F.filt_neighbors, F.filt_weights, F.filt_n_nb,
                problem.n_elem,
            )
            _tmp_swap(K, F)  # F.rho <-> F.rho_filt (backup orig in rho_new)

            n_cg = pcg.solve(
                E_min=problem.E_min, penalty=problem.penalty, dim=problem.dim,
                max_iter=problem.max_cg_iter, tol=problem.cg_tol,
                preconditioner=problem.preconditioner, warm_start=(it > 1),
            )

            # Compute sensitivities with rho_filt (still in F.rho slot)
            compliance = K.compute_sensitivity(
                F.u, F.rho, F.dc, F.dv,
                F.Ke, F.edof,
                problem.E_min, problem.penalty, problem.n_elem, problem.dim,
            )
            _tmp_restore(K, F)  # restore original rho

            # Chain-rule projection: dc_design = H * dc_phys / Hs
            filt.apply_dc_filter_density_mode(problem.n_elem)

        else:
            # Sensitivity filter workflow:
            # 1. FEA with original rho
            # 2. Compute dc with original rho
            # 3. Sigmund's sensitivity filter
            n_cg = pcg.solve(
                E_min=problem.E_min, penalty=problem.penalty, dim=problem.dim,
                max_iter=problem.max_cg_iter, tol=problem.cg_tol,
                preconditioner=problem.preconditioner, warm_start=(it > 1),
            )
            compliance = K.compute_sensitivity(
                F.u, F.rho, F.dc, F.dv,
                F.Ke, F.edof,
                problem.E_min, problem.penalty, problem.n_elem, problem.dim,
            )
            filt.apply_dc_filter_sensitivity_mode(problem.n_elem)

        # --- OC density update ---
        max_change = oc_mod.oc_update(problem)
        volume = float(K.compute_volume(F.rho, problem.n_elem))

        elapsed = time.perf_counter() - t0
        compliance_history.append(compliance)
        volume_history.append(volume)

        print(f"{it:>5d} {compliance:>14.6f} {volume:>8.4f} {max_change:>9.4f} "
              f"{n_cg:>6d} {elapsed:>7.2f}")

        # --- Live visualization ---
        if live_viz and axes is not None and it % max(1, problem.save_interval // 5) == 0:
            import matplotlib.pyplot as plt
            _update_live_fig(axes, F.rho.to_numpy(), problem, it, compliance_history)
            plt.pause(0.001)

        # --- Save intermediate ---
        if it % problem.save_interval == 0:
            _save_iter_png(problem, F.rho.to_numpy(), it)

        # --- Optional user callback ---
        if viz_callback is not None:
            viz_callback(F.rho.to_numpy(), it, compliance, volume)

        # --- Convergence ---
        if it > 5 and max_change < problem.tol:
            print(f"\n[toptimization] Converged at iter {it} (change={max_change:.6f})")
            break

    print(f"\n[toptimization] Done.")

    rho_np = F.rho.to_numpy()
    _save_final(problem, rho_np, compliance_history, volume_history)

    if live_viz:
        import matplotlib.pyplot as plt
        plt.ioff()
        plt.show(block=True)

    return _reshape_result(rho_np, problem)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_backend(backend: str):
    import taichi as ti
    if backend == "auto":
        if platform.system() == "Darwin":
            return ti.metal
        try:
            return ti.cuda
        except Exception:
            return ti.cpu
    mapping = {"metal": ti.metal, "cuda": ti.cuda, "cpu": ti.cpu, "vulkan": ti.vulkan}
    if backend not in mapping:
        raise ValueError(f"Unknown backend '{backend}'")
    return mapping[backend]


def _tmp_swap(K, F):
    """Temporarily move rho_filt into rho (save original in rho_new)."""
    K.copy_field(F.rho, F.rho_new)      # rho_new = orig rho
    K.copy_field(F.rho_filt, F.rho)     # rho = filtered rho


def _tmp_restore(K, F):
    """Restore rho from rho_new."""
    K.copy_field(F.rho_new, F.rho)


def _update_live_fig(axes, rho_np, problem, it, compliance_hist):
    import matplotlib.pyplot as plt
    axes[0].cla()
    img = rho_np.reshape(problem.nelx, problem.nely).T
    axes[0].imshow(1.0 - img, cmap="gray", vmin=0, vmax=1,
                   origin="lower", aspect="equal")
    axes[0].set_title(f"iter {it}")
    axes[0].axis("off")

    axes[1].cla()
    axes[1].semilogy(compliance_hist, linewidth=1.2, color="steelblue")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Compliance")
    axes[1].set_title("Convergence")
    axes[1].grid(True, alpha=0.3)
    plt.gcf().tight_layout()


def _save_iter_png(problem, rho_np, it):
    if problem.dim != 2:
        return
    try:
        from toptimization.viz.plot2d import save_density_png
        rho_2d = rho_np.reshape(problem.nelx, problem.nely)
        save_density_png(rho_2d, problem.output_dir / f"iter_{it:04d}.png",
                         title=f"{problem.name} — iter {it}")
    except Exception:
        pass


def _save_final(problem, rho_np, compliance_hist, volume_hist):
    np.save(problem.output_dir / "density_final.npy", rho_np)
    if problem.dim == 2:
        try:
            from toptimization.viz.plot2d import save_density_png, save_convergence_png
            rho_2d = rho_np.reshape(problem.nelx, problem.nely)
            save_density_png(rho_2d, problem.output_dir / "density_final.png",
                             title=problem.name)
            save_convergence_png(compliance_hist, volume_hist,
                                 problem.output_dir / "convergence.png")
            print(f"[toptimization] Saved results to {problem.output_dir}/")
        except Exception as e:
            print(f"[warn] Save failed: {e}")


def _reshape_result(rho_np, problem) -> np.ndarray:
    if problem.dim == 2:
        return rho_np.reshape(problem.nelx, problem.nely)
    return rho_np.reshape(problem.nelx, problem.nely, problem.nelz)
