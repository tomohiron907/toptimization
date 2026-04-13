"""Tests for structured mesh indexing."""

import numpy as np
import pytest
from toptimization.mesh import build_edof_2d, element_centers_2d
from toptimization.problem import Problem
from pathlib import Path


def make_problem_2d(nelx=4, nely=3):
    return Problem(
        name="test",
        dim=2,
        nelx=nelx,
        nely=nely,
        nelz=1,
        E=1.0, nu=0.3, E_min=1e-9,
        fixed_dofs=np.array([0, 1], dtype=int),
        force_dofs=np.array([2 * (nelx + 1) * (nely + 1) - 1], dtype=int),
        force_values=np.array([-1.0]),
        volfrac=0.5, penalty=3.0, rmin=1.5,
        filter_type="density", max_iter=100, tol=0.01, move_limit=0.2,
        max_cg_iter=500, cg_tol=1e-8, preconditioner="jacobi",
        solver_mode="scipy",
        output_dir=Path("results"),
        save_interval=10, output_format="png",
    )


class TestEdof2D:
    def setup_method(self):
        self.nelx, self.nely = 4, 3
        self.p = make_problem_2d(self.nelx, self.nely)
        self.edof = build_edof_2d(self.p)

    def test_shape(self):
        n_elem = self.nelx * self.nely
        assert self.edof.shape == (n_elem, 8)

    def test_dtype(self):
        assert self.edof.dtype == np.int32

    def test_dof_range(self):
        """All DOF indices must be in [0, n_dofs)."""
        n_dofs = 2 * (self.nelx + 1) * (self.nely + 1)
        assert self.edof.min() >= 0
        assert self.edof.max() < n_dofs

    def test_element_zero(self):
        """Element (0,0) bottom-left corner: nodes n1=(0,0), n2=(1,0), n3=(1,1), n4=(0,1)."""
        nely = self.nely
        n1 = 0 * (nely + 1) + 0
        n2 = 1 * (nely + 1) + 0
        n3 = 1 * (nely + 1) + 1
        n4 = 0 * (nely + 1) + 1
        expected = np.array([
            2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1, 2*n4, 2*n4+1
        ], dtype=np.int32)
        np.testing.assert_array_equal(self.edof[0], expected)

    def test_unique_per_element(self):
        """Each element should reference 8 distinct DOFs."""
        for e in range(self.nelx * self.nely):
            assert len(set(self.edof[e])) == 8, f"Element {e} has duplicate DOFs"

    def test_adjacent_elements_share_dofs(self):
        """Two horizontally adjacent elements share 2 nodes (4 DOFs)."""
        e0 = 0  # element (0,0)
        e1 = self.nely  # element (1,0)
        shared = set(self.edof[e0].tolist()) & set(self.edof[e1].tolist())
        assert len(shared) == 4, f"Expected 4 shared DOFs, got {len(shared)}"


class TestElementCenters2D:
    def test_shape(self):
        p = make_problem_2d(4, 3)
        centers = element_centers_2d(p)
        assert centers.shape == (12, 2)

    def test_first_element_center(self):
        p = make_problem_2d(4, 3)
        centers = element_centers_2d(p)
        np.testing.assert_allclose(centers[0], [0.5, 0.5], atol=1e-6)

    def test_last_element_center(self):
        p = make_problem_2d(4, 3)
        centers = element_centers_2d(p)
        e = 3 * 3 + 2
        np.testing.assert_allclose(centers[e], [3.5, 2.5], atol=1e-6)
