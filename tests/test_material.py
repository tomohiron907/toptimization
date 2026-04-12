"""Tests for element stiffness matrix computation."""

import numpy as np
import pytest
from toptimization.material import compute_Ke_2d, compute_Ke_3d


class TestKe2D:
    def setup_method(self):
        self.Ke = compute_Ke_2d(E=1.0, nu=0.3)

    def test_shape(self):
        assert self.Ke.shape == (8, 8)

    def test_symmetry(self):
        np.testing.assert_allclose(self.Ke, self.Ke.T, atol=1e-12)

    def test_positive_semidefinite(self):
        eigvals = np.linalg.eigvalsh(self.Ke)
        assert np.all(eigvals >= -1e-10), f"Negative eigenvalue: {eigvals.min()}"

    def test_rank(self):
        """Q4 element has 3 rigid body modes -> rank = 8 - 3 = 5."""
        rank = np.linalg.matrix_rank(self.Ke, tol=1e-10)
        assert rank == 5, f"Expected rank 5, got {rank}"

    def test_scales_with_E(self):
        Ke2 = compute_Ke_2d(E=2.0, nu=0.3)
        np.testing.assert_allclose(Ke2, 2.0 * self.Ke, rtol=1e-10)

    def test_known_diagonal_entry(self):
        assert np.all(np.diag(self.Ke) > 0)


class TestKe3D:
    def setup_method(self):
        self.Ke = compute_Ke_3d(E=1.0, nu=0.3)

    def test_shape(self):
        assert self.Ke.shape == (24, 24)

    def test_symmetry(self):
        np.testing.assert_allclose(self.Ke, self.Ke.T, atol=1e-12)

    def test_positive_semidefinite(self):
        eigvals = np.linalg.eigvalsh(self.Ke)
        assert np.all(eigvals >= -1e-8), f"Negative eigenvalue: {eigvals.min()}"

    def test_rank(self):
        """H8 element has 6 rigid body modes -> rank = 24 - 6 = 18."""
        rank = np.linalg.matrix_rank(self.Ke, tol=1e-8)
        assert rank == 18, f"Expected rank 18, got {rank}"
