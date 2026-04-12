"""Tests for YAML problem definition parser."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from toptimization.problem import load_problem


CANTILEVER_YAML = """\
problem:
  name: "Test Cantilever"
  dimension: 2

domain:
  nelx: 10
  nely: 5

material:
  E: 1.0
  nu: 0.3
  E_min: 1.0e-9

boundary_conditions:
  fixed:
    - nodes:
        type: "edge"
        edge: "left"
      dofs: ["x", "y"]

loads:
  - nodes:
      type: "point"
      location: [10, 2]
    force: [0.0, -1.0]

optimization:
  volume_fraction: 0.5
  penalty: 3.0
  filter_radius: 1.5
  filter_type: "density"
  max_iterations: 100
  tolerance: 0.01
  move_limit: 0.2

solver:
  max_cg_iterations: 500
  cg_tolerance: 1.0e-8
  preconditioner: "jacobi"

output:
  directory: "results/test"
  save_interval: 5
  format: "png"
"""


def write_yaml(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    f.write(content)
    f.close()
    return Path(f.name)


class TestLoadProblem:
    def setup_method(self):
        path = write_yaml(CANTILEVER_YAML)
        self.p = load_problem(path)

    def test_name(self):
        assert self.p.name == "Test Cantilever"

    def test_dim(self):
        assert self.p.dim == 2

    def test_mesh_size(self):
        assert self.p.nelx == 10
        assert self.p.nely == 5

    def test_n_nodes(self):
        assert self.p.n_nodes == 66

    def test_n_dofs(self):
        assert self.p.n_dofs == 132

    def test_n_elem(self):
        assert self.p.n_elem == 50

    def test_fixed_dofs_nonempty(self):
        assert len(self.p.fixed_dofs) > 0

    def test_fixed_dofs_left_edge(self):
        """Left edge has (nely+1)=6 nodes, each fixing 2 DOFs -> 12 fixed."""
        assert len(self.p.fixed_dofs) == 12

    def test_fixed_dofs_in_range(self):
        assert self.p.fixed_dofs.min() >= 0
        assert self.p.fixed_dofs.max() < self.p.n_dofs

    def test_force_dofs_nonempty(self):
        assert len(self.p.force_dofs) > 0

    def test_force_values_nonzero(self):
        assert np.any(self.p.force_values != 0)

    def test_volfrac(self):
        assert self.p.volfrac == pytest.approx(0.5)

    def test_free_dofs(self):
        all_dofs = np.union1d(self.p.free_dofs, self.p.fixed_dofs)
        np.testing.assert_array_equal(all_dofs, np.arange(self.p.n_dofs))

    def test_free_fixed_disjoint(self):
        common = np.intersect1d(self.p.free_dofs, self.p.fixed_dofs)
        assert len(common) == 0
