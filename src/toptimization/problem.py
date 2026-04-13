"""Problem definition: YAML parser and validation.

Converts a human-readable YAML problem file into concrete numerical arrays
consumed by the FEA solver and optimizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import yaml


@dataclass
class Problem:
    """All numerical parameters needed to run a topology optimization."""

    name: str
    dim: Literal[2, 3]

    # Mesh
    nelx: int
    nely: int
    nelz: int  # 1 for 2D

    # Material
    E: float
    nu: float
    E_min: float

    # Boundary conditions (global DOF indices)
    fixed_dofs: np.ndarray  # shape (n_fixed,), dtype int

    # Loads (global DOF indices and force values)
    force_dofs: np.ndarray   # shape (n_force,), dtype int
    force_values: np.ndarray  # shape (n_force,), dtype float

    # Optimization parameters
    volfrac: float
    penalty: float
    rmin: float
    filter_type: Literal["density", "sensitivity"]
    max_iter: int
    tol: float
    move_limit: float

    # Solver parameters
    max_cg_iter: int
    cg_tol: float
    preconditioner: Literal["jacobi", "block_jacobi", "none"]

    # Output
    output_dir: Path
    save_interval: int
    output_format: Literal["png", "vtk"]

    # Derived quantities (set post-init)
    n_nodes: int = field(init=False)
    n_dofs: int = field(init=False)
    n_elem: int = field(init=False)
    free_dofs: np.ndarray = field(init=False)
    dofs_per_node: int = field(init=False)

    def __post_init__(self) -> None:
        self.dofs_per_node = self.dim
        if self.dim == 2:
            self.n_nodes = (self.nelx + 1) * (self.nely + 1)
            self.n_elem = self.nelx * self.nely
        else:
            self.n_nodes = (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)
            self.n_elem = self.nelx * self.nely * self.nelz
        self.n_dofs = self.dofs_per_node * self.n_nodes
        all_dofs = np.arange(self.n_dofs, dtype=int)
        self.free_dofs = np.setdiff1d(all_dofs, self.fixed_dofs)


def load_problem(yaml_path: str | Path) -> Problem:
    """Parse and validate a YAML problem definition file."""
    yaml_path = Path(yaml_path)
    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    prob = raw.get("problem", {})
    domain = raw.get("domain", {})
    mat = raw.get("material", {})
    opt = raw.get("optimization", {})
    solver = raw.get("solver", {})
    output = raw.get("output", {})

    name = prob.get("name", "Topology Optimization")
    dim = int(prob.get("dimension", 2))
    if dim not in (2, 3):
        raise ValueError(f"dimension must be 2 or 3, got {dim}")

    nelx = int(domain["nelx"])
    nely = int(domain["nely"])
    nelz = int(domain.get("nelz", 1)) if dim == 3 else 1

    E = float(mat.get("E", 1.0))
    nu = float(mat.get("nu", 0.3))
    E_min = float(mat.get("E_min", 1e-9))

    volfrac = float(opt.get("volume_fraction", 0.5))
    penalty = float(opt.get("penalty", 3.0))
    rmin = float(opt.get("filter_radius", 1.5))
    filter_type = opt.get("filter_type", "density")
    if filter_type not in ("density", "sensitivity"):
        raise ValueError(f"filter_type must be 'density' or 'sensitivity', got {filter_type}")
    max_iter = int(opt.get("max_iterations", 200))
    tol = float(opt.get("tolerance", 0.01))
    move_limit = float(opt.get("move_limit", 0.2))

    max_cg_iter = int(solver.get("max_cg_iterations", 2000))
    cg_tol = float(solver.get("cg_tolerance", 1e-6))
    preconditioner = solver.get("preconditioner", "jacobi")
    if preconditioner not in ("jacobi", "block_jacobi", "none"):
        raise ValueError(
            f"preconditioner must be 'jacobi', 'block_jacobi', or 'none', got {preconditioner}"
        )

    output_dir = Path(output.get("directory", "results"))
    save_interval = int(output.get("save_interval", 10))
    output_format = output.get("format", "png")
    if output_format not in ("png", "vtk"):
        raise ValueError(f"output format must be 'png' or 'vtk', got {output_format}")

    # Parse boundary conditions into DOF index arrays
    bc_specs = raw.get("boundary_conditions", {})
    fixed_dofs = _parse_fixed_dofs(bc_specs, nelx, nely, nelz, dim)

    # Parse loads into DOF index and value arrays
    load_specs = raw.get("loads", [])
    force_dofs, force_values = _parse_loads(load_specs, nelx, nely, nelz, dim)

    return Problem(
        name=name,
        dim=dim,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        E=E,
        nu=nu,
        E_min=E_min,
        fixed_dofs=fixed_dofs,
        force_dofs=force_dofs,
        force_values=force_values,
        volfrac=volfrac,
        penalty=penalty,
        rmin=rmin,
        filter_type=filter_type,
        max_iter=max_iter,
        tol=tol,
        move_limit=move_limit,
        max_cg_iter=max_cg_iter,
        cg_tol=cg_tol,
        preconditioner=preconditioner,
        output_dir=output_dir,
        save_interval=save_interval,
        output_format=output_format,
    )


def _node_index_2d(ix: int, iy: int, nelx: int, nely: int) -> int:
    """Global node index from (ix, iy) grid position (0-based).

    Nodes are numbered column-major: node (ix, iy) -> ix*(nely+1) + iy.
    ix in [0, nelx], iy in [0, nely].
    """
    return ix * (nely + 1) + iy


def _parse_fixed_dofs(
    bc_specs: dict,
    nelx: int,
    nely: int,
    nelz: int,
    dim: int,
) -> np.ndarray:
    """Convert boundary condition specs to array of fixed global DOF indices."""
    fixed_set: set[int] = set()
    nely1 = nely + 1
    nelx1 = nelx + 1

    for entry in bc_specs.get("fixed", []):
        node_spec = entry["nodes"]
        dof_labels = entry.get("dofs", ["x", "y"])
        dof_offsets = _dof_labels_to_offsets(dof_labels, dim)

        node_indices = _resolve_node_spec(node_spec, nelx, nely, nelz, dim)
        for ni in node_indices:
            for off in dof_offsets:
                fixed_set.add(dim * ni + off)

    return np.array(sorted(fixed_set), dtype=int)


def _parse_loads(
    load_specs: list,
    nelx: int,
    nely: int,
    nelz: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert load specs to (force_dofs, force_values) arrays."""
    dof_list: list[int] = []
    val_list: list[float] = []

    for entry in load_specs:
        node_spec = entry["nodes"]
        force_vec = entry["force"]
        if len(force_vec) != dim:
            raise ValueError(f"force vector length {len(force_vec)} != dim {dim}")

        node_indices = _resolve_node_spec(node_spec, nelx, nely, nelz, dim)
        for ni in node_indices:
            for d, fval in enumerate(force_vec):
                if fval != 0.0:
                    dof_list.append(dim * ni + d)
                    val_list.append(float(fval))

    if not dof_list:
        raise ValueError("No nonzero forces specified in problem definition")

    return np.array(dof_list, dtype=int), np.array(val_list, dtype=float)


def _dof_labels_to_offsets(labels: list[str], dim: int) -> list[int]:
    label_map = {"x": 0, "y": 1, "z": 2}
    offsets = []
    for label in labels:
        if label not in label_map:
            raise ValueError(f"Unknown DOF label '{label}'. Use 'x', 'y', or 'z'.")
        off = label_map[label]
        if off >= dim:
            raise ValueError(f"DOF '{label}' not valid for {dim}D problem")
        offsets.append(off)
    return offsets


def _resolve_node_spec(
    spec: dict,
    nelx: int,
    nely: int,
    nelz: int,
    dim: int,
) -> list[int]:
    """Resolve a node specification dict to a list of global node indices."""
    spec_type = spec["type"]

    if spec_type == "edge" and dim == 2:
        return _nodes_on_edge_2d(spec["edge"], nelx, nely)
    elif spec_type == "face" and dim == 3:
        return _nodes_on_face_3d(spec["face"], nelx, nely, nelz)
    elif spec_type == "point":
        return [_resolve_point_node(spec, nelx, nely, nelz, dim)]
    elif spec_type == "region":
        return _nodes_in_region(spec, nelx, nely, nelz, dim)
    else:
        raise ValueError(f"Unknown node spec type '{spec_type}' for dim={dim}")


def _nodes_on_edge_2d(edge: str, nelx: int, nely: int) -> list[int]:
    """All node indices on a specified edge of the 2D domain."""
    nely1 = nely + 1
    if edge == "left":
        return [_node_index_2d(0, iy, nelx, nely) for iy in range(nely + 1)]
    elif edge == "right":
        return [_node_index_2d(nelx, iy, nelx, nely) for iy in range(nely + 1)]
    elif edge == "bottom":
        return [_node_index_2d(ix, 0, nelx, nely) for ix in range(nelx + 1)]
    elif edge == "top":
        return [_node_index_2d(ix, nely, nelx, nely) for ix in range(nelx + 1)]
    else:
        raise ValueError(f"Unknown edge '{edge}'. Use 'left', 'right', 'top', 'bottom'.")


def _nodes_on_face_3d(face: str, nelx: int, nely: int, nelz: int) -> list[int]:
    nely1 = nely + 1
    nelz1 = nelz + 1
    nodes = []
    if face == "left":
        for iy in range(nely + 1):
            for iz in range(nelz + 1):
                nodes.append(_node_index_3d(0, iy, iz, nely, nelz))
    elif face == "right":
        for iy in range(nely + 1):
            for iz in range(nelz + 1):
                nodes.append(_node_index_3d(nelx, iy, iz, nely, nelz))
    elif face == "bottom":
        for ix in range(nelx + 1):
            for iz in range(nelz + 1):
                nodes.append(_node_index_3d(ix, 0, iz, nely, nelz))
    elif face == "top":
        for ix in range(nelx + 1):
            for iz in range(nelz + 1):
                nodes.append(_node_index_3d(ix, nely, iz, nely, nelz))
    elif face == "front":
        for ix in range(nelx + 1):
            for iy in range(nely + 1):
                nodes.append(_node_index_3d(ix, iy, 0, nely, nelz))
    elif face == "back":
        for ix in range(nelx + 1):
            for iy in range(nely + 1):
                nodes.append(_node_index_3d(ix, iy, nelz, nely, nelz))
    else:
        raise ValueError(f"Unknown face '{face}'.")
    return nodes


def _node_index_3d(ix: int, iy: int, iz: int, nely: int, nelz: int) -> int:
    return ix * (nely + 1) * (nelz + 1) + iy * (nelz + 1) + iz


def _resolve_point_node(
    spec: dict,
    nelx: int,
    nely: int,
    nelz: int,
    dim: int,
) -> int:
    """Resolve a point node spec to a global node index."""
    if "location" in spec:
        loc = spec["location"]
        if dim == 2:
            # loc = [x_elem, y_elem] -> node at that element's corner
            # We interpret location as node grid coordinates (0..nelx, 0..nely)
            ix, iy = int(loc[0]), int(loc[1])
            ix = min(ix, nelx)
            iy = min(iy, nely)
            return _node_index_2d(ix, iy, nelx, nely)
        else:
            ix, iy, iz = int(loc[0]), int(loc[1]), int(loc[2])
            return _node_index_3d(ix, iy, iz, nely, nelz)
    elif "corner" in spec:
        return _resolve_corner(spec["corner"], nelx, nely, nelz, dim)
    else:
        raise ValueError("Point node spec requires 'location' or 'corner'")


def _resolve_corner(corner: str, nelx: int, nely: int, nelz: int, dim: int) -> int:
    """Resolve a named corner to a global node index."""
    if dim == 2:
        corners = {
            "bottom-left": (0, 0),
            "bottom-right": (nelx, 0),
            "top-left": (0, nely),
            "top-right": (nelx, nely),
        }
        if corner not in corners:
            raise ValueError(f"Unknown 2D corner '{corner}'")
        ix, iy = corners[corner]
        return _node_index_2d(ix, iy, nelx, nely)
    else:
        corners = {
            "front-bottom-left": (0, 0, 0),
            "front-bottom-right": (nelx, 0, 0),
            "back-bottom-left": (0, 0, nelz),
            "back-bottom-right": (nelx, 0, nelz),
        }
        if corner not in corners:
            raise ValueError(f"Unknown 3D corner '{corner}'")
        ix, iy, iz = corners[corner]
        return _node_index_3d(ix, iy, iz, nely, nelz)


def _nodes_in_region(
    spec: dict,
    nelx: int,
    nely: int,
    nelz: int,
    dim: int,
) -> list[int]:
    """Nodes within a rectangular region [x0,x1] x [y0,y1] (x [z0,z1] for 3D)."""
    x0, x1 = spec.get("x", [0, nelx])
    y0, y1 = spec.get("y", [0, nely])
    nodes = []
    if dim == 2:
        for ix in range(int(x0), int(x1) + 1):
            for iy in range(int(y0), int(y1) + 1):
                nodes.append(_node_index_2d(ix, iy, nelx, nely))
    else:
        z0, z1 = spec.get("z", [0, nelz])
        for ix in range(int(x0), int(x1) + 1):
            for iy in range(int(y0), int(y1) + 1):
                for iz in range(int(z0), int(z1) + 1):
                    nodes.append(_node_index_3d(ix, iy, iz, nely, nelz))
    return nodes
