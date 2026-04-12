"""Element stiffness matrix computation.

Computes the element stiffness matrix Ke for:
  - Q4 bilinear quadrilateral element (plane stress, 2D)
  - H8 trilinear hexahedral element (3D)

Both assume unit element side length and unit Young's modulus.
The actual stiffness is scaled by (E / E_ref) * rho_e^penalty during the solve.

References:
  - Felippa, C.A., "Introduction to Finite Element Methods" (ASEN 5007 course notes)
  - Sigmund, O. (2001), "A 99 line topology optimization code written in Matlab"
"""

from __future__ import annotations

import numpy as np


def compute_Ke_2d(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """8x8 element stiffness matrix for a unit Q4 plane-stress element.

    Integration is performed analytically over the 2x2 Gauss quadrature.
    The element occupies [0,1]x[0,1] in physical coordinates.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    Ke : ndarray, shape (8, 8), dtype float64
    """
    # Constitutive matrix for plane stress
    c = E / (1.0 - nu**2)
    D = c * np.array([
        [1.0,  nu,          0.0         ],
        [nu,   1.0,         0.0         ],
        [0.0,  0.0,  (1.0 - nu) / 2.0  ],
    ])

    Ke = np.zeros((8, 8), dtype=np.float64)

    # 2x2 Gauss quadrature points and weights on [-1,1]^2
    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [-gp, gp]
    gauss_wts = [1.0, 1.0]

    for xi, wi in zip(gauss_pts, gauss_wts):
        for eta, wj in zip(gauss_pts, gauss_wts):
            # Shape function derivatives w.r.t. xi, eta (Q4 bilinear)
            # N1 = (1-xi)(1-eta)/4, N2 = (1+xi)(1-eta)/4,
            # N3 = (1+xi)(1+eta)/4, N4 = (1-xi)(1+eta)/4
            dN_dxi  = np.array([
                -(1.0 - eta) / 4.0,
                 (1.0 - eta) / 4.0,
                 (1.0 + eta) / 4.0,
                -(1.0 + eta) / 4.0,
            ])
            dN_deta = np.array([
                -(1.0 - xi) / 4.0,
                -(1.0 + xi) / 4.0,
                 (1.0 + xi) / 4.0,
                 (1.0 - xi) / 4.0,
            ])

            # Jacobian for unit square element: J = 0.5 * I_2 (since mapped from [-1,1]^2 to [0,1]^2)
            # det(J) = 0.25, J_inv = diag(2, 2)
            det_J = 0.25
            dN_dx = 2.0 * dN_dxi    # = J_inv[0,0] * dN_dxi
            dN_dy = 2.0 * dN_deta   # = J_inv[1,1] * dN_deta

            # Strain-displacement matrix B (3 x 8)
            B = np.zeros((3, 8))
            for i in range(4):
                B[0, 2 * i]     = dN_dx[i]   # epsilon_xx
                B[1, 2 * i + 1] = dN_dy[i]   # epsilon_yy
                B[2, 2 * i]     = dN_dy[i]   # gamma_xy
                B[2, 2 * i + 1] = dN_dx[i]

            Ke += wi * wj * det_J * (B.T @ D @ B)

    return Ke


def compute_Ke_3d(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """24x24 element stiffness matrix for a unit H8 hexahedral element.

    Uses 2x2x2 Gauss quadrature. The element occupies [0,1]^3.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    Ke : ndarray, shape (24, 24), dtype float64
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    # 3D isotropic elasticity constitutive matrix (6x6, Voigt notation)
    D = np.array([
        [lam + 2*mu, lam,        lam,        0,   0,   0  ],
        [lam,        lam + 2*mu, lam,        0,   0,   0  ],
        [lam,        lam,        lam + 2*mu, 0,   0,   0  ],
        [0,          0,          0,          mu,  0,   0  ],
        [0,          0,          0,          0,   mu,  0  ],
        [0,          0,          0,          0,   0,   mu ],
    ], dtype=np.float64)

    Ke = np.zeros((24, 24), dtype=np.float64)

    gp = 1.0 / np.sqrt(3.0)
    gauss_pts = [-gp, gp]
    gauss_wts = [1.0, 1.0]

    for xi, wi in zip(gauss_pts, gauss_wts):
        for eta, wj in zip(gauss_pts, gauss_wts):
            for zeta, wk in zip(gauss_pts, gauss_wts):
                # Shape function derivatives for H8 trilinear element
                # Nodes ordered: (−,−,−)(+,−,−)(+,+,−)(−,+,−)(−,−,+)(+,−,+)(+,+,+)(−,+,+)
                # mapped from [-1,1]^3 to [0,1]^3 -> det(J) = 0.125
                dN_dxi   = np.zeros(8)
                dN_deta  = np.zeros(8)
                dN_dzeta = np.zeros(8)

                signs = np.array([
                    [-1, -1, -1],
                    [ 1, -1, -1],
                    [ 1,  1, -1],
                    [-1,  1, -1],
                    [-1, -1,  1],
                    [ 1, -1,  1],
                    [ 1,  1,  1],
                    [-1,  1,  1],
                ], dtype=float)

                for k in range(8):
                    a, b, c = signs[k]
                    dN_dxi[k]   = a * (1 + b * eta) * (1 + c * zeta) / 8.0
                    dN_deta[k]  = b * (1 + a * xi)  * (1 + c * zeta) / 8.0
                    dN_dzeta[k] = c * (1 + a * xi)  * (1 + b * eta)  / 8.0

                det_J = 0.125
                dN_dx = 2.0 * dN_dxi
                dN_dy = 2.0 * dN_deta
                dN_dz = 2.0 * dN_dzeta

                # Strain-displacement matrix B (6 x 24)
                B = np.zeros((6, 24))
                for k in range(8):
                    B[0, 3*k]     = dN_dx[k]   # eps_xx
                    B[1, 3*k + 1] = dN_dy[k]   # eps_yy
                    B[2, 3*k + 2] = dN_dz[k]   # eps_zz
                    B[3, 3*k]     = dN_dy[k]   # gamma_xy (= 2*eps_xy)
                    B[3, 3*k + 1] = dN_dx[k]
                    B[4, 3*k + 1] = dN_dz[k]   # gamma_yz
                    B[4, 3*k + 2] = dN_dy[k]
                    B[5, 3*k]     = dN_dz[k]   # gamma_xz
                    B[5, 3*k + 2] = dN_dx[k]

                Ke += wi * wj * wk * det_J * (B.T @ D @ B)

    return Ke


def compute_Ke(E: float, nu: float, dim: int) -> np.ndarray:
    """Dispatch element stiffness computation on dimension."""
    if dim == 2:
        return compute_Ke_2d(E, nu)
    else:
        return compute_Ke_3d(E, nu)
