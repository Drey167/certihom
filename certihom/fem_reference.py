"""Periodic Q1 (bilinear) FEM reference for 2D elasticity (VALIDATION ONLY).

One element per pixel, constant Mandel stiffness per element, periodic node
identification, translations pinned. Because Q1 displacements form a
CONFORMING subspace of the admissible fluctuation space, the FEM effective
tensor is itself an upper bound:

        C_FEM(h)  >=  C_eff        (Loewner),

which gives a rigorous one-sided audit: the certified lower bound must
satisfy C_low <= C_FEM(h) for EVERY h, with no tolerance haggling. For the
laminate, the exact solution (piecewise-linear displacement with kinks on
element edges) lies inside the Q1 space, so C_FEM equals the exact tensor —
a machine-precision test of this module itself.

2x2 Gauss integration is exact for the bilinear element energy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)  # Gauss points on [-1, 1]
SQ2 = np.sqrt(2.0)


def _bmat(xi: float, eta: float, h: float) -> np.ndarray:
    """Mandel strain-displacement matrix (3 x 8) for the bilinear quad of
    side h, dof order (u1,v1,...,u4,v4), nodes CCW from (0,0)."""
    dN = np.array([
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
    ]) / (2.0 * h)  # physical derivatives (d/dx, d/dy) x 4 nodes
    B = np.zeros((3, 8))
    for a in range(4):
        B[0, 2 * a] = dN[0, a]
        B[1, 2 * a + 1] = dN[1, a]
        B[2, 2 * a] = dN[1, a] / SQ2
        B[2, 2 * a + 1] = dN[0, a] / SQ2
    return B


def fem_effective(Cfield: np.ndarray) -> np.ndarray:
    """Effective Mandel stiffness from periodic Q1 FEM.

    Cfield: (n, n, 3, 3) per-pixel Mandel stiffnesses. Solves the three unit
    macroscopic-strain load cases with one factorization and assembles
    C_FEM[k,l] = < (E_k + B u_k) . C (E_l + B u_l) > element-wise.
    """
    n = Cfield.shape[0]
    h = 1.0 / n
    ndof = 2 * n * n

    def node(i, j):
        return (i % n) * n + (j % n)

    # Element Gauss data (shared geometry): B at 4 Gauss points, weight h^2/4.
    Bs = [_bmat(xi, eta, h) for xi in _GP for eta in _GP]
    w = (h * h) / 4.0

    rows, cols, vals = [], [], []
    F = np.zeros((ndof, 3))
    elems = []
    for i in range(n):
        for j in range(n):
            nd = [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            dofs = np.array([[2 * a, 2 * a + 1] for a in nd]).ravel()
            C = Cfield[i, j]
            Ke = sum(B.T @ C @ B for B in Bs) * w
            fe = -sum(B.T @ C for B in Bs) * w  # (8,3): loads for E_1..E_3
            elems.append((dofs, C))
            for a in range(8):
                F[dofs[a], :] += fe[a, :]
                for b in range(8):
                    rows.append(dofs[a]); cols.append(dofs[b]); vals.append(Ke[a, b])
    K = sp.csr_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tolil()
    for d in (0, 1):  # pin translations: node 0, both components
        K[d, :] = 0.0
        K[d, d] = 1.0
        F[d, :] = 0.0
    lu = spla.splu(K.tocsc())
    U = np.column_stack([lu.solve(F[:, k]) for k in range(3)])

    E = np.eye(3)
    C_hom = np.zeros((3, 3))
    for dofs, C in elems:
        for B in Bs:
            G = [E[k] + B @ U[dofs, k] for k in range(3)]
            for k in range(3):
                CG = C @ G[k]
                for l in range(k, 3):
                    C_hom[k, l] += w * (CG @ G[l])
    C_hom = C_hom + np.triu(C_hom, 1).T
    return C_hom
