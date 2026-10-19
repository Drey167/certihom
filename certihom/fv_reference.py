"""Periodic finite-volume reference solver (VALIDATION ONLY).

Solves the scalar cell problem  div( k(y) (ebar + grad theta) ) = 0 with
Y-periodic theta on the pixel grid itself: one unknown per pixel, two-point
flux approximation with HARMONIC face conductivities

    k_face = 2 k_L k_R / (k_L + k_R),

which is exact for one-dimensional series transport — hence the solver
reproduces the laminate's harmonic branch to machine precision (unit test),
and its arithmetic branch as well (uniform tangential gradient). For general
microstructures it carries O(h) - O(h^2) discretization error, so bracketing
tests against it must use a tolerance informed by mesh refinement
(``fv_effective`` at two resolutions, or the exact value when known).

The role of this module in the research plan is strictly auditing: the
certified bounds are mathematical statements independent of any reference;
the reference only detects implementation bugs.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _face_conductivities(K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic means across the two face families (periodic wrap).

    kx[i, j] is the conductivity of the face between cells (i-1, j) and
    (i, j) modulo n (faces normal to y1); ky analogous for y2.
    """
    Km = np.roll(K, 1, axis=0)
    kx = 2.0 * K * Km / (K + Km)
    Km = np.roll(K, 1, axis=1)
    ky = 2.0 * K * Km / (K + Km)
    return kx, ky


def fv_solve(K: np.ndarray, ebar: np.ndarray) -> np.ndarray:
    """Solve the periodic FV cell problem for mean gradient ``ebar``.

    Returns theta (n, n) with theta[0, 0] pinned to zero (the constant
    nullspace of the periodic problem).
    """
    n = K.shape[0]
    h = 1.0 / n
    kx, ky = _face_conductivities(K)
    N = n * n

    def idx(i, j):
        return (i % n) * n + (j % n)

    rows, cols, vals = [], [], []
    b = np.zeros(N)
    for i in range(n):
        for j in range(n):
            c = idx(i, j)
            # x-faces: left face kx[i,j] (normal -e1), right face kx[i+1,j]
            # y-faces: bottom ky[i,j], top ky[i,j+1]
            kl, kr = kx[i, j], kx[(i + 1) % n, j]
            kb, kt = ky[i, j], ky[i, (j + 1) % n]
            diag = -(kl + kr + kb + kt)
            for cc, vv in (
                (idx(i - 1, j), kl),
                (idx(i + 1, j), kr),
                (idx(i, j - 1), kb),
                (idx(i, j + 1), kt),
                (c, diag),
            ):
                rows.append(c)
                cols.append(cc)
                vals.append(vv)
            # RHS: -h * sum_faces k_f (ebar . n_f); outward normals.
            b[c] = -h * (kr * ebar[0] - kl * ebar[0] + kt * ebar[1] - kb * ebar[1])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    # Pin theta[0,0] = 0 to remove the constant nullspace.
    A = A.tolil()
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = 0.0
    theta = spla.spsolve(A.tocsr(), b)
    return theta.reshape(n, n)


def fv_effective(K: np.ndarray) -> np.ndarray:
    """Discrete effective tensor from the FV energy bilinear form.

    K_ref[a, b] = sum over faces of k_f * g_f^(a) * g_f^(b) * h^2, where the
    face gradient g_f^(a) = (theta^(a)_R - theta^(a)_L)/h + ebar_a . n_f and
    theta^(a) solves the cell problem for unit load e_a. Symmetric by
    construction; exact for laminates (unit test).
    """
    n = K.shape[0]
    h = 1.0 / n
    kx, ky = _face_conductivities(K)
    thetas = [fv_solve(K, e) for e in (np.array([1.0, 0.0]), np.array([0.0, 1.0]))]

    def face_grads(theta, ebar):
        gx = (theta - np.roll(theta, 1, axis=0)) / h + ebar[0]  # on x-faces
        gy = (theta - np.roll(theta, 1, axis=1)) / h + ebar[1]  # on y-faces
        return gx, gy

    E = np.eye(2)
    G = [face_grads(thetas[a], E[a]) for a in range(2)]
    K_ref = np.empty((2, 2))
    for a in range(2):
        for b in range(2):
            K_ref[a, b] = h * h * (
                (kx * G[a][0] * G[b][0]).sum() + (ky * G[a][1] * G[b][1]).sum()
            )
    return 0.5 * (K_ref + K_ref.T)
