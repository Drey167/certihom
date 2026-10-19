"""Bogner-Fox-Schmit (BFS) C^1 elements — exact certification of C_low.

The elasticity dual bound needs an EXACTLY equilibrated stress field, built
as the Airy Hessian of a potential. Q1 cannot carry it (bilinears have
phi,11 = phi,22 = 0), so the projection uses the BFS bicubic Hermite space:
tensor-product cubic Hermite with nodal degrees of freedom

        (phi, phi_,x, phi_,y, phi_,xy)      (4 per node, 16 per element),

all four obtainable from the network by automatic differentiation.

*Why interpolation preserves static admissibility.* BFS fields are globally
C^1. The Airy stress sigma = (phi,22, phi,11, -sqrt2 phi,12) is
divergence-free pointwise inside each element (identity), and across an
edge with normal e1 the traction components are (phi,22, -phi,12): phi is
continuous, so its second TANGENTIAL derivative phi,22 is continuous; phi,1
is continuous by C^1, so its tangential derivative phi,12 is continuous.
Traction is therefore continuous, the interpolated stress lies in H(div)
with div sigma = 0, and its cell mean is exactly the prescribed one
(integrals of derivatives of periodic piecewise polynomials vanish).

*Exactness of integration.* Second derivatives of BFS fields are degree <= 3
per coordinate, so the complementary-energy integrand is degree <= 6:
4-point Gauss per direction (exact to degree 7) integrates it exactly.

1D cubic Hermite on [0,h], t = x/h, DOFs = (value, physical derivative):
    H1 = 1 - 3t^2 + 2t^3        H2 = h (t - 2t^2 + t^3)
    H3 = 3t^2 - 2t^3            H4 = h (-t^2 + t^3)
"""

from __future__ import annotations

import numpy as np
import torch

SQ2 = np.sqrt(2.0)

# 4-point Gauss-Legendre on [-1, 1] (nodes/weights), mapped to t in [0,1].
_G4 = np.array([-0.8611363115940526, -0.3399810435848563,
                0.3399810435848563, 0.8611363115940526])
_W4 = np.array([0.3478548451374538, 0.6521451548625461,
                0.6521451548625461, 0.3478548451374538])


def _hermite(t: float, h: float):
    """Return (H, dH, ddH), each length-4, for value/deriv DOF pairs
    ordered (v0, d0, v1, d1). Derivatives are with respect to PHYSICAL x."""
    H = np.array([1 - 3 * t**2 + 2 * t**3,
                  h * (t - 2 * t**2 + t**3),
                  3 * t**2 - 2 * t**3,
                  h * (-t**2 + t**3)])
    dH = np.array([(-6 * t + 6 * t**2) / h,
                   1 - 4 * t + 3 * t**2,
                   (6 * t - 6 * t**2) / h,
                   -2 * t + 3 * t**2])
    ddH = np.array([(-6 + 12 * t) / h**2,
                    (-4 + 6 * t) / h,
                    (6 - 12 * t) / h**2,
                    (-2 + 6 * t) / h])
    return H, dH, ddH


def bfs_nodal_dofs(fn, n_elem: int) -> np.ndarray:
    """Nodal BFS DOFs of a scalar-per-head network.

    Returns (n_elem, n_elem, n_out, 4) with the last axis
    (phi, phi_,x, phi_,y, phi_,xy), obtained by autograd — exact, not
    finite-differenced.
    """
    c = torch.arange(n_elem, dtype=torch.float64) / n_elem
    Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
    y = torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1).requires_grad_(True)
    out = fn(y)
    dofs = []
    for i in range(out.shape[1]):
        g = torch.autograd.grad(out[:, i].sum(), y, create_graph=True,
                                retain_graph=True)[0]
        gxy = torch.autograd.grad(g[:, 0].sum(), y, retain_graph=True)[0][:, 1]
        dofs.append(torch.stack([out[:, i], g[:, 0], g[:, 1], gxy], dim=1))
    D = torch.stack(dofs, dim=1).detach().numpy()  # (N, n_out, 4)
    return D.reshape(n_elem, n_elem, out.shape[1], 4)


def _second_derivs(D_head: np.ndarray, tx: float, ty: float, h: float):
    """Element-wise (phi_,xx, phi_,yy, phi_,xy) at local coordinates
    (tx, ty) in [0,1]^2, for one head. D_head: (ne, ne, 4) nodal DOFs."""
    Hx, dHx, ddHx = _hermite(tx, h)
    Hy, dHy, ddHy = _hermite(ty, h)
    xx = np.zeros(D_head.shape[:2])
    yy = np.zeros(D_head.shape[:2])
    xy = np.zeros(D_head.shape[:2])
    for a in (0, 1):
        for b in (0, 1):
            S = np.roll(np.roll(D_head, -a, axis=0), -b, axis=1)
            V, Dx, Dy, Dxy = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
            ix, jx = (0, 1) if a == 0 else (2, 3)   # (value, deriv) slots, x
            iy, jy = (0, 1) if b == 0 else (2, 3)   # ... y
            # phi = V Hx_i Hy_i + Dx Hx_j Hy_i + Dy Hx_i Hy_j + Dxy Hx_j Hy_j
            xx += (V * ddHx[ix] * Hy[iy] + Dx * ddHx[jx] * Hy[iy]
                   + Dy * ddHx[ix] * Hy[jy] + Dxy * ddHx[jx] * Hy[jy])
            yy += (V * Hx[ix] * ddHy[iy] + Dx * Hx[jx] * ddHy[iy]
                   + Dy * Hx[ix] * ddHy[jy] + Dxy * Hx[jx] * ddHy[jy])
            xy += (V * dHx[ix] * dHy[iy] + Dx * dHx[jx] * dHy[iy]
                   + Dy * dHx[ix] * dHy[jy] + Dxy * dHx[jx] * dHy[jy])
    return xx, yy, xy


def certified_lower_elasticity(dofs: np.ndarray, Cfield: np.ndarray) -> np.ndarray:
    """Exactly-integrated certified C_low from BFS nodal DOFs.

    dofs   : (n_elem, n_elem, 3, 4) — three Airy potentials (one per unit
             Mandel stress) with their (phi, phi_,x, phi_,y, phi_,xy).
    Cfield : (n, n, 3, 3) pixel Mandel stiffnesses, n | n_elem.
    """
    n_elem = dofs.shape[0]
    n = Cfield.shape[0]
    assert n_elem % n == 0, "FE mesh must be a refinement of the pixel grid"
    r = n_elem // n
    Ce = np.repeat(np.repeat(Cfield, r, axis=0), r, axis=1)
    Se = np.linalg.inv(Ce)                       # (ne, ne, 3, 3)
    h = 1.0 / n_elem
    I3 = np.eye(3)

    S_up = np.zeros((3, 3))
    for gx, wx in zip(_G4, _W4):
        for gy, wy in zip(_G4, _W4):
            tx, ty = 0.5 * (gx + 1.0), 0.5 * (gy + 1.0)
            w = wx * wy * (h * h) / 4.0
            Sig = []
            for k in range(3):
                xx, yy, xy = _second_derivs(dofs[:, :, k, :], tx, ty, h)
                s = np.stack([yy, xx, -SQ2 * xy], axis=-1)   # Airy
                Sig.append(s + I3[k])
            for k in range(3):
                SS = np.einsum("ijab,ijb->ija", Se, Sig[k])
                for l in range(k, 3):
                    S_up[k, l] += w * np.sum(SS * Sig[l])
    S_up = S_up + np.triu(S_up, 1).T
    return np.linalg.inv(S_up)
