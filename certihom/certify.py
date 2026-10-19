"""Phase 1 — EXACT-INTEGRATION certificates (removes the quadrature caveat).

Phases 0/2/3 evaluate the bound integrals with a midpoint rule, so their
certificates read "guaranteed up to quadrature error". This module removes
that caveat: network fields are interpolated into conforming finite-element
spaces on a pixel-aligned mesh, and the resulting energies are integrated
EXACTLY. The bounds then hold as mathematical statements, subject only to
floating-point arithmetic.

Why interpolation preserves admissibility
-----------------------------------------
*Primal.* Nodal Q1 interpolation of a continuous periodic field is
continuous, piecewise bilinear and periodic (identified nodes carry equal
values because the network is exactly periodic). Hence chi_h is in the
conforming subspace of H^1_per and the primal principle applies verbatim.

*Dual (the elegant part).* Let psi_h be continuous piecewise bilinear and
set q = curl psi_h = (d psi_h/dy2, -d psi_h/dy1). Within an element,
div q = psi,21 - psi,12 = 0 pointwise. Across an element edge, continuity
of psi_h forces its TANGENTIAL derivative to be continuous; curl rotates
tangential into normal, so the NORMAL component of q is continuous — which
is precisely the condition for q in H(div) with div q = 0 in the
distributional sense. The mean is exact too: the integral of a derivative
of a periodic piecewise polynomial vanishes identically. So the
interpolated dual field is fully statically admissible.

*Exactness of the integration.* With a pixel-aligned mesh the coefficient
(k or the Mandel C) is constant per element, and gradients of Q1 fields are
bilinear, so each energy integrand is a polynomial of degree <= 2 in each
coordinate: 2x2 Gauss is exact.

Scope note (honest): the ELASTICITY dual side needs second derivatives of
the Airy potential, which vanish identically for Q1 (span{1,x,y,xy}). Its
exact certification requires C^1 elements (Bogner-Fox-Schmit bicubic
Hermite), for which the same argument goes through — C^1 continuity makes
the Airy traction (phi,22, -phi,12) continuous across edges. That is
implemented in `bfs.py`. This module therefore certifies: scalar primal,
scalar dual, and elasticity primal.
"""

from __future__ import annotations

import numpy as np
import torch

from .bounds import CertifiedBounds

SQ2 = np.sqrt(2.0)
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _dN(xi: float, eta: float, h: float) -> np.ndarray:
    """Physical gradients of the 4 bilinear shape functions, shape (2, 4).
    Node order CCW from the element origin: (0,0), (h,0), (h,h), (0,h)."""
    return np.array([
        [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
        [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
    ]) / (2.0 * h)


def _corners(V: np.ndarray) -> list[np.ndarray]:
    """Periodic gather of the four element-corner nodal arrays."""
    return [V,
            np.roll(V, -1, axis=0),
            np.roll(np.roll(V, -1, axis=0), -1, axis=1),
            np.roll(V, -1, axis=1)]


def node_grid(n_elem: int) -> torch.Tensor:
    """(n_elem^2, 2) node coordinates y = (i/n, j/n), i,j = 0..n-1.
    Periodicity identifies i=n with i=0, so only these nodes are stored."""
    c = torch.arange(n_elem, dtype=torch.float64) / n_elem
    Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1)


def nodal_values(fn, n_elem: int) -> np.ndarray:
    """Evaluate a field callable at the node grid -> (n_elem, n_elem, n_out).

    ``fn`` maps (N,2) float64 tensor -> (N, n_out); it may be a plain
    FieldPair or a closure around a conditioned network (Phase 3).
    """
    with torch.no_grad():
        out = fn(node_grid(n_elem))
    return out.reshape(n_elem, n_elem, -1).numpy()


def _elementwise_k(K_field: np.ndarray, n_elem: int) -> np.ndarray:
    """Constant coefficient per element (mesh must refine the pixel grid)."""
    n = K_field.shape[0]
    assert n_elem % n == 0, "FE mesh must be a refinement of the pixel grid"
    r = n_elem // n
    return np.kron(K_field, np.ones((r, r)))


def certified_bounds_scalar(chi_vals: np.ndarray, psi_vals: np.ndarray,
                            K_field: np.ndarray) -> CertifiedBounds:
    """Exactly-integrated certified interval for scalar conductivity.

    chi_vals, psi_vals : (n_elem, n_elem, 2) nodal values of the primal
        fluctuations and the dual stream functions.
    K_field            : (n, n) pixel conductivities, n | n_elem.
    """
    n_elem = chi_vals.shape[0]
    h = 1.0 / n_elem
    ke = _elementwise_k(K_field, n_elem)
    w = (h * h) / 4.0
    E = np.eye(2)

    K_up = np.zeros((2, 2))
    R_up = np.zeros((2, 2))
    for xi in _GP:
        for eta in _GP:
            dN = _dN(xi, eta, h)
            prim, dual = [], []
            for a in range(2):
                cs = _corners(chi_vals[:, :, a])
                g = [sum(dN[c, m] * cs[m] for m in range(4)) for c in (0, 1)]
                prim.append((E[a, 0] + g[0], E[a, 1] + g[1]))
                cs = _corners(psi_vals[:, :, a])
                g = [sum(dN[c, m] * cs[m] for m in range(4)) for c in (0, 1)]
                # curl psi = (d psi/dy2, -d psi/dy1)
                dual.append((E[a, 0] + g[1], E[a, 1] - g[0]))
            for a in range(2):
                for b in range(a, 2):
                    K_up[a, b] += w * np.sum(
                        ke * (prim[a][0] * prim[b][0] + prim[a][1] * prim[b][1]))
                    R_up[a, b] += w * np.sum(
                        (dual[a][0] * dual[b][0] + dual[a][1] * dual[b][1]) / ke)
    K_up = K_up + np.triu(K_up, 1).T
    R_up = R_up + np.triu(R_up, 1).T
    return CertifiedBounds(K_low=np.linalg.inv(R_up), K_up=K_up,
                           meta={"integration": "exact (2x2 Gauss, Q1)",
                                 "n_elem": n_elem})


def certified_upper_elasticity(chi_vals: np.ndarray,
                               Cfield: np.ndarray) -> np.ndarray:
    """Exactly-integrated certified C_up from nodal displacement values.

    chi_vals : (n_elem, n_elem, 6) nodal values, head 2k+c = component c of
        the fluctuation for unit Mandel strain E_k.
    Cfield   : (n, n, 3, 3) pixel Mandel stiffnesses, n | n_elem.
    """
    n_elem = chi_vals.shape[0]
    n = Cfield.shape[0]
    assert n_elem % n == 0, "FE mesh must be a refinement of the pixel grid"
    r = n_elem // n
    Ce = np.repeat(np.repeat(Cfield, r, axis=0), r, axis=1)  # (ne,ne,3,3)
    h = 1.0 / n_elem
    w = (h * h) / 4.0
    E3 = np.eye(3)

    C_up = np.zeros((3, 3))
    for xi in _GP:
        for eta in _GP:
            dN = _dN(xi, eta, h)
            F = []
            for k in range(3):
                cu = _corners(chi_vals[:, :, 2 * k])
                cv = _corners(chi_vals[:, :, 2 * k + 1])
                du = [sum(dN[c, m] * cu[m] for m in range(4)) for c in (0, 1)]
                dv = [sum(dN[c, m] * cv[m] for m in range(4)) for c in (0, 1)]
                eps = np.stack([du[0], dv[1], (du[1] + dv[0]) / SQ2], axis=-1)
                F.append(eps + E3[k])
            for k in range(3):
                CF = np.einsum("ijab,ijb->ija", Ce, F[k])
                for l in range(k, 3):
                    C_up[k, l] += w * np.sum(CF * F[l])
    return C_up + np.triu(C_up, 1).T
