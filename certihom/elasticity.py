"""Phase 2 — certified bounds on effective ELASTICITY tensors (2D).

All tensor algebra uses Mandel notation, in which the Loewner (quadratic-
form) order on stiffness matrices is literal 3x3 matrix ordering:

    strain  eps_hat = (e11, e22, sqrt(2) e12)
    stress  sig_hat = (s11, s22, sqrt(2) s12)
    energy density  = (1/2) eps_hat . C_hat eps_hat,   sig_hat = C_hat eps_hat
    compliance      S_hat = C_hat^{-1}  (a plain matrix inverse — the point
                    of Mandel over Voigt notation).

Bound structure (proofs mirror MATH.md sections 2-3 verbatim):

* Primal: three periodic displacement-fluctuation heads chi_k (one per unit
  Mandel strain E_k). Total strains F_k = E_k + eps(chi_k). The Gram matrix
  C_up[k,l] = < F_k . C(y) F_l >  satisfies  C_up >= C_eff.

* Dual: three periodic Airy potentials phi_k. The induced stress

      sig'11 = phi,22   sig'22 = phi,11   sig'12 = -phi,12

  is symmetric, divergence-free identically, and has EXACTLY zero mean
  (means of second derivatives of periodic functions vanish), so
  Sig_k = T_k + sig'(phi_k) is statically admissible with mean T_k (unit
  Mandel stress). Then S_up[k,l] = < Sig_k . S(y) Sig_l > >= C_eff^{-1} and
  C_low = S_up^{-1} <= C_eff. In 2D the Airy representation is COMPLETE for
  zero-mean periodic equilibrated stresses (Fourier argument, MATH.md), so
  the dual side is not structurally handicapped.

Zero-initialized heads reproduce the elasticity Voigt <C> and Reuss
<S>^{-1} bounds at step 0, exactly as in Phase 0.
"""

from __future__ import annotations

import numpy as np
import torch

from .fields import heads_and_grads

SQ2 = np.sqrt(2.0)
def _eye3(t: torch.Tensor) -> torch.Tensor:
    """Mandel unit loads matching the dtype/device of a field tensor."""
    return torch.eye(3, dtype=t.dtype, device=t.device)


# ----------------------------------------------------------------------------
# Materials (isotropic, plane strain) in Mandel notation
# ----------------------------------------------------------------------------

def iso_stiffness_mandel(E: float, nu: float) -> np.ndarray:
    """Plane-strain isotropic stiffness, Mandel 3x3.

    lam = E nu / ((1+nu)(1-2 nu)),  mu = E / (2(1+nu));
    C = [[lam+2mu, lam, 0], [lam, lam+2mu, 0], [0, 0, 2mu]].
    (The Mandel shear entry is 2*mu because sig_hat3 = sqrt2*s12 =
    2 mu * sqrt2 * e12 = 2 mu * eps_hat3.)
    """
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.array(
        [[lam + 2 * mu, lam, 0.0],
         [lam, lam + 2 * mu, 0.0],
         [0.0, 0.0, 2 * mu]], dtype=np.float64)
    return C


def mandel_field_from_mask(mask: np.ndarray, C_in: np.ndarray,
                           C_out: np.ndarray) -> np.ndarray:
    """Pixel field of Mandel stiffnesses: C_in where mask, else C_out.
    Returns (n, n, 3, 3) float64."""
    n = mask.shape[0]
    F = np.empty((n, n, 3, 3), dtype=np.float64)
    F[mask] = C_in
    F[~mask] = C_out
    return F


def sample_mandel(Cfield: np.ndarray, y: torch.Tensor) -> torch.Tensor:
    """Nearest-pixel lookup of the (n,n,3,3) Mandel field at points y
    (same convention/caveats as microstructures.sample_k). Returns
    (N, 3, 3) torch float64 with no gradient."""
    n = Cfield.shape[0]
    idx = torch.clamp((y * n).long(), 0, n - 1)
    Ct = torch.as_tensor(Cfield, dtype=y.dtype, device=y.device)
    return Ct[idx[:, 0], idx[:, 1]]


# ----------------------------------------------------------------------------
# Matrix-weighted Gram — Phase 2's single source of truth
# ----------------------------------------------------------------------------

def gram_mandel(fields: list[torch.Tensor], M: torch.Tensor) -> torch.Tensor:
    """G[k,l] = mean_p( fields[k]_p . M_p fields[l]_p ),  fields[k]: (N,3),
    M: (N,3,3) SPD. Both bounds and every analytic-field unit test route
    through this function."""
    d = len(fields)
    G = torch.empty((d, d), dtype=fields[0].dtype, device=fields[0].device)
    for i in range(d):
        Mi = torch.einsum("pij,pj->pi", M, fields[i])
        for j in range(i, d):
            val = (Mi * fields[j]).sum(dim=1).mean()
            G[i, j] = val
            G[j, i] = val
    return G


# ----------------------------------------------------------------------------
# Trial fields
# ----------------------------------------------------------------------------

def primal_strain_fields(chi_net, pts: torch.Tensor,
                         create_graph: bool) -> list[torch.Tensor]:
    """Total Mandel strains F_k = E_k + eps(chi_k), k = 1..3.

    chi_net has 6 outputs, interpreted as (chi_k)_c = out[:, 2k + c],
    c in {0,1} the displacement components. eps in Mandel:
    (d1 u1, d2 u2, (d2 u1 + d1 u2)/sqrt2).
    """
    _, g = heads_and_grads(chi_net, pts, create_graph)  # 6 grads, each (N,2)
    I3 = _eye3(g[0])
    fields = []
    for k in range(3):
        gu, gv = g[2 * k], g[2 * k + 1]  # grads of chi_k components
        eps = torch.stack(
            [gu[:, 0], gv[:, 1], (gu[:, 1] + gv[:, 0]) / SQ2], dim=1)
        fields.append(I3[k] + eps)
    return fields


def _hessians(net, pts: torch.Tensor, create_graph: bool) -> list[torch.Tensor]:
    """Per-head Hessians of a scalar-output-per-head network.

    Returns a list over heads of (N, 2, 2) tensors H with H[p] the Hessian
    at point p. First derivatives are built with create_graph=True (needed
    to differentiate again); the second pass honors ``create_graph`` so
    evaluation does not drag a training graph around.
    """
    y = pts.detach().clone().requires_grad_(True)
    out = net(y)
    hessians = []
    for i in range(out.shape[1]):
        gi = torch.autograd.grad(out[:, i].sum(), y,
                                 create_graph=True, retain_graph=True)[0]
        rows = []
        for c in range(2):
            r = torch.autograd.grad(gi[:, c].sum(), y,
                                    create_graph=create_graph,
                                    retain_graph=True)[0]
            rows.append(r)
        hessians.append(torch.stack(rows, dim=1))  # (N, 2, 2)
    return hessians


def dual_stress_fields(phi_net, pts: torch.Tensor,
                       create_graph: bool) -> list[torch.Tensor]:
    """Total Mandel stresses Sig_k = T_k + Airy(phi_k), k = 1..3, with
    Airy(phi) = (phi,22, phi,11, -sqrt2 phi,12) — equilibrated and mean-zero
    by construction (module docstring)."""
    H = _hessians(phi_net, pts, create_graph)
    I3 = _eye3(H[0])
    fields = []
    for k in range(3):
        Hk = H[k]
        sig = torch.stack(
            [Hk[:, 1, 1], Hk[:, 0, 0], -SQ2 * Hk[:, 0, 1]], dim=1)
        fields.append(I3[k] + sig)
    return fields


# ----------------------------------------------------------------------------
# Bound assembly + certified engineering constants
# ----------------------------------------------------------------------------

def upper_gram_elast(chi_net, pts, Cvals, create_graph=False) -> torch.Tensor:
    return gram_mandel(primal_strain_fields(chi_net, pts, create_graph), Cvals)


def dual_gram_elast(phi_net, pts, Svals, create_graph=False) -> torch.Tensor:
    return gram_mandel(dual_stress_fields(phi_net, pts, create_graph), Svals)


def youngs_interval(C_low: np.ndarray, C_up: np.ndarray,
                    d: np.ndarray) -> tuple[float, float]:
    """Certified interval for the directional Young's modulus E(d).

    With a(d) = Mandel(d (x) d) = (d1^2, d2^2, sqrt2 d1 d2) and
    1/E(d) = a . S_eff a:
        S_eff <= C_low^{-1}  ==>  E(d) >= 1 / (a . C_low^{-1} a)
        S_eff >= C_up^{-1}   ==>  E(d) <= 1 / (a . C_up^{-1} a).
    (Plane-strain modulus, consistent with the constitutive convention.)
    """
    d = np.asarray(d, dtype=np.float64)
    d = d / np.linalg.norm(d)
    a = np.array([d[0] ** 2, d[1] ** 2, SQ2 * d[0] * d[1]])
    e_lo = 1.0 / float(a @ np.linalg.solve(C_low, a))
    e_hi = 1.0 / float(a @ np.linalg.solve(C_up, a))
    return e_lo, e_hi


# ----------------------------------------------------------------------------
# Exact two-phase elasticity laminate (unit-test ground truth)
# ----------------------------------------------------------------------------

def laminate_exact_elasticity(C1: np.ndarray, C2: np.ndarray, f1: float):
    """Exact effective Mandel stiffness of a laminate stacked along y1,
    plus the per-phase exact strain/stress states for each unit load.

    For fields varying in y1 only: (i) the tangential strain e22 is common
    to both phases; (ii) equilibrium forces s11 and s12 to be constant;
    (iii) phase strains volume-average to the load. Per unit Mandel strain
    E_k this is a 6x6 linear solve for (eps^1, eps^2); the mean stress
    column then gives C_exact, and the piecewise-constant fields are
    returned for the machine-precision Gram test (midpoint quadrature is
    exact for them).

    Returns (C_exact, eps_phase[2,3,3], sig_phase[2,3,3]) where
    eps_phase[p, :, k] is phase-p strain under load E_k.
    """
    f2 = 1.0 - f1
    C_exact = np.empty((3, 3))
    eps_ph = np.empty((2, 3, 3))
    sig_ph = np.empty((2, 3, 3))
    for k in range(3):
        ebar = np.zeros(3)
        ebar[k] = 1.0
        A = np.zeros((6, 6))
        b = np.zeros(6)
        # (i) e22 continuity: eps^p_2 = ebar_2
        A[0, 1] = 1.0;                 b[0] = ebar[1]
        A[1, 4] = 1.0;                 b[1] = ebar[1]
        # (ii) traction continuity across x1-interfaces: components 1 and 3
        A[2, 0:3] = C1[0];  A[2, 3:6] = -C2[0]
        A[3, 0:3] = C1[2];  A[3, 3:6] = -C2[2]
        # (iii) mean strain, components 1 and 3 (component 2 implied by (i))
        A[4, 0] = f1;  A[4, 3] = f2;   b[4] = ebar[0]
        A[5, 2] = f1;  A[5, 5] = f2;   b[5] = ebar[2]
        x = np.linalg.solve(A, b)
        e1, e2 = x[:3], x[3:]
        s1, s2 = C1 @ e1, C2 @ e2
        assert np.allclose([s1[0], s1[2]], [s2[0], s2[2]], atol=1e-12)
        eps_ph[0, :, k], eps_ph[1, :, k] = e1, e2
        sig_ph[0, :, k], sig_ph[1, :, k] = s1, s2
        C_exact[:, k] = f1 * s1 + f2 * s2
    C_exact = 0.5 * (C_exact + C_exact.T)
    return C_exact, eps_ph, sig_ph
