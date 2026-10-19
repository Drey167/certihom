"""Phase 1 verification ladder — exact-integration certificates.

Decisive claims:
 (a) Q1 projection of the EXACT laminate corrector reproduces K_exact /
     C_exact to machine precision with exact integration (the analytic
     correctors are elementwise polynomial, so interpolation is exact);
 (b) BFS projection of the exact laminate Airy potential reproduces
     C_exact to machine precision on the DUAL side (C^1 certification);
 (c) projecting an actually TRAINED network yields bounds that are valid
     with no quadrature caveat and close to the midpoint-rule values.
"""

import numpy as np
import torch

from certihom.bfs import bfs_nodal_dofs, certified_lower_elasticity
from certihom.bounds import CertifiedBounds
from certihom.certify import (certified_bounds_scalar,
                              certified_upper_elasticity, nodal_values)
from certihom.elasticity import (iso_stiffness_mandel,
                                 laminate_exact_elasticity,
                                 mandel_field_from_mask)
from certihom.exact import laminate_exact
from certihom.microstructures import boolean_disks, laminate
from certihom.train import fit_bounds
from certihom.train_elast import fit_bounds_elasticity

C_SOFT = iso_stiffness_mandel(1.0, 0.3)
C_STIFF = iso_stiffness_mandel(10.0, 0.2)


# ---------------------------------------------------------------- scalar ----

def test_q1_projection_scalar_laminate_machine_precision():
    """Exact laminate correctors, nodal-interpolated, exactly integrated."""
    k1, k2, f1, n = 1.0, 6.0, 0.25, 8
    K = laminate(n, k1, k2, f1, axis=0)
    ne = 16
    h = 1.0 / ne
    ke = np.kron(K, np.ones((ne // n, ne // n)))[:, 0]  # k per element row
    harm = 1.0 / (f1 / k1 + (1 - f1) / k2)
    arith = f1 * k1 + (1 - f1) * k2

    # primal, load e1: d(chi)/dy1 = harm/k - 1 (piecewise constant)
    slope = harm / ke - 1.0
    chi1 = np.concatenate([[0.0], np.cumsum(slope * h)])[:ne]
    chi = np.zeros((ne, ne, 2))
    chi[:, :, 0] = chi1[:, None]          # head 0 varies with y1
    # primal, load e2: chi_2 = 0
    # dual, mean flux e1: psi_1 = 0
    # dual, mean flux e2: curl psi = (k/arith - 1) e2 -> -d psi/dy1 = k/a - 1
    dslope = -(ke / arith - 1.0)
    psi2 = np.concatenate([[0.0], np.cumsum(dslope * h)])[:ne]
    psi = np.zeros((ne, ne, 2))
    psi[:, :, 1] = psi2[:, None]

    b = certified_bounds_scalar(chi, psi, K)
    K_ex = laminate_exact(k1, k2, f1, axis=0)
    assert np.allclose(b.K_up, K_ex, atol=1e-12)
    assert np.allclose(b.K_low, K_ex, atol=1e-12)
    assert b.loewner_valid(tol=1e-11)


def test_projected_trained_network_is_valid_and_close():
    """Project a trained Phase-0 pair: exactly-integrated bounds must be
    Loewner-valid, bracket the exact laminate tensor, and land near the
    midpoint-rule values (the 'certification price')."""
    K = laminate(8, 1.0, 4.0, 0.5, axis=0)
    b_quad, _, (chi_net, psi_net) = fit_bounds(
        K, nq=32, n_harmonics=3, width=32, depth=2, steps=200, lr=3e-3,
        seed=0, verbose=False)
    chi = nodal_values(chi_net, 32)
    psi = nodal_values(psi_net, 32)
    b_cert = certified_bounds_scalar(chi, psi, K)
    K_ex = laminate_exact(1.0, 4.0, 0.5, axis=0)
    assert b_cert.loewner_valid(tol=1e-10)
    assert b_cert.brackets(K_ex, tol=1e-9)          # no quadrature slack
    assert abs(b_cert.gap_rel() - b_quad.gap_rel()) < 0.05


# ------------------------------------------------------------ elasticity ----

def _laminate_elastic(n=8, f1=0.25):
    mask = laminate(n, 1.0, 2.0, f1, axis=0) < 1.5
    Cf = mandel_field_from_mask(mask, C_STIFF, C_SOFT)
    C_ex, eps_ph, sig_ph = laminate_exact_elasticity(C_STIFF, C_SOFT, f1)
    return Cf, C_ex, eps_ph, sig_ph


def test_q1_projection_elasticity_upper_machine_precision():
    """Exact laminate displacement correctors -> C_up = C_exact exactly."""
    n, f1, ne = 8, 0.25, 16
    Cf, C_ex, eps_ph, _ = _laminate_elastic(n, f1)
    h = 1.0 / ne
    r = ne // n
    phase = np.repeat((laminate(n, 1.0, 2.0, f1, axis=0) < 1.5)[:, 0], r)
    chi = np.zeros((ne, ne, 6))
    for k in range(3):
        # fluctuation strain = eps_phase - E_k ; only y1-dependence, so
        # du/dy1 = eps11 - delta, dv/dy1 = sqrt2 eps12 - delta  (e22 has no
        # fluctuation because e22 is common to both phases)
        e = np.where(phase[:, None], eps_ph[0][:, k], eps_ph[1][:, k])  # (ne,3)
        du = e[:, 0] - (1.0 if k == 0 else 0.0)
        # Mandel: eps_hat3 = sqrt2 * eps12 and dv/dy1 = 2*eps12 = sqrt2*eps_hat3
        dv = e[:, 2] * np.sqrt(2.0) - (np.sqrt(2.0) if k == 2 else 0.0)
        u = np.concatenate([[0.0], np.cumsum(du * h)])[:ne]
        v = np.concatenate([[0.0], np.cumsum(dv * h)])[:ne]
        chi[:, :, 2 * k] = u[:, None]
        chi[:, :, 2 * k + 1] = v[:, None]
    C_up = certified_upper_elasticity(chi, Cf)
    assert np.allclose(C_up, C_ex, atol=1e-11)


def test_bfs_projection_elasticity_lower_machine_precision():
    """Exact laminate Airy potentials -> C_low = C_exact exactly (C^1)."""
    n, f1, ne = 8, 0.25, 16
    Cf, C_ex, _, sig_ph = _laminate_elastic(n, f1)
    h = 1.0 / ne
    r = ne // n
    phase = np.repeat((laminate(n, 1.0, 2.0, f1, axis=0) < 1.5)[:, 0], r)
    Minv = np.linalg.inv(C_ex)
    dofs = np.zeros((ne, ne, 3, 4))
    for j in range(3):
        # unit-mean-stress basis (MATH.md 9b normalization lemma)
        s_phase = [sum(Minv[k, j] * sig_ph[p][:, k] for k in range(3))
                   for p in (0, 1)]
        s22 = np.where(phase, s_phase[0][1], s_phase[1][1])
        c = s22 - (1.0 if j == 1 else 0.0)      # fluctuation: phi,11 = s'22
        # phi' piecewise linear, choose constant so phi is periodic
        dphi = np.concatenate([[0.0], np.cumsum(c * h)])[:ne + 1]
        seg = 0.5 * h * (dphi[:-1] + dphi[1:])
        dphi = dphi - seg.sum()                  # enforce mean(phi') = 0
        seg = 0.5 * h * (dphi[:-1] + dphi[1:])
        phi = np.concatenate([[0.0], np.cumsum(seg)])[:ne]
        dofs[:, :, j, 0] = phi[:, None]
        dofs[:, :, j, 1] = dphi[:ne][:, None]
    C_low = certified_lower_elasticity(dofs, Cf)
    assert np.allclose(C_low, C_ex, atol=1e-10)


def test_projected_trained_elasticity_valid():
    """Trained Phase-2 pair projected onto Q1 (primal) and BFS (dual):
    exactly-integrated bounds must be Loewner-valid."""
    mask = boolean_disks(8, 1.0, 2.0, n_disks=3, radius=0.2, seed=2) > 1.5
    Cf = mandel_field_from_mask(mask, C_STIFF, C_SOFT)
    _, _, (chi_net, phi_net) = fit_bounds_elasticity(
        Cf, nq=16, n_harmonics=2, width=24, depth=2, steps=60, lr=3e-3,
        seed=0, verbose=False)
    C_up = certified_upper_elasticity(nodal_values(chi_net, 16), Cf)
    C_low = certified_lower_elasticity(bfs_nodal_dofs(phi_net, 16), Cf)
    b = CertifiedBounds(C_low, C_up)
    assert b.loewner_valid(tol=1e-9)
