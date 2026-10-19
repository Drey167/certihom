"""Certificate-machinery tests.

Philosophy: test the EXACT production code path (``energy.gram`` and
friends) by feeding it fields whose energies are known in closed form. If
these pass at machine precision, the bound assembly, quadrature alignment,
Loewner logic, and dual inversion are all correct; NN training can then only
affect tightness, never validity.
"""

import numpy as np
import torch

from certihom.bounds import CertifiedBounds
from certihom.energy import as_numpy, dual_gram, gram, lower_from_dual, upper_gram
from certihom.exact import laminate_exact, reuss, voigt
from certihom.fields import FieldPair
from certihom.microstructures import homogeneous, laminate, boolean_disks, sample_k
from certihom.quadrature import midpoint_grid

E = torch.eye(2, dtype=torch.float64)


def test_homogeneous_zero_fields_collapse():
    """k = k0: zero fluctuations are optimal; both bounds equal k0*I."""
    k0 = 2.7
    K = homogeneous(8, k0)
    pts = midpoint_grid(32)
    kv = sample_k(K, pts)
    fields = [E[0].expand_as(pts), E[1].expand_as(pts)]
    K_up = gram(fields, kv)
    R_up = gram(fields, 1.0 / kv)
    b = CertifiedBounds(as_numpy(lower_from_dual(R_up)), as_numpy(K_up))
    assert np.allclose(b.K_up, k0 * np.eye(2), atol=1e-12)
    assert np.allclose(b.K_low, k0 * np.eye(2), atol=1e-12)
    assert b.loewner_valid()


def test_zero_init_networks_give_voigt_reuss():
    """Untrained (zero-init head) networks must reproduce the classical
    Voigt and Reuss bounds exactly — the 'valid from step 0' property."""
    K = boolean_disks(16, 1.0, 8.0, n_disks=5, radius=0.15, seed=3)
    pts = midpoint_grid(64)
    kv = sample_k(K, pts)
    chi, psi = FieldPair(3, 16, 2), FieldPair(3, 16, 2)
    K_up = as_numpy(upper_gram(chi, pts, kv))
    K_low = as_numpy(lower_from_dual(dual_gram(psi, pts, kv)))
    assert np.allclose(K_up, voigt(K), atol=1e-12)
    assert np.allclose(K_low, reuss(K), atol=1e-12)


def _laminate_exact_fields(K, pts, k1, k2, f1):
    """Closed-form OPTIMAL admissible fields for a laminate stacked along
    axis 0 (derivations in MATH.md, 'Laminate worked example'):

    primal, load e1 (across): total gradient (harm/k(y)) e1  (uniform flux);
    primal, load e2 (along) : total gradient e2               (chi_2 = 0);
    dual,   mean flux e1    : total flux e1                    (psi_1 = 0);
    dual,   mean flux e2    : total flux (k(y)/arith) e2 — the fluctuation
        depends on y1 only, so it is the curl of a periodic stream function
        psi(y1); admissibility is exact.
    All four have gradients constant per pixel, so midpoint quadrature is
    EXACT and machine precision is a fair demand.
    """
    kv = sample_k(K, pts)
    f2 = 1.0 - f1
    harm = 1.0 / (f1 / k1 + f2 / k2)
    arith = f1 * k1 + f2 * k2
    prim = [ (harm / kv).unsqueeze(1) * E[0], E[1].expand_as(pts).clone() ]
    dual = [ E[0].expand_as(pts).clone(), (kv / arith).unsqueeze(1) * E[1] ]
    return prim, dual, kv


def test_laminate_analytic_fields_machine_precision():
    """The money test: exact optimal fields through the production Gram
    assembly must give K_low == K_up == K_exact to ~1e-12."""
    k1, k2, f1, n = 1.0, 6.0, 0.25, 8
    K = laminate(n, k1, k2, f1, axis=0)
    pts = midpoint_grid(32)  # multiple of n -> pixel-aligned, rule exact
    prim, dual, _ = _laminate_exact_fields(K, pts, k1, k2, f1)
    kv = sample_k(K, pts)
    K_up = as_numpy(gram(prim, kv))
    K_low = as_numpy(lower_from_dual(gram(dual, 1.0 / kv)))
    K_ex = laminate_exact(k1, k2, f1, axis=0)
    assert np.allclose(K_up, K_ex, atol=1e-12)
    assert np.allclose(K_low, K_ex, atol=1e-12)
    b = CertifiedBounds(K_low, K_up)
    assert b.loewner_valid(tol=1e-11)
    assert b.gap_rel() < 1e-11


def test_arbitrary_admissible_fields_still_bracket():
    """Validity must be independent of field quality: random periodic
    Fourier trial fields (analytic gradients/curls, NOT the optimum) have to
    bracket the exact laminate tensor. This is the 'a bad model widens, never
    invalidates' invariant."""
    k1, k2, f1, n = 1.0, 9.0, 0.5, 8
    K = laminate(n, k1, k2, f1, axis=0)
    pts = midpoint_grid(64)
    kv = sample_k(K, pts)
    rng = np.random.default_rng(0)

    def random_periodic_grad():
        """Gradient of a random low-order Fourier scalar (hence exactly the
        gradient of a periodic field): sum_m a_m grad sin(2pi m.y + phase)."""
        g = torch.zeros_like(pts)
        for _ in range(4):
            m = torch.tensor(rng.integers(1, 3, size=2), dtype=torch.float64)
            a, ph = 0.2 * rng.standard_normal(), 2 * np.pi * rng.random()
            phase = 2 * np.pi * (pts * m).sum(1) + ph
            g += a * 2 * np.pi * torch.cos(phase).unsqueeze(1) * m
        return g

    prim = [E[i] + random_periodic_grad() for i in range(2)]
    def rot(g):  # curl of the same random scalars: rotate the gradient
        return torch.stack([g[:, 1], -g[:, 0]], dim=1)
    dual = [E[i] + rot(random_periodic_grad()) for i in range(2)]

    K_up = as_numpy(gram(prim, kv))
    K_low = as_numpy(lower_from_dual(gram(dual, 1.0 / kv)))
    b = CertifiedBounds(K_low, K_up)
    K_ex = laminate_exact(k1, k2, f1, axis=0)
    assert b.loewner_valid(tol=1e-10)
    assert b.brackets(K_ex, tol=1e-10)
    assert b.gap_rel() > 1e-3  # and they are strictly suboptimal fields
