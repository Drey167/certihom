"""Phase 2 verification ladder (elasticity).

Same philosophy as Phase 0: certify the production Gram path with fields
whose energies are known exactly, audit the auditor (FEM), then let a short
training run demonstrate tightening without ever risking validity.
"""

import numpy as np
import torch

from certihom.bounds import CertifiedBounds
from certihom.elasticity import (SQ2, dual_gram_elast, gram_mandel,
                                 iso_stiffness_mandel,
                                 laminate_exact_elasticity,
                                 mandel_field_from_mask, sample_mandel,
                                 upper_gram_elast, youngs_interval)
from certihom.energy import as_numpy
from certihom.fem_reference import fem_effective
from certihom.fields import FieldPair
from certihom.microstructures import boolean_disks, laminate
from certihom.quadrature import midpoint_grid
from certihom.train_elast import fit_bounds_elasticity

C_SOFT = iso_stiffness_mandel(1.0, 0.3)
C_STIFF = iso_stiffness_mandel(10.0, 0.2)
I3 = torch.eye(3, dtype=torch.float64)


def test_mandel_material_sanity():
    """Shear entry 2*mu; plane-strain uniaxial compliance (1-nu^2)/E."""
    E, nu = 3.0, 0.25
    C = iso_stiffness_mandel(E, nu)
    mu = E / (2 * (1 + nu))
    assert abs(C[2, 2] - 2 * mu) < 1e-14
    S = np.linalg.inv(C)
    assert abs(S[0, 0] - (1 - nu**2) / E) < 1e-14
    # SPD
    assert np.linalg.eigvalsh(C).min() > 0


def test_homogeneous_collapse_elasticity():
    """Uniform material: zero trial fields give C_low = C_up = C exactly."""
    n = 8
    Cf = mandel_field_from_mask(np.zeros((n, n), bool), C_STIFF, C_SOFT)
    pts = midpoint_grid(16)
    Cv = sample_mandel(Cf, pts)
    Sv = torch.linalg.inv(Cv)
    fields = [I3[k].expand(pts.shape[0], 3).clone() for k in range(3)]
    C_up = as_numpy(gram_mandel(fields, Cv))
    C_low = as_numpy(torch.linalg.inv(gram_mandel(fields, Sv)))
    assert np.allclose(C_up, C_SOFT, atol=1e-12)
    assert np.allclose(C_low, C_SOFT, atol=1e-12)


def test_zero_init_networks_give_voigt_reuss_elasticity():
    mask = boolean_disks(16, 1.0, 2.0, n_disks=4, radius=0.16, seed=5) > 1.5
    Cf = mandel_field_from_mask(mask, C_STIFF, C_SOFT)
    pts = midpoint_grid(32)
    Cv = sample_mandel(Cf, pts)
    Sv = torch.linalg.inv(Cv)
    chi = FieldPair(2, 16, 2, n_out=6)
    phi = FieldPair(2, 16, 2, n_out=3)
    C_up = as_numpy(upper_gram_elast(chi, pts, Cv))
    C_low = as_numpy(torch.linalg.inv(dual_gram_elast(phi, pts, Sv)))
    voigt = as_numpy(Cv.mean(dim=0))
    reuss = np.linalg.inv(as_numpy(Sv.mean(dim=0)))
    assert np.allclose(C_up, voigt, atol=1e-12)
    assert np.allclose(C_low, reuss, atol=1e-11)


def _laminate_setup(n=8, f1=0.25):
    mask = laminate(n, 1.0, 2.0, f1, axis=0) < 1.5  # phase 1 where k==1
    Cf = mandel_field_from_mask(mask, C_STIFF, C_SOFT)  # phase1=STIFF
    C_ex, eps_ph, sig_ph = laminate_exact_elasticity(C_STIFF, C_SOFT, f1)
    return mask, Cf, C_ex, eps_ph, sig_ph


def test_laminate_analytic_fields_machine_precision_elasticity():
    """The Phase-2 money test: exact per-phase strain/stress states pushed
    through the production Mandel Gram give C_up = C_low = C_exact to
    ~1e-11; the two bounds coinciding independently certifies C_exact."""
    n, f1 = 8, 0.25
    mask, Cf, C_ex, eps_ph, sig_ph = _laminate_setup(n, f1)
    pts = midpoint_grid(32)
    Cv = sample_mandel(Cf, pts)
    Sv = torch.linalg.inv(Cv)
    phase = (pts[:, 0] < f1).long()  # 0 -> phase2? careful: mask True where y1<f1
    # mask True (stiff) where y1 < f1; index eps_ph rows: 0=phase1(stiff)
    sel = torch.where(phase == 1, 0, 1)  # point -> phase row
    prim = [torch.as_tensor(eps_ph[:, :, k])[sel] for k in range(3)]
    # The strain-basis stress solutions have mean stress C_exact[:, k]; the
    # dual Gram expects UNIT mean stress, so recombine linearly (the stress
    # solution map is linear in the mean stress): sigma(T_j) =
    # sum_k [C_exact^{-1}]_{k j} sigma^{(k)}. Their S-weighted Gram is then
    # S_eff itself and C_low = C_exact — the squeeze that certifies both
    # sides at machine precision.
    sig_fields = [torch.as_tensor(sig_ph[:, :, k])[sel] for k in range(3)]
    Minv = torch.as_tensor(np.linalg.inv(C_ex))
    dual = [sum(Minv[k, j] * sig_fields[k] for k in range(3)) for j in range(3)]
    C_up = as_numpy(gram_mandel(prim, Cv))
    C_low = as_numpy(torch.linalg.inv(gram_mandel(dual, Sv)))
    assert np.allclose(C_up, C_ex, atol=1e-11)
    assert np.allclose(C_low, C_ex, atol=1e-11)
    b = CertifiedBounds(C_low, C_up)
    assert b.loewner_valid(tol=1e-10)
    # sanity: exact tensor sits inside Voigt-Reuss
    voigt = as_numpy(Cv.mean(dim=0))
    reuss = np.linalg.inv(as_numpy(Sv.mean(dim=0)))
    assert np.linalg.eigvalsh(voigt - C_ex).min() > -1e-10
    assert np.linalg.eigvalsh(C_ex - reuss).min() > -1e-10


def test_fem_laminate_exact_and_upper_bound_property():
    """Q1 FEM contains the exact laminate solution -> machine precision;
    and on a random microstructure C_FEM must itself be an upper bound
    (checked against the certified lower bound in the smoke test)."""
    _, Cf, C_ex, _, _ = _laminate_setup(8, 0.25)
    C_fem = fem_effective(Cf)
    assert np.allclose(C_fem, C_ex, atol=1e-9)


def test_training_smoke_elasticity():
    """Bounds tighten from Voigt-Reuss, stay Loewner-valid, and respect the
    rigorous one-sided audit C_low <= C_FEM(h) at two mesh levels."""
    mask = boolean_disks(8, 1.0, 2.0, n_disks=3, radius=0.2, seed=2) > 1.5
    Cf = mandel_field_from_mask(mask, C_STIFF, C_SOFT)
    b, hist, _ = fit_bounds_elasticity(
        Cf, nq=24, n_harmonics=2, width=24, depth=2,
        steps=120, lr=3e-3, seed=0, log_every=60, verbose=False,
    )
    assert b.loewner_valid(tol=1e-9)
    assert b.gap_rel() < 0.7 * hist[0]["gap_rel"]
    for Cref in (fem_effective(Cf),
                 fem_effective(np.kron(Cf, np.ones((2, 2, 1, 1))).reshape(16, 16, 3, 3))):
        # rigorous one-sided audit: C_FEM >= C_eff >= C_low, no tolerance games
        assert np.linalg.eigvalsh(Cref - b.K_low).min() >= -1e-9


def test_youngs_interval_orders():
    """Certified E(d) interval must be ordered and contain the FEM value's
    implied modulus for a quick random case."""
    mask = boolean_disks(8, 1.0, 2.0, n_disks=3, radius=0.2, seed=2) > 1.5
    Cf = mandel_field_from_mask(mask, C_STIFF, C_SOFT)
    b, _, _ = fit_bounds_elasticity(Cf, nq=16, n_harmonics=2, width=16,
                                    depth=2, steps=60, lr=3e-3, seed=0,
                                    verbose=False)
    lo, hi = youngs_interval(b.K_low, b.K_up, np.array([1.0, 0.0]))
    assert lo <= hi
    S_fem = np.linalg.inv(fem_effective(np.kron(Cf, np.ones((2, 2, 1, 1))).reshape(16, 16, 3, 3)))
    a = np.array([1.0, 0.0, 0.0])
    E_fem = 1.0 / float(a @ S_fem @ a)
    # Rigorous side: C_FEM >= C_eff  =>  E_fem >= E_eff >= lo, exactly.
    assert lo <= E_fem * (1 + 1e-9)
    # Loose side: E_fem and hi both over-estimate E_eff with no mutual
    # ordering; a barely-trained C_up sits near Voigt, so 5% slack suffices.
    assert E_fem <= hi * 1.05
