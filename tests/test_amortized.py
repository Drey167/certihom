"""Phase 3 verification ladder (amortization).

The decisive claims tested here:
 (a) conditioned formulas are bit-identical to the certified Phase 0/2
     paths on a shared network (kills transcription risk);
 (b) validity-from-step-0 survives conditioning, for every instance;
 (c) a model trained on one family gives, for HELD-OUT instances,
     Loewner-valid certificates that bracket the FV reference and are much
     tighter than Voigt-Reuss — at inference cost;
 (d) on OUT-OF-DISTRIBUTION microstructures with EXACT effective tensors
     (laminates), certificates remain valid (they must, by admissibility)
     while the gap widens — the self-diagnosis property, measured.
"""

import numpy as np
import torch

from certihom.amortized import (AmortizedConductivity, AmortizedElasticity,
                                cond_heads_and_grads, fields_mandel_dual,
                                fields_mandel_primal, fields_scalar_dual,
                                fields_scalar_primal, fit_amortized)
from certihom.elasticity import (dual_stress_fields, iso_stiffness_mandel,
                                 laminate_exact_elasticity,
                                 mandel_field_from_mask, primal_strain_fields,
                                 _hessians)
from certihom.energy import dual_fields, primal_fields
from certihom.exact import laminate_exact, reuss, voigt
from certihom.fem_reference import fem_effective
from certihom.fields import FieldPair, heads_and_grads
from certihom.fv_reference import fv_effective
from certihom.microstructures import boolean_disks, laminate
from certihom.quadrature import midpoint_grid


def test_formula_consistency_with_phase_0_2():
    """Same random-initialized nets, two code paths, identical fields."""
    pts = midpoint_grid(8)
    torch.manual_seed(1)
    net2 = FieldPair(2, 12, 2, n_out=2)
    for p in net2.head.parameters():
        torch.nn.init.normal_(p, std=0.1)
    _, g = heads_and_grads(net2, pts, False)
    for A, B in zip(fields_scalar_primal(g), primal_fields(net2, pts, False)):
        assert torch.allclose(A, B, atol=0)
    for A, B in zip(fields_scalar_dual(g), dual_fields(net2, pts, False)):
        assert torch.allclose(A, B, atol=0)

    net6 = FieldPair(2, 12, 2, n_out=6)
    net3 = FieldPair(2, 12, 2, n_out=3)
    for net in (net6, net3):
        for p in net.head.parameters():
            torch.nn.init.normal_(p, std=0.1)
    _, g6 = heads_and_grads(net6, pts, False)
    for A, B in zip(fields_mandel_primal(g6),
                    primal_strain_fields(net6, pts, False)):
        assert torch.allclose(A, B, atol=0)
    H = _hessians(net3, pts, False)
    for A, B in zip(fields_mandel_dual(H), dual_stress_fields(net3, pts, False)):
        assert torch.allclose(A, B, atol=0)


def test_amortized_zero_init_voigt_reuss_per_instance():
    """Untrained conditioned operator: exact Voigt-Reuss for EVERY input."""
    model = AmortizedConductivity(kmax=3, n_harmonics=2, width=16, depth=2)
    for seed in (0, 3, 7):
        K = boolean_disks(16, 1.0, 10.0, n_disks=6, radius=0.13, seed=seed)
        b = model.bounds_for(K, nq=32)
        assert np.allclose(b.K_up, voigt(K), atol=1e-11)
        assert np.allclose(b.K_low, reuss(K), atol=1e-10)


def test_amortized_generalization_preserves_validity():
    """Train once on a disk family; certify held-out disks AND OOD
    laminates. Runtime-budgeted for a 1-core CPU box."""
    fam = [boolean_disks(16, 1.0, 10.0, n_disks=nd, radius=r, seed=s)
           for nd, r, s in [(5, .12, 0), (6, .13, 1), (7, .11, 2), (5, .15, 3),
                            (6, .12, 4), (7, .14, 5), (5, .13, 6), (6, .15, 7),
                            (7, .12, 8), (6, .11, 9), (5, .14, 10), (7, .13, 11)]]
    test = [boolean_disks(16, 1.0, 10.0, n_disks=nd, radius=r, seed=s)
            for nd, r, s in [(6, .13, 101), (5, .14, 102), (7, .12, 103)]]
    model = AmortizedConductivity(kmax=4, n_harmonics=3, width=48, depth=3)
    model, hist = fit_amortized(model, fam, nq=32, steps=150, batch=6,
                                lr=2e-3, seed=0, log_every=75, verbose=False)

    id_gaps = []
    for K in test:
        b = model.bounds_for(K, nq=32)
        assert b.loewner_valid(tol=1e-9)
        assert b.brackets(fv_effective(K), tol=2e-2)
        vr_gap = np.linalg.norm(voigt(K) - reuss(K)) / np.linalg.norm(
            0.5 * (voigt(K) + reuss(K)))
        assert b.gap_rel() < 0.6 * vr_gap
        id_gaps.append(b.gap_rel())

    # Near-OOD (unseen FAMILY, same contrast): validity is unconditional —
    # audited against EXACT tensors with no reference-solver slack. Finding
    # (kept as data, not asserted as ordering): local grad-k~ conditioning
    # TRANSFERS — laminate gaps come out comparable to ID gaps.
    near_gaps = []
    for f1, axis in [(0.25, 0), (0.5, 1)]:
        K = laminate(16, 1.0, 10.0, f1, axis=axis)
        b = model.bounds_for(K, nq=32)
        assert b.loewner_valid(tol=1e-9)
        assert b.brackets(laminate_exact(1.0, 10.0, f1, axis), tol=1e-2)
        near_gaps.append(b.gap_rel())

    # Far-OOD (contrast shift x10 beyond anything trained): still VALID
    # against the exact tensor, and the certificate widens past the ID
    # level — width flags the SEVERITY of distribution shift.
    K = laminate(16, 1.0, 100.0, 0.25, axis=0)
    b = model.bounds_for(K, nq=32)
    assert b.loewner_valid(tol=1e-8)
    assert b.brackets(laminate_exact(1.0, 100.0, 0.25, 0), tol=0.1)
    assert b.gap_rel() > np.mean(id_gaps)


def test_amortized_elasticity_smoke():
    """Machinery generality: tiny elastic family; held-out validity with the
    rigorous one-sided FEM audit; tightening vs elasticity Voigt-Reuss."""
    C_soft = iso_stiffness_mandel(1.0, 0.3)
    C_stiff = iso_stiffness_mandel(10.0, 0.2)
    masks = [boolean_disks(8, 1.0, 2.0, n_disks=3, radius=0.2, seed=s) > 1.5
             for s in range(6)]
    fam = [mandel_field_from_mask(m, C_stiff, C_soft) for m in masks]
    held = mandel_field_from_mask(
        boolean_disks(8, 1.0, 2.0, n_disks=3, radius=0.2, seed=99) > 1.5,
        C_stiff, C_soft)
    model = AmortizedElasticity(kmax=3, n_harmonics=2, width=32, depth=2)
    b0 = model.bounds_for(held, nq=16)
    model, _ = fit_amortized(model, fam, nq=16, steps=80, batch=3,
                             lr=3e-3, seed=0, log_every=40, verbose=False)
    b = model.bounds_for(held, nq=16)
    assert b.loewner_valid(tol=1e-9)
    assert b.gap_rel() < 0.75 * b0.gap_rel()
    # rigorous one-sided audit on the held-out instance
    assert np.linalg.eigvalsh(fem_effective(held) - b.K_low).min() >= -1e-9
