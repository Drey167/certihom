"""THE DECISIVE TEST vs the Voigt-Reuss Net class of surrogates.

Prior-art context. Keshav, Herb & Fritzen's "Voigt-Reuss Net"
(GAMM-Mitteilungen 2025; IJNME 2026) is an amortized neural surrogate for
effective stiffness that guarantees, by spectral normalisation, that its
predictions lie inside the analytic VOIGT-REUSS envelope in the Loewner
sense -- during training, inference and gradient-based design. That
overlaps this project on four axes at once: amortization, tensor
elasticity, Loewner bracketing, and differentiable certified design.

The ONE thing that distinguishes a variational certificate from that
construction: the Voigt-Reuss envelope depends only on VOLUME FRACTION, so
it is identical for every microstructure with the same phase fractions and
carries no geometric information. A variational certificate is computed
from admissible trial fields for THIS microstructure, so it can be strictly
tighter -- and it must be, or this project is a subset of theirs.

This script measures exactly that, on UNSEEN microstructures, with
exact-integration certificates (no quadrature caveat):

    tightening factor = (Voigt-Reuss relative width) / (certified width)

A factor near 1 would be fatal. A large factor, plus the demonstration that
geometry MOVES the certificate while Voigt-Reuss stays fixed, is the
project's core defence.

Run:  python examples/run_vs_voigt_reuss.py [--steps 300]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.amortized import AmortizedConductivity, fit_amortized
from certihom.certify import certified_bounds_scalar, node_grid
from certihom.exact import checkerboard_exact, laminate_exact, reuss, voigt
from certihom.microstructures import boolean_disks, checkerboard, laminate


def rel_width(lo, up):
    mid = 0.5 * (up + lo)
    return float(np.linalg.norm(up - lo) / np.linalg.norm(mid))


def certify_exact(model, K, ne):
    """Exact-integration certificate from the amortized model's fields."""
    import torch
    coeffs, glob = model.condition(K)
    y = node_grid(ne)
    c = coeffs.unsqueeze(0).expand(y.shape[0], -1, -1, -1)
    g = glob.unsqueeze(0).expand(y.shape[0], -1)

    def nod(net):
        with torch.no_grad():
            out = net(y, model.cond(y, c, g))
        return out.reshape(ne, ne, -1).numpy()

    return certified_bounds_scalar(nod(model.chi), nod(model.dual_net), K)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--nq", type=int, default=32)
    args = p.parse_args()

    C = 10.0
    fam = [boolean_disks(16, 1.0, C, n_disks=int(d), radius=float(r), seed=s)
           for d, r, s in [(5, .12, 0), (6, .13, 1), (7, .11, 2), (6, .15, 3),
                           (5, .14, 4), (7, .12, 5), (6, .11, 6), (5, .15, 7)]]
    print(f"# Amortized operator, contrast {C:g}, trained on {len(fam)} disk "
          f"microstructures, {args.steps} steps")
    model = AmortizedConductivity(kmax=4, n_harmonics=3, width=64, depth=3)
    model, _ = fit_amortized(model, fam, nq=args.nq, steps=args.steps,
                             batch=4, lr=2e-3, seed=0, log_every=150,
                             verbose=False)

    # Held-out instances, deliberately spanning morphologies and including
    # ones far outside the training family.
    tests = [
        ("disks (held out)", boolean_disks(16, 1., C, 6, .13, 101), None),
        ("disks (held out)", boolean_disks(16, 1., C, 5, .14, 102), None),
        ("disks (held out)", boolean_disks(16, 1., C, 7, .12, 103), None),
        ("laminate f=.25 (OOD)", laminate(16, 1., C, .25, 0),
         laminate_exact(1., C, .25, 0)),
        ("checkerboard (OOD)", checkerboard(16, 1., C), checkerboard_exact(1., C)),
    ]

    print(f"\n{'microstructure':>22} | {'V-R width':>10} {'certified':>10} "
          f"{'tightening':>11} | {'inside V-R':>10} {'valid':>6}")
    print("-" * 82)
    factors = []
    for name, K, K_ex in tests:
        b = certify_exact(model, K, args.nq)
        vr = rel_width(reuss(K), voigt(K))
        ce = b.gap_rel()
        f = vr / ce
        factors.append(f)
        # is the certified interval strictly inside the Voigt-Reuss envelope?
        inside = (np.linalg.eigvalsh(b.K_low - reuss(K)).min() >= -1e-9 and
                  np.linalg.eigvalsh(voigt(K) - b.K_up).min() >= -1e-9)
        ok = b.loewner_valid(1e-9) and (K_ex is None or b.brackets(K_ex, 1e-2))
        print(f"{name:>22} | {vr:9.1%} {ce:9.1%} {f:10.1f}x | "
              f"{str(inside):>10} {str(ok):>6}")

    print(f"\nmedian tightening factor: {np.median(factors):.1f}x")

    # The geometric-information argument, made concrete.
    print("\n## Voigt-Reuss carries NO geometric information — proof by example")
    print(f"{'microstructure':>22} | {'volume frac':>11} {'V-R width':>10} "
          f"{'certified':>10}")
    print("-" * 60)
    same = [("laminate f=.50", laminate(16, 1., C, .5, 0)),
            ("checkerboard", checkerboard(16, 1., C))]
    for name, K in same:
        b = certify_exact(model, K, args.nq)
        vf = float((K > 0.5 * (1 + C)).mean())
        print(f"{name:>22} | {vf:10.3f} {rel_width(reuss(K), voigt(K)):9.1%} "
              f"{b.gap_rel():9.1%}")
    print("\nIdentical volume fraction -> IDENTICAL Voigt-Reuss envelope, but")
    print("different geometry -> different certified interval. A guarantee")
    print("that depends only on volume fraction cannot distinguish these.")


if __name__ == "__main__":
    main()
