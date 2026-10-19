"""THE DECISIVE TEST — is the amortized certificate tighter than Voigt-Reuss?

Prior-art context. The Voigt-Reuss Net (Keshav, Herb & Fritzen, GAMM-Mitt.
2025 / IJNME 2026) is an amortized neural surrogate for effective stiffness
that GUARANTEES its predictions lie inside the analytic Voigt-Reuss
envelope, by construction, including during gradient-driven design. It
overlaps this project on amortization, tensor elasticity, Loewner
bracketing and differentiable certified design.

The distinction this project must earn is that its interval is a
PER-MICROSTRUCTURE certified bracket derived from admissible trial fields,
which TIGHTENS with the actual geometry — whereas the Voigt-Reuss envelope
depends only on volume fraction and phase properties, and knows nothing
about arrangement.

So the make-or-break number is the TIGHTNESS RATIO

        R = (Voigt-Reuss relative gap) / (certified relative gap)

R = 1 means we have added nothing over the trivial envelope and the
contribution collapses into prior work. R >> 1 means the certificate is
extracting real geometric information, at inference cost, with a guarantee.

Prior art also tests only ~10:1 contrast on one-to-few inclusions
(Gaynutdinova et al.), and the group's classical stiffness work explicitly
avoids high contrast. So high contrast on varied morphologies is open
ground and is tested here deliberately.

Run:  python examples/run_voigt_reuss_benchmark.py [--steps 300]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.amortized import AmortizedConductivity, fit_amortized
from certihom.exact import (checkerboard_exact, laminate_exact, reuss, voigt)
from certihom.fft_reference import fft_effective
from certihom.microstructures import boolean_disks, checkerboard, laminate


def vr_gap(K):
    """Relative width of the trivial Voigt-Reuss envelope."""
    v, r = voigt(K), reuss(K)
    return float(np.linalg.norm(v - r) / np.linalg.norm(0.5 * (v + r)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--nq", type=int, default=32)
    args = p.parse_args()

    # Training family spans contrast, so the operator is not implicitly
    # specialised to a single ratio (prior art tests ~10:1 only).
    rng = np.random.default_rng(0)
    fam = []
    for s in range(24):
        c = float(np.exp(rng.uniform(np.log(5), np.log(100))))
        fam.append(boolean_disks(16, 1.0, c, n_disks=int(rng.integers(5, 9)),
                                 radius=float(rng.uniform(0.10, 0.16)),
                                 seed=s))
    print(f"# amortized operator, 24 training disks, contrast 5-100, "
          f"{args.steps} steps")

    model = AmortizedConductivity(kmax=4, n_harmonics=3, width=64, depth=3)
    model, _ = fit_amortized(model, fam, nq=args.nq, steps=args.steps,
                             batch=6, lr=1.5e-3, seed=0, log_every=150)

    # Held-out cases: unseen disks across contrast, plus morphologies never
    # seen in training (laminate, checkerboard) with EXACT references.
    cases = []
    for c in (10.0, 50.0, 100.0, 200.0):
        cases.append((f"disks c={c:g}", boolean_disks(16, 1.0, c, n_disks=6,
                                                      radius=0.14, seed=901),
                      None))
    cases.append(("laminate c=100", laminate(16, 1.0, 100.0, 0.25, 0),
                  laminate_exact(1.0, 100.0, 0.25, 0)))
    cases.append(("checkerboard c=100", checkerboard(16, 1.0, 100.0),
                  checkerboard_exact(1.0, 100.0)))

    print(f"\n{'case':>20} | {'V-R gap':>8} {'certified':>10} "
          f"{'RATIO':>7} | valid | contains ref")
    print("-" * 72)
    ratios = []
    for name, K, K_ex in cases:
        b = model.bounds_for(K, args.nq)
        ref = K_ex if K_ex is not None else fft_effective(K)
        g_vr, g_c = vr_gap(K), b.gap_rel()
        R = g_vr / max(g_c, 1e-12)
        ratios.append(R)
        tol = 1e-9 if K_ex is not None else 5e-2
        print(f"{name:>20} | {g_vr:7.1%} {g_c:9.1%} {R:7.1f}x | "
              f"{str(b.loewner_valid(1e-9)):>5} | {b.brackets(ref, tol=tol)}")

    print(f"\nmedian tightness ratio over held-out cases: "
          f"{np.median(ratios):.1f}x")
    print("\nR = 1x would mean the certificate adds nothing over the trivial")
    print("Voigt-Reuss envelope, and the contribution collapses into the")
    print("Voigt-Reuss Net. R >> 1x means the certificate extracts genuine")
    print("geometric information at inference cost, with a guarantee.")


if __name__ == "__main__":
    main()
