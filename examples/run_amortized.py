"""Phase 3 demonstration: one operator, certificates at inference cost.

Trains an amortized conductivity operator on a family of random disk
microstructures (contrast 10), then — with NO further training — certifies:
  * held-out in-distribution instances (validity audited vs FV),
  * near-OOD laminates and a checkerboard (validity audited vs EXACT
    tensors — no reference-solver slack),
  * a far-OOD contrast-100 laminate (validity vs exact; width widens =
    self-diagnosed distribution shift).
Also measures the amortization economics: inference seconds per certificate
vs a per-instance Phase-0 training run on the same instance, and the
tightness price of amortization.

Run:  python examples/run_amortized.py [--steps 450]
"""

import argparse

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.amortized import AmortizedConductivity, fit_amortized
from certihom.exact import checkerboard_exact, laminate_exact, reuss, voigt
from certihom.fv_reference import fv_effective
from certihom.microstructures import boolean_disks, checkerboard, laminate
from certihom.train import fit_bounds


def vr_gap(K):
    v, r = voigt(K), reuss(K)
    return float(np.linalg.norm(v - r) / np.linalg.norm(0.5 * (v + r)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--nq", type=int, default=32)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    fam = [boolean_disks(16, 1.0, 10.0,
                         n_disks=int(rng.integers(5, 9)),
                         radius=float(rng.uniform(0.10, 0.16)), seed=s)
           for s in range(40)]
    test = [boolean_disks(16, 1.0, 10.0,
                          n_disks=int(rng.integers(5, 9)),
                          radius=float(rng.uniform(0.10, 0.16)), seed=1000 + s)
            for s in range(8)]

    print(f"# Amortized operator: 40 training disks (contrast 10), "
          f"{len(test)} held-out\n")
    model = AmortizedConductivity(kmax=4, n_harmonics=3, width=64, depth=3)
    model, hist = fit_amortized(model, fam, nq=args.nq, steps=args.steps,
                                batch=8, lr=1.5e-3, seed=0, log_every=150)

    # ---- held-out ID: validity + tightness + inference cost --------------
    print("\n## Held-out in-distribution (no training on these):")
    t0 = time.time()
    gaps, ok = [], True
    for K in test:
        b = model.bounds_for(K, args.nq)
        ok &= b.loewner_valid(1e-9) and b.brackets(fv_effective(K), tol=2e-2)
        gaps.append(b.gap_rel())
    t_inf = (time.time() - t0) / len(test)
    print(f"validity (Loewner + FV bracket, 8/8) : {ok}")
    print(f"certified gap  mean {np.mean(gaps):.2%}  max {np.max(gaps):.2%}  "
          f"(Voigt-Reuss level: {np.mean([vr_gap(K) for K in test]):.2%})")
    print(f"inference cost per certificate       : {t_inf*1e3:.0f} ms "
          f"(includes the FV audit)")

    # ---- amortization price vs per-instance ------------------------------
    K0 = test[0]
    b_am = model.bounds_for(K0, args.nq)
    t0 = time.time()
    b_pi, _, _ = fit_bounds(K0, nq=64, n_harmonics=4, width=64, depth=3,
                            steps=400, lr=1e-3, seed=0, verbose=False)
    t_pi = time.time() - t0
    print(f"\n## Amortization economics on one instance:")
    print(f"per-instance Phase-0 fit : gap {b_pi.gap_rel():.2%} in {t_pi:.0f} s")
    print(f"amortized inference      : gap {b_am.gap_rel():.2%} in "
          f"{t_inf*1e3:.0f} ms  (~{t_pi/max(t_inf,1e-9):.0f}x faster)")

    # ---- OOD: validity vs EXACT tensors ----------------------------------
    print("\n## Out-of-distribution (audited against EXACT tensors):")
    for name, K, K_ex, tol in [
        ("laminate f=0.25 (near-OOD)", laminate(16, 1, 10, 0.25, 0),
         laminate_exact(1, 10, 0.25, 0), 1e-2),
        ("laminate f=0.50 (near-OOD)", laminate(16, 1, 10, 0.5, 1),
         laminate_exact(1, 10, 0.5, 1), 1e-2),
        ("checkerboard (near-OOD)", checkerboard(16, 1, 10),
         checkerboard_exact(1, 10), 1e-2),
        ("laminate contrast 100 (far-OOD)", laminate(16, 1, 100, 0.25, 0),
         laminate_exact(1, 100, 0.25, 0), 0.1),
    ]:
        b = model.bounds_for(K, args.nq)
        print(f"{name:34s}: valid {b.loewner_valid(1e-8)}, "
              f"contains exact {b.brackets(K_ex, tol=tol)}, "
              f"gap {b.gap_rel():.2%}")
    print("\n(near-OOD gaps at the ID level = the local-feature conditioning "
          "TRANSFERS;\n far-OOD widening = self-diagnosed contrast shift)")


if __name__ == "__main__":
    main()
