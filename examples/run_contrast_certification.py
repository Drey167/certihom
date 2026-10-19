"""Study: phase contrast sweep + the price of EXACT certification.

Answers two reviewer questions with one experiment on a fixed geometry:

1. "Do the bounds degrade uselessly at high contrast?" — sweep contrast and
   report the certified gap against the Voigt-Reuss level it starts from.
2. "Your certificates are only guaranteed up to quadrature error." — no
   longer: project the trained fields onto conforming Q1 (Phase 1) and
   integrate exactly. The table reports both numbers, so the price of
   removing the caveat is measured, not asserted.

Run:  python examples/run_contrast_certification.py [--steps 250]
"""

import argparse

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.exact import reuss, voigt
from certihom.fv_reference import fv_effective
from certihom.microstructures import boolean_disks
from certihom.train import fit_bounds


def vr_gap(K):
    v, r = voigt(K), reuss(K)
    return float(np.linalg.norm(v - r) / np.linalg.norm(0.5 * (v + r)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--contrasts", type=float, nargs="+",
                   default=[2.0, 10.0, 50.0, 200.0])
    args = p.parse_args()

    print("# Fixed disk geometry (6 disks, r=0.14), contrast swept.")
    print("# 'quad' = midpoint-rule bound (Phases 0-3);")
    print("# 'exact' = Q1-projected, exactly integrated (Phase 1) — no caveat.\n")
    print(f"{'contrast':>9} | {'V-R gap':>8} | {'quad gap':>8} | "
          f"{'exact gap':>9} | {'price':>7} | valid | brackets FV")
    print("-" * 78)

    for c in args.contrasts:
        K = boolean_disks(16, 1.0, c, n_disks=6, radius=0.14, seed=1)
        b_q, _, (chi, psi) = fit_bounds(
            K, nq=32, n_harmonics=4, width=48, depth=3, steps=args.steps,
            lr=2e-3, seed=0, verbose=False)
        b_e = certified_bounds_scalar(nodal_values(chi, 32),
                                      nodal_values(psi, 32), K)
        price = b_e.gap_rel() - b_q.gap_rel()
        ok = b_e.loewner_valid(1e-10)
        br = b_e.brackets(fv_effective(K), tol=3e-2)
        print(f"{c:9.0f} | {vr_gap(K):7.1%} | {b_q.gap_rel():7.1%} | "
              f"{b_e.gap_rel():8.1%} | {price:+6.1%} | {str(ok):>5} | {br}")

    print("\nReading: the certified gap grows with contrast (expected — the "
          "\nprimal-dual pair is tightest near contrast 1), but stays far "
          "\nbelow the Voigt-Reuss envelope, and removing the quadrature "
          "\ncaveat costs only the 'price' column.")


if __name__ == "__main__":
    main()
