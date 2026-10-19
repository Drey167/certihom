"""Checkerboard demonstration: certified interval vs. the exact Dykhne value.

The 2x2 checkerboard is the canonical hard case (corner-singular local
fields) with a closed-form answer K_eff = sqrt(k1*k2) * I. This script
trains the primal-dual pair label-free and prints the resulting certified
interval, the classical Voigt-Reuss envelope it started from, and an
FV-reference audit.

Run:  python examples/run_checkerboard.py [--contrast 5] [--steps 1500]
"""

import argparse

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import numpy as np

from certihom.exact import checkerboard_exact, reuss, voigt
from certihom.fv_reference import fv_effective
from certihom.microstructures import checkerboard
from certihom.train import fit_bounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contrast", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--nq", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    k1, k2 = 1.0, args.contrast
    K = checkerboard(16, k1, k2)
    exact = checkerboard_exact(k1, k2)

    print(f"# Checkerboard, contrast {k2:g}: exact K_eff = sqrt(k1 k2) I "
          f"= {exact[0,0]:.6f} I")
    print(f"# Classical envelope: Reuss {reuss(K)[0,0]:.6f}  <=  K_eff  "
          f"<=  Voigt {voigt(K)[0,0]:.6f}\n")

    b, hist, _ = fit_bounds(
        K, nq=args.nq, n_harmonics=4, width=64, depth=3,
        steps=args.steps, lr=1e-3, seed=args.seed, log_every=250,
    )

    print("\n" + b.summary())
    lo, up = np.linalg.eigvalsh(b.K_low), np.linalg.eigvalsh(b.K_up)
    ok = lo.max() <= exact[0, 0] + 1e-3 and up.min() >= exact[0, 0] - 1e-3
    print(f"exact value       : {exact[0,0]:.6f}")
    print(f"contains exact    : {ok}")

    K_fv = fv_effective(checkerboard(64, k1, k2))
    print(f"FV(64) reference  : {K_fv[0,0]:.6f}  "
          f"(bracketed: {b.brackets(K_fv, tol=5e-3)})")


if __name__ == "__main__":
    main()
