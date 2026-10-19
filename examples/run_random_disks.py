"""Random Boolean-disk microstructure: full-matrix bracket audit.

Unlike the symmetric checkerboard, a random disk arrangement has no special
symmetry, so the certified interval must bracket a genuinely anisotropic
2x2 reference tensor (off-diagonals included) — this exercises the Loewner
machinery beyond eigenvalue comparisons on multiples of the identity.

Run:  python examples/run_random_disks.py [--steps 1000]
"""

import argparse

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import numpy as np

from certihom.fv_reference import fv_effective
from certihom.microstructures import boolean_disks
from certihom.train import fit_bounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    K = boolean_disks(32, k_matrix=1.0, k_inclusion=10.0,
                      n_disks=7, radius=0.14, seed=args.seed)
    print(f"# Boolean disks, contrast 10, inclusion fraction "
          f"{(K > 1).mean():.3f}\n")

    b, hist, _ = fit_bounds(
        K, nq=64, n_harmonics=4, width=64, depth=3,
        steps=args.steps, lr=1e-3, seed=0, log_every=250,
    )

    print("\n" + b.summary())
    with np.printoptions(precision=5):
        print("K_low =\n", b.K_low)
        print("K_up  =\n", b.K_up)

    # FV audit: solve the SAME piecewise-constant geometry on its native
    # pixel grid and on a Kron-refined grid (identical conductivity field,
    # halved mesh size) so the reference's own discretization error is
    # visible while the microstructure stays fixed.
    K_fv32 = fv_effective(K)
    K_fv64 = fv_effective(np.kron(K, np.ones((2, 2))))
    with np.printoptions(precision=5):
        print("FV(h)   =\n", K_fv32)
        print("FV(h/2) =\n", K_fv64)
    print(f"brackets FV(h)  : {b.brackets(K_fv32, tol=2e-2)}")
    print(f"brackets FV(h/2): {b.brackets(K_fv64, tol=2e-2)}")


if __name__ == "__main__":
    main()
