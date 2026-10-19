"""Phase 2 demonstration: certified elasticity-tensor interval for a random
stiff-particle composite, plus certified directional Young's moduli.

Material: soft matrix (E=1, nu=0.3), stiff inclusions (E=10, nu=0.2),
plane strain. Outputs the certified Loewner interval [C_low, C_up], the
derived certified intervals for E_x and E_y (engineering deliverable), and
the RIGOROUS one-sided audit against conforming periodic Q1 FEM:
C_FEM(h) >= C_eff >= C_low must hold at every mesh level with no tolerance.

Run:  python examples/run_elasticity_disks.py [--steps 450]
"""

import argparse

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.elasticity import (iso_stiffness_mandel, mandel_field_from_mask,
                                 youngs_interval)
from certihom.fem_reference import fem_effective
from certihom.microstructures import boolean_disks
from certihom.train_elast import fit_bounds_elasticity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=450)
    p.add_argument("--nq", type=int, default=48)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    C_soft = iso_stiffness_mandel(1.0, 0.3)
    C_stiff = iso_stiffness_mandel(10.0, 0.2)
    mask = boolean_disks(16, 1.0, 2.0, n_disks=6, radius=0.15,
                         seed=args.seed) > 1.5
    Cf = mandel_field_from_mask(mask, C_stiff, C_soft)
    print(f"# Stiff disks (E contrast 10), inclusion fraction "
          f"{mask.mean():.3f}, plane strain\n")

    b, hist, _ = fit_bounds_elasticity(
        Cf, nq=args.nq, n_harmonics=3, width=48, depth=3,
        steps=args.steps, lr=1.5e-3, seed=0, log_every=150,
    )

    print("\n" + b.summary())
    with np.printoptions(precision=4, suppress=True):
        print("C_low =\n", b.K_low)
        print("C_up  =\n", b.K_up)

    for name, d in (("E_x", [1.0, 0.0]), ("E_y", [0.0, 1.0])):
        lo, hi = youngs_interval(b.K_low, b.K_up, np.array(d))
        print(f"certified {name} interval : [{lo:.4f}, {hi:.4f}]  "
              f"(width {(hi-lo)/(0.5*(hi+lo)):.2%})")

    # Rigorous one-sided audit at two mesh levels: min eig(C_FEM - C_low) >= 0.
    for lbl, Cref in (("h", fem_effective(Cf)),
                      ("h/2", fem_effective(
                          np.kron(Cf, np.ones((2, 2, 1, 1))).reshape(32, 32, 3, 3)))):
        gap = np.linalg.eigvalsh(Cref - b.K_low).min()
        inside = b.brackets(Cref, tol=5e-2)
        print(f"FEM({lbl}): min eig(C_FEM - C_low) = {gap:+.4e} "
              f"(must be >= 0) | inside certified interval: {inside}")


if __name__ == "__main__":
    main()
