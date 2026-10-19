"""The limits table in the HEADLINE PHYSICS: elasticity.

The conductivity map showed morphology dominates contrast, with
corner-singular microstructures failing above ~50:1. Elasticity has a
richer failure surface (three loading modes, and the Airy dual carries
second derivatives), so the same map must be measured, not assumed.

All certificates by exact integration: Q1 for the primal displacement
fields, Bogner-Fox-Schmit C1 for the Airy dual. Every entry carries a
label-free quality check (primal bandwidth vs Nyquist, and the refinement
ratio of the primal energy) so a wide result can be attributed either to a
hard problem or to a bad run.

Run:  python examples/run_contrast_morphology_elastic.py [--steps 150]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from certihom.bfs import bfs_nodal_dofs, certified_lower_elasticity
from certihom.bounds import CertifiedBounds
from certihom.certify import certified_upper_elasticity, nodal_values
from certihom.diagnostics import spectral_bandwidth
from certihom.elasticity import (iso_stiffness_mandel,
                                 laminate_exact_elasticity,
                                 mandel_field_from_mask, sample_mandel,
                                 upper_gram_elast)
from certihom.energy import as_numpy
from certihom.microstructures import boolean_disks, checkerboard, laminate
from certihom.quadrature import midpoint_grid
from certihom.train_elast import fit_bounds_elasticity


def voigt_reuss(Cf):
    V = Cf.reshape(-1, 3, 3).mean(axis=0)
    R = np.linalg.inv(np.linalg.inv(Cf.reshape(-1, 3, 3)).mean(axis=0))
    return V, R


def rel_width(lo, up):
    return float(np.linalg.norm(up - lo) / np.linalg.norm(0.5 * (up + lo)))


def refine_ratio_elastic(chi, Cf, nq, factor=2):
    """Trace of C_up at nq vs at factor*nq — the honesty check."""
    vals = []
    for q in (nq, factor * nq):
        pts = midpoint_grid(q)
        vals.append(float(np.trace(as_numpy(
            upper_gram_elast(chi, pts, sample_mandel(Cf, pts))))))
    return vals[1] / max(vals[0], 1e-30)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--nq", type=int, default=32)
    p.add_argument("--ne", type=int, default=32)
    p.add_argument("--contrasts", type=float, nargs="+", default=[10., 50., 200.])
    args = p.parse_args()

    C_SOFT = iso_stiffness_mandel(1.0, 0.3)
    print(f"# Elasticity limits map (nq={args.nq}, {args.steps} steps, "
          f"exact integration: Q1 primal + BFS C1 dual)")
    print(f"# matrix E=1 nu=0.3; inclusions E=contrast nu=0.2; plane strain\n")
    print(f"{'contrast':>9} {'morphology':>13} | {'V-R gap':>8} "
          f"{'certified':>10} {'tightening':>11} | {'headroom':>8} "
          f"{'ratio':>6} {'gate':>7} | valid")
    print("-" * 88)

    table = {}
    for c in args.contrasts:
        C_STIFF = iso_stiffness_mandel(c, 0.2)
        cases = [
            ("disks", mandel_field_from_mask(
                boolean_disks(16, 1., 2., 6, 0.14, 1) > 1.5, C_STIFF, C_SOFT),
             None),
            ("laminate", mandel_field_from_mask(
                laminate(16, 1., 2., 0.25, 0) < 1.5, C_STIFF, C_SOFT),
             laminate_exact_elasticity(C_STIFF, C_SOFT, 0.25)[0]),
            ("checkerboard", mandel_field_from_mask(
                checkerboard(16, 1., 2.) < 1.5, C_STIFF, C_SOFT), None),
        ]
        for name, Cf, C_ex in cases:
            _, _, (chi, phi) = fit_bounds_elasticity(
                Cf, nq=args.nq, n_harmonics=3, width=48, depth=3,
                steps=args.steps, lr=2e-3, seed=0, verbose=False)
            b = CertifiedBounds(
                certified_lower_elasticity(bfs_nodal_dofs(phi, args.ne), Cf),
                certified_upper_elasticity(nodal_values(chi, args.ne), Cf))
            B = spectral_bandwidth(chi)
            head = (args.nq / 2) / max(B, 1e-9)
            rr = refine_ratio_elastic(chi, Cf, args.nq)
            gate = head >= 2.0 and rr <= 1.05
            V, R = voigt_reuss(Cf)
            v = rel_width(R, V)
            valid = b.loewner_valid(1e-9) and (C_ex is None
                                               or b.brackets(C_ex, tol=5e-2))
            table[(c, name)] = (b.gap_rel(), v / b.gap_rel(), gate)
            print(f"{c:9.0f} {name:>13} | {v:7.1%} {b.gap_rel():9.1%} "
                  f"{v / b.gap_rel():10.1f}x | {head:8.2f} {rr:6.3f} "
                  f"{'ACCEPT' if gate else 'REJECT':>7} | {valid}")
        print()

    print("## Certified gap map")
    print(f"{'contrast':>9} | " + " ".join(f"{m:>14}" for m in
                                           ("disks", "laminate", "checkerboard")))
    print("-" * 60)
    for c in args.contrasts:
        cells = [f"{table[(c, m)][0]:13.1%}" + ("" if table[(c, m)][2] else "*")
                 for m in ("disks", "laminate", "checkerboard")]
        print(f"{c:9.0f} | " + " ".join(f"{x:>14}" for x in cells))
    print("\n* = failed the quality check")


if __name__ == "__main__":
    main()
