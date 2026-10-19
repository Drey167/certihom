"""The honest limits table: certified tightness vs CONTRAST and MORPHOLOGY.

Why this exists. The prior-art assessment flagged that the nearest
competing per-instance work tops out at 10:1 phase contrast and the
group's classical bounds work explicitly avoids high contrast. Real
composites run 10:1 to 1000:1. A map that includes the regime where WE
degrade is both honest and occupies open ground — and it decides whether
the Hashin-Shtrikman polarisation variant is optional or required.

Every entry is an EXACT-INTEGRATION certificate (never the midpoint rule)
and carries the label-free quality gate, so a wide result is reported as
either "genuinely hard" (accepted, wide) or "bad run" (rejected).

Run:  python examples/run_contrast_morphology.py [--steps 200]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.diagnostics import quality_gate
from certihom.exact import checkerboard_exact, laminate_exact, reuss, voigt
from certihom.microstructures import boolean_disks, checkerboard, laminate
from certihom.train import fit_bounds


def vr_gap(K):
    v, r = voigt(K), reuss(K)
    return float(np.linalg.norm(v - r) / np.linalg.norm(0.5 * (v + r)))


def morphologies(c):
    """Three morphology classes at the same phase contrast c."""
    return [
        ("disks", boolean_disks(16, 1.0, c, n_disks=6, radius=0.14, seed=1),
         None, 3e-2),
        ("laminate", laminate(16, 1.0, c, 0.25, axis=0),
         laminate_exact(1.0, c, 0.25, 0), 1e-8),
        ("checkerboard", checkerboard(16, 1.0, c),
         checkerboard_exact(1.0, c), 5e-2),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--nq", type=int, default=64)
    p.add_argument("--contrasts", type=float, nargs="+",
                   default=[10.0, 50.0, 100.0, 200.0])
    p.add_argument("--seeds", type=int, default=1)
    args = p.parse_args()

    print(f"# Certified gap vs contrast and morphology "
          f"(nq={args.nq}, {args.steps} steps, exact integration)\n")
    print(f"{'contrast':>9} {'morphology':>13} | {'V-R gap':>8} "
          f"{'median':>10} {'tighten':>11} | {'full range':>15} "
          f"{'acc':>5} | valid")
    print("-" * 84)

    table = {}
    for c in args.contrasts:
        for name, K, K_ex, tol in morphologies(c):
            gaps, accs, valids, factors = [], [], [], []
            for sd in range(args.seeds):
                _, _, (chi, psi) = fit_bounds(K, nq=args.nq, n_harmonics=4,
                                              width=48, depth=3,
                                              steps=args.steps, lr=2e-3,
                                              seed=sd, verbose=False)
                b = certified_bounds_scalar(nodal_values(chi, args.nq),
                                            nodal_values(psi, args.nq), K)
                qg = quality_gate(chi, K, args.nq)
                gaps.append(b.gap_rel()); accs.append(qg["accept"])
                factors.append(vr_gap(K) / b.gap_rel())
                valids.append(b.loewner_valid(1e-9) and
                              (K_ex is None or b.brackets(K_ex, tol=tol)))
            g = np.array(gaps)
            keep = np.array(accs, dtype=bool)
            med = float(np.median(g[keep])) if keep.any() else float("nan")
            fac = float(np.median(np.array(factors)[keep])) if keep.any() else float("nan")
            table[(c, name)] = (med, fac, int(keep.sum()), g.min(), g.max())
            print(f"{c:9.0f} {name:>13} | {vr_gap(K):7.1%} {med:9.1%} "
                  f"{fac:10.1f}x | [{g.min():5.1%},{g.max():6.1%}] "
                  f"{int(keep.sum())}/{args.seeds:<2} | {all(valids)}")
        print()

    print("## Certified gap map (accepted runs only)")
    print(f"{'contrast':>9} | " + " ".join(f"{m:>14}" for m in
                                           ("disks", "laminate", "checkerboard")))
    print("-" * 60)
    for c in args.contrasts:
        cells = []
        for m in ("disks", "laminate", "checkerboard"):
            g, f, nacc, _, _ = table[(c, m)]
            cells.append(f"{g:13.1%}")
        print(f"{c:9.0f} | " + " ".join(f"{x:>14}" for x in cells))
    print("\nMedians are over ACCEPTED runs only; the full range covers all "
          "seeds.\nRejected runs stay VALID — they are wide, not wrong.")


if __name__ == "__main__":
    main()
