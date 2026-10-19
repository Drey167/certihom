"""T0.2b — multi-seed study. RUN THIS FIRST.

NOTE (nq default raised 32 -> 64). At nq=32 the laminate case sits at
quadrature headroom 0.64 — BELOW the safety threshold of 1 — and training
overfits the rule, so the honest exact-integration certificate blows up to
~160% while remaining valid. Measured: at nq=32 the gaps across seeds were
101.5% / 172.8% / 89.3%; at nq=64 (headroom 4.57) they were 0.8% / 1.7% /
0.4%. A 127x improvement from the quadrature rule alone. Always check
headroom before trusting a tightness number.

Every headline number in RESULTS.md is single-seed, which will not survive
review. This harness re-runs the core cases across several seeds and
reports medians and ranges, plus a CSV you can plot.

It also separates the two things that must never be confused:
  * VALIDITY  — does the certified interval contain the truth? (must be
                100%; a single False is a bug, not a tuning issue)
  * TIGHTNESS — how wide is it? (this is what varies with seed)

Usage:
    python examples/run_multiseed.py --seeds 5 --steps 400
    python examples/run_multiseed.py --seeds 3 --steps 150   # quick check
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.diagnostics import quality_gate
from certihom.exact import checkerboard_exact, laminate_exact, reuss, voigt
from certihom.fv_reference import fv_effective
from certihom.microstructures import boolean_disks, checkerboard, laminate
from certihom.train import fit_bounds


def cases():
    """(name, K field, exact tensor or None, bracket tolerance)."""
    return [
        ("checkerboard_c5", checkerboard(16, 1.0, 5.0),
         checkerboard_exact(1.0, 5.0), 1e-3),
        ("laminate_c6", laminate(16, 1.0, 6.0, 0.25, axis=0),
         laminate_exact(1.0, 6.0, 0.25, 0), 1e-9),
        ("disks_c10", boolean_disks(16, 1.0, 10.0, n_disks=6, radius=0.14,
                                    seed=1), None, 2e-2),
    ]


def vr_gap(K):
    v, r = voigt(K), reuss(K)
    return float(np.linalg.norm(v - r) / np.linalg.norm(0.5 * (v + r)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--nq", type=int, default=64)
    p.add_argument("--out", type=str, default="multiseed.csv")
    args = p.parse_args()

    rows = []
    print(f"# T0.2b multi-seed study: {args.seeds} seeds x {args.steps} steps\n")

    for name, K, K_ex, tol in cases():
        ref = K_ex if K_ex is not None else fv_effective(np.kron(K, np.ones((4, 4))))
        print(f"## {name}   (Voigt-Reuss gap {vr_gap(K):.1%})")
        for s in range(args.seeds):
            t0 = time.time()
            b_q, _, (chi, psi) = fit_bounds(
                K, nq=args.nq, n_harmonics=4, width=48, depth=3,
                steps=args.steps, lr=2e-3, seed=s, verbose=False)
            b_e = certified_bounds_scalar(nodal_values(chi, args.nq),
                                          nodal_values(psi, args.nq), K)
            qg = quality_gate(chi, K, args.nq)
            row = {
                "case": name, "seed": s,
                "gap_quad": b_q.gap_rel(), "gap_exact": b_e.gap_rel(),
                "loewner_valid": b_e.loewner_valid(1e-10),
                "contains_ref": b_e.brackets(ref, tol=tol),
                "vr_gap": vr_gap(K), "wall_s": time.time() - t0,
                "accept": qg["accept"], "headroom": qg["headroom"],
                "refine_ratio": qg["ratio"],
            }
            rows.append(row)
            print(f"  seed {s}: exact gap {row['gap_exact']:7.1%} | "
                  f"valid {row['loewner_valid']} | headroom "
                  f"{row['headroom']:5.2f} | "
                  f"{'ACCEPT' if row['accept'] else 'REJECT (' + qg['reason'] + ')'}"
                  f" | {row['wall_s']:.0f}s")

        g = np.array([r["gap_exact"] for r in rows if r["case"] == name])
        ok = all(r["loewner_valid"] and r["contains_ref"]
                 for r in rows if r["case"] == name)
        print(f"  -> median {np.median(g):.1%}  range [{g.min():.1%}, "
              f"{g.max():.1%}]  spread {g.max()-g.min():.1%}")
        acc = [r for r in rows if r["case"] == name and r["accept"]]
        ga = np.array([r["gap_exact"] for r in acc]) if acc else np.array([])
        print(f"  -> VALIDITY {args.seeds}/{args.seeds}: {ok}")
        print(f"  -> ACCEPTED runs: {len(acc)}/{args.seeds}"
              + (f"; median gap over accepted {np.median(ga):.1%}" if len(ga) else "")
              + "\n")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    all_ok = all(r["loewner_valid"] and r["contains_ref"] for r in rows)
    print(f"CSV written to {args.out}")
    print(f"\nOVERALL VALIDITY across every run: {all_ok}")
    if not all_ok:
        print("!! A validity failure is a BUG. Do not tune it away — find it.")
    print("\nWhat to do with this:")
    print(" 1. Report medians and ranges (not single seeds) in RESULTS.md.")
    print(" 2. Note which cases are seed-robust and which are volatile.")
    print(" 3. Volatile cases = a convergence problem to diagnose")
    print("    (longer schedule, L-BFGS polish, different init), not just")
    print("    a reporting problem.")


if __name__ == "__main__":
    main()
