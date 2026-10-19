"""Ten-cell audit of the public Voigt--Reuss Net 2D elasticity benchmark.

This is a *data and convention* pilot.  It reconstructs the native labelled
100x100 periodic microstructures, converts reference labels from engineering Voigt to
Mandel notation, and checks every label is within the analytic phase-wise
Voigt--Reuss envelope.  Passing this audit is necessary before investing in
the expensive trained CertiHOM run; it is not itself a tightened certificate.

Example:
  python examples/run_vrnet_elasticity_pilot.py ^
    --h5 data/vrnet_2d/feature_engineering_mechanical_2D.h5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from certihom.amortized import AmortizedElasticity
from certihom.vrnet_benchmark import load_vrnet_mechanical_samples, voigt_reuss_bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="DaRUS feature_engineering_mechanical_2D.h5")
    parser.add_argument("--split", default="val_set", choices=("train_set", "val_set"))
    parser.add_argument("--mode-grid", type=int, default=3, choices=(3, 5, 7, 9, 11))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--certihom", action="store_true",
                        help="also run the actual zero-initialized CertiHOM certificate")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    samples = load_vrnet_mechanical_samples(
        args.h5, split=args.split, mode_grid=(args.mode_grid, args.mode_grid),
        indices=range(args.count), resolution=args.resolution)
    print("# Public VRNet 2D elasticity benchmark: convention pilot")
    print("# DaRUS-5730 | plane strain | E_solid=1000, nu_solid=0.3 | "
          "E_compliant=1, nu_compliant=0.49")
    print(f"# native periodic reconstruction: {args.resolution}x{args.resolution}\n")
    print(f"{'cell':>5} | {'solid f':>8} | {'Cref11':>10} | {'min(Cref-R)':>13} "
          f"{'min(V-Cref)':>13} | {'inside VR':>9}")
    print("-" * 82)
    all_inside = True
    n_inside = 0
    for s in samples:
        C_low, C_up = voigt_reuss_bounds(s.stiffness_field())
        lower_margin = float(np.linalg.eigvalsh(s.reference_mandel - C_low).min())
        upper_margin = float(np.linalg.eigvalsh(C_up - s.reference_mandel).min())
        inside = lower_margin >= -1e-8 and upper_margin >= -1e-8
        all_inside &= inside
        n_inside += int(inside)
        print(f"{s.index:5d} | {s.declared_solid_fraction:8.4f} | "
              f"{s.reference_mandel[0,0]:10.4f} | {lower_margin:13.4e} "
              f"{upper_margin:13.4e} | {str(inside):>9}")
    print(f"\nVoigt--Reuss envelope audit: {all_inside} ({n_inside}/{len(samples)})")
    if not all_inside:
        raise SystemExit("STOP: source convention/geometry mismatch; do not train CertiHOM.")
    if args.certihom:
        # The zero-initialized admissible fields reproduce the phase-wise
        # Voigt--Reuss bounds.  This is deliberately a certification smoke
        # test, not an inflated claim of learned tightening.
        model = AmortizedElasticity(kmax=3, n_harmonics=2, width=40, depth=3)
        print("\n# Actual CertiHOM certificate (zero-initialized admissible fields)")
        print(f"{'cell':>5} | {'C_low11':>10} {'Cref11':>10} {'C_up11':>10} | "
              f"{'min(Cref-low)':>14} {'min(up-Cref)':>14} | {'inside':>6}")
        print("-" * 100)
        n_certified = 0
        for s in samples:
            b = model.bounds_for(s.stiffness_field(), nq=args.resolution)
            lower_margin = float(np.linalg.eigvalsh(s.reference_mandel - b.K_low).min())
            upper_margin = float(np.linalg.eigvalsh(b.K_up - s.reference_mandel).min())
            inside = lower_margin >= -1e-8 and upper_margin >= -1e-8
            n_certified += int(inside)
            print(f"{s.index:5d} | {b.K_low[0,0]:10.4f} {s.reference_mandel[0,0]:10.4f} "
                  f"{b.K_up[0,0]:10.4f} | {lower_margin:14.4e} "
                  f"{upper_margin:14.4e} | {str(inside):>6}")
        print(f"\nCertiHOM/reference bracketing: {n_certified}/{len(samples)}")
    print("Next: train CertiHOM on train_set only and evaluate tightened "
          f"{args.resolution}x{args.resolution} certificates on a pre-registered val_set subset.")


if __name__ == "__main__":
    main()
