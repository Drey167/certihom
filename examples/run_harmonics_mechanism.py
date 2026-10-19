"""Closing the loop: are bandwidth blow-up and seed volatility ONE mechanism?

Two symptoms were observed separately in this project:

  (S1) seed volatility — at n_harmonics=3 the certified gap on a laminate
       varied wildly across seeds (50.8% / 1.1% / 148.8%), while at
       n_harmonics=4 it was stable (spread 0.1%);
  (S2) bandwidth non-monotonicity — the trained field's spectral bandwidth
       was minimised at n_harmonics=4 (B=9) and much larger at 2 (B=36)
       and 12 (B=34), driving quadrature overfitting.

Proposed single mechanism: an UNDER-PARAMETERISED embedding cannot
represent the target smoothly, so optimisation compensates by inflating the
weights; the tanh nonlinearity then manufactures high harmonics. High
bandwidth and unstable optimisation are then two faces of the same
pathology, mediated by weight magnitude.

Test: sweep n_harmonics, several seeds each, and measure jointly
  * certified gap (EXACT integration) -> median and spread  (symptom S1)
  * spectral bandwidth B                                     (symptom S2)
  * trunk gain = product of spectral norms of the linear layers, and
    the derivative-amplification proxy (2 pi M) * gain       (the mediator)

If the mechanism is real, all three must move together, with a common
minimum, rather than being independent quirks.

Run:  python examples/run_harmonics_mechanism.py [--seeds 3] [--steps 250]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.diagnostics import spectral_bandwidth
from certihom.exact import laminate_exact
from certihom.microstructures import laminate
from certihom.train import fit_bounds

K = laminate(8, 1.0, 6.0, 0.25, axis=0)     # the configuration that was volatile
K_EX = laminate_exact(1.0, 6.0, 0.25, 0)


def trunk_gain(net) -> float:
    """Product of spectral norms of all Linear layers — an upper-bound
    proxy for how much the network can amplify input variation."""
    g = 1.0
    for m in net.modules():
        if isinstance(m, nn.Linear):
            g *= float(torch.linalg.matrix_norm(m.weight.detach(), ord=2))
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--harmonics", type=int, nargs="+", default=[2, 3, 4, 6, 10])
    p.add_argument("--nq", type=int, default=32)
    args = p.parse_args()

    print(f"# laminate contrast 6, nq={args.nq}, width 48, depth 3, "
          f"{args.steps} steps, {args.seeds} seeds\n")
    print(f"{'n_harm':>7} | {'gap median':>11} {'gap spread':>11} | "
          f"{'bandwidth':>9} | {'trunk gain':>11} {'(2piM)*gain':>12} | valid")
    print("-" * 82)

    rows = []
    for nh in args.harmonics:
        gaps, bws, gains = [], [], []
        allok = True
        for s in range(args.seeds):
            _, _, (chi, psi) = fit_bounds(K, nq=args.nq, n_harmonics=nh,
                                          width=48, depth=3, steps=args.steps,
                                          lr=2e-3, seed=s, verbose=False)
            b = certified_bounds_scalar(nodal_values(chi, args.nq),
                                        nodal_values(psi, args.nq), K)
            gaps.append(b.gap_rel())
            bws.append(spectral_bandwidth(chi))
            gains.append(trunk_gain(chi))
            allok &= b.loewner_valid(1e-10) and b.brackets(K_EX, tol=1e-9)
        g = np.array(gaps)
        amp = float(np.median(gains)) * 2 * np.pi * nh
        rows.append(dict(nh=nh, med=np.median(g), spread=g.max() - g.min(),
                         B=np.median(bws), gain=np.median(gains), amp=amp,
                         ok=allok))
        r = rows[-1]
        print(f"{nh:>7} | {r['med']:10.1%} {r['spread']:10.1%} | "
              f"{r['B']:9.1f} | {r['gain']:11.1f} {r['amp']:12.1f} | {r['ok']}")

    # --- does the mechanism hold together? -------------------------------
    sp = np.array([r["spread"] for r in rows])
    B = np.array([r["B"] for r in rows])
    amp = np.array([r["amp"] for r in rows])
    print("\nrank correlations across the sweep (Spearman-style, n small):")

    def rho(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"  bandwidth  vs seed spread : {rho(B, sp):+.2f}")
    print(f"  amplification vs bandwidth: {rho(amp, B):+.2f}")
    print(f"  amplification vs spread   : {rho(amp, sp):+.2f}")
    print("\nAll three moving together supports ONE mechanism (weight "
          "inflation).\nIndependent movement would falsify it — report "
          "either result honestly.")
    print(f"\nvalidity across every run: {all(r['ok'] for r in rows)} "
          "(exact-integration certificates)")


if __name__ == "__main__":
    main()
