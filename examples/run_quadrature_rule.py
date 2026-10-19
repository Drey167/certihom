"""T0.5.3 — WHEN is a collocation rule fine enough? A measurable rule.

Background: a network trained against fixed collocation points can hide
energy between them, so its reported energy (and any bound built from it)
can be far below the truth. "Use enough points" is useless advice unless
"enough" is a number. This script produces one.

Method: train, then measure
  * bandwidth B  — the radius in frequency space containing 99.9% of the
    field's spectral energy (certihom.diagnostics.spectral_bandwidth);
  * Nyquist  nq/2 — the highest frequency an nq-point rule can resolve;
  * headroom = Nyquist / B;
  * refinement ratio — energy at 4x the rule divided by energy at the rule.

Measured outcome (1 CPU core, laminate contrast 6, width 48, depth 3,
400 steps, lr 2e-3; true tr(K_eff) = 7.4167):

    nq  n_harm | bandwidth Nyquist headroom | reported  refined     ratio | invalid
    16       4 |      16.0     8.0     0.50 |    0.031  109.853  3498.46 | YES
    32       4 |       9.0    16.0     1.78 |    7.271    7.921     1.09 | yes (mild)
    64       4 |       7.0    32.0     4.57 |    7.394    7.612     1.03 | yes (mild)
   128       4 |       6.0    64.0    10.67 |    7.434    7.446     1.00 | no
    32       2 |      36.0    16.0     0.44 |    4.767  102.982    21.60 | YES
    32       8 |      30.0    16.0     0.53 |    4.758   46.811     9.84 | YES
    32      12 |      34.0    16.0     0.47 |    0.015   16.524  1093.23 | YES

Two rules fall out:

RULE 1 (hard). headroom < 1 — the field carries frequencies the rule cannot
see — is catastrophic: refinement ratios of 10x to 3500x, i.e. the reported
"bound" is fiction.

RULE 2 (practical). headroom > 1 keeps the ratio near 1, but a strictly
VALID midpoint bound needed headroom of order 10 here. Below that the
residual quadrature error, though small, is enough to push the bound past
the exact value.

RULE 3 (the one to actually follow). Certify by exact integration
(certihom.certify / certihom.bfs) and none of this applies: interpolation
cannot hide energy between sample points because there are none.

A non-obvious observation: bandwidth is NON-MONOTONIC in n_harmonics, with
a minimum at 4. Too FEW embedding harmonics is as dangerous as too many —
an under-parameterised embedding cannot fit the target smoothly, so
training drives the weights large and the tanh nonlinearity manufactures
high harmonics. This is very likely the same root cause as the seed
volatility observed at n_harmonics=3 (see TASKS T0.2b): both are symptoms
of an under-resolved embedding forcing the network into wild solutions.

Run:  python examples/run_quadrature_rule.py [--steps 400]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from certihom.diagnostics import quadrature_safety_report
from certihom.energy import as_numpy, upper_gram
from certihom.exact import laminate_exact
from certihom.microstructures import laminate, sample_k
from certihom.quadrature import midpoint_grid
from certihom.train import fit_bounds

K = laminate(16, 1.0, 6.0, 0.25, axis=0)
K_EX = laminate_exact(1.0, 6.0, 0.25, 0)


def run(nq, nh, steps, width=48, depth=3, lr=2e-3, seed=0):
    _, _, (chi, _) = fit_bounds(K, nq=nq, n_harmonics=nh, width=width,
                                depth=depth, steps=steps, lr=lr, seed=seed,
                                verbose=False)
    rep = quadrature_safety_report(chi, K, nq)
    pts = midpoint_grid(nq)
    Kup = as_numpy(upper_gram(chi, pts, sample_k(K, pts)))
    rep["invalid"] = bool(Kup[0, 0] < K_EX[0, 0] - 1e-9)
    return rep


def header():
    print(f"{'nq':>5} {'n_harm':>7} | {'bandwidth':>9} {'Nyquist':>8} "
          f"{'headroom':>8} | {'reported':>9} {'refined':>9} {'ratio':>9} | "
          f"invalid?")
    print("-" * 90)


def line(nq, nh, r):
    print(f"{nq:>5} {nh:>7} | {r['bandwidth']:9.1f} {r['nyquist']:8.1f} "
          f"{r['headroom']:8.2f} | {r['reported']:9.3f} {r['refined']:9.3f} "
          f"{r['ratio']:9.2f} | {r['invalid']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--quick", action="store_true",
                   help="one row only, for a fast sanity check")
    args = p.parse_args()

    print(f"true tr(K_eff) = {np.trace(K_EX):.4f}\n")
    header()
    if args.quick:
        line(32, 4, run(32, 4, args.steps))
        return
    for nq in (16, 32, 64, 128):
        line(nq, 4, run(nq, 4, args.steps))
    print()
    for nh in (2, 4, 8, 12):
        line(32, nh, run(32, nh, args.steps))
    print("\nheadroom < 1  -> reported bound is fiction (ratio 10x-3500x)")
    print("headroom ~ 10 -> midpoint bound was reliably valid here")
    print("exact integration -> no headroom requirement at all; use it")


if __name__ == "__main__":
    main()
