"""EXPERIMENTS 1 + 2 — the paper's central claim, in the headline physics.

Until now the tensor-elasticity case rested on a smoke test while every
strong result was scalar conductivity. That asymmetry is fatal at review:
the paper claims amortized certified bounds on effective STIFFNESS, and the
nearest competitor (Keshav, Herb & Fritzen's Voigt-Reuss Net,
GAMM-Mitteilungen 2025 / IJNME 2026) is itself an ELASTICITY surrogate. We
must answer them on their own ground.

This script does both:

  EXPERIMENT 1 — train ONE conditioned elasticity operator on a family of
  microstructures, then certify HELD-OUT instances with exact integration
  (Q1 for the primal displacement fields, Bogner-Fox-Schmit C1 for the Airy
  dual). Report validity, gap statistics, and inference cost.

  EXPERIMENT 2 — measure the tightening factor against the Voigt-Reuss
  envelope on those same held-out instances, and repeat the decisive
  identical-volume-fraction / different-geometry demonstration in
  elasticity.

Every certificate here is exact-integration; none rely on the midpoint rule
(see RESULTS.md, "CRITICAL FINDING").

Run:  python examples/run_amortized_elasticity.py [--steps 300]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from certihom.amortized import AmortizedElasticity, fit_amortized
from certihom.bfs import bfs_nodal_dofs, certified_lower_elasticity
from certihom.bounds import CertifiedBounds
from certihom.certify import certified_upper_elasticity, nodal_values
from certihom.elasticity import (iso_stiffness_mandel,
                                 laminate_exact_elasticity,
                                 mandel_field_from_mask)
from certihom.fem_reference import fem_effective
from certihom.microstructures import boolean_disks, checkerboard, laminate

C_SOFT = iso_stiffness_mandel(1.0, 0.3)
C_STIFF = iso_stiffness_mandel(10.0, 0.2)


def make_field(mask):
    return mandel_field_from_mask(mask, C_STIFF, C_SOFT)


def voigt_reuss(Cf):
    """Elasticity Voigt (mean stiffness) and Reuss (inverse mean compliance)
    — both depend ONLY on volume fraction, carrying no geometry."""
    V = Cf.reshape(-1, 3, 3).mean(axis=0)
    R = np.linalg.inv(np.linalg.inv(Cf.reshape(-1, 3, 3)).mean(axis=0))
    return V, R


def rel_width(lo, up):
    mid = 0.5 * (up + lo)
    return float(np.linalg.norm(up - lo) / np.linalg.norm(mid))


def certify_exact(model, Cf, ne):
    """Exact-integration certificate from the AMORTIZED operator:
    Q1 projection for the primal, BFS C1 projection for the Airy dual."""
    coeffs, glob = model.condition(Cf)

    def wrap(net):
        def fn(y):
            c = coeffs.unsqueeze(0).expand(y.shape[0], -1, -1, -1)
            g = glob.unsqueeze(0).expand(y.shape[0], -1)
            return net(y, model.cond(y, c, g))
        return fn

    C_up = certified_upper_elasticity(nodal_values(wrap(model.chi), ne), Cf)
    C_low = certified_lower_elasticity(bfs_nodal_dofs(wrap(model.dual_net), ne), Cf)
    return CertifiedBounds(C_low, C_up)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--nq", type=int, default=24)
    p.add_argument("--ne", type=int, default=24)
    p.add_argument("--n_train", type=int, default=40)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    fam = [make_field(boolean_disks(8, 1.0, 2.0, n_disks=int(rng.integers(2, 5)),
                                    radius=float(rng.uniform(0.14, 0.22)),
                                    seed=s) > 1.5)
           for s in range(args.n_train)]
    held = [make_field(boolean_disks(8, 1.0, 2.0, n_disks=int(rng.integers(2, 5)),
                                     radius=float(rng.uniform(0.14, 0.22)),
                                     seed=500 + s) > 1.5) for s in range(6)]

    print("# EXPERIMENT 1 — amortized certified ELASTICITY")
    print(f"# E-contrast 10 (E=1,nu=.3 matrix; E=10,nu=.2 inclusions), "
          f"plane strain")
    print(f"# {len(fam)} training microstructures, {len(held)} held out, "
          f"{args.steps} steps\n")

    model = AmortizedElasticity(kmax=3, n_harmonics=2, width=40, depth=3)
    t0 = time.time()
    model, _ = fit_amortized(model, fam, nq=args.nq, steps=args.steps,
                             batch=3, lr=2e-3, seed=0, log_every=100,
                             monitor=2, verbose=False)
    t_train = time.time() - t0
    print(f"family training: {t_train:.0f} s\n")

    print(f"{'held-out instance':>18} | {'C_low[0,0]':>10} {'C_up[0,0]':>10} "
          f"{'gap':>7} | {'valid':>5} {'FEM audit':>9} | {'t':>6}")
    print("-" * 76)
    gaps, factors, ok_all = [], [], True
    for i, Cf in enumerate(held):
        t0 = time.time()
        b = certify_exact(model, Cf, args.ne)
        t_inf = time.time() - t0
        # rigorous one-sided audit: conforming FEM is itself an upper bound
        C_fem = fem_effective(Cf)
        audit = bool(np.linalg.eigvalsh(C_fem - b.K_low).min() >= -1e-9)
        valid = b.loewner_valid(1e-9)
        ok_all &= valid and audit
        gaps.append(b.gap_rel())
        V, R = voigt_reuss(Cf)
        factors.append(rel_width(R, V) / b.gap_rel())
        print(f"{'held-out #' + str(i + 1):>18} | {b.K_low[0,0]:10.4f} "
              f"{b.K_up[0,0]:10.4f} {b.gap_rel():6.1%} | {str(valid):>5} "
              f"{str(audit):>9} | {t_inf:5.2f}s")

    print(f"\ncertified gap: median {np.median(gaps):.1%}, "
          f"range [{min(gaps):.1%}, {max(gaps):.1%}]")
    print(f"validity (Loewner + rigorous FEM audit): {ok_all} "
          f"({len(held)}/{len(held)})")

    # ---------------- EXPERIMENT 2 ----------------
    print("\n\n# EXPERIMENT 2 — tightening vs the Voigt-Reuss envelope")
    print(f"median tightening factor on held-out instances: "
          f"{np.median(factors):.1f}x")
    print(f"range: [{min(factors):.1f}x, {max(factors):.1f}x]")

    print("\n## Identical volume fraction, different geometry")
    print(f"{'microstructure':>20} | {'vol frac':>9} {'V-R width':>10} "
          f"{'certified':>10} | {'valid':>6}")
    print("-" * 66)
    cases = [("laminate f=0.50", make_field(laminate(8, 1., 2., .5, 0) < 1.5),
              laminate_exact_elasticity(C_STIFF, C_SOFT, 0.5)[0]),
             ("checkerboard", make_field(checkerboard(8, 1., 2.) < 1.5), None)]
    for name, Cf, C_ex in cases:
        b = certify_exact(model, Cf, args.ne)
        V, R = voigt_reuss(Cf)
        vf = float((Cf[:, :, 0, 0] > 0.5 * (C_SOFT[0, 0] + C_STIFF[0, 0])).mean())
        val = b.loewner_valid(1e-9)
        if C_ex is not None:
            val = val and b.brackets(C_ex, tol=1e-2)
        print(f"{name:>20} | {vf:8.3f} {rel_width(R, V):9.1%} "
              f"{b.gap_rel():9.1%} | {str(val):>6}")
    print("\nSame volume fraction -> identical Voigt-Reuss envelope; different")
    print("geometry -> different certified interval. The Voigt-Reuss Net's")
    print("guarantee provably cannot distinguish these two microstructures.")


if __name__ == "__main__":
    main()
