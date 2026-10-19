"""CERTIFIED DESIGN IN ELASTICITY — optimise a GUARANTEED stiffness floor.

The conductivity version raised a guaranteed conductivity floor by 57%, but
carried a known blemish: the material budget was enforced by a soft penalty
and drifted ~6%. Here the budget is enforced by PROJECTION — after every
design step a global radius scale is bisected so the rendered volume
fraction returns exactly to target. The reported gain therefore cannot be
bought with extra material.

Objective. The certified floor on the directional Young's modulus is

    E(d) >= 1 / (a . C_low^{-1} a),    a = Mandel(d (x) d),

and since C_low = S_up^{-1}, that is simply  E(d) >= 1 / (a . S_up a).
So maximising the guaranteed floor means MINIMISING the complementary
energy a . S_up a — the design objective is a quadratic form in the dual
Gram matrix, requiring no inversion in the loop.

The objective is self-penalising: a loose dual field gives a larger
a . S_up a and hence a LOWER guaranteed floor. The optimiser cannot win by
fooling the surrogate — the failure mode of ordinary surrogate-driven
design is structurally absent.

Final design certified by exact integration (Q1 primal + BFS C1 dual).

Run:  python examples/run_certified_design_elastic.py [--steps 250]
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
from certihom.elasticity import (dual_stress_fields, gram_mandel,
                                 iso_stiffness_mandel, primal_strain_fields,
                                 youngs_interval)
from certihom.energy import as_numpy
from certihom.fem_reference import fem_effective
from certihom.fields import FieldPair
from certihom.quadrature import midpoint_grid

N_PIX = 16
BETA = 40.0
C_SOFT = torch.as_tensor(iso_stiffness_mandel(1.0, 0.3))
C_STIFF = torch.as_tensor(iso_stiffness_mandel(10.0, 0.2))
A_X = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)   # Mandel(e1 (x) e1)


def indicator(centers, rscale, radius):
    """Differentiable smooth union of periodic disks -> (N_PIX, N_PIX)."""
    c = (torch.arange(N_PIX, dtype=torch.float64) + 0.5) / N_PIX
    Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
    keep = torch.ones(N_PIX, N_PIX, dtype=torch.float64)
    for i in range(centers.shape[0]):
        d1 = torch.abs(Y1 - centers[i, 0]) % 1.0
        d2 = torch.abs(Y2 - centers[i, 1]) % 1.0
        d1 = torch.minimum(d1, 1.0 - d1)
        d2 = torch.minimum(d2, 1.0 - d2)
        d = torch.sqrt(d1**2 + d2**2 + 1e-12)
        keep = keep * (1.0 - torch.sigmoid(BETA * (radius * rscale - d)))
    return 1.0 - keep


def stiffness_field(chi):
    """Linear interpolation between phases; (N_PIX, N_PIX, 3, 3)."""
    return C_SOFT + chi[..., None, None] * (C_STIFF - C_SOFT)


def project_fraction(centers, radius, target, lo=0.5, hi=2.0, iters=30):
    """Bisect a global radius scale so the rendered fraction == target.
    This is a PROJECTION onto the material-budget constraint, applied
    outside the gradient — i.e. projected gradient ascent."""
    with torch.no_grad():
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            f = float(indicator(centers, mid, radius).mean())
            if f < target:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


def sample(Cf_t, pts):
    """Differentiable nearest-pixel lookup of a torch stiffness field."""
    idx = torch.clamp((pts * N_PIX).long(), 0, N_PIX - 1)
    return Cf_t[idx[:, 0], idx[:, 1]]


def grams(chi_net, phi_net, Cf_t, pts, create_graph):
    Cv = sample(Cf_t, pts)
    Sv = torch.linalg.inv(Cv)
    C_up = gram_mandel(primal_strain_fields(chi_net, pts, create_graph), Cv)
    S_up = gram_mandel(dual_stress_fields(phi_net, pts, create_graph), Sv)
    return C_up, S_up


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--n_disks", type=int, default=8)
    p.add_argument("--radius", type=float, default=0.10)
    p.add_argument("--nq", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    centers = torch.rand((args.n_disks, 2), dtype=torch.float64, generator=g)
    centers.requires_grad_(True)
    chi_net = FieldPair(3, 48, 3, n_out=6)
    phi_net = FieldPair(3, 48, 3, n_out=3)
    pts = midpoint_grid(args.nq)

    opt_f = torch.optim.Adam(list(chi_net.parameters()) + list(phi_net.parameters()),
                             lr=2e-3)
    opt_d = torch.optim.Adam([centers], lr=6e-3)

    target = float(indicator(centers, 1.0, args.radius).mean())
    print("# Certified design in ELASTICITY: maximise the GUARANTEED E_x")
    print(f"# {args.n_disks} inclusions, E-contrast 10, plane strain")
    print(f"# material budget FIXED BY PROJECTION at fraction {target:.4f}\n")

    def certify_exact():
        rs = project_fraction(centers, args.radius, target)
        Cf = as_numpy(stiffness_field(indicator(centers, rs, args.radius)).detach())
        b = CertifiedBounds(
            certified_lower_elasticity(bfs_nodal_dofs(phi_net, args.nq), Cf),
            certified_upper_elasticity(nodal_values(chi_net, args.nq), Cf))
        return b, Cf

    # warm up the trial fields on the initial design
    rs = 1.0
    for _ in range(80):
        opt_f.zero_grad(set_to_none=True)
        Cf_t = stiffness_field(indicator(centers.detach(), rs, args.radius))
        C_up, S_up = grams(chi_net, phi_net, Cf_t, pts, True)
        (torch.trace(C_up) + torch.trace(S_up)).backward()
        opt_f.step()
    b0, Cf0 = certify_exact()
    e0_lo, e0_hi = youngs_interval(b0.K_low, b0.K_up, np.array([1.0, 0.0]))
    print(f"initial : guaranteed E_x >= {e0_lo:.4f}   "
          f"(certified interval [{e0_lo:.4f}, {e0_hi:.4f}])")

    for step in range(1, args.steps + 1):
        rs = project_fraction(centers, args.radius, target)
        # (i) tighten the certificate
        opt_f.zero_grad(set_to_none=True)
        Cf_t = stiffness_field(indicator(centers.detach(), rs, args.radius))
        C_up, S_up = grams(chi_net, phi_net, Cf_t, pts, True)
        (torch.trace(C_up) + torch.trace(S_up)).backward()
        opt_f.step()
        # (ii) raise the guaranteed floor: minimise a . S_up a
        opt_d.zero_grad(set_to_none=True)
        Cf_t = stiffness_field(indicator(centers, rs, args.radius))
        _, S_up = grams(chi_net, phi_net, Cf_t, pts, True)
        (A_X @ S_up @ A_X).backward()
        opt_d.step()

        if step % 50 == 0 or step == args.steps:
            b, _ = certify_exact()
            lo, hi = youngs_interval(b.K_low, b.K_up, np.array([1.0, 0.0]))
            rs2 = project_fraction(centers, args.radius, target)
            frac = float(indicator(centers, rs2, args.radius).mean())
            print(f"step {step:4d}: guaranteed E_x >= {lo:.4f}   "
                  f"gap {b.gap_rel():5.1%}   fraction {frac:.4f}")

    b, Cf = certify_exact()
    lo, hi = youngs_interval(b.K_low, b.K_up, np.array([1.0, 0.0]))
    frac = float(indicator(centers, project_fraction(centers, args.radius, target),
                           args.radius).mean())
    audit = bool(np.linalg.eigvalsh(fem_effective(Cf) - b.K_low).min() >= -1e-9)
    print(f"\n## Final design, exact-integration certificate")
    print(f"guaranteed E_x  : {e0_lo:.4f} -> {lo:.4f}   "
          f"({(lo / e0_lo - 1):+.1%})")
    print(f"certified E_x   : [{lo:.4f}, {hi:.4f}]")
    print(f"material budget : {frac:.4f} (target {target:.4f}, "
          f"drift {abs(frac - target) / target:+.2%})")
    print(f"Loewner valid   : {b.loewner_valid(1e-9)}   "
          f"rigorous FEM audit: {audit}")
    print("\nThe gain is a guaranteed FLOOR at FIXED material budget: a looser")
    print("dual field would have produced a LOWER number, never a false one.")


if __name__ == "__main__":
    main()
