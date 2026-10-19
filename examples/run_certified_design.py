"""Certified inverse design — optimize the GUARANTEED performance floor.

Standard surrogate-driven design maximizes a point estimate and hopes the
surrogate was right. Here the objective IS the certificate: we maximize
K_low[0,0], the mathematically guaranteed effective conductivity in x. The
optimizer cannot win by fooling the model, because a loose model produces a
LOWER guaranteed value — self-penalizing by construction.

Setup. Inclusion centers are the design variables (count and radius fixed).
The microstructure is rendered differentiably as a smooth union of disks,

    chi(y) = 1 - prod_i [1 - sigmoid(beta (r - dist_per(y, c_i)))],
    k(y)   = k_matrix + (k_incl - k_matrix) chi(y),

so gradients flow from the certified bound back to the centers. Trial
fields and design are updated alternately: the fields minimize (complemen-
tary) energy to tighten the certificate, the design maximizes the certified
floor. Every iterate remains a valid certificate throughout.

The final design is re-certified with EXACT integration (Phase 1) and
audited against the FV reference.

Run:  python examples/run_certified_design.py [--steps 300]
"""

import argparse

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from certihom.bounds import CertifiedBounds
from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.energy import as_numpy, dual_gram, lower_from_dual, upper_gram
from certihom.fields import FieldPair
from certihom.fv_reference import fv_effective
from certihom.quadrature import midpoint_grid

N_PIX = 32
K_MAT, K_INC = 1.0, 10.0
RADIUS, BETA = 0.10, 60.0


def render(centers: torch.Tensor, rscale: float = 1.0) -> torch.Tensor:
    """Differentiable periodic pixel field from disk centers (n_pix^2)."""
    c = (torch.arange(N_PIX, dtype=torch.float64) + 0.5) / N_PIX
    Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
    keep = torch.ones(N_PIX, N_PIX, dtype=torch.float64)
    for i in range(centers.shape[0]):
        d1 = torch.abs(Y1 - centers[i, 0]) % 1.0
        d2 = torch.abs(Y2 - centers[i, 1]) % 1.0
        d1 = torch.minimum(d1, 1.0 - d1)
        d2 = torch.minimum(d2, 1.0 - d2)
        d = torch.sqrt(d1**2 + d2**2 + 1e-12)
        keep = keep * (1.0 - torch.sigmoid(BETA * (RADIUS * rscale - d)))
    chi = 1.0 - keep
    return K_MAT + (K_INC - K_MAT) * chi


def project_fraction(centers, target, lo=0.5, hi=2.0, iters=30):
    """Bisect a global radius scale so the rendered inclusion fraction
    equals ``target`` exactly. This is a PROJECTION onto the material-budget
    constraint (applied outside the gradient), replacing the soft penalty
    whose ~6% drift was a documented limitation of the earlier version."""
    with torch.no_grad():
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            K = render(centers, mid)
            f = float(((K - K_MAT) / (K_INC - K_MAT)).mean())
            if f < target:
                lo = mid
            else:
                hi = mid
    return 0.5 * (lo + hi)


def sample_pixels(Kt: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Nearest-pixel lookup that keeps gradients w.r.t. pixel VALUES."""
    idx = torch.clamp((pts * N_PIX).long(), 0, N_PIX - 1)
    return Kt[idx[:, 0], idx[:, 1]]


def certify(chi_net, psi_net, Kt, pts):
    kv = sample_pixels(Kt, pts)
    K_up = upper_gram(chi_net, pts, kv, create_graph=True)
    R_up = dual_gram(psi_net, pts, kv, create_graph=True)
    return K_up, R_up


def certify_eval(chi_net, psi_net, Kt, pts) -> CertifiedBounds:
    """Evaluate the certificate. NOTE: spatial autograd is still required
    here, so torch.no_grad() must NOT wrap this (it would kill the input
    gradients); we simply avoid building a training graph and detach."""
    kv = sample_pixels(Kt.detach(), pts)
    K_up = upper_gram(chi_net, pts, kv, create_graph=False)
    R_up = dual_gram(psi_net, pts, kv, create_graph=False)
    return CertifiedBounds(as_numpy(lower_from_dual(R_up)), as_numpy(K_up))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--n_disks", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    centers = torch.rand((args.n_disks, 2), dtype=torch.float64, generator=g)
    centers.requires_grad_(True)
    chi_net = FieldPair(4, 48, 3)
    psi_net = FieldPair(4, 48, 3)
    pts = midpoint_grid(N_PIX)

    opt_f = torch.optim.Adam(list(chi_net.parameters()) + list(psi_net.parameters()),
                             lr=2e-3)
    opt_d = torch.optim.Adam([centers], lr=8e-3)

    K0_np = as_numpy(render(centers).detach())
    vf0 = float((K0_np > 0.5 * (K_MAT + K_INC)).mean())
    print(f"# Certified design: maximize the GUARANTEED K_low[0,0]")
    print(f"# {args.n_disks} disks, r={RADIUS}, contrast {K_INC:.0f}, "
          f"initial inclusion fraction {vf0:.3f}\n")

    # --- warm up the trial fields on the initial design -------------------
    for _ in range(60):
        opt_f.zero_grad(set_to_none=True)
        K_up, R_up = certify(chi_net, psi_net,
                             render(centers, project_fraction(centers, vf0)).detach(),
                             pts)
        (torch.trace(K_up) + torch.trace(R_up)).backward()
        opt_f.step()
    b_init = certify_eval(chi_net, psi_net, render(centers), pts)
    vf0 = float(((as_numpy(render(centers).detach()) - K_MAT) / (K_INC - K_MAT)).mean())
    print(f"initial : K_low[0,0] = {b_init.K_low[0,0]:.4f}   "
          f"K_up[0,0] = {b_init.K_up[0,0]:.4f}   gap {b_init.gap_rel():.1%}")

    # --- alternating design / field optimization --------------------------
    for step in range(1, args.steps + 1):
        opt_f.zero_grad(set_to_none=True)
        K_up, R_up = certify(chi_net, psi_net,
                             render(centers, project_fraction(centers, vf0)).detach(),
                             pts)
        (torch.trace(K_up) + torch.trace(R_up)).backward()
        opt_f.step()

        rs = project_fraction(centers, vf0)      # exact budget, no penalty
        opt_d.zero_grad(set_to_none=True)
        Kt = render(centers, rs)
        K_up, R_up = certify(chi_net, psi_net, Kt, pts)
        K_low = torch.linalg.inv(R_up)
        (-K_low[0, 0]).backward()
        opt_d.step()

        if step % 75 == 0 or step == args.steps:
            b = certify_eval(chi_net, psi_net,
                             render(centers, project_fraction(centers, vf0)), pts)
            print(f"step {step:4d}: K_low[0,0] = {b.K_low[0,0]:.4f}   "
                  f"K_up[0,0] = {b.K_up[0,0]:.4f}   gap {b.gap_rel():.1%}   "
                  f"incl. fraction {float(((as_numpy(render(centers, project_fraction(centers, vf0)).detach()) - K_MAT) / (K_INC - K_MAT)).mean()):.3f}")

    # --- final design: EXACT-integration certificate + FV audit -----------
    K_final = as_numpy(render(centers, project_fraction(centers, vf0)).detach())
    b_cert = certified_bounds_scalar(nodal_values(chi_net, N_PIX),
                                     nodal_values(psi_net, N_PIX), K_final)
    F = fv_effective(K_final)
    print(f"\n## Final design, exactly-integrated certificate (Phase 1):")
    print(f"guaranteed K_low[0,0] = {b_cert.K_low[0,0]:.4f}   "
          f"(initial {b_init.K_low[0,0]:.4f}, "
          f"gain {(b_cert.K_low[0,0]/b_init.K_low[0,0]-1):+.1%})")
    print(f"certified interval [{b_cert.K_low[0,0]:.4f}, {b_cert.K_up[0,0]:.4f}], "
          f"Loewner valid: {b_cert.loewner_valid(1e-10)}")
    print(f"FV reference K[0,0] = {F[0,0]:.4f}  "
          f"(inside certified interval: {b_cert.brackets(F, tol=2e-2)})")
    print(f"inclusion fraction {float(((K_final - K_MAT)/(K_INC-K_MAT)).mean()):.3f} "
          f"(budget {vf0:.3f})")
    print("\nThe reported gain is a guaranteed floor, not a surrogate "
          "estimate:\na looser model would have produced a LOWER number, "
          "never a false one.")


if __name__ == "__main__":
    main()
