"""Label-free training of the primal-dual trial-field pair.

The loss is simply

    L = trace(K_up) + trace(R_up)
      = sum_i [ energy(e_i + grad chi_i) + comp_energy(e_i + curl psi_i) ],

i.e. the sum of the four basis (complementary) energies. Each head decouples
and is minimized exactly by the corresponding true corrector field, so
minimizing L tightens BOTH bounds simultaneously; no FEM/FFT labels are used
anywhere. Because admissibility is architectural, every intermediate iterate
already yields a valid certificate — a mediocre run gives a wider interval,
never a wrong one.

Quadrature points are FIXED (midpoint grid): the certificate statement is
tied to a definite quadrature rule, and stochastic collocation would blur
it. Phase 1 replaces "up to quadrature error" with exact-integration
certificates via FE projection (MATH.md).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .bounds import CertifiedBounds
from .energy import as_numpy, dual_gram, lower_from_dual, upper_gram
from .fields import FieldPair
from .microstructures import sample_k
from .quadrature import midpoint_grid, midpoint_grid_chunk


def _chunks(n_points: int, batch_points: int | None):
    """Slices that bound autograd memory without changing the quadrature."""
    size = n_points if batch_points is None else batch_points
    if size <= 0:
        raise ValueError("batch_points must be positive")
    return (slice(i, min(i + size, n_points)) for i in range(0, n_points, size))


def evaluate_bounds(chi_net, psi_net, pts, kvals,
                    batch_points: int | None = None) -> CertifiedBounds:
    """Assemble the certified interval from the current fields (no training
    graph). Also used standalone to certify any saved/loaded field pair."""
    n_points = len(pts)
    K_up = torch.zeros((2, 2), dtype=pts.dtype, device=pts.device)
    R_up = torch.zeros_like(K_up)
    with torch.enable_grad():  # input-gradients are needed even in eval
        for s in _chunks(n_points, batch_points):
            weight = (s.stop - s.start) / n_points
            K_up += weight * upper_gram(chi_net, pts[s], kvals[s],
                                         create_graph=False).detach()
            R_up += weight * dual_gram(psi_net, pts[s], kvals[s],
                                        create_graph=False).detach()
    K_low = lower_from_dual(R_up)
    return CertifiedBounds(K_low=as_numpy(K_low), K_up=as_numpy(K_up))


def fit_bounds(
    K_field: np.ndarray,
    nq: int = 64,
    n_harmonics: int = 3,
    width: int = 64,
    depth: int = 3,
    steps: int = 1500,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 250,
    verbose: bool = True,
    batch_points: int | None = None,
):
    """Train the primal-dual pair on one microstructure; return
    (CertifiedBounds, history, nets).

    Parameters mirror the research plan's reproducibility standard: a run is
    fully determined by (K_field, nq, n_harmonics, width, depth, steps, lr,
    seed). ``nq`` should be a multiple of K_field.shape[0] (asserted) so the
    midpoint rule respects pixel boundaries.

    history : list of dicts with keys step, tr_K_up, tr_R_up, gap_rel,
              wall_s — enough to regenerate every convergence figure.
    """
    n = K_field.shape[0]
    assert nq % n == 0, "quadrature resolution must be a multiple of the pixel grid"
    torch.manual_seed(seed)

    pts = midpoint_grid(nq)
    kvals = sample_k(K_field, pts)

    chi_net = FieldPair(n_harmonics, width, depth)
    psi_net = FieldPair(n_harmonics, width, depth)
    params = list(chi_net.parameters()) + list(psi_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    history = []
    t0 = time.time()

    def log(step):
        b = evaluate_bounds(chi_net, psi_net, pts, kvals, batch_points)
        rec = {
            "step": step,
            "tr_K_up": float(np.trace(b.K_up)),
            "tr_K_low": float(np.trace(b.K_low)),
            "gap_rel": b.gap_rel(),
            "wall_s": time.time() - t0,
        }
        history.append(rec)
        if verbose:
            print(
                f"step {step:5d} | tr K_up {rec['tr_K_up']:.6f} | "
                f"tr K_low {rec['tr_K_low']:.6f} | gap {rec['gap_rel']:.3%} | "
                f"{rec['wall_s']:.1f}s"
            )

    log(0)  # Voigt-Reuss by zero-initialization — valid before any training.
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        # The loss is the mean of independent pointwise contributions.
        # Accumulating its gradient over fixed chunks is algebraically the
        # same quadrature sum as a full batch, but keeps each autograd graph
        # bounded by ``batch_points``.
        for s in _chunks(len(pts), batch_points):
            weight = (s.stop - s.start) / len(pts)
            K_up = upper_gram(chi_net, pts[s], kvals[s], create_graph=True)
            R_up = dual_gram(psi_net, pts[s], kvals[s], create_graph=True)
            (weight * (torch.trace(K_up) + torch.trace(R_up))).backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            log(step)

    final = evaluate_bounds(chi_net, psi_net, pts, kvals, batch_points)
    final.meta = {
        "nq": nq, "n_harmonics": n_harmonics, "width": width, "depth": depth,
        "steps": steps, "lr": lr, "seed": seed, "batch_points": batch_points,
    }
    return final, history, (chi_net, psi_net)
