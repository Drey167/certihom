"""Label-free training of the elasticity primal-dual pair (Phase 2).

Identical contract to Phase 0's train.py: fixed midpoint quadrature, seeded,
zero-init heads (step 0 = elasticity Voigt-Reuss), loss = tr C_up + tr S_up
(each head decoupled and minimized by its true corrector), every iterate
carries a valid certificate. Cost note: the dual side differentiates the
Airy potentials twice, so a step runs ~15 autograd passes; sizes in tests
and examples are chosen for a 1-core CPU box and scale trivially on a GPU.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .bounds import CertifiedBounds
from .elasticity import (dual_gram_elast, sample_mandel, upper_gram_elast)
from .energy import as_numpy
from .fields import FieldPair
from .quadrature import midpoint_grid


def evaluate_bounds_elast(chi_net, phi_net, pts, Cvals, Svals) -> CertifiedBounds:
    with torch.enable_grad():
        C_up = upper_gram_elast(chi_net, pts, Cvals, create_graph=False)
        S_up = dual_gram_elast(phi_net, pts, Svals, create_graph=False)
    C_low = torch.linalg.inv(S_up)
    return CertifiedBounds(K_low=as_numpy(C_low), K_up=as_numpy(C_up))


def fit_bounds_elasticity(
    Cfield: np.ndarray,
    nq: int = 48,
    n_harmonics: int = 3,
    width: int = 48,
    depth: int = 3,
    steps: int = 600,
    lr: float = 1e-3,
    seed: int = 0,
    log_every: int = 150,
    verbose: bool = True,
):
    """Train on one microstructure; return (CertifiedBounds, history, nets).

    Cfield: (n, n, 3, 3) per-pixel Mandel stiffnesses; nq must be a multiple
    of n. Reproducibility contract identical to Phase 0.
    """
    n = Cfield.shape[0]
    assert nq % n == 0, "quadrature resolution must be a multiple of the pixel grid"
    torch.manual_seed(seed)

    pts = midpoint_grid(nq)
    Cvals = sample_mandel(Cfield, pts)
    Svals = torch.linalg.inv(Cvals)

    chi_net = FieldPair(n_harmonics, width, depth, n_out=6)  # 3 loads x (u,v)
    phi_net = FieldPair(n_harmonics, width, depth, n_out=3)  # 3 Airy heads
    params = list(chi_net.parameters()) + list(phi_net.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    history = []
    t0 = time.time()

    def log(step):
        b = evaluate_bounds_elast(chi_net, phi_net, pts, Cvals, Svals)
        rec = {
            "step": step,
            "tr_C_up": float(np.trace(b.K_up)),
            "tr_C_low": float(np.trace(b.K_low)),
            "gap_rel": b.gap_rel(),
            "wall_s": time.time() - t0,
        }
        history.append(rec)
        if verbose:
            print(
                f"step {step:5d} | tr C_up {rec['tr_C_up']:.6f} | "
                f"tr C_low {rec['tr_C_low']:.6f} | gap {rec['gap_rel']:.3%} | "
                f"{rec['wall_s']:.1f}s"
            )

    log(0)  # elasticity Voigt-Reuss by zero-initialization
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        C_up = upper_gram_elast(chi_net, pts, Cvals, create_graph=True)
        S_up = dual_gram_elast(phi_net, pts, Svals, create_graph=True)
        loss = torch.trace(C_up) + torch.trace(S_up)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            log(step)

    final = evaluate_bounds_elast(chi_net, phi_net, pts, Cvals, Svals)
    final.meta = {
        "nq": nq, "n_harmonics": n_harmonics, "width": width, "depth": depth,
        "steps": steps, "lr": lr, "seed": seed,
    }
    return final, history, (chi_net, phi_net)
