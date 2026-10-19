"""Assembly of the certified Gram-matrix bounds.

Mathematical core (full derivations and proofs in MATH.md):

Primal (upper) bound. For unit mean gradients e_1, e_2 and ANY periodic
trial fluctuations chi_1, chi_2, the total-gradient fields
G_i = e_i + grad(chi_i) satisfy, by the minimum-energy characterization of
K_eff and linear superposition (the trial map ebar -> sum_i ebar_i chi_i is
linear in the load),

    K_up[i,j] := < k * G_i . G_j >   ==>   K_up >= K_eff  (Loewner order).

Dual (lower) bound. For unit mean fluxes e_1, e_2 and ANY periodic stream
functions psi_1, psi_2, the fields Q_i = e_i + curl(psi_i), with
curl(psi) := (d(psi)/dy2, -d(psi)/dy1), are exactly divergence-free with
exactly the prescribed mean (the mean of derivatives of a periodic function
vanishes identically). The complementary-energy principle then gives

    R_up[i,j] := < (1/k) * Q_i . Q_j >  >=  K_eff^{-1}
    ==>  K_low := R_up^{-1}  <=  K_eff,

because matrix inversion reverses the Loewner order on SPD matrices.

The single function ``gram`` below implements < w * F_i . F_j > for a pair
of vector fields and a positive weight; both bounds, and the analytic-field
unit tests, go through it. That is deliberate: the tests then certify the
exact code path used in production.
"""

from __future__ import annotations

import numpy as np
import torch

from .fields import heads_and_grads

def _eye2(t: torch.Tensor) -> torch.Tensor:
    """Unit loads e_1, e_2 matching the dtype/device of a field tensor."""
    return torch.eye(2, dtype=t.dtype, device=t.device)


def gram(fields: list[torch.Tensor], weight: torch.Tensor) -> torch.Tensor:
    """G[i,j] = mean_p( weight_p * fields[i]_p . fields[j]_p ).

    fields : list of (N, 2) tensors (total gradient or total flux fields).
    weight : (N,) positive tensor (k for the primal side, 1/k for the dual).

    With uniform midpoint weights the mean over points IS the cell integral
    (up to the quadrature caveat documented in quadrature.py).
    """
    d = len(fields)
    G = torch.empty((d, d), dtype=fields[0].dtype, device=fields[0].device)
    for i in range(d):
        for j in range(i, d):
            val = (weight * (fields[i] * fields[j]).sum(dim=1)).mean()
            G[i, j] = val
            G[j, i] = val
    return G


def primal_fields(chi_net, pts: torch.Tensor, create_graph: bool) -> list[torch.Tensor]:
    """Total-gradient trial fields G_i = e_i + grad(chi_i)."""
    _, grads = heads_and_grads(chi_net, pts, create_graph)
    E = _eye2(grads[0])
    return [E[i] + grads[i] for i in range(2)]


def dual_fields(psi_net, pts: torch.Tensor, create_graph: bool) -> list[torch.Tensor]:
    """Total-flux trial fields Q_i = e_i + curl(psi_i), divergence-free by
    construction: curl(psi) = (d psi / d y2, -d psi / d y1)."""
    _, grads = heads_and_grads(psi_net, pts, create_graph)
    E = _eye2(grads[0])
    out = []
    for i in range(2):
        g = grads[i]
        curl = torch.stack([g[:, 1], -g[:, 0]], dim=1)
        out.append(E[i] + curl)
    return out


def upper_gram(chi_net, pts: torch.Tensor, kvals: torch.Tensor,
               create_graph: bool = False) -> torch.Tensor:
    """K_up from the primal network (>= K_eff up to quadrature error)."""
    return gram(primal_fields(chi_net, pts, create_graph), kvals)


def dual_gram(psi_net, pts: torch.Tensor, kvals: torch.Tensor,
              create_graph: bool = False) -> torch.Tensor:
    """R_up from the dual network (>= K_eff^{-1} up to quadrature error)."""
    return gram(dual_fields(psi_net, pts, create_graph), 1.0 / kvals)


def lower_from_dual(R_up: torch.Tensor) -> torch.Tensor:
    """K_low = R_up^{-1}; SPD inversion reverses the Loewner order."""
    return torch.linalg.inv(R_up)


def as_numpy(M: torch.Tensor) -> np.ndarray:
    return M.detach().cpu().numpy()
