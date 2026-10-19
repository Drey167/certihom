"""Neural trial fields that are admissible BY CONSTRUCTION.

Admissibility is the whole game: the bound theorems (MATH.md) hold for any
field in the admissible set, so the architecture must place every
representable function inside that set, exactly.

* Periodicity (both the primal fluctuation chi and the dual stream function
  psi must be Y-periodic): enforced by feeding the network only the
  Fourier features sin(2*pi*m*y_c), cos(2*pi*m*y_c), m = 1..M, c = 1,2.
  Any smooth function of these features is Y-periodic, and mean-zero of the
  gradient/curl of a periodic field is automatic.

* Zero initialization: the last linear layer starts at exactly zero, so at
  step 0 the trial fluctuations vanish and the certificates equal the
  classical Voigt (upper) and Reuss (lower) bounds. Training can only
  tighten from there — the certificate is valid from the first iteration.

The same network class serves both sides; only the *interpretation* of the
two output heads differs (chi_1, chi_2 for the primal side; psi_1, psi_2 for
the dual side), which is handled in energy.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PeriodicEmbedding(nn.Module):
    """y in [0,1)^2  ->  [sin(2 pi m y_c), cos(2 pi m y_c)]_{m<=M, c<=2}.

    Output dimension is 4*M. Differentiable, so autograd gradients of the
    downstream MLP with respect to y are exact.
    """

    def __init__(self, n_harmonics: int = 3, dtype=torch.float64):
        super().__init__()
        self.M = int(n_harmonics)
        freqs = 2.0 * torch.pi * torch.arange(1, self.M + 1, dtype=dtype)
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 4 * self.M

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (N, 2) -> phases: (N, 2, M)
        ph = y.unsqueeze(-1) * self.freqs  # broadcast over harmonics
        feats = torch.cat([torch.sin(ph), torch.cos(ph)], dim=-1)  # (N, 2, 2M)
        return feats.reshape(y.shape[0], -1)  # (N, 4M)


class FieldPair(nn.Module):
    """``n_out`` scalar periodic fields (one per unit load / component) in
    one MLP.

    forward(y) -> (N, n_out); column i is the trial field associated with
    the i-th unit macroscopic load (Phase 0: 2 heads; Phase 2 primal: 6 =
    3 strain loads x 2 displacement components; Phase 2 dual: 3 Airy
    potentials). tanh activations keep the fields smooth (bounded
    derivatives of all orders — the dual side differentiates twice), float64
    throughout for certificate arithmetic.
    """

    def __init__(self, n_harmonics: int = 3, width: int = 64, depth: int = 3,
                 n_out: int = 2, dtype=torch.float64):
        super().__init__()
        self.embed = PeriodicEmbedding(n_harmonics, dtype=dtype)
        layers: list[nn.Module] = []
        d_in = self.embed.out_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, width, dtype=dtype), nn.Tanh()]
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d_in, n_out, dtype=dtype)
        # Zero-init: certificates start at Voigt-Reuss (see module docstring).
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(self.embed(y)))


def heads_and_grads(
    net: nn.Module, pts: torch.Tensor, create_graph: bool
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Evaluate a FieldPair and the spatial gradient of each head.

    Parameters
    ----------
    net : FieldPair
    pts : (N, 2) quadrature points (no gradient history required).
    create_graph : True during training (gradients of the energy w.r.t.
        network parameters flow through these spatial derivatives), False
        for evaluation.

    Returns
    -------
    out   : (N, 2) field values.
    grads : list of two (N, 2) tensors; grads[i][p] = d(head_i)/dy at point p.

    Implementation note: because the network acts pointwise, the gradient of
    sum_p head_i(y_p) with respect to the batch of points equals the batch of
    per-point gradients — the standard PINN trick, one backward pass per head.
    """
    y = pts.detach().clone().requires_grad_(True)
    out = net(y)
    grads = []
    for i in range(out.shape[1]):
        gi = torch.autograd.grad(
            out[:, i].sum(), y, create_graph=create_graph, retain_graph=True
        )[0]
        grads.append(gi)
    return out, grads
