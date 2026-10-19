"""Quadrature on the periodic unit cell.

Phase 0 uses a fixed tensor-product midpoint rule. Two facts make this an
honest choice:

1. If the quadrature resolution nq is an integer multiple of the
   microstructure resolution n, every quadrature point lies strictly inside
   one pixel, so the piecewise-constant conductivity is sampled
   unambiguously and the integrand (which is smooth *within* each pixel for
   smooth trial fields) is integrated with O(nq^-2) error.

2. For trial fields whose gradient is constant per pixel — which includes
   the exact optimal fields of the laminate — the midpoint rule is EXACT,
   which is what lets the unit tests demand machine precision.

The certificates produced with this rule are therefore "guaranteed up to
quadrature error". Phase 1 removes even that caveat by projecting network
fields onto conforming finite-element spaces with exact per-element
integration (see MATH.md).
"""

from __future__ import annotations

import torch


def midpoint_grid(nq: int, dtype=torch.float64, device=None) -> torch.Tensor:
    """Return the (nq*nq, 2) float64 tensor of midpoints ((i+.5)/nq, (j+.5)/nq).

    The associated quadrature weight is uniform, 1/nq^2, so cell averages are
    plain means over the returned points.
    """
    c = (torch.arange(nq, dtype=dtype, device=device) + 0.5) / nq
    Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
    return torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1)


def midpoint_grid_chunk(nq: int, start: int, stop: int,
                        dtype=torch.float64, device=None) -> torch.Tensor:
    """Return rows ``start:stop`` of :func:`midpoint_grid` without
    materialising its full ``nq*nq`` tensor.

    The flattened ordering is exactly the one used by ``midpoint_grid``.
    This is important for large diagnostic refinements: it changes only the
    memory schedule, not the quadrature rule or its points.
    """
    if not (0 <= start <= stop <= nq * nq):
        raise ValueError("chunk must lie inside the nq*nq midpoint grid")
    idx = torch.arange(start, stop, device=device)
    i = torch.div(idx, nq, rounding_mode="floor")
    j = torch.remainder(idx, nq)
    return torch.stack([(i.to(dtype) + 0.5) / nq,
                        (j.to(dtype) + 0.5) / nq], dim=1)
