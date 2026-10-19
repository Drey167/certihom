"""Diagnostics for quadrature safety.

Motivation (RESULTS.md, "CRITICAL FINDING"): a network trained against a
fixed collocation rule can place energy BETWEEN the sample points, so the
reported energy — and any bound built from it — can be far below the truth.
This module provides the two checks that detect it.

1. ``refinement_ratio`` — evaluate the same field at nq and at f*nq. If the
   energy rises materially, the training rule was under-resolved. Cheap,
   assumption-free, and should accompany every collocation-trained result.

2. ``spectral_bandwidth`` — measure where the field's energy actually lives
   in frequency space. A midpoint rule with nq points per direction can
   only resolve frequencies up to nq/2 (Nyquist); if the field carries
   significant content beyond that, aliasing is not a risk but a
   certainty. This turns "use a fine enough rule" into a number.

Neither is specific to homogenization: both apply to any physics-informed
network trained on fixed collocation points.
"""

from __future__ import annotations

import numpy as np
import torch

from .energy import as_numpy, upper_gram
from .microstructures import sample_k
from .quadrature import midpoint_grid, midpoint_grid_chunk


def _upper_trace_batched(chi_net, K: np.ndarray, nq: int,
                         batch_points: int | None) -> float:
    """Exact same midpoint average as ``upper_gram``, evaluated in chunks."""
    n_points = nq * nq
    size = n_points if batch_points is None else batch_points
    if size <= 0:
        raise ValueError("batch_points must be positive")
    total = 0.0
    with torch.enable_grad():
        for start in range(0, n_points, size):
            stop = min(start + size, n_points)
            pts = midpoint_grid_chunk(nq, start, stop)
            kv = sample_k(K, pts)
            total += (stop - start) * float(torch.trace(
                upper_gram(chi_net, pts, kv)).detach())
    return total / n_points


def refinement_ratio(chi_net, K: np.ndarray, nq: int, factor: int = 4,
                     batch_points: int | None = None):
    """Energy at the training rule vs at a refined rule.

    Returns dict with the reported and refined upper bounds (trace) and
    ``ratio`` = refined / reported. A ratio near 1 means the rule resolved
    the field; a large ratio means energy was hidden between points and the
    reported bound is NOT trustworthy.
    """
    out = {}
    for tag, q in (("reported", nq), ("refined", factor * nq)):
        out[tag] = _upper_trace_batched(chi_net, K, q, batch_points)
    out["ratio"] = out["refined"] / max(out["reported"], 1e-30)
    out["nq"] = nq
    return out


def spectral_bandwidth(net, n_probe: int = 128, quantile: float = 0.999,
                       head: int = 0) -> float:
    """Effective bandwidth of a trained periodic field, in cycles per cell.

    Evaluates the field on an n_probe x n_probe grid, takes the 2D FFT,
    discards the DC term, and returns the smallest radius (in integer
    frequency units) whose disc contains ``quantile`` of the spectral
    energy. Compare against the Nyquist limit nq/2 of the quadrature rule.
    """
    c = torch.arange(n_probe, dtype=torch.float64) / n_probe
    Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
    pts = torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1)
    with torch.no_grad():
        f = net(pts)[:, head].reshape(n_probe, n_probe)
    F = torch.fft.fft2(f)
    P = (F.abs() ** 2).numpy()
    P[0, 0] = 0.0
    k = np.fft.fftfreq(n_probe, d=1.0 / n_probe)
    R = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    order = np.argsort(R, axis=None)
    cum = np.cumsum(P.ravel()[order])
    tot = cum[-1]
    if tot <= 0:
        return 0.0
    idx = int(np.searchsorted(cum, quantile * tot))
    return float(R.ravel()[order][min(idx, R.size - 1)])


def quadrature_safety_report(chi_net, K: np.ndarray, nq: int,
                             factor: int = 4,
                             batch_points: int | None = None) -> dict:
    """Combined verdict: bandwidth vs Nyquist, and the refinement ratio."""
    B = spectral_bandwidth(chi_net)
    r = refinement_ratio(chi_net, K, nq, factor, batch_points)
    nyquist = nq / 2.0
    r.update(bandwidth=B, nyquist=nyquist, headroom=nyquist / max(B, 1e-9),
             safe=(r["ratio"] < 1.05) and (B < nyquist))
    return r


# ----------------------------------------------------------------------------
# Quality gate — a LABEL-FREE accept/reject test for a trained field
# ----------------------------------------------------------------------------

HEADROOM_MIN = 2.0     # Nyquist / bandwidth
RATIO_MAX = 1.05       # energy at 4x the rule / energy at the rule


def quality_gate(chi_net, K: np.ndarray, nq: int,
                 headroom_min: float = HEADROOM_MIN,
                 ratio_max: float = RATIO_MAX,
                 batch_points: int | None = None) -> dict:
    """Decide whether a trained field can be trusted — WITHOUT ground truth.

    Motivation (measured, RESULTS.md): at a fixed configuration and
    quadrature rule, individual training runs can still collapse. On the
    laminate case at nq=64, four seeds gave certified gaps of 0.4-1.7% with
    bandwidth ~6.6 (headroom ~4.9, refinement ratio ~1.02), while one seed
    gave 97.8% with bandwidth 41.0 (headroom 0.78, refinement ratio 7.08).
    Headroom is therefore a property of the TRAINED FIELD, not of the
    configuration, and must be checked per run.

    The check uses no reference solution and no exact answer, so it is
    available in exactly the situation where it is needed.

    Returns the diagnostic dict with an added boolean ``accept``. A rejected
    run should be retrained (different seed) or run at higher nq; its
    certificate is still VALID if computed by exact integration, merely
    uselessly wide.
    """
    rep = quadrature_safety_report(chi_net, K, nq,
                                   batch_points=batch_points)
    rep["accept"] = bool(rep["headroom"] >= headroom_min
                         and rep["ratio"] <= ratio_max)
    rep["reason"] = ("ok" if rep["accept"] else
                     f"headroom {rep['headroom']:.2f} < {headroom_min}"
                     if rep["headroom"] < headroom_min else
                     f"refinement ratio {rep['ratio']:.2f} > {ratio_max}")
    return rep
