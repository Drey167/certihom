"""Pixel-grid microstructures on the periodic unit cell Y = [0,1)^2.

A microstructure is a numpy array ``K`` of shape (n, n), float64, where cell
(i, j) covers the square [i/n, (i+1)/n) x [j/n, (j+1)/n) and holds the
(isotropic, scalar) conductivity of that pixel. Axis 0 is the y1 coordinate,
axis 1 is y2.

All generators are deterministic given their arguments (and ``seed`` where
applicable) — reproducibility is a hard requirement of this project.
"""

from __future__ import annotations

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------

def homogeneous(n: int, k0: float) -> np.ndarray:
    """Uniform conductivity k0. K_eff = k0 * I exactly; optimal fluctuations
    vanish, so trained bounds must collapse onto k0*I (unit test)."""
    return np.full((n, n), float(k0), dtype=np.float64)


def laminate(n: int, k1: float, k2: float, f1: float, axis: int = 0) -> np.ndarray:
    """Two-phase laminate with layers stacked along ``axis``.

    Phase 1 (conductivity k1) occupies volume fraction f1; interfaces are
    pixel-aligned, which requires f1*n to be an integer (asserted). The exact
    effective tensor is diagonal: harmonic mean along the stacking axis
    (transport crosses the layers in series), arithmetic mean along the other
    axis (layers in parallel). See exact.laminate_exact.
    """
    m = f1 * n
    assert abs(m - round(m)) < 1e-12, "f1*n must be an integer (pixel-aligned interfaces)"
    m = int(round(m))
    K = np.full((n, n), float(k2), dtype=np.float64)
    if axis == 0:
        K[:m, :] = k1
    elif axis == 1:
        K[:, :m] = k1
    else:
        raise ValueError("axis must be 0 or 1")
    return K


def checkerboard(n: int, k1: float, k2: float) -> np.ndarray:
    """2x2 periodic checkerboard (each phase occupies two diagonal quadrants).

    Exact effective conductivity (Dykhne/Keller duality): sqrt(k1*k2) * I.
    Local fields are singular at the corners where four cells meet, which is
    exactly why this is the standard stress test for neural trial fields.
    ``n`` must be even.
    """
    assert n % 2 == 0, "n must be even for a 2x2 checkerboard"
    K = np.empty((n, n), dtype=np.float64)
    h = n // 2
    K[:h, :h] = k1
    K[h:, h:] = k1
    K[:h, h:] = k2
    K[h:, :h] = k2
    return K


def boolean_disks(
    n: int,
    k_matrix: float,
    k_inclusion: float,
    n_disks: int = 6,
    radius: float = 0.12,
    seed: int = 0,
) -> np.ndarray:
    """Random periodic Boolean model: disks of ``k_inclusion`` in a matrix.

    Disk centers are i.i.d. uniform on the torus; overlap is allowed; the
    indicator is evaluated at pixel centers with periodic (torus) distance.
    Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    centers = rng.random((n_disks, 2))
    yc = (np.arange(n) + 0.5) / n
    Y1, Y2 = np.meshgrid(yc, yc, indexing="ij")
    K = np.full((n, n), float(k_matrix), dtype=np.float64)
    for cx, cy in centers:
        d1 = np.minimum(np.abs(Y1 - cx), 1.0 - np.abs(Y1 - cx))
        d2 = np.minimum(np.abs(Y2 - cy), 1.0 - np.abs(Y2 - cy))
        K[d1**2 + d2**2 <= radius**2] = k_inclusion
    return K


# ----------------------------------------------------------------------------
# REALISTIC morphologies (standard models used in the composites literature)
# ----------------------------------------------------------------------------

def gaussian_random_field(n: int, k_a: float, k_b: float, volume_fraction: float
                          = 0.5, correlation: float = 6.0, seed: int = 0):
    """Spinodal / bicontinuous morphology: level set of a filtered periodic
    Gaussian field. This is the standard model for spinodally decomposed
    alloys, polymer blends, aerogels and bicontinuous porous media, and is
    far more realistic than isolated inclusions: the phases percolate,
    interfaces are curved and rough, and there is no characteristic
    inclusion shape.

    ``correlation`` sets the dominant wavenumber (larger = finer features).
    The level is chosen so the phase fraction is exactly as requested.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    xi = np.fft.fftfreq(n, d=1.0 / n)
    X1, X2 = np.meshgrid(xi, xi, indexing="ij")
    R = np.sqrt(X1**2 + X2**2)
    # band-pass filter peaked at the correlation wavenumber
    filt = np.exp(-((R - correlation) ** 2) / (2.0 * (correlation / 3.0) ** 2))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
    level = np.quantile(field, 1.0 - volume_fraction)
    return np.where(field >= level, k_a, k_b).astype(np.float64)


def voronoi_polycrystal(n: int, k_a: float, k_b: float, n_grains: int = 12,
                        volume_fraction: float = 0.5, seed: int = 0):
    """Polycrystalline / cellular morphology: periodic Voronoi tessellation
    with grains assigned to phases. Models sintered ceramics, metal
    polycrystals and closed-cell foams — straight grain boundaries meeting
    at TRIPLE JUNCTIONS, which are milder singularities than the
    checkerboard's four-phase corner but harsher than smooth inclusions.
    """
    rng = np.random.default_rng(seed)
    sites = rng.random((n_grains, 2))
    c = (np.arange(n) + 0.5) / n
    Y1, Y2 = np.meshgrid(c, c, indexing="ij")
    best = np.full((n, n), np.inf)
    label = np.zeros((n, n), dtype=int)
    for g, (sx, sy) in enumerate(sites):
        d1 = np.abs(Y1 - sx); d1 = np.minimum(d1, 1 - d1)
        d2 = np.abs(Y2 - sy); d2 = np.minimum(d2, 1 - d2)
        d = d1**2 + d2**2
        m = d < best
        best[m] = d[m]; label[m] = g
    order = rng.permutation(n_grains)
    n_hard = max(1, int(round(volume_fraction * n_grains)))
    hard = set(order[:n_hard].tolist())
    return np.where(np.isin(label, list(hard)), k_a, k_b).astype(np.float64)


def fiber_packing(n: int, k_matrix: float, k_fiber: float, n_fibers: int = 10,
                  radius: float = 0.09, seed: int = 0, max_tries: int = 4000):
    """Non-overlapping (random sequential adsorption) fiber packing — the
    standard transverse cross-section model for unidirectional
    fibre-reinforced composites. Unlike the Boolean model used elsewhere in
    this project, fibres may not overlap, which produces the narrow matrix
    ligaments between near-touching fibres that dominate real transverse
    properties."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(max_tries):
        if len(centers) == n_fibers:
            break
        p = rng.random(2)
        ok = True
        for q in centers:
            d1 = abs(p[0] - q[0]); d1 = min(d1, 1 - d1)
            d2 = abs(p[1] - q[1]); d2 = min(d2, 1 - d2)
            if d1**2 + d2**2 < (2.05 * radius) ** 2:
                ok = False
                break
        if ok:
            centers.append(p)
    c = (np.arange(n) + 0.5) / n
    Y1, Y2 = np.meshgrid(c, c, indexing="ij")
    K = np.full((n, n), float(k_matrix))
    for cx, cy in centers:
        d1 = np.abs(Y1 - cx); d1 = np.minimum(d1, 1 - d1)
        d2 = np.abs(Y2 - cy); d2 = np.minimum(d2, 1 - d2)
        K[d1**2 + d2**2 <= radius**2] = k_fiber
    return K


# ----------------------------------------------------------------------------
# Point lookup (numpy microstructure -> conductivity at quadrature points)
# ----------------------------------------------------------------------------

def sample_k(K: np.ndarray, y: torch.Tensor) -> torch.Tensor:
    """Nearest-pixel lookup of the conductivity field at points ``y``.

    Parameters
    ----------
    K : (n, n) float64 array, pixel conductivities.
    y : (N, 2) torch tensor with entries in [0, 1). Points must not sit
        exactly on pixel boundaries for the lookup to be unambiguous; the
        midpoint quadrature grids used throughout guarantee this whenever
        the quadrature resolution is a multiple of n.

    Returns
    -------
    (N,) float64 torch tensor of conductivities (constant, no gradient).
    """
    n = K.shape[0]
    idx = torch.clamp((y * n).long(), 0, n - 1)
    Kt = torch.as_tensor(K, dtype=y.dtype, device=y.device)
    return Kt[idx[:, 0], idx[:, 1]]
