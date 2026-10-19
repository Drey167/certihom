"""Closed-form effective conductivities used as ground truth.

These are the falsifiable anchors of Phase 0: any certified interval that
fails to contain them (beyond quadrature/reference tolerance) proves a bug.
"""

from __future__ import annotations

import numpy as np


def laminate_exact(k1: float, k2: float, f1: float, axis: int = 0) -> np.ndarray:
    """Exact effective tensor of a two-phase laminate stacked along ``axis``.

    Transport ACROSS the layers (along the stacking axis) sees the phases in
    series -> harmonic mean; transport ALONG the layers sees them in
    parallel -> arithmetic mean. Classical result (e.g. Milton, "The Theory
    of Composites", ch. 9).
    """
    f2 = 1.0 - f1
    harm = 1.0 / (f1 / k1 + f2 / k2)
    arith = f1 * k1 + f2 * k2
    if axis == 0:
        return np.diag([harm, arith]).astype(np.float64)
    return np.diag([arith, harm]).astype(np.float64)


def checkerboard_exact(k1: float, k2: float) -> np.ndarray:
    """Dykhne/Keller duality: the 2D two-phase checkerboard has
    K_eff = sqrt(k1*k2) * I exactly, independent of contrast."""
    return np.sqrt(k1 * k2) * np.eye(2, dtype=np.float64)


def voigt(K_field: np.ndarray) -> np.ndarray:
    """Arithmetic-mean (zero primal fluctuation) upper bound <k> I."""
    return float(K_field.mean()) * np.eye(2, dtype=np.float64)


def reuss(K_field: np.ndarray) -> np.ndarray:
    """Harmonic-mean (uniform-flux) lower bound <1/k>^{-1} I."""
    return float(1.0 / (1.0 / K_field).mean()) * np.eye(2, dtype=np.float64)
