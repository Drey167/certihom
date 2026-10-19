"""Regression tests for the independent periodic Q1 audit solver."""

import numpy as np

from certihom.elasticity import (iso_stiffness_mandel, laminate_exact_elasticity,
                                 mandel_field_from_mask)
from certihom.fem_independent import PeriodicQ1Elasticity, periodic_q1_effective


def test_independent_q1_annihilates_translations_and_solves_laminate():
    """An exact layer solution validates the audit implementation itself."""
    n, f_stiff = 8, 0.25
    C_stiff = iso_stiffness_mandel(10.0, 0.2)
    C_soft = iso_stiffness_mandel(1.0, 0.3)
    mask = np.zeros((n, n), dtype=bool)
    mask[: int(n * f_stiff)] = True
    field = mandel_field_from_mask(mask, C_stiff, C_soft)

    op = PeriodicQ1Elasticity(field)
    translation = np.ones((n, n, 2))
    assert np.linalg.norm(op.apply(translation)) < 1e-11

    got, report = periodic_q1_effective(field, rtol=1e-11)
    expected, _, _ = laminate_exact_elasticity(C_stiff, C_soft, f_stiff)
    assert np.allclose(got, expected, atol=1e-10)
    assert max(report.relative_residuals) < 1e-11
