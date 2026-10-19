"""Audit the auditor: the FV reference must be exact where exactness is
provable (laminates, homogeneous) and convergent toward the Dykhne value on
the checkerboard."""

import numpy as np

from certihom.exact import checkerboard_exact, laminate_exact
from certihom.fv_reference import fv_effective
from certihom.microstructures import checkerboard, homogeneous, laminate


def test_fv_homogeneous_exact():
    K_ref = fv_effective(homogeneous(8, 3.3))
    assert np.allclose(K_ref, 3.3 * np.eye(2), atol=1e-10)


def test_fv_laminate_exact_both_axes():
    for axis in (0, 1):
        K = laminate(16, 1.0, 7.0, 0.25, axis=axis)
        K_ref = fv_effective(K)
        assert np.allclose(K_ref, laminate_exact(1.0, 7.0, 0.25, axis), atol=1e-9)


def test_fv_checkerboard_converges_to_dykhne():
    """Corner singularities make this slow to converge; demand monotone
    improvement and <=5% error at n=64 for contrast 5."""
    exact = checkerboard_exact(1.0, 5.0)[0, 0]
    errs = []
    for n in (16, 32, 64):
        K_ref = fv_effective(checkerboard(n, 1.0, 5.0))
        errs.append(abs(K_ref[0, 0] - exact) / exact)
        # isotropy of the discrete tensor
        assert abs(K_ref[0, 0] - K_ref[1, 1]) < 1e-8
        assert abs(K_ref[0, 1]) < 1e-8
    assert errs[2] < errs[0]
    assert errs[2] < 0.05
