"""Quadrature safety — WHY Phase 1 exact integration is mandatory.

Discovered experimentally, not anticipated: a network trained against a
FIXED midpoint rule can learn to place its energy BETWEEN the quadrature
points. The rule then reports an energy far below the field's true energy,
and the resulting "upper bound" can fall BELOW the exact effective tensor —
a false certificate.

Measured instances:
  * adversarial (600 steps, primal-only, lr 3e-3, nq=32): the rule reported
    K_up[0,0] = 0.0014 while refinement to nq>=128 revealed ~58.9, against
    a true value of 3.0769;
  * ordinary settings (laminate c6, nq=32, 400 steps): K_up[0,0] = 2.5207
    vs true 2.6667 — mildly but genuinely invalid.
  * by contrast the checkerboard at nq=64 was stable under refinement
    (2.2786 -> 2.2906), so this is configuration-dependent, not universal.

The fix is structural, not a tolerance: certify by projecting onto a
conforming FE space and integrating EXACTLY (certihom.certify /
certihom.bfs). Interpolation cannot hide energy between sample points
because there are no sample points — the integral is exact.

RULE FOR THIS PROJECT: never report a midpoint-rule bound as a
certificate. Train against it if convenient; certify with exact
integration always.
"""

import numpy as np

from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.energy import as_numpy, dual_gram, lower_from_dual, upper_gram
from certihom.exact import laminate_exact
from certihom.microstructures import laminate, sample_k
from certihom.quadrature import midpoint_grid
from certihom.train import fit_bounds


def test_exact_integration_is_valid_where_midpoint_may_not_be():
    """The exactly-integrated certificate must contain the exact tensor.

    The midpoint bound at training resolution is NOT asserted valid here —
    that is precisely the failure mode this module documents. What must
    always hold is the exact-integration certificate.
    """
    K = laminate(16, 1.0, 6.0, 0.25, axis=0)
    K_ex = laminate_exact(1.0, 6.0, 0.25, 0)
    NQ = 32
    _, _, (chi, psi) = fit_bounds(K, nq=NQ, n_harmonics=4, width=48, depth=3,
                                  steps=400, lr=2e-3, seed=0, verbose=False)

    b_exact = certified_bounds_scalar(nodal_values(chi, 64),
                                      nodal_values(psi, 64), K)
    assert b_exact.loewner_valid(1e-10)
    assert b_exact.brackets(K_ex, tol=1e-9), (
        "exact-integration certificate must contain the exact tensor")


def test_refinement_detects_quadrature_overfitting():
    """Refining the rule must not LOWER the reported upper bound much.

    A large increase on refinement means the training resolution was hiding
    energy. This is a diagnostic every study should run; here we simply
    assert the diagnostic itself works by checking the refined value is a
    valid bound.
    """
    K = laminate(16, 1.0, 6.0, 0.25, axis=0)
    K_ex = laminate_exact(1.0, 6.0, 0.25, 0)
    NQ = 32
    _, _, (chi, psi) = fit_bounds(K, nq=NQ, n_harmonics=4, width=48, depth=3,
                                  steps=400, lr=2e-3, seed=0, verbose=False)

    vals = []
    for nq in (NQ, 4 * NQ):
        pts = midpoint_grid(nq)
        kv = sample_k(K, pts)
        vals.append(as_numpy(upper_gram(chi, pts, kv))[0, 0])
        low = as_numpy(lower_from_dual(dual_gram(psi, pts, kv)))[0, 0]
        assert low <= K_ex[0, 0] + 1e-6      # dual side stayed honest here

    # The refined evaluation is the trustworthy one and must be a bound.
    assert vals[1] >= K_ex[0, 0] - 1e-6, (
        "even the refined midpoint value is below the exact tensor — "
        "investigate before trusting anything from this run")
