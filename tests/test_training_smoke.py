"""End-to-end smoke test, sized for a 1-core CPU box (< ~60 s).

Checks the three properties that make this project falsifiable:
  1. step 0 equals Voigt-Reuss (valid before training),
  2. training tightens the certified gap substantially,
  3. the trained interval remains Loewner-valid and still brackets the
     EXACT checkerboard tensor (small tolerance for quadrature error).
"""

import numpy as np

from certihom.exact import checkerboard_exact, reuss, voigt
from certihom.microstructures import checkerboard
from certihom.train import fit_bounds


def test_training_tightens_and_stays_valid():
    K = checkerboard(16, 1.0, 4.0)
    b, hist, _ = fit_bounds(
        K, nq=32, n_harmonics=3, width=32, depth=2,
        steps=300, lr=2e-3, seed=0, log_every=150, verbose=False,
    )
    # 1. step 0 == Voigt-Reuss
    tr_v = np.trace(voigt(K))
    tr_r = np.trace(reuss(K))
    assert abs(hist[0]["tr_K_up"] - tr_v) < 1e-10
    assert abs(hist[0]["tr_K_low"] - tr_r) < 1e-10
    # 2. substantial tightening
    gap0 = hist[0]["gap_rel"]
    assert b.gap_rel() < 0.6 * gap0
    # 3. validity and bracketing of the exact value (quadrature slack 1e-3)
    assert b.loewner_valid(tol=1e-9)
    assert b.brackets(checkerboard_exact(1.0, 4.0), tol=1e-3)
