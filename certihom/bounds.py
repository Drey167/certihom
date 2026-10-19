"""Container and validity logic for the certified interval [K_low, K_up].

The falsification rules of the research plan live here:

* ``loewner_valid``  — K_up - K_low must be PSD (a violation beyond
  tolerance means the implementation, not the model, is wrong).
* ``brackets``       — a trusted reference K_ref must satisfy
  K_low <= K_ref <= K_up within a tolerance that accounts for the
  reference's own discretization error. A single violation on a converged
  reference falsifies the pipeline; this is a feature, not a risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _min_eig(M: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())


@dataclass
class CertifiedBounds:
    K_low: np.ndarray
    K_up: np.ndarray
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- checks
    def loewner_valid(self, tol: float = 1e-10) -> bool:
        """K_up >= K_low in the Loewner sense, within ``tol``."""
        return _min_eig(self.K_up - self.K_low) >= -tol

    def brackets(self, K_ref: np.ndarray, tol: float = 1e-8) -> bool:
        """K_low <= K_ref <= K_up within ``tol`` (choose tol from the
        reference's own error, e.g. its mesh-convergence study)."""
        return (
            _min_eig(self.K_up - K_ref) >= -tol
            and _min_eig(K_ref - self.K_low) >= -tol
        )

    # --------------------------------------------------------------- metrics
    def gap_rel(self) -> float:
        """Relative certificate width: ||K_up - K_low||_F / ||K_mid||_F."""
        mid = 0.5 * (self.K_up + self.K_low)
        return float(
            np.linalg.norm(self.K_up - self.K_low) / np.linalg.norm(mid)
        )

    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.K_up + self.K_low)

    # --------------------------------------------------------------- display
    def summary(self) -> str:
        lo = np.linalg.eigvalsh(self.K_low)
        up = np.linalg.eigvalsh(self.K_up)
        return (
            f"K_low eigenvalues : [{lo[0]:.6f}, {lo[1]:.6f}]\n"
            f"K_up  eigenvalues : [{up[0]:.6f}, {up[1]:.6f}]\n"
            f"relative gap      : {self.gap_rel():.4%}\n"
            f"Loewner valid     : {self.loewner_valid()}"
        )
