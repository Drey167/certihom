"""certihom — Phase 0: certified two-sided bounds on effective conductivity.

Scalar (thermal/electrical conductivity) warm-up for the research program
"Certified Amortized Homogenization". This package computes, for a periodic
two-phase microstructure k(y) on the unit cell Y=[0,1)^2, a pair of 2x2
SPD matrices

        K_low  <=  K_eff  <=  K_up      (Loewner / quadratic-form order)

where K_up comes from a kinematically admissible (periodic) temperature
fluctuation field and K_low from a statically admissible (divergence-free,
mean-controlled) flux field built from a stream function. Both trial fields
are neural networks trained *label-free* by minimizing (complementary)
energy; the bounds are valid for ANY admissible field, trained or not,
up to quadrature error (see MATH.md, "Certificate caveat").

Modules
-------
microstructures : pixel-grid microstructure generators + point lookup
fields          : periodic-by-construction neural trial fields
quadrature      : midpoint quadrature grids on the unit cell
energy          : Gram-matrix bound assembly (single source of truth)
bounds          : CertifiedBounds container + validity checks
train           : self-supervised training loop producing bounds
fv_reference    : periodic finite-volume reference solver (validation only)
exact           : closed-form effective tensors (laminate, checkerboard, ...)
"""

from .bounds import CertifiedBounds
from .train import fit_bounds

__all__ = ["CertifiedBounds", "fit_bounds"]
__version__ = "0.1.0"
