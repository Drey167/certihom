"""Requirements check for certifying a real microstructure image.

Certifying a micrograph is not "load image, run solver". Two failures found
in this project's own experiments motivate every check below:

  * a 504-pixel image with 4-pixel features was downsampled to 32x32 and
    certified, giving the TIGHTEST number in the study (19.6%) for a
    geometry that was not the one in the image;
  * the quality gate then rejected that run anyway, on headroom.

This script runs the preflight checks and tells you what settings the image
actually requires, before you spend a training run on it.

    python examples/check_micrograph_requirements.py --image path.png

Requirements, in order of how badly they bite:

  R1  BINARY (or cleanly segmentable). The certificate is for a two-phase
      problem with a definite coefficient. A greyscale image must be
      segmented first, and the segmentation is a modelling choice you own.
  R2  RESOLVED. The grid must resolve the smallest features. We measure the
      autocorrelation half-width; if features span fewer than ~3 pixels on
      the certification grid, the geometry is aliased and the certificate
      bounds the wrong problem.
  R3  SQUARE and PERIODIC-COMPATIBLE. The cell must be square. A micrograph
      is not periodic; either take the crop as a cell directly (artificial
      boundary interfaces) or mirror it (imposes symmetry). Report which.
  R4  REPRESENTATIVE. The crop must contain enough features that the
      effective property is not dominated by one inclusion. Below ~10
      correlation lengths across, treat the result as a statistical volume
      element, not an RVE.
  R5  CONTRAST ASSIGNED. Phase conductivities/stiffnesses are yours to
      choose; the certificate is conditional on them.
  R6  QUADRATURE HEADROOM. After training, headroom >= 2 and refinement
      ratio <= 1.05, or the number is not trustworthy (see the paper's
      Section 7). This is checked after the run, not before.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image


def correlation_half_width(mask):
    """Pixels at which the two-point autocorrelation of the phase indicator
    first falls below 0.5. This is the feature scale."""
    f = mask.astype(float) - mask.mean()
    F = np.fft.fft2(f)
    ac = np.real(np.fft.ifft2(F * np.conj(F)))
    ac /= ac[0, 0]
    prof = ac[0, : mask.shape[1] // 2]
    below = np.where(prof < 0.5)[0]
    return int(below[0]) if below.size else mask.shape[1] // 2


def check(path, size, border):
    a = np.array(Image.open(path).convert("L"), dtype=np.float64)
    print(f"image                : {os.path.basename(path)}  {a.shape}")

    # R1 — binary?
    u = np.unique(a)
    lo, hi = a.min(), a.max()
    frac_mid = float(((a > lo + 0.1 * (hi - lo)) &
                      (a < hi - 0.1 * (hi - lo))).mean())
    binary = frac_mid < 0.05
    print(f"R1 binary            : {len(u)} grey levels, "
          f"{frac_mid:.1%} of pixels mid-tone -> "
          f"{'OK (segmented)' if binary else 'SEGMENT FIRST — this is greyscale'}")

    if border:
        a = a[border:-border, border:-border]
    n = min(a.shape)
    a = a[:n, :n]
    mask = a > 0.5 * (a.min() + a.max())
    print(f"R3 square crop       : {a.shape}, phase fraction {mask.mean():.3f}")

    # R2 — resolved?
    hw_full = correlation_half_width(mask)
    print(f"R2 feature scale     : correlation half-width {hw_full} px "
          f"at native {n} px")
    feats = n / max(hw_full, 1)
    hw_grid = hw_full * size / n
    print(f"   at grid {size:>4}      : features span {hw_grid:.1f} px "
          f"-> {'OK' if hw_grid >= 3 else 'ALIASED — increase --size or crop smaller'}")
    if hw_grid < 3:
        need = int(np.ceil(3 * n / max(hw_full, 1)))
        print(f"   minimum grid needed: {need} px "
              f"(or crop to {int(hw_full*30)} px at full resolution)")

    # R4 — representative?
    print(f"R4 representativeness: ~{feats:.0f} correlation lengths across")
    if feats < 10:
        print("   WARNING: fewer than 10 — report as a statistical volume "
              "element, not an RVE")
    else:
        print("   OK")

    # R6 — what to expect
    print(f"R6 quadrature        : start at nq = {2*size}; if the gate "
          f"rejects, double it")
    print("\nrecommended command:")
    print(f"  python examples/run_micrograph.py --image {path} "
          f"--size {max(size, int(np.ceil(3*n/max(hw_full,1))))} --steps 300")
    print("\nThen CHECK THE GATE. A rejected run is valid but its width is "
          "not trustworthy;\nraise nq until it accepts, and quote the "
          "accepted number even if it is wider.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--size", type=int, default=64,
                   help="intended certification grid")
    p.add_argument("--border", type=int, default=3,
                   help="pixels to trim from each edge")
    args = p.parse_args()
    check(args.image, args.size, args.border)
