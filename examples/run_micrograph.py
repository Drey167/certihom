"""Certifying a REAL segmented micrograph.

NOTE on the bracket test: real micrographs are anisotropic, so a reference
must be compared as a FULL MATRIX in the Loewner sense. Comparing against a
fabricated isotropic diag([K11,K11]) is the wrong test and will report
spurious failures — an error made and caught during this project.

Referee objection R3.2: every microstructure in this work is synthetic.
This script answers it in the narrow, defensible way: it takes a published
segmented BINARY IMAGE, treats its geometry as the microstructure, and
certifies the effective properties of the resulting idealised two-phase
problem.

What this does and does not claim.
  DOES:     the method operates on genuine micrograph geometry -- rough
            interfaces, polydispersity, clustering, percolation -- rather
            than on procedurally generated shapes.
  DOES NOT: assert anything about a physical specimen. No measured
            properties are used and no model-adequacy claim is made. The
            certificate bounds the idealised periodic linear problem posed
            on this geometry.

Periodicity. A micrograph is not periodic. We therefore report both
conventions used in the literature:
  (a) direct   -- the crop is taken as a unit cell, introducing artificial
                  phase discontinuities at the cell boundary;
  (b) mirrored -- the crop is reflected to build a genuinely periodic cell
                  of twice the side length, which removes the artificial
                  interfaces at the cost of imposing mirror symmetry.
Both are certified; agreement between them indicates the boundary
convention is not driving the result.

Usage:
    python examples/run_micrograph.py --image path/to/segmented.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image

from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.diagnostics import quality_gate
from certihom.exact import reuss, voigt
from certihom.fft_reference import fft_effective
from certihom.fv_reference import fv_effective
from certihom.train import fit_bounds


def load_binary(path, panel, size, invert=False):
    """Extract a binary phase indicator from a (possibly multi-panel) image."""
    a = np.array(Image.open(path).convert("L"), dtype=np.float64)
    if panel is not None:
        w = a.shape[1] // 3
        a = a[:, panel * w:(panel + 1) * w]
    n = min(a.shape)
    a = a[:n, :n]                                   # square crop
    im = Image.fromarray(a.astype(np.uint8)).resize((size, size), Image.NEAREST)
    b = np.array(im, dtype=np.float64)
    mask = b > 0.5 * (b.min() + b.max())            # Otsu-free midpoint split
    return ~mask if invert else mask


def load_crop(path, panel, size, offset, invert=False, trim=0):
    """Take a size x size window at NATIVE resolution — no downsampling, so
    the geometry is preserved exactly. This is the recommended mode: a
    downsampled frame aliases fine features and the certificate then bounds
    a geometry that is not the one in the image."""
    a = np.array(Image.open(path).convert("L"), dtype=np.float64)
    if panel is not None:
        w = a.shape[1] // 3
        a = a[:, panel * w:(panel + 1) * w]
    if trim:
        a = a[trim:-trim, trim:-trim]
    n = min(a.shape)
    a = a[:n, :n]
    if offset + size > n:
        raise SystemExit(f"crop {size} at offset {offset} exceeds image "
                         f"({n} px). Reduce --size or --offset.")
    w = a[offset:offset + size, offset:offset + size]
    mask = w > 0.5 * (w.min() + w.max())
    return ~mask if invert else mask


def mirror(mask):
    """Reflect into a genuinely periodic cell of twice the side length."""
    top = np.concatenate([mask, mask[:, ::-1]], axis=1)
    return np.concatenate([top, top[::-1, :]], axis=0)


def certify(K, nq, steps, label, batch_points):
    _, _, (chi, psi) = fit_bounds(K, nq=nq, n_harmonics=6, width=64, depth=3,
                                  steps=steps, lr=2e-3, seed=0, verbose=False,
                                  batch_points=batch_points)
    b = certified_bounds_scalar(nodal_values(chi, nq), nodal_values(psi, nq), K)
    qg = quality_gate(chi, K, nq, batch_points=batch_points)
    v, r = voigt(K), reuss(K)
    vr = float(np.linalg.norm(v - r) / np.linalg.norm(0.5 * (v + r)))
    FV = fv_effective(K)
    FF = fft_effective(K)
    fv, ff = FV[0, 0], FF[0, 0]
    print(f"{label:>10} | {nq:>4} | {b.gap_rel():8.1%} "
          f"{vr / b.gap_rel():7.1f}x | {qg['headroom']:7.2f} "
          f"{'ACCEPT' if qg['accept'] else 'REJECT':>7} | "
          f"{b.K_low[0,0]:7.4f} {b.K_up[0,0]:7.4f} | "
          f"{fv:7.4f} {ff:7.4f} | "
          f"{str(b.brackets(FV, tol=1e-9) and b.brackets(FF, tol=1e-9)):>5}")
    return b, qg["accept"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--panel", type=int, default=None,
                   help="which third of a 3-panel figure holds the segmentation; "
                        "omit for a standalone image")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--contrast", type=float, default=10.0)
    p.add_argument("--reference-k11", type=float, default=None,
                   help="optional published K11 used only for a componentwise "
                        "bracketing report; it does not enter training or certification")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--invert", action="store_true")
    p.add_argument("--nq", type=int, default=None,
                   help="quadrature resolution; default 2*size. Must be a "
                        "multiple of the grid. If the gate rejects, double it "
                        "(or use --auto-nq).")
    p.add_argument("--auto-nq", action="store_true",
                   help="keep doubling nq until the quality gate accepts")
    p.add_argument("--crop", action="store_true",
                   help="take a NATIVE-RESOLUTION crop instead of "
                        "downsampling the whole frame. Strongly recommended: "
                        "downsampling aliases fine features and certifies the "
                        "wrong geometry.")
    p.add_argument("--offset", type=int, default=0,
                   help="top-left corner of the crop, in native pixels")
    p.add_argument("--trim", type=int, default=0,
                   help="pixels to remove from each image edge before cropping; "
                        "use only to remove figure borders")
    p.add_argument("--batch-points", type=int, default=65536,
                   help="maximum quadrature points per autograd batch. This changes "
                        "only the memory schedule, not the training or certification "
                        "quadrature. Set 0 to disable batching.")
    args = p.parse_args()

    if args.crop:
        mask = load_crop(args.image, args.panel, args.size, args.offset,
                         args.invert, args.trim)
    else:
        mask = load_binary(args.image, args.panel, args.size, args.invert)
    f = float(mask.mean())
    print(f"# Real segmented micrograph: {os.path.basename(args.image)}")
    print(f"# {args.size}x{args.size} crop, phase fraction {f:.3f}, "
          f"contrast {args.contrast:g}\n")

    direct = np.where(mask, args.contrast, 1.0)
    mirrored = np.where(mirror(mask), args.contrast, 1.0)

    print(f"{'cell':>10} | {'nq':>4} | {'gap':>8} {'tighten':>7} | "
          f"{'headroom':>7} {'gate':>7} | {'K_low':>7} {'K_up':>7} | "
          f"{'FV':>7} {'FFT':>7} | brack")
    print("-" * 104)
    nq = args.nq if args.nq else 2 * args.size
    if nq % args.size:
        raise SystemExit(f"--nq must be a multiple of the grid ({args.size})")

    batch_points = None if args.batch_points == 0 else args.batch_points
    b, ok = certify(direct, nq=nq, steps=args.steps, label="direct",
                    batch_points=batch_points)
    if args.reference_k11 is not None:
        inside = b.K_low[0, 0] <= args.reference_k11 <= b.K_up[0, 0]
        print(f"published K11 = {args.reference_k11:.7f} | "
              f"inside direct certificate: {inside}")
    while args.auto_nq and not ok and nq < 16 * args.size:
        nq *= 2
        print(f"{'':>10}   gate rejected — doubling nq to {nq}")
        _, ok = certify(direct, nq=nq, steps=args.steps, label="direct",
                        batch_points=batch_points)
    if not ok:
        print("\n!! The gate still rejects. The certificate is VALID but its")
        print("   WIDTH is not trustworthy. Either raise --nq further, or the")
        print("   geometry is under-resolved — check --crop and the")
        print("   requirements script.")

    if not args.crop:
        certify(mirrored, nq=2 * mirrored.shape[0], steps=args.steps,
                label="mirrored", batch_points=batch_points)

    print("\nThe certificate bounds the idealised periodic two-phase problem")
    print("posed on this geometry. No measured property is used and no claim")
    print("is made about the physical specimen.")


if __name__ == "__main__":
    main()
