"""Audit the spatial-axis convention of the public VRNet 2D data.

The public HDF5 stores an image as a raster (row, column), whereas the
CertiHOM FE routines interpret array axes as mathematical (x1, x2).  A
vertical image axis can therefore carry the opposite orientation from x2.
This script keeps the geometry and the published label unchanged, evaluates
both tensor coordinate conventions explicitly, and records which one is
compatible with a certificate made on the raster geometry.

It is a diagnostic only: no model fitting and no reference label is used to
construct the trial fields.  The only label-dependent operation is the final
containment audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from certihom.amortized import AmortizedElasticity
from certihom.bfs import bfs_nodal_dofs, certified_lower_elasticity
from certihom.bounds import CertifiedBounds
from certihom.certify import certified_upper_elasticity, nodal_values
from certihom.vrnet_benchmark import (VRNET_RASTER_TO_CARTESIAN_MANDEL,
                                      load_vrnet_mechanical_samples,
                                      voigt_reuss_bounds)

# Mandel representation of either x1 -> -x1 or x2 -> -x2.  Normal strains
# are unchanged; the Mandel shear component changes sign.
REFLECTION_MANDEL = VRNET_RASTER_TO_CARTESIAN_MANDEL


def reflected_mandel_stiffness(C: np.ndarray) -> np.ndarray:
    """Express stiffness in the reflected two-dimensional coordinate frame."""
    return REFLECTION_MANDEL @ C @ REFLECTION_MANDEL


def certify(model: AmortizedElasticity, Cfield: np.ndarray, elements: int) -> CertifiedBounds:
    """The exact Q1/BFS path used for the reported external experiment."""
    coeffs, glob = model.condition(Cfield)

    def wrap(net):
        def fn(y):
            c = coeffs.unsqueeze(0).expand(y.shape[0], -1, -1, -1)
            g = glob.unsqueeze(0).expand(y.shape[0], -1)
            return net(y, model.cond(y, c, g))
        return fn

    C_up = certified_upper_elasticity(nodal_values(wrap(model.chi), elements), Cfield)
    C_low = certified_lower_elasticity(bfs_nodal_dofs(wrap(model.dual_net), elements), Cfield)
    return CertifiedBounds(C_low, C_up)


def min_margins(bounds: CertifiedBounds, reference: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.eigvalsh(reference - bounds.K_low).min()),
        float(np.linalg.eigvalsh(bounds.K_up - reference).min()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True,
                        help="protocol.json written by run_vrnet_elasticity_training.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--elements", type=int, default=100)
    args = parser.parse_args()
    if args.elements != 100:
        parser.error("this audit is for the released native 100x100 raster")

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = AmortizedElasticity(kmax=3, n_harmonics=2, width=40, depth=3)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    for mode_text, indices in protocol["selection"]["val_set"].items():
        mode = int(mode_text)
        samples = load_vrnet_mechanical_samples(
            args.h5, split="val_set", mode_grid=(mode, mode),
            indices=indices, resolution=args.elements)
        for sample in samples:
            field = sample.stiffness_field()
            bounds = certify(model, field, args.elements)
            # The reader returns the matching Cartesian-frame tensor.  Apply
            # the same involution once more solely to reconstruct the raw
            # raster-frame HDF5 target for the diagnostic comparison.
            reflected = sample.reference_mandel
            raw = reflected_mandel_stiffness(reflected)
            raw_lo, raw_up = min_margins(bounds, raw)
            ref_lo, ref_up = min_margins(bounds, reflected)
            vr_lo, vr_up = voigt_reuss_bounds(field)
            rows.append({
                "mode_grid": mode,
                "index": sample.index,
                "raw_lower_margin": raw_lo,
                "raw_upper_margin": raw_up,
                "raw_inside": raw_lo >= -1e-8 and raw_up >= -1e-8,
                "reflected_lower_margin": ref_lo,
                "reflected_upper_margin": ref_up,
                "reflected_inside": ref_lo >= -1e-8 and ref_up >= -1e-8,
                "raw_in_vr_lower": float(np.linalg.eigvalsh(raw - vr_lo).min()),
                "raw_in_vr_upper": float(np.linalg.eigvalsh(vr_up - raw).min()),
                "Cref11": raw[0, 0],
                "Cref13": raw[0, 2],
                "Cref23": raw[1, 2],
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    raw_coverage = sum(bool(row["raw_inside"]) for row in rows)
    reflected_coverage = sum(bool(row["reflected_inside"]) for row in rows)
    vr_coverage = sum(
        row["raw_in_vr_lower"] >= -1e-8 and row["raw_in_vr_upper"] >= -1e-8
        for row in rows)
    print(f"raw label coverage:       {raw_coverage}/{len(rows)}")
    print(f"reflected label coverage: {reflected_coverage}/{len(rows)}")
    print(f"analytic VR coverage:     {vr_coverage}/{len(rows)}")
    if reflected_coverage != len(rows):
        raise SystemExit("reflected-coordinate audit still has escaped references")


if __name__ == "__main__":
    main()
