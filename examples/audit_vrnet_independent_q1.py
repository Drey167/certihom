"""Independent periodic-Q1 audit for selected VRNet mechanical cells.

This is deliberately separate from CertiHOM's trial-field certificate path.
It reads the native 100 x 100 public raster, solves the periodic Q1 energy
minimisation by matrix-free PCG, and compares the result to the HDF5 target
after the reader's documented raster-to-Cartesian tensor conversion.

The defaults include the first formerly escaped grid-3 cell, another grid-3
escape, and the grid-5 escape from the fixed 100-cell protocol.  No target
is used by the solve itself; it is read only for the final numerical audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from certihom.fem_independent import periodic_q1_effective
from certihom.vrnet_benchmark import load_vrnet_mechanical_samples


def parse_cell(text: str) -> tuple[int, int]:
    """Parse ``mode_grid:index``."""
    try:
        mode, index = text.split(":", 1)
        return int(mode), int(index)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("cell must be MODE_GRID:INDEX") from exc


def tensor_entries(C: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_11": float(C[0, 0]),
        f"{prefix}_12": float(C[0, 1]),
        f"{prefix}_13": float(C[0, 2]),
        f"{prefix}_22": float(C[1, 1]),
        f"{prefix}_23": float(C[1, 2]),
        f"{prefix}_33": float(C[2, 2]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--cell", type=parse_cell, action="append",
                        default=None,
                        help="MODE_GRID:INDEX; may be repeated")
    parser.add_argument("--output", required=True,
                        help="JSON output; a same-stem CSV is also written")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--maxiter", type=int, default=10000)
    args = parser.parse_args()

    cells = args.cell or [(3, 1491), (3, 3926), (5, 4601)]
    rows: list[dict[str, object]] = []
    for mode, index in cells:
        sample = load_vrnet_mechanical_samples(
            args.h5, split="val_set", mode_grid=(mode, mode),
            indices=[index], resolution=100)[0]
        C_q1, report = periodic_q1_effective(sample.stiffness_field(),
                                               rtol=args.rtol, maxiter=args.maxiter)
        C_ref = sample.reference_mandel
        delta = C_q1 - C_ref
        row: dict[str, object] = {
            "mode_grid": mode,
            "index": index,
            "solid_fraction": sample.declared_solid_fraction,
            "max_abs_tensor_error": float(np.abs(delta).max()),
            "fro_relative_tensor_error": float(np.linalg.norm(delta) / np.linalg.norm(C_ref)),
            "min_eig_q1_minus_reference": float(np.linalg.eigvalsh(delta).min()),
            "pcg_iterations": list(report.iterations),
            "pcg_relative_residuals": list(report.relative_residuals),
        }
        row.update(tensor_entries(C_ref, "reference"))
        row.update(tensor_entries(C_q1, "q1"))
        rows.append(row)
        print(
            f"grid {mode:2d}, cell {index:5d}: max |Q1-ref|={row['max_abs_tensor_error']:.3e}; "
            f"rel={row['fro_relative_tensor_error']:.3e}; "
            f"PCG residual max={max(report.relative_residuals):.3e}")

    payload = {
        "description": "Independent matrix-free periodic-Q1 reference audit; labels only compared after solve.",
        "native_resolution": 100,
        "tensor_convention": "Mandel, Cartesian frame matched to returned image raster",
        "cells": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [k for k in rows[0] if k not in {"pcg_iterations", "pcg_relative_residuals"}]
        fieldnames += ["pcg_iterations", "pcg_relative_residuals"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["pcg_iterations"] = ";".join(map(str, row["pcg_iterations"]))
            flat["pcg_relative_residuals"] = ";".join(
                f"{value:.17g}" for value in row["pcg_relative_residuals"])
            writer.writerow(flat)


if __name__ == "__main__":
    main()
