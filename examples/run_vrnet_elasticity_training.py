"""Reproducible external elasticity experiment on DaRUS-5730.

Protocol (fixed in this source before execution):
* Train CertiHOM without reference labels on 40 seeded ``train_set`` cells
  from each of the five Fourier mode grids.
* Evaluate 20 disjoint, seeded ``val_set`` cells per mode grid.
* Reconstruct the released 100x100 binary preview and use phase-aligned
  conforming projection for the reported certificate.

The phase law and coordinate-aware Voigt-to-Mandel conversion are supplied by
``certihom.vrnet_benchmark``. This program never uses the reference stiffness
while optimising the admissible fields; it is read only after certification for
the external-bracketing audit. The released native 100x100 raster is used
unchanged, and the reader converts the released stiffness target into the
same Cartesian tensor frame as that raster. See
``examples/audit_vrnet_orientation.py`` for the explicit convention audit.

Example:
  python examples/run_vrnet_elasticity_training.py ^
    --h5 data/vrnet_2d/feature_engineering_mechanical_2D.h5 ^
    --steps 300
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from certihom.amortized import AmortizedElasticity, fit_amortized
from certihom.bfs import bfs_nodal_dofs, certified_lower_elasticity
from certihom.bounds import CertifiedBounds
from certihom.certify import certified_upper_elasticity, nodal_values
from certihom.vrnet_benchmark import load_vrnet_mechanical_samples, voigt_reuss_bounds

SEED = 20261010
MODE_GRIDS = (3, 5, 7, 9, 11)
TRAIN_PER_GRID = 40
VAL_PER_GRID = 20


def selected_indices(h5_path: str, seed: int, train_per_grid: int,
                     val_per_grid: int) -> dict[str, dict[str, list[int]]]:
    """Draw and return the fixed stratified indices without reading labels."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("This benchmark experiment needs h5py") from exc
    rng = np.random.default_rng(seed)
    selection: dict[str, dict[str, list[int]]] = {"train_set": {}, "val_set": {}}
    with h5py.File(h5_path, "r") as f:
        for split, n_per_grid in (("train_set", train_per_grid),
                                  ("val_set", val_per_grid)):
            for mode in MODE_GRIDS:
                count = f[f"{split}/mode_grid_{mode}x{mode}/image_data"].shape[0]
                selection[split][str(mode)] = sorted(
                    int(i) for i in rng.choice(count, size=n_per_grid, replace=False))
    return selection


def certify_with_conforming_projection(model: AmortizedElasticity,
                                       Cfield: np.ndarray, ne: int) -> CertifiedBounds:
    """Certification path, distinct from the inexact training quadrature.

    Neural fields are sampled and projected into periodic Q1 displacement and
    BFS Airy-potential spaces.  The ensuing energies are integrated exactly
    for the phase-aligned piecewise-constant pixel stiffness, so this remains
    a certificate after neural-field training.
    """
    coeffs, glob = model.condition(Cfield)

    def wrap(net):
        def fn(y):
            repeated_coeffs = coeffs.unsqueeze(0).expand(y.shape[0], -1, -1, -1)
            repeated_globals = glob.unsqueeze(0).expand(y.shape[0], -1)
            return net(y, model.cond(y, repeated_coeffs, repeated_globals))
        return fn

    C_up = certified_upper_elasticity(nodal_values(wrap(model.chi), ne), Cfield)
    C_low = certified_lower_elasticity(bfs_nodal_dofs(wrap(model.dual_net), ne), Cfield)
    return CertifiedBounds(C_low, C_up)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--train-nq", type=int, default=24,
                        help="training quadrature; does not affect reported bounds")
    parser.add_argument("--elements", type=int, default=100,
                        help="phase-aligned Q1/BFS element count; must equal released image resolution")
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--train-per-grid", type=int, default=TRAIN_PER_GRID)
    parser.add_argument("--val-per-grid", type=int, default=VAL_PER_GRID)
    parser.add_argument("--output-dir", default="outputs/vrnet_elasticity")
    args = parser.parse_args()
    if args.elements != 100:
        parser.error("this diagnostic reconstructs the released 100x100 preview")
    if args.steps < 1:
        parser.error("--steps must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = selected_indices(args.h5, SEED, args.train_per_grid, args.val_per_grid)
    (output_dir / "protocol.json").write_text(json.dumps({
        "seed": SEED, "mode_grids": MODE_GRIDS,
        "train_per_grid": args.train_per_grid, "val_per_grid": args.val_per_grid,
        "steps": args.steps, "train_nq": args.train_nq, "elements": args.elements,
        "selection": selection,
    }, indent=2), encoding="utf-8")

    train_samples, val_samples = [], []
    for mode in MODE_GRIDS:
        train_samples.extend(load_vrnet_mechanical_samples(
            args.h5, split="train_set", mode_grid=(mode, mode),
            indices=selection["train_set"][str(mode)], resolution=100))
        val_samples.extend(load_vrnet_mechanical_samples(
            args.h5, split="val_set", mode_grid=(mode, mode),
            indices=selection["val_set"][str(mode)], resolution=100))

    # Deliberately pass only coefficient fields: no reference C enters fitting.
    train_fields = [s.stiffness_field() for s in train_samples]
    model = AmortizedElasticity(kmax=3, n_harmonics=2, width=40, depth=3)
    print("# CertiHOM external 2D elasticity experiment")
    print(f"# seed={SEED}; train={len(train_fields)} (label-free); "
          f"val={len(val_samples)}; steps={args.steps}")
    started = time.time()
    model, history = fit_amortized(model, train_fields, nq=args.train_nq,
                                   steps=args.steps, batch=args.batch,
                                   lr=2e-3, seed=SEED, log_every=50,
                                   monitor=5, verbose=True)
    train_seconds = time.time() - started
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "seed": SEED,
                "steps": args.steps, "train_nq": args.train_nq},
               output_dir / "trained_operator.pt")

    print("\n# Held-out reference audit")
    print(f"{'grid':>4} {'cell':>6} | {'C_low11':>10} {'Cref11':>10} {'C_up11':>10} | "
          f"{'min(ref-low)':>12} {'min(up-ref)':>12} | {'gap':>7} {'inside':>6}")
    print("-" * 108)
    rows = []
    for sample in val_samples:
        Cfield = sample.stiffness_field()
        t0 = time.time()
        bounds = certify_with_conforming_projection(model, Cfield, ne=args.elements)
        seconds = time.time() - t0
        lower_margin = float(np.linalg.eigvalsh(sample.reference_mandel - bounds.K_low).min())
        upper_margin = float(np.linalg.eigvalsh(bounds.K_up - sample.reference_mandel).min())
        inside = lower_margin >= -1e-8 and upper_margin >= -1e-8
        C_reuss, C_voigt = voigt_reuss_bounds(Cfield)
        vr_gap = np.linalg.norm(C_voigt - C_reuss) / np.linalg.norm(0.5 * (C_voigt + C_reuss))
        row = {
            "mode_grid": sample.mode_grid[0], "index": sample.index,
            "solid_fraction": sample.declared_solid_fraction,
            "C_low_11": bounds.K_low[0, 0], "C_ref_11": sample.reference_mandel[0, 0],
            "C_up_11": bounds.K_up[0, 0], "lower_margin_min_eig": lower_margin,
            "upper_margin_min_eig": upper_margin, "inside_reference": inside,
            "certified_relative_gap": bounds.gap_rel(), "voigt_reuss_relative_gap": vr_gap,
            "tightening_factor": vr_gap / bounds.gap_rel(), "certificate_seconds": seconds,
        }
        rows.append(row)
        print(f"{row['mode_grid']:4d} {row['index']:6d} | {row['C_low_11']:10.4f} "
              f"{row['C_ref_11']:10.4f} {row['C_up_11']:10.4f} | "
              f"{lower_margin:12.3e} {upper_margin:12.3e} | "
              f"{row['certified_relative_gap']:6.1%} {str(inside):>6}")
    with (output_dir / "heldout_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    coverage = sum(r["inside_reference"] for r in rows)
    print(f"\ncoverage: {coverage}/{len(rows)}")
    print(f"median certified gap: {np.median([r['certified_relative_gap'] for r in rows]):.1%}")
    print(f"median Voigt--Reuss tightening: {np.median([r['tightening_factor'] for r in rows]):.2f}x")
    print(f"training wall time: {train_seconds:.1f} s")
    if coverage != len(rows):
        raise SystemExit("Reference escaped a bound: inspect discretisation and solver convention before reporting.")


if __name__ == "__main__":
    main()
