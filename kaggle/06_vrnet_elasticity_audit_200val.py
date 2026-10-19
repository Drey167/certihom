"""Kaggle driver for a scaled 200-cell external CertiHOM audit.

Attach a writable copy of this repository and the public DaRUS-5730 HDF5
file, select a GPU if one is available, then use ``Save & Run All``.  The
driver trains the label-free CertiHOM operator on 200 fixed training cells
(40 per Fourier mode grid) and audits it on 200 distinct held-out cells
(40 per grid).  It retains the released native 100x100 geometry and writes
both the certified coverage result and a coordinate-convention diagnostic.

This is a certificate-coverage audit.  It is deliberately separate from the
point-prediction baseline driver ``05_vrnet_elasticity_baselines.py``.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import subprocess
import sys


def stage_source() -> str:
    """Return a safe writable repository checkout without overwriting data."""
    destination = "/kaggle/working/certihom"
    if os.path.isdir(os.path.join(destination, "certihom")):
        return destination
    candidates: list[str] = []
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        candidates.extend(glob.glob(f"/kaggle/input/{depth}/certihom"))
    source = next((p for p in sorted(candidates)
                   if os.path.isdir(os.path.join(p, "certihom"))), None)
    if source is None:
        raise RuntimeError(
            "Attach the CertiHOM source Dataset containing a certihom/ directory."
        )
    if os.path.exists(destination):
        raise RuntimeError(
            f"Refusing to overwrite existing staging directory: {destination}. "
            "Use the checkout already there or start a fresh Kaggle session."
        )
    shutil.copytree(source, destination)
    return destination


def locate_h5(root: str) -> str:
    """Prefer the attached benchmark file over a staged source duplicate."""
    attached = sorted(set(glob.glob(
        "/kaggle/input/**/feature_engineering_mechanical_2D.h5", recursive=True)))
    if len(attached) == 1:
        return attached[0]
    if len(attached) > 1:
        raise RuntimeError(
            "Found multiple attached DaRUS-5730 HDF5 inputs named "
            "feature_engineering_mechanical_2D.h5; attach only one: "
            + repr(attached))

    bundled = os.path.join(root, "data", "vrnet_2d", "feature_engineering_mechanical_2D.h5")
    if os.path.isfile(bundled):
        return bundled
    raise RuntimeError(
        "Could not find DaRUS-5730 HDF5 input named "
        "feature_engineering_mechanical_2D.h5. Attach it as a Kaggle input.")


def coverage_summary(csv_path: str) -> dict[str, object]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No audit rows written: {csv_path}")

    def flag(value: str) -> bool:
        return value.strip().lower() == "true"

    raw = sum(flag(row["raw_inside"]) for row in rows)
    reflected = sum(flag(row["reflected_inside"]) for row in rows)
    return {
        "cells": len(rows),
        "raw_hdf5_frame_coverage": raw,
        "cartesian_frame_coverage": reflected,
        "minimum_cartesian_lower_margin": min(float(row["reflected_lower_margin"]) for row in rows),
        "minimum_cartesian_upper_margin": min(float(row["reflected_upper_margin"]) for row in rows),
        "coordinate_note": (
            "The Cartesian-frame target is the released HDF5 tensor after the "
            "documented one-axis Mandel shear reflection; it is not a fitted label."
        ),
    }


ROOT = stage_source()
H5 = locate_h5(ROOT)
OUT = "/kaggle/working/vrnet_elasticity_audit_200x200"

import h5py  # noqa: E402 -- fail before the long run if the input is wrong
import torch  # noqa: E402

print("torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("h5py:", h5py.__version__)

train_command = [
    sys.executable,
    os.path.join(ROOT, "examples", "run_vrnet_elasticity_training.py"),
    "--h5", H5,
    "--steps", "500",
    "--train-nq", "24",
    "--batch", "3",
    "--train-per-grid", "40",
    "--val-per-grid", "40",
    "--elements", "100",
    "--output-dir", OUT,
]
print("Running label-free CertiHOM audit:\n ", " ".join(train_command))
subprocess.run(train_command, check=True, cwd=ROOT)

orientation_csv = os.path.join(OUT, "orientation_audit.csv")
orientation_command = [
    sys.executable,
    os.path.join(ROOT, "examples", "audit_vrnet_orientation.py"),
    "--h5", H5,
    "--checkpoint", os.path.join(OUT, "trained_operator.pt"),
    "--protocol", os.path.join(OUT, "protocol.json"),
    "--output", orientation_csv,
    "--elements", "100",
]
print("Running coordinate-convention audit:\n ", " ".join(orientation_command))
subprocess.run(orientation_command, check=True, cwd=ROOT)

summary = coverage_summary(orientation_csv)
with open(os.path.join(OUT, "coverage_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print("\nScaled external audit summary:")
print(json.dumps(summary, indent=2))
print("\nSaved reproducibility bundle to", OUT)
