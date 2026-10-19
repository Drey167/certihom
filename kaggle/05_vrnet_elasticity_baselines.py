"""Kaggle driver: matched 200/100 external-elasticity point predictors.

Attach both (1) a zip/Dataset containing this repository and (2) the public
DaRUS-5730 mechanical HDF5 file.  Select a GPU accelerator and use
``Save & Run All``.  This driver trains the vanilla, Cholesky/SPD, and
Voigt--Reuss constrained predictors on exactly the fixed protocol used by
``examples/run_vrnet_elasticity_training.py``.

It intentionally does *not* certify or report CertiHOM coverage.  The three
models here make point predictions; their structural rates are saved for a
fairly matched comparison after the external reference audit is resolved.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys


def _writable(path: str) -> bool:
    try:
        probe = os.path.join(path, ".certihom_write_probe")
        with open(probe, "w", encoding="utf-8"):
            pass
        os.remove(probe)
        return True
    except OSError:
        return False


def stage_source() -> str:
    """Return a writable repository root, copying an attached Dataset if needed."""
    destination = "/kaggle/working/certihom"
    if os.path.isdir(os.path.join(destination, "certihom")):
        return destination
    candidates = []
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        candidates.extend(glob.glob(f"/kaggle/input/{depth}/certihom"))
    source = next((p for p in sorted(candidates)
                   if os.path.isdir(os.path.join(p, "certihom"))), None)
    if source is None:
        raise RuntimeError(
            "Could not find the source archive. Attach the CertiHOM Kaggle Dataset "
            "containing a certihom/ directory."
        )
    if os.path.exists(destination):
        raise RuntimeError(
            f"Refusing to overwrite existing staging directory: {destination}. "
            "Use the existing CertiHOM checkout there, or start a fresh Kaggle session."
        )
    shutil.copytree(source, destination)
    if not _writable(destination):  # Defensive: should always be true in /working.
        raise RuntimeError(f"Staged repository is not writable: {destination}")
    return destination


def locate_h5(root: str) -> str:
    """Prefer the attached benchmark file over a staged source duplicate."""
    attached = sorted(set(glob.glob(
        "/kaggle/input/**/feature_engineering_mechanical_2D.h5", recursive=True
    )))
    if len(attached) == 1:
        return attached[0]
    if len(attached) > 1:
        raise RuntimeError(
            "Found multiple attached DaRUS-5730 HDF5 inputs named "
            "feature_engineering_mechanical_2D.h5; attach only one: "
            + repr(attached)
        )

    bundled = os.path.join(root, "data", "vrnet_2d", "feature_engineering_mechanical_2D.h5")
    if os.path.isfile(bundled):
        return bundled
    raise RuntimeError(
        "Could not find DaRUS-5730 HDF5 input named "
        "feature_engineering_mechanical_2D.h5. Attach it as a Kaggle input."
    )


ROOT = stage_source()
H5 = locate_h5(ROOT)
OUT = "/kaggle/working/vrnet_baselines_200x100"

import h5py  # noqa: E402  -- explicit preflight before a long run
import torch  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("Choose a Kaggle GPU accelerator before running this driver.")
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "h5py:", h5py.__version__)

command = [
    sys.executable,
    os.path.join(ROOT, "examples", "run_vrnet_elasticity_baselines.py"),
    "--h5", H5,
    "--device", "cuda",
    "--train-per-grid", "40",
    "--val-per-grid", "20",
    "--epochs", "1200",
    "--batch", "64",
    "--seed", "20261010",
    "--output-dir", OUT,
]
print("Running:\n ", " ".join(command))
subprocess.run(command, check=True, cwd=ROOT)
print("\nSaved reproducibility bundle to", OUT)
print("Required files: protocol.json, baseline_summary.csv, validation_predictions.csv, "
      "history.json, and one checkpoint per baseline.")
