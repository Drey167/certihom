"""Kaggle cell 00 — SETUP AND VERIFY.  Run this first, every session.

Cells are delimited by `# %%`. You can either
  * run the whole file:      python kaggle/00_setup_and_verify.py
  * or paste each `# %%` block into its own Kaggle notebook cell.

Accelerator: choose CPU. You do not need a GPU for any of Blocks 0-3, and
a T4 is actively bad for this float64 codebase (see KAGGLE.md).
"""

# %% [cell 1] --- locate the package -----------------------------------------
import os
import subprocess
import sys

def _here():
    try:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    except NameError:
        return ""


def _writable(d):
    try:
        t = os.path.join(d, ".wtest_certihom")
        open(t, "w").close(); os.remove(t); return True
    except OSError:
        return False


def bootstrap():
    """Return (ROOT, OUT).

    ROOT = importable, WRITABLE copy of the package. Kaggle Datasets live
    under /kaggle/input which is READ-ONLY, so if the source is found there
    it is staged into /kaggle/working (drill 02 edits source files, and
    nothing can be written next to a read-only dataset).
    OUT  = writable directory for CSVs, figures, logs, REPRO.md.
    """
    import glob, shutil, tempfile
    cands = ["/kaggle/working/certihom", os.getcwd(), _here()]
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        cands += sorted(glob.glob(f"/kaggle/input/{depth}/certihom"))
    src = next((p for p in cands
                if p and os.path.isdir(os.path.join(p, "certihom"))), None)
    assert src, ("certihom not found. Attach certihom_full.zip as a Kaggle "
                 "Dataset, or run from the repo root.")
    src = os.path.abspath(src)
    if not _writable(src):
        dest = ("/kaggle/working/certihom" if os.path.isdir("/kaggle/working")
                else os.path.join(tempfile.gettempdir(), "certihom"))
        if os.path.abspath(dest) != src:
            if os.path.isdir(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            print(f"source was read-only; staged copy -> {dest}")
        src = dest
    os.chdir(src)
    if src not in sys.path:
        sys.path.insert(0, src)
    out = "/kaggle/working" if (os.path.isdir("/kaggle/working")
                                and _writable("/kaggle/working")) else src
    print(f"ROOT (code) : {src}\nOUT  (files): {out}")
    return src, out


ROOT, OUT = bootstrap()

# %% [cell 2] --- record the environment (paste this into REPRO.md) ----------
import numpy
import scipy
import torch

ENV = (f"python {sys.version.split()[0]} | torch {torch.__version__} | "
       f"numpy {numpy.__version__} | scipy {scipy.__version__} | "
       f"cuda {torch.cuda.is_available()}")
print(ENV)
print("cpu count:", os.cpu_count())
print("\nNOTE: certificates are float64 throughout. Do NOT switch to "
      "float32 without reading KAGGLE.md — training may, certification "
      "may not.")

# %% [cell 3] --- install pytest if missing ----------------------------------
try:
    import pytest  # noqa: F401
    print("pytest present")
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"],
                   check=True)
    print("pytest installed")

# %% [cell 4] --- T0.1: the full verification ladder --------------------------
# Expect: 24 passed. Anything else means STOP and investigate before
# running any science. The machine-precision anchors (laminate, BFS) are
# the ones that must never go red.
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"],
                   capture_output=True, text=True)
print(r.stdout[-1500:])
print(r.stderr[-800:])
PASSED = "24 passed" in r.stdout
print("\n" + ("VERIFIED — proceed to 01_multiseed.py"
              if PASSED else "FAILED — do not proceed until this is green"))

# %% [cell 5] --- write REPRO.md ---------------------------------------------
with open(os.path.join(OUT, "REPRO.md"), "a") as f:
    f.write(f"\n## session\n- env: {ENV}\n- cpu count: {os.cpu_count()}\n"
            f"- test suite: {'24 passed' if PASSED else 'FAILED'}\n")
print("appended to REPRO.md")
