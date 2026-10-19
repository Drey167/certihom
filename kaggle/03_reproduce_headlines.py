"""Kaggle cell 03 — REPRODUCE THE HEADLINE RESULTS.

Runs every example script and saves the console output, so you can diff
your numbers against RESULTS.md. Anything that disagrees by more than a
percent or so is worth understanding before you cite it.

Total runtime ~12 min on a Kaggle CPU session at these settings.
Run this AFTER 00 (verify) and 01 (multi-seed).
"""

# %% [cell 1] --- setup -------------------------------------------------------
import os
import subprocess
import sys
import time

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

RUNS = [
    # (script, args, what it demonstrates, RESULTS.md anchor)
    ("run_checkerboard.py", ["--contrast", "5", "--steps", "900"],
     "Phase 0: scalar bounds vs exact sqrt(5)", "gap ~3.0%"),
    ("run_elasticity_disks.py", ["--steps", "400"],
     "Phase 2: full elasticity tensor + certified E_x, E_y", "gap ~10.7%"),
    ("run_amortized.py", ["--steps", "400"],
     "Phase 3: certificates at inference cost + OOD validity", "~19 ms each"),
    ("run_contrast_certification.py", ["--steps", "200"],
     "contrast sweep + price of exact integration", "1.1/10.3/27.8/58.6%"),
    ("run_certified_design.py", ["--steps", "240"],
     "certified inverse design (guaranteed floor)", "+57% K_low"),
]

# %% [cell 2] --- run them ----------------------------------------------------
LOGS = os.path.join(OUT, "logs"); os.makedirs(LOGS, exist_ok=True)
for script, args, what, anchor in RUNS:
    print(f"\n{'='*70}\n{script}  —  {what}\n  RESULTS.md says: {anchor}\n{'='*70}")
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join("examples", script), *args],
                       capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + ("\n[stderr]\n" + r.stderr if r.returncode else "")
    print(out[-2200:])
    with open(os.path.join(LOGS, script.replace(".py", ".log")), "w") as f:
        f.write(out)
    print(f"[{time.time()-t0:.0f}s, exit {r.returncode}] "
          f"-> {LOGS}/{script.replace('.py', '.log')}")

# %% [cell 3] --- what to check ----------------------------------------------
print("""
CHECKLIST — go through this before citing any of these numbers:

 1. Did every script exit 0?
 2. Does every 'Loewner valid' / 'contains exact' read True? Validity is
    binary; a single False is a bug, not variance.
 3. Are your gaps within ~1 point of RESULTS.md? Small drift from BLAS and
    thread counts is expected; large drift is not.
 4. In run_contrast_certification, does the contrast-200 row still show
    'brackets FV: False'? That is the FEATURE (the reference solver is
    under-converged, and the rigorous bound detects it), not a failure.
 5. In run_certified_design, note the inclusion fraction drift
    (0.256 -> 0.272). That is a known limitation of the soft budget
    penalty. Fixing it with a projection onto the constraint set is a
    good, small, self-contained task.

Then: rewrite the RESULTS.md tables using YOUR numbers, over 5 seeds
(from 01_multiseed.py), and never cite a figure you did not regenerate.
""")
