"""Kaggle cell 02 — T0.2 FALSIFICATION DRILLS.

You cannot defend this work at review unless you can break it on purpose.
Each drill introduces one targeted bug, runs a relevant subset of the test
suite, and restores the source. Sources are backed up and restored in a
`finally` block, so an interrupted run cannot leave your tree modified —
but commit to git first anyway.

HOW TO USE THIS PROPERLY
1. Read each drill's `patch` and WRITE DOWN your prediction: which tests
   fail, and by what mechanism?
2. Run the drill.
3. Compare. A drill you predicted correctly proves you understand that
   mechanism. A drill you got wrong is the thing to go study.

Do not skip step 1. Reading the answer first teaches you nothing.

A DELIBERATE TRAP: at least one of these drills may NOT break validity,
for a real mathematical reason. Working out which one, and why, is the
most valuable half-hour in this file. (Hint for afterwards: in 2D there
is a duality relating the effective conductivity of k and of 1/k.)
"""

# %% [cell 1] --- setup -------------------------------------------------------
import os
import shutil
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

# %% [cell 2] --- the drills --------------------------------------------------
# Each: (name, file, old_string, new_string, tests_to_run, what_it_breaks)
DRILLS = [
    ("a_no_zero_init", "certihom/fields.py",
     "        nn.init.zeros_(self.head.weight)",
     "        nn.init.normal_(self.head.weight, std=0.1)",
     ["tests/test_certificates.py::test_zero_init_networks_give_voigt_reuss"],
     "the 'valid from step 0' property (untrained nets = Voigt-Reuss)"),

    ("b_break_periodicity", "certihom/fields.py",
     "        freqs = 2.0 * torch.pi * torch.arange(1, self.M + 1, dtype=torch.float64)",
     "        freqs = 2.0 * torch.pi * (torch.arange(1, self.M + 1, dtype=torch.float64) + 0.5)",
     ["tests/test_training_smoke.py"],
     "Y-periodicity of the trial fields (half-integer frequencies)"),

    ("c_break_curl", "certihom/energy.py",
     "        curl = torch.stack([g[:, 1], -g[:, 0]], dim=1)",
     "        curl = torch.stack([g[:, 0], g[:, 1]], dim=1)",
     ["tests/test_training_smoke.py"],
     "divergence-freeness of the dual flux (gradient instead of curl)"),

    ("d_break_mandel_shear", "certihom/elasticity.py",
     "            [gu[:, 0], gv[:, 1], (gu[:, 1] + gv[:, 0]) / SQ2], dim=1)",
     "            [gu[:, 0], gv[:, 1], (gu[:, 1] + gv[:, 0])], dim=1)",
     ["tests/test_elasticity.py::test_zero_init_networks_give_voigt_reuss_elasticity",
      "tests/test_elasticity.py::test_training_smoke_elasticity"],
     "the Mandel shear normalisation (Loewner order stops being matrix order)"),

    ("e_break_airy_sign", "certihom/elasticity.py",
     "            [Hk[:, 1, 1], Hk[:, 0, 0], -SQ2 * Hk[:, 0, 1]], dim=1)",
     "            [Hk[:, 1, 1], Hk[:, 0, 0], SQ2 * Hk[:, 0, 1]], dim=1)",
     ["tests/test_elasticity.py::test_training_smoke_elasticity"],
     "the Airy sign convention (stress no longer equilibrated)"),
]

for i, (n, f, _, _, t, why) in enumerate(DRILLS):
    print(f"{i}. {n:22s} breaks {why}")
print("\nWrite your predictions down NOW, before running cell 3.")

# %% [cell 3] --- run one drill (change INDEX and re-run) --------------------
INDEX = 0          # <-- change me: 0..4, or set RUN_ALL = True below
RUN_ALL = False

def run_drill(name, path, old, new, tests, why):
    full = os.path.join(ROOT, path)
    backup = full + ".drillbak"
    src = open(full).read()
    if old not in src:
        print(f"[{name}] SKIP: anchor string not found in {path} "
              f"(did the file change?)")
        return
    shutil.copy2(full, backup)
    try:
        open(full, "w").write(src.replace(old, new, 1))
        print(f"\n=== DRILL {name} ===\nbreaks: {why}")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                           capture_output=True, text=True, cwd=ROOT)
        tail = [l for l in r.stdout.splitlines() if l.strip()][-6:]
        print("\n".join(tail))
        broke = ("failed" in r.stdout) or ("error" in r.stdout.lower())
        print(f"--> tests DETECTED the bug: {broke}")
        if not broke:
            print("    Interesting. The suite did not catch this. Either the")
            print("    bug is mathematically harmless here, or you have found")
            print("    a genuine GAP IN TEST COVERAGE. Both are worth writing")
            print("    up. Work out which before moving on.")
    finally:
        shutil.move(backup, full)          # always restore
        print(f"[{name}] source restored")

if RUN_ALL:
    for d in DRILLS:
        run_drill(*d)
else:
    run_drill(*DRILLS[INDEX])

# %% [cell 4] --- confirm the tree is clean ----------------------------------
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/"],
                   capture_output=True, text=True, cwd=ROOT)
print(r.stdout.strip().splitlines()[-1])
print("Expect '24 passed'. If not, restore from git before doing anything else.")

# %% [cell 5] --- record what you learned -------------------------------------
# Fill this in yourself and paste it into REPRO.md. This table is the
# evidence that you understand the machinery, and it is the raw material
# for the paper's limitations section.
print("""
| drill | my prediction | actual | mechanism |
|---|---|---|---|
| a_no_zero_init |  |  |  |
| b_break_periodicity |  |  |  |
| c_break_curl |  |  |  |
| d_break_mandel_shear |  |  |  |
| e_break_airy_sign |  |  |  |
""")
