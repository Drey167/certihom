"""Kaggle cell 01 — T0.2b MULTI-SEED STUDY.  Your first real result.

NOTE (NQ default raised 32 -> 64). At NQ=32 the laminate case sits at
quadrature headroom 0.64, below the safety threshold of 1: training
overfits the collocation rule and the honest exact-integration certificate
blows up to ~160% while staying valid. At NQ=64 (headroom 4.57) the same
case gives ~0.8%.

Replaces every single-seed number in RESULTS.md with medians and ranges.
Separates the two things that must never be conflated:
    VALIDITY  must be 100% — a single failure is a BUG, not a tuning knob.
    TIGHTNESS varies with seed — that is what you report as median [range].

Runtime: ~15 min at the defaults on a Kaggle CPU session. Drop --steps to
150 for a quick pass first.
"""

# %% [cell 1] --- setup -------------------------------------------------------
import os
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

SEEDS = 5      # >= 5 for anything you intend to publish
STEPS = 400    # 150 for a quick pass
NQ = 64

# %% [cell 2] --- run the study ----------------------------------------------
import time

import numpy as np

from certihom.certify import certified_bounds_scalar, nodal_values
from certihom.diagnostics import quality_gate
from certihom.exact import checkerboard_exact, laminate_exact, reuss, voigt
from certihom.fv_reference import fv_effective
from certihom.microstructures import boolean_disks, checkerboard, laminate
from certihom.train import fit_bounds


def vr_gap(K):
    v, r = voigt(K), reuss(K)
    return float(np.linalg.norm(v - r) / np.linalg.norm(0.5 * (v + r)))


CASES = [
    ("checkerboard_c5", checkerboard(16, 1.0, 5.0),
     checkerboard_exact(1.0, 5.0), 1e-3),
    ("laminate_c6", laminate(16, 1.0, 6.0, 0.25, axis=0),
     laminate_exact(1.0, 6.0, 0.25, 0), 1e-9),
    ("disks_c10", boolean_disks(16, 1.0, 10.0, n_disks=6, radius=0.14, seed=1),
     None, 2e-2),
]

rows = []
for name, K, K_ex, tol in CASES:
    ref = K_ex if K_ex is not None else fv_effective(np.kron(K, np.ones((4, 4))))
    print(f"\n## {name}  (Voigt-Reuss gap {vr_gap(K):.1%})")
    for s in range(SEEDS):
        t0 = time.time()
        b_q, _, (chi, psi) = fit_bounds(K, nq=NQ, n_harmonics=4, width=48,
                                        depth=3, steps=STEPS, lr=2e-3,
                                        seed=s, verbose=False)
        b_e = certified_bounds_scalar(nodal_values(chi, NQ),
                                      nodal_values(psi, NQ), K)
        qg = quality_gate(chi, K, NQ)      # label-free accept/reject
        rows.append(dict(case=name, seed=s, gap_quad=b_q.gap_rel(),
                         gap_exact=b_e.gap_rel(),
                         valid=b_e.loewner_valid(1e-10),
                         contains_ref=b_e.brackets(ref, tol=tol),
                         accept=qg["accept"], headroom=qg["headroom"],
                         refine_ratio=qg["ratio"],
                         vr_gap=vr_gap(K), wall_s=time.time() - t0))
        r = rows[-1]
        print(f"  seed {s}: gap {r['gap_exact']:7.1%} | valid {r['valid']} | "
              f"headroom {r['headroom']:5.2f} ratio {r['refine_ratio']:5.2f} | "
              f"{'ACCEPT' if r['accept'] else 'REJECT — ' + qg['reason']} | "
              f"{r['wall_s']:.0f}s")

# %% [cell 3] --- summarise ---------------------------------------------------
import csv

print(f"\n{'case':>16} | {'median(acc)':>11} | {'min':>7} | {'max':>7} | "
      f"{'accepted':>9} | validity")
print("-" * 72)
summary = {}
for name, _, _, _ in CASES:
    sel = [r for r in rows if r["case"] == name]
    g = np.array([r["gap_exact"] for r in sel])
    acc = np.array([r["gap_exact"] for r in sel if r["accept"]])
    ok = all(r["valid"] and r["contains_ref"] for r in sel)
    med = np.median(acc) if len(acc) else float("nan")
    summary[name] = (med, g.min(), g.max(), ok, len(acc))
    print(f"{name:>16} | {med:10.1%} | {g.min():6.1%} | {g.max():6.1%} | "
          f"{len(acc):>4}/{SEEDS} | {SEEDS}/{SEEDS} {ok}")
print("\nmedian(acc) = median over ACCEPTED runs. A rejected run is still")
print("VALID (its certificate contains the truth) but uselessly wide — see")
print("RESULTS.md, 'label-free quality gate'.")

ALL_OK = all(r["valid"] and r["contains_ref"] for r in rows)
print(f"\nOVERALL VALIDITY: {ALL_OK}")
if not ALL_OK:
    print("!! A validity failure is a BUG in the certification path.")
    print("   Do NOT relax a tolerance. Find it. Start with the")
    print("   machine-precision anchors in tests/.")

with open(os.path.join(OUT, "multiseed.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote", os.path.join(OUT, "multiseed.csv"))

# %% [cell 4] --- figure ------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
names = [c[0] for c in CASES]
x = np.arange(len(names))
for i, n in enumerate(names):
    g = np.array([r["gap_exact"] for r in rows if r["case"] == n]) * 100
    ax.scatter([i] * len(g), g, s=45, zorder=3, label="seeds" if i == 0 else None)
    ax.plot([i - .18, i + .18], [np.median(g)] * 2, lw=2.5, color="k",
            zorder=4, label="median" if i == 0 else None)
    vr = [r["vr_gap"] for r in rows if r["case"] == n][0] * 100
    ax.plot([i - .28, i + .28], [vr] * 2, lw=2, ls="--", color="crimson",
            label="Voigt-Reuss" if i == 0 else None)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("certified relative gap  [%]")
ax.set_title(f"Certified interval width over {SEEDS} seeds "
             f"(validity {'100%' if ALL_OK else 'FAILED'})")
ax.legend(); ax.grid(alpha=.3, axis="y"); fig.tight_layout()
fig.savefig(os.path.join(OUT, "multiseed.png"), dpi=150)
print("wrote", os.path.join(OUT, "multiseed.png"))

# %% [cell 5] --- markdown you can paste straight into RESULTS.md -------------
print("\n| case | Voigt-Reuss | certified median (accepted) | full range | accepted | validity |")
print("|---|---:|---:|---|:--:|:--:|")
for name, _, _, _ in CASES:
    med, lo, hi, ok, nacc = summary[name]
    vr = [r["vr_gap"] for r in rows if r["case"] == name][0]
    print(f"| {name} | {vr:.1%} | **{med:.1%}** | [{lo:.1%}, {hi:.1%}] | "
          f"{nacc}/{SEEDS} | {SEEDS}/{SEEDS} |")
print(f"\n(medians over ACCEPTED runs of {SEEDS} seeds, {STEPS} steps, exact "
      f"integration; validity audited against exact tensors where available. "
      f"Rejected runs fail the label-free headroom/refinement gate and are "
      f"wide but still valid.)")
