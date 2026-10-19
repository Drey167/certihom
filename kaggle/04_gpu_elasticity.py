"""Kaggle cell 04 — GPU amortized ELASTICITY (the load-bearing experiment).

This is the paper's central claim: certified two-sided bounds on effective
STIFFNESS TENSORS for unseen microstructures at inference cost. On one CPU
core the result had not converged at 450 steps (gap 37.5% -> 28.1% and
still improving), so GPU training is load-bearing rather than optional.

PRECISION SPLIT — read this before changing anything.
Certificates are float64. A T4 runs FP64 at 1:32 of FP32, so we
    * TRAIN in float32 on the GPU (fast), and
    * CERTIFY in float64 on the CPU (exact integration, Q1 + BFS C1).
This is safe because the bound holds for ANY admissible field: reduced
training precision can only WIDEN the interval, never invalidate it.
Verified in-sandbox — float32-trained fields certified in float64 gave
gap 35.1% vs float64's 36.6%, with validity confirmed against exact
laminate tensors in both cases.

Runtime: ~30-60 min on a T4 at the settings below. Use "Save & Run All"
(commit) so the job survives the browser tab closing.
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
    import glob, shutil, tempfile
    cands = ["/kaggle/working/certihom", os.getcwd(), _here()]
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        cands += sorted(glob.glob(f"/kaggle/input/{depth}/certihom"))
    src = next((p for p in cands
                if p and os.path.isdir(os.path.join(p, "certihom"))), None)
    assert src, "certihom not found — attach certihom_full.zip as a Dataset"
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

# %% [cell 2] --- configuration ----------------------------------------------
import torch

N_TRAIN = 200      # training microstructures (CPU run used 40)
N_HELD = 12        # held-out instances to certify
STEPS = 4000       # CPU run used 450 and had NOT converged
BATCH = 16         # push this up on GPU; small tensors leave a T4 idle
NQ = 32            # collocation resolution (multiple of the pixel grid)
NE = 32            # FE mesh for exact certification
PIX = 8            # microstructure pixel grid
LR = 2e-3
SEED = 0

# %% [cell 3] --- build the microstructure family ----------------------------
import numpy as np

from certihom.elasticity import iso_stiffness_mandel, mandel_field_from_mask
from certihom.microstructures import boolean_disks

C_SOFT = iso_stiffness_mandel(1.0, 0.3)
C_STIFF = iso_stiffness_mandel(10.0, 0.2)
rng = np.random.default_rng(SEED)


def sample(seed):
    m = boolean_disks(PIX, 1.0, 2.0, n_disks=int(rng.integers(2, 5)),
                      radius=float(rng.uniform(0.14, 0.22)), seed=seed) > 1.5
    return mandel_field_from_mask(m, C_STIFF, C_SOFT)


fam = [sample(s) for s in range(N_TRAIN)]
held = [sample(5000 + s) for s in range(N_HELD)]
print(f"{len(fam)} training / {len(held)} held-out microstructures")

# %% [cell 4] --- train in float32 on the GPU --------------------------------
import time

from certihom.amortized import AmortizedElasticity, fit_amortized
from certihom.gpu import pick_device, report_precision, to_certification, to_training

dev = pick_device()
model = AmortizedElasticity(kmax=3, n_harmonics=2, width=64, depth=3,
                            dtype=torch.float32)
model = to_training(model, dev, torch.float32)
print(report_precision(model))

t0 = time.time()
model, hist = fit_amortized(model, fam, nq=NQ, steps=STEPS, batch=BATCH,
                            lr=LR, seed=SEED, log_every=max(1, STEPS // 10),
                            monitor=2, verbose=True)
print(f"\ntraining wall clock: {time.time() - t0:.0f} s")
torch.save({"model": model.state_dict(), "history": hist,
            "config": dict(N_TRAIN=N_TRAIN, STEPS=STEPS, BATCH=BATCH, NQ=NQ)},
           os.path.join(OUT, "amortized_elasticity.pt"))
print("checkpoint written")

# %% [cell 5] --- certify in float64 on the CPU (EXACT integration) ----------
from certihom.bfs import bfs_nodal_dofs, certified_lower_elasticity
from certihom.bounds import CertifiedBounds
from certihom.certify import certified_upper_elasticity, nodal_values
from certihom.fem_reference import fem_effective

cert = to_certification(model)           # float64 on CPU — never certify in fp32
print(report_precision(cert))


def certify(Cf, ne=NE):
    co, gl = cert.condition(Cf)

    def wrap(net):
        def fn(y):
            c = co.unsqueeze(0).expand(y.shape[0], -1, -1, -1)
            g = gl.unsqueeze(0).expand(y.shape[0], -1)
            return net(y, cert.cond(y, c, g))
        return fn

    return CertifiedBounds(
        certified_lower_elasticity(bfs_nodal_dofs(wrap(cert.dual_net), ne), Cf),
        certified_upper_elasticity(nodal_values(wrap(cert.chi), ne), Cf))


def voigt_reuss(Cf):
    V = Cf.reshape(-1, 3, 3).mean(axis=0)
    R = np.linalg.inv(np.linalg.inv(Cf.reshape(-1, 3, 3)).mean(axis=0))
    return V, R


def rel_width(lo, up):
    return float(np.linalg.norm(up - lo) / np.linalg.norm(0.5 * (up + lo)))


rows = []
print(f"\n{'instance':>10} | {'gap':>7} | {'Loewner':>7} {'FEM audit':>9} | "
      f"{'V-R width':>9} {'tightening':>10} | {'t':>6}")
print("-" * 74)
for i, Cf in enumerate(held):
    t0 = time.time()
    b = certify(Cf)
    dt = time.time() - t0
    audit = bool(np.linalg.eigvalsh(fem_effective(Cf) - b.K_low).min() >= -1e-9)
    V, R = voigt_reuss(Cf)
    f = rel_width(R, V) / b.gap_rel()
    rows.append(dict(gap=b.gap_rel(), valid=b.loewner_valid(1e-9),
                     audit=audit, factor=f, t=dt))
    print(f"{'#' + str(i + 1):>10} | {b.gap_rel():6.1%} | "
          f"{str(rows[-1]['valid']):>7} {str(audit):>9} | "
          f"{rel_width(R, V):8.1%} {f:9.1f}x | {dt:5.2f}s")

g = np.array([r["gap"] for r in rows])
fa = np.array([r["factor"] for r in rows])
ok = all(r["valid"] and r["audit"] for r in rows)
print(f"\ncertified gap : median {np.median(g):.1%}  "
      f"range [{g.min():.1%}, {g.max():.1%}]")
print(f"tightening    : median {np.median(fa):.1f}x  "
      f"range [{fa.min():.1f}x, {fa.max():.1f}x]")
print(f"validity      : {ok}  ({len(rows)}/{len(rows)})")
print(f"inference     : {np.mean([r['t'] for r in rows]) * 1000:.0f} ms "
      f"per certificate")

# %% [cell 6] --- what to check ----------------------------------------------
print("""
CHECKLIST
 1. Validity must be 12/12. A single False is a BUG in the certification
    path, not a tuning issue — do not relax the tolerance, find it.
 2. Compare the median gap against the CPU baseline of 28.1% (450 steps,
    40 microstructures). If it has not improved, the bottleneck is
    capacity/family size, not steps.
 3. Compare tightening against the CPU baseline of 2.6x and against the
    conductivity result of 3.4x. This is the number that answers the
    Voigt-Reuss Net.
 4. Check the training history for a plateau. The CPU run never plateaued;
    if this one does not either, report the trend honestly and say the
    numbers are a lower bound.
 5. Re-run with >= 5 seeds before putting any of these numbers in a paper.
""")
