# Running certihom on Kaggle (and where NOT to)

## The float64 rule (read this first)

Every certificate in this project is float64. The machine-precision anchors
assert agreement to ~1e-12, which float32 cannot represent, and bound
validity in float32 would be decided by rounding noise. **Do not convert to
float32 to make it fast.**

That single fact decides your hardware:

| Kaggle accelerator | FP64 rate | verdict |
|---|---|---|
| **GPU P100** | 1:2 (~4.7 TFLOPS) | **use this** |
| GPU T4 ×2 | 1:32 (~0.25 TFLOPS) | avoid — slower than P100 by ~19× here |
| CPU (4 vCPU, ~30 GB RAM) | — | fine, and preferable, for Blocks 0–2 |

## If you are stuck on T4: the precision split (verified)

T4 FP64 is hopeless, but you do not need FP64 to TRAIN — only to CERTIFY.
The bound holds for any admissible field, so a float32-trained network
defines a legitimate field whose energy you then integrate exactly in
float64. Imprecise training widens the interval; it cannot invalidate it.

**Verified in-sandbox:** float32 training -> float64 exact-integration
certificate was Loewner-valid and bracketed the EXACT laminate tensor at
1e-9 tolerance in 9/9 runs across 3 seeds and 2 Adam epsilon settings.
Tightness was somewhat worse than float64 training, but seed variance
(50.8% / 1.1% / 148.8% on the fp64 baseline across seeds) dominated the
precision effect — so treat "fp32 costs tightness" as plausible but
unquantified until T0.2b is done.

Practical shape:
* networks + training loop in float32 on the T4 (8.1 TFLOPS vs 0.25 FP64);
* nodal extraction, Q1/BFS projection and Gauss integration in float64
  on CPU (`certify.py` / `bfs.py` are already float64 numpy — leave them);
* keep the machine-precision anchors in float64 on CPU, unchanged.

You will need to thread a `dtype` argument through `FieldPair`,
`midpoint_grid`, `sample_k` and `sample_mandel`, which currently hardcode
float64. Watch for silent upcasting: `torch.eye(2, dtype=float64) + fp32`
promotes to float64 and quietly destroys the speedup.

## What to run where

| Work | Where | Why |
|---|---|---|
| Block 0 (repro, falsification drills, derivations) | **your own machine** | you need a debugger, fast edit-run loops, and git. Tests take ~60 s on one core |
| Block 1 (conformity-trap figure) | local CPU or Kaggle CPU | small runs, many edits |
| Block 2 (HS polarization) | local CPU; Kaggle CPU for the sweep | derivation-heavy, compute-light |
| Block 3 (RVE, ablations, cost curves) | **Kaggle CPU commits** | many independent runs; parallelise across notebook versions |
| Block 4 (amortized scale-up, 3D) | **Kaggle GPU P100** | this is the only part that truly needs a GPU |

Honest note: nothing in Blocks 0–2 needs Kaggle. Notebooks are a poor home
for a codebase — keep the code in git and keep notebooks thin.

## Setup pattern (thin driver, code in git)

Two options; the first is cleaner.

**A. Attach the repo as a private Kaggle Dataset.**
Upload `certihom_full.zip` as a Dataset (e.g. `certihom-src`), then:

```python
!pip -q install pytest
!cp -r /kaggle/input/certihom-src/certihom /kaggle/working/
%cd /kaggle/working/certihom
!python -m pytest -q          # expect 24 passed
```

**B. Install from GitHub** (needs Internet enabled in notebook settings):

```python
!pip -q install git+https://github.com/<you>/certihom.git
```

Re-uploading the Dataset on each change is friction; option B is better
once the repo is on GitHub.

## Session limits and checkpointing

Kaggle sessions are capped (roughly 12 h CPU / 9 h GPU) and interactive
sessions die when the tab closes. Two consequences:

1. **Use "Save & Run All" (commit)** for anything long — it runs headless
   in the background and saves outputs. Do not babysit an interactive tab.
2. **Checkpoint everything.** Nothing in this codebase currently saves
   state, which is fine for 2-minute runs and fatal for a 6-hour Phase 3
   training. Before Block 4, add to `fit_amortized`:

```python
torch.save({"step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "history": history},
           "/kaggle/working/ckpt.pt")
```

and a resume path that reloads it. Write files to `/kaggle/working`
(~20 GB, persisted as notebook output); everything else vanishes.

Also pin your environment and record it in `REPRO.md`:

```python
import torch, numpy, scipy, sys
print(sys.version, torch.__version__, numpy.__version__, scipy.__version__)
```

## GPU support is now IMPLEMENTED

The codebase is dtype- and device-aware, and `certihom/gpu.py` provides the
precision split:

```python
from certihom.gpu import pick_device, to_training, to_certification
dev   = pick_device()
model = AmortizedElasticity(dtype=torch.float32)
model = to_training(model, dev)          # fp32 on GPU
model, hist = fit_amortized(model, fam, ...)
cert  = to_certification(model)          # fp64 on CPU for certification
```

**Verified in-sandbox for amortized elasticity:** float32-trained fields
certified in float64 gave gap 35.1% vs float64 training's 36.6% (within
noise), with Loewner validity, the rigorous FEM audit, and containment of
the EXACT laminate tensor confirmed in both. Reduced training precision
widens intervals at worst; it cannot invalidate them.

Ready-to-run driver: `kaggle/04_gpu_elasticity.py`.

Remaining caveats:
1. **CUDA itself is untested** — the sandbox has no GPU, so only the dtype
   path was exercised. Run the test suite once on Kaggle before trusting a
   long job.
2. **`fv_reference` / `fem_reference` are SciPy sparse** — CPU only, and
   correctly so. Only the neural side moves.
3. **Batch size is the real win.** Push `BATCH` up on GPU; small tensors
   leave the card idle.
4. **Never certify from the float32 model.** `certify.py` and `bfs.py` are
   float64 numpy by design. Always go through `to_certification`.

## Alternatives worth knowing

- **Google Colab** — free tier is T4 (bad FP64, same trap as above); paid
  A100 has good FP64 (1:2). Session limits are similar.
- **University HPC** — usually the right answer for Block 4 if you have
  access, since there are no session caps and you can queue sweeps.
- **A local workstation GPU** — check its FP64 ratio before assuming it
  helps. Most consumer RTX cards are 1:64 for FP64 and will be *slower*
  than your CPU here.
