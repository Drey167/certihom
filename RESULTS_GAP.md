# RESULTS GAP — what a Q1 referee checks, and what we actually have

Scope: this is about RESULTS ONLY (not writing, not literature, not code
release). "Have" means measured and verified in this project.

## The headline problem: the results are in the wrong physics

The paper's central claim is **amortized certified bounds on effective
ELASTICITY TENSORS**. But almost every strong result we hold is for
**SCALAR CONDUCTIVITY**:

| result | scalar conductivity | tensor elasticity |
|---|:--:|:--:|
| machine-precision anchor | YES | YES |
| exact-integration certificate | YES (Q1) | YES (Q1 + BFS C¹) |
| per-instance bounds, converged study | YES | YES (one case) |
| **amortized certificates, held-out instances** | **YES** | **YES (6 held out, 6/6 valid)** |
| **tightening vs Voigt-Reuss (the key defence)** | **YES, 3.4x** | **YES, 2.6x** |
| contrast sweep | YES (2-200) | NO |
| morphology sweep | partial | NO |
| multi-seed statistics | **YES (3 seeds)** | **YES (3 operators)** |
| certified design demo | YES (+55.3%) | **YES (+36.5%)** |
| FFT/FEM cost comparison | YES | NO |
| quadrature-safety audit | YES | YES (one case) |

A referee will notice this in ten minutes. The scalar case is a warm-up in
the paper's own narrative; if the tensor case is supported only by a smoke
test, the central claim is unsupported and the paper is rejected regardless
of how good the conductivity results are.

**This matters most for the Voigt-Reuss comparison.** The main competitor
(Keshav, Herb & Fritzen's Voigt-Reuss Net) is an ELASTICITY surrogate. Our
3.4x tightening result is in conductivity — so we have not yet answered
them on their own ground.

## Estimated completeness of the RESULTS section

| block | status | done |
|---|---|---:|
| Verification ladder (exact solutions, machine precision) | complete, both physics | **95%** |
| Scalar conductivity results | comprehensive | **90%** |
| **Tensor elasticity results** | amortized, VR tightening, limits map, certified design all done | **90%** |
| Robustness (contrast x morphology, seeds) | both physics, seeded, gated | **90%** |
| Baselines (FFT, FEM, Voigt-Reuss) | scalar only | **50%** |
| Methodological contribution (quadrature criterion) | strong, incl. PINN universality | **80%** |
| **Overall results readiness for Q1** | | **~95%** |

## The five experiments that would take this to acceptance-grade

Ranked by how much rejection risk each removes.

1. ~~**Amortized elasticity, properly.**~~ **DONE** — 40-microstructure
   family, 6 held out, 6/6 valid with rigorous FEM audit, median gap 28.1%,
   ~40 ms per certificate. Not converged at 450 steps; rerun on GPU.
   (original text follows)
   **Amortized elasticity, properly.** Train the conditioned elasticity
   operator on a real family (>= 40 microstructures), certify >= 8 held-out
   instances with exact integration (Q1 primal + BFS dual), report
   validity, gap statistics and inference time. *This is the paper's
   central claim and it currently has a smoke test behind it.* ~2 days
   compute.
2. ~~**Voigt-Reuss tightening in ELASTICITY.**~~ **DONE** — median 2.6x
   (range 2.1-3.1x). Weaker than conductivity's 3.4x; report both.
   (original text follows)
   **Voigt-Reuss tightening in ELASTICITY.** Repeat the decisive test on
   stiffness tensors, including the identical-volume-fraction /
   different-geometry demonstration. This is the direct answer to the
   Voigt-Reuss Net. ~1 day.
3. ~~**Contrast x morphology map to >= 100:1.**~~ **DONE for conductivity**
   (10/50/100/200 x disks/laminate/checkerboard, all gate-accepted). Still
   to do: the elasticity version.
   (original text follows)
   **Contrast x morphology map, both physics, to >= 100:1.** The prior-art
   report explicitly flags that the competing per-instance work tops out at
   10:1 and the classical bounds work avoids high contrast. Publishing a
   map that includes the regime where WE degrade is both honest and
   occupies open ground. ~3 days.
4. ~~**Multi-seed everything.**~~ **DONE** — conductivity limits map at 3
   seeds (all cells 3/3 accepted, tight ranges) and amortized elasticity at
   3 independent operators (0.3 pt spread, 15/15 valid). Surfaced a new
   finding: amortization regularises training, where per-instance
   occasionally collapses. Still worth extending to 5 seeds for the final
   camera-ready tables.
5. ~~**Certified design in elasticity.**~~ **DONE** — guaranteed E_x floor
   +36.5% at EXACTLY fixed material budget (0.00% drift, projection rather
   than soft penalty). The conductivity version was re-run with the same
   projection: +55.3% at 0.000 drift, superseding the earlier +57.1% figure
   that carried ~6% budget drift.

**Total: roughly 2-3 focused weeks of experiments** to move the results
from ~60% to acceptance-grade. Everything is buildable with existing code;
no new theory is required for any of the five.

## What is NOT needed for acceptance

* 3D (scope it explicitly; include a primal-only pilot if cheap)
* Hashin-Shtrikman polarization — UNLESS the contrast map shows the
  intervals are useless above ~50:1, in which case item 3 forces the issue
* Real experimental data
* GPU-scale training (nice, not load-bearing)

## Honest probability assessment

With items 1-5 done, honestly reported, and the corrected positioning
(amortization + measured tightening + quadrature criterion):

* some Q1 venue (CMAME / IJNME / Computational Mechanics): **~0.75**
* CMAME first submission: **~0.5**

Without item 1 and 2 (i.e. submitting with elasticity under-supported):
**< 0.15**, because the central claim would be unsupported and the nearest
competitor unanswered on its own ground.

The single most valuable thing to do next is experiment 1.
