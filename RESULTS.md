# Measured results — Phase 0 verification (1 CPU core, float64)

Regenerate with the commands shown; all runs are seeded and deterministic.

## Test suite

`pytest -q` → **8 passed, 83 s.** Includes the machine-precision laminate
certificate (K_low = K_up = K_exact to ~1e-12 through the production Gram
assembly), Voigt–Reuss at zero-initialization, the bracket invariant for
arbitrary admissible fields, FV-reference exactness on laminates, FV
convergence to the Dykhne value, and the training smoke.

## Checkerboard, contrast 5 (`run_checkerboard.py --contrast 5 --steps 900`)

Exact: K_eff = √5 I = 2.236068 I. Classical envelope: [1.6667, 3.0000].

| step | tr K_up | tr K_low | relative gap |
|-----:|--------:|---------:|-------------:|
| 0    | 6.0000 (= 2×Voigt) | 3.3333 (= 2×Reuss) | 57.14% |
| 250  | 4.6289  | 4.3204   | 6.90% |
| 500  | 4.5935  | 4.3551   | 5.33% |
| 900  | 4.5586  | 4.4230   | **3.03%** |

Final eigenvalue intervals: K_low ∈ [2.2043, 2.2187], K_up ∈ [2.2784,
2.2802]; contains the exact value: **True**; Loewner valid: **True**;
brackets FV(64) = 2.2141: **True**. Wall clock 90 s, label-free.

## Random Boolean disks, contrast 10, 36% inclusions
(`run_random_disks.py --steps 700`)

Genuinely anisotropic case — full 2×2 matrices with off-diagonals:

```
K_low = [[1.9013, -0.0045], [-0.0045, 1.9267]]
K_up  = [[2.0575, -0.0068], [-0.0068, 2.1061]]      gap 8.42%
FV(h)   = [[1.9200, -0.0057], [-0.0057, 1.9435]]    bracketed: True
FV(h/2) = [[1.9405, -0.0061], [-0.0061, 1.9673]]    bracketed: True
```

Note that mesh refinement moves the FV reference *deeper into* the
certified interval — the expected signature of a consistent reference
audited by rigorous bounds.

## Observations worth carrying into the paper

1. **Certificates never break, they widen.** Every logged iterate of every
   run was Loewner-valid — the architectural-admissibility design doing its
   job. Optimization quality is visible only as interval width.
2. **The dual side lags at high contrast.** On the contrast-10 disks the
   lower bound plateaus earlier than the upper bound (tr K_low stalls near
   3.83 while tr K_up keeps improving). This is exactly the regime the
   planned Hashin–Shtrikman polarization-field variant targets; the effect
   is now measured, not hypothesized.
3. **Cost profile.** ~0.1 s/step at 4096 quadrature points, width-64
   depth-3 nets, on a single CPU core. A single consumer GPU trivially
   supports the Phase-2 elasticity Gram (6 heads per side in 3D) and the
   Phase-3 amortized batches.

---

# Phase 2 — elasticity (measured)

## Test suite

`pytest -q` → **15 passed, 14 s** (Phase 0 remains green). New anchors:
Mandel material sanity; homogeneous collapse; elasticity Voigt–Reuss at
zero-init; the elasticity money test — exact laminate per-phase states
through the production Mandel Gram give **C_up = C_low = C_exact to
~1e-11** (with the dual trials recombined to unit mean stress; see MATH.md
§9b); Q1-FEM exactness on the laminate; training smoke with the rigorous
one-sided audit C_low ⪯ C_FEM(h).

## Stiff-disk composite, plane strain
(`run_elasticity_disks.py --steps 400`; matrix E=1, ν=0.3; disks E=10,
ν=0.2; inclusion fraction 0.340)

| step | tr C_up | tr C_low | relative gap |
|-----:|--------:|---------:|-------------:|
| 0    | 12.6693 (= Voigt) | 4.9440 (= Reuss) | 84.51% |
| 150  | 6.4866  | 5.4421   | 16.41% |
| 400  | 6.1817  | 5.5183   | **10.71%** |

Full 3×3 Mandel interval (off-diagonals and shear coupling included):

```
C_low = [[2.1141, 0.8305,  0.0121],        C_up = [[2.3552, 0.8794,  0.0214],
         [0.8305, 2.1394, -0.0167],                [0.8794, 2.3758, -0.0178],
         [0.0121,-0.0167,  1.2647]]                [0.0214,-0.0178,  1.4507]]
```

Certified engineering deliverables:
**E_x ∈ [1.7914, 2.0291]**, **E_y ∈ [1.8128, 2.0470]** (plane-strain
directional moduli, widths ≈ 12%). Wall clock 80 s, 1 CPU core, label-free.

Rigorous one-sided FEM audit (no tolerance): min eig(C_FEM − C_low) =
**+6.66e-2 at h** and **+4.40e-2 at h/2** — positive at both levels, with
refinement moving the conforming reference downward toward C_eff while
staying above the certified floor, exactly the required signature. The FEM
values also lie inside the certified interval at both levels.

## Observations carried forward

4. **The dual lag persists in elasticity.** tr C_low moves 4.94 → 5.52
   while tr C_up covers 12.67 → 6.18: the Airy side (second derivatives)
   optimizes more slowly — now measured in both physics, sharpening the
   case for the Hashin–Shtrikman polarization variant and for giving the
   dual net more capacity/steps.
5. **Cost.** ~0.20 s/step at 2304 points (15 autograd passes/step) on one
   CPU core; the 3D Beltrami version (6 heads/side) is a GPU afternoon,
   not a cluster problem.

---

# Phase 3 — amortization (measured)

## Test suite

`pytest -q` → **19 passed, 48 s** (Phases 0 and 2 remain green). New
anchors: bit-for-bit formula pinning of the conditioned field paths against
the certified Phase 0/2 code on shared networks; family-wide Voigt–Reuss at
zero-init for every input; held-out generalization with FV audits; OOD
validity audited against EXACT tensors; amortized-elasticity smoke with the
rigorous one-sided FEM audit on a held-out instance.

## One operator, 40 training disks, certificates at inference cost
(`run_amortized.py --steps 400`; contrast 10, n=16 grids, 1 CPU core)

Family training (label-free): monitor mean gap 89.39% (= Voigt–Reuss) →
**26.05%** in 400 steps / 123 s.

**Held-out in-distribution (8 unseen instances, zero further training):**
validity 8/8 (Loewner + FV bracket); certified gap mean **28.0%**, max
40.2%, against a Voigt–Reuss level of 90.1%; **19 ms per certificate**
(including the FV audit in the timing).

**Amortization economics (same instance, both routes):**

| route | certified gap | wall clock |
|------|--------------:|-----------:|
| per-instance Phase-0 fit (400 steps) | 9.91% | 25 s |
| amortized inference | 26.58% | **19 ms (~1,340× faster)** |

Three orders of magnitude in speed for ~2.7× in width — and validity is
never part of the trade.

**Out-of-distribution, audited against EXACT effective tensors (no
reference-solver slack anywhere in this table):**

| input | valid | contains exact | gap |
|-------|:-----:|:--------------:|----:|
| laminate f=0.25 (unseen family) | True | True | 12.04% |
| laminate f=0.50 (unseen family) | True | True | 11.65% |
| checkerboard (corner-singular) | True | True | 39.04% |
| laminate, contrast 100 (far-OOD) | True | True | 174.64% |

## Observations carried forward

6. **Conditioning must preserve conformity.** Pixel-lookup features would
   make the trial fields discontinuous (not H¹) and void every guarantee;
   the smooth in-graph Fourier-feature conditioning is what lets all
   Phase 0–2 theorems survive amortization untouched (MATH.md §10). This
   design constraint is, to our knowledge, unstated in prior work — the
   per-instance literature never has to face it.
7. **Local features transfer; width tracks shift severity.** Unseen-family
   laminates certify at ID-level width (interface correctors generalize),
   the corner-singular checkerboard widens moderately, and a 10× contrast
   shift widens the certificate by ~6× — while every one of them remains
   valid against the exact tensor. The certificate is a trust signal with
   no false alarm on benign shifts and a loud one on severe shifts.
8. **The economics motivate the hybrid workflow.** Amortized screening at
   ~50 certificates/s, then per-instance refinement (optionally warm-started
   from the amortized fields) on the shortlisted designs.

---

# Phase 1 + application studies (measured)

`pytest -q` → **24 passed, 63 s** (all earlier phases green).

## Phase 1 — certificates with the quadrature caveat REMOVED

Network fields are interpolated into conforming FE spaces on a
pixel-aligned mesh and the energies are integrated exactly (2×2 Gauss for
Q1, 4×4 for BFS). Machine-precision anchors, all passing:

| projection | space | anchor | result |
|---|---|---|---|
| scalar primal + dual | Q1 (C⁰) | exact laminate correctors | K_low = K_up = K_exact, 1e-12 |
| elasticity primal | Q1 vector | exact laminate displacements | C_up = C_exact, 1e-11 |
| **elasticity dual** | **BFS bicubic Hermite (C¹)** | exact laminate Airy potential | C_low = C_exact, 1e-10 |

The BFS result is the technically hardest piece: Q1 cannot carry an Airy
stress (bilinears have φ,11 = φ,22 = 0), and C¹ continuity is exactly what
makes the Airy traction (φ,22, −φ,12) continuous across element edges, so
the interpolated stress stays in H(div) with zero divergence.

## Contrast sweep + the price of exact certification
(`run_contrast_certification.py --steps 200`, fixed 6-disk geometry)

| contrast | Voigt–Reuss gap | quadrature gap | **exact gap** | price | valid |
|---:|---:|---:|---:|---:|:--:|
| 2 | 10.1% | 1.0% | **1.1%** | +0.1% | True |
| 10 | 92.4% | 9.3% | **10.3%** | +1.1% | True |
| 50 | 167.1% | 25.7% | **27.8%** | +2.1% | True |
| 200 | 190.9% | 55.5% | **58.6%** | +3.0% | True |

Removing the quadrature caveat costs 0.1–3.0 percentage points of width —
the certification price, measured rather than assumed. Bounds degrade
gracefully with contrast and stay far inside the Voigt–Reuss envelope.

## The certificate audits the auditor (contrast 200)

At contrast 200 the FV reference on the native mesh fell **outside** the
certified interval (min eig(K_FV − K_low) = −0.031). Since the bounds are
rigorous, this proves the *reference* was under-converged — not the
certificate. Mesh refinement confirms it:

| FV mesh | K_FV[0,0] | inside certified interval |
|---|---:|:--:|
| h | 1.8894 | **False** (−0.031) |
| h/2 | 1.9534 | True (+0.043) |
| h/4 | 1.9830 | True (+0.074) |

A rigorous two-sided certificate detects an unconverged conventional solver
without knowing the true answer — a use case beyond surrogate modelling.

## Certified inverse design (`run_certified_design.py --steps 240`)

Design variables: 8 inclusion centers (differentiable smooth-union
rendering); objective: maximize the **guaranteed** K_low[0,0]; contrast 10.

| | K_low[0,0] (guaranteed) | K_up[0,0] | inclusion fraction |
|---|---:|---:|---:|
| initial (random) | 1.9159 | — | 0.256 |
| optimized, exactly certified | **3.0099** | 3.0780 | 0.272 |

**Gain +57.1% in the guaranteed floor**, Loewner-valid, with the FV
reference inside the final interval. *Volume-fraction control:* the
fraction drifted 0.256 → 0.272, which by itself moves Voigt by only +4.4%
and Reuss by +1.9% — so the gain comes from arrangement (percolating
chains), not from extra material. The drift is nonetheless a limitation of
the soft budget penalty; a projection onto the constraint set would remove
it.

The design objective is self-penalizing: a looser trial-field model yields
a LOWER guaranteed value, so the optimizer cannot win by fooling the
surrogate — the failure mode that makes conventional surrogate-driven
design untrustworthy is structurally absent.

---

# CRITICAL FINDING — quadrature overfitting can produce FALSE certificates

Discovered while testing the conformity trap; not anticipated.

**A network trained against a FIXED midpoint rule can learn to place its
energy between the quadrature points.** The rule then reports an energy far
below the field's true energy, and the "upper bound" can fall BELOW the
exact effective tensor.

| case | evaluated at | K_up[0,0] | true | verdict |
|---|---|---:|---:|---|
| adversarial (primal-only, 600 steps, lr 3e-3) | nq=32 (as trained) | **0.0014** | 3.0769 | **FALSE BOUND** |
| same field | nq=64 | 53.25 | 3.0769 | valid |
| same field | nq>=128 | ~58.9 | 3.0769 | valid (converged) |
| **laminate c6, ordinary settings** | **nq=32 (as trained)** | **2.5207** | **2.6667** | **FALSE BOUND** |
| laminate c6 | nq=64 / 128 | 3.107 / 3.171 | 2.6667 | valid |
| laminate c6 | **Q1 exact, ne=64** | 3.0429 | 2.6667 | **valid** |
| checkerboard c5 | nq=64 -> 256 | 2.2786 -> 2.2906 | 2.2361 | valid throughout |

So it is configuration-dependent, not universal — the headline checkerboard
result is clean under refinement, but the laminate at nq=32 is genuinely
invalid at its own training resolution.

**Consequences.**
1. **Phase 1 is not a nicety, it is mandatory.** Exact integration cannot
   hide energy between sample points because there are no sample points.
   Every "exact" column in this file remains trustworthy.
2. **Every midpoint-rule number in Phases 0/2/3 must be re-certified** by
   projection before being cited. Treat the "quad" columns as training
   diagnostics, not results.
3. **Standing rule added:** never report a midpoint bound as a certificate.
   Train against it if convenient; certify with exact integration always.
4. Regression tests added in `tests/test_quadrature_safety.py`, and a
   refinement diagnostic (does K_up jump when the rule is refined?) should
   accompany every study.

This is exactly the failure mode a referee would hunt for, and it was found
by trying to break the method rather than by running it.

# Conformity trap (Block 1) — partial result, claim SOFTENED

Lookup (non-conforming) conditioning was compared against smooth
(conforming) conditioning, verdict against exact laminate tensors:

| conditioning | midpoint bound | Q1-projected bound |
|---|---|---|
| smooth (conforming) | valid, gaps 46-66% | valid, gaps 46-66% |
| lookup (non-conforming) | valid, gaps 90-107% | valid, gaps 115-125% |

**No validity violations were observed from non-conformity alone.** The
lookup conditioner was markedly LOOSER (roughly double the gap) but not
wrong in these runs. The honest statement for the paper is therefore:
non-conformity removes the GUARANTEE (the theorem no longer applies) and
measurably degrades tightness; it did not, in this configuration, produce
an observed violation. MATH.md §10's stronger phrasing has been softened
accordingly.

---

# T0.5.3 — a MEASURABLE rule for "is the collocation rule fine enough?"

`examples/run_quadrature_rule.py`. Diagnostics in `certihom/diagnostics.py`:
spectral bandwidth B (radius containing 99.9% of the field's spectral
energy), Nyquist limit nq/2, headroom = Nyquist/B, and the refinement ratio
(energy at 4x the rule / energy at the rule). Laminate contrast 6, width 48,
depth 3, 400 steps; true tr(K_eff) = 7.4167.

| nq | n_harm | bandwidth | Nyquist | headroom | reported | refined | ratio | invalid |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 16 | 4 | 16.0 | 8.0 | 0.50 | 0.031 | 109.853 | **3498.46** | YES |
| 32 | 4 | 9.0 | 16.0 | 1.78 | 7.271 | 7.921 | 1.09 | mild |
| 64 | 4 | 7.0 | 32.0 | 4.57 | 7.394 | 7.612 | 1.03 | mild |
| 128 | 4 | 6.0 | 64.0 | 10.67 | 7.434 | 7.446 | 1.00 | **no** |
| 32 | 2 | 36.0 | 16.0 | 0.44 | 4.767 | 102.98 | 21.6 | YES |
| 32 | 8 | 30.0 | 16.0 | 0.53 | 4.758 | 46.81 | 9.84 | YES |
| 32 | 12 | 34.0 | 16.0 | 0.47 | 0.015 | 16.524 | **1093.23** | YES |

**Rule 1 (hard).** headroom < 1 is catastrophic — refinement ratios of
10x-3500x. The reported "bound" is fiction.
**Rule 2 (practical).** headroom > 1 keeps the ratio near 1, but a strictly
valid midpoint bound required headroom of order 10 here.
**Rule 3 (follow this).** Certify by exact integration and no headroom
requirement applies at all.

**Non-obvious finding: bandwidth is NON-MONOTONIC in n_harmonics**, with a
minimum at 4 (B = 9) and much larger values at 2 (B = 36) and 12 (B = 34).
Too FEW embedding harmonics is as dangerous as too many: an
under-parameterised embedding cannot fit the target smoothly, so training
drives the weights up and the tanh nonlinearity manufactures high
harmonics. This is very likely the same root cause as the seed volatility
seen earlier at n_harmonics=3 — one mechanism, two symptoms.

**Why this travels beyond homogenization.** Every physics-informed network
trained on fixed collocation points is exposed to exactly this failure, and
the diagnostics are three lines of FFT. The bandwidth-vs-Nyquist check is
not specific to bounds, homogenization, or even elliptic PDEs.

# Phase 2 quadrature audit — prediction FALSIFIED, Phase 2 is safe

Prediction on record: the Airy dual differentiates twice, so it should be
MORE exposed to quadrature overfitting than the primal. Measured on the
exact elasticity laminate (nq=32, n_harm=3, width 48, depth 3, 250 steps;
exact C_eff[0,0] = 1.7252):

| field | bandwidth | headroom at nq=32 |
|---|---:|---:|
| primal displacement chi | 6.0 | 2.67 |
| **dual Airy potential phi** | **3.0** | **5.33** |

| evaluation | C_up[0,0] | valid | C_low[0,0] | valid |
|---|---:|:--:|---:|:--:|
| midpoint nq=32 | 1.7311 | yes | 1.7251 | yes |
| midpoint nq=64 | 1.7587 | yes | 1.7250 | yes |
| midpoint nq=128 | 1.7665 | yes | 1.7250 | yes |
| **exact (Q1 / BFS)** | **1.7592** | **yes** | **1.7250** | **yes** |

**The dual is SMOOTHER, not rougher — the opposite of the prediction.**
The explanation is that the Airy parameterisation is self-regularising:
the complementary energy is quadratic in the SECOND derivatives of phi, so
a Fourier mode of frequency xi is penalised like |xi|^4 rather than the
|xi|^2 of a first-derivative energy. High-frequency content is therefore
far more expensive for the dual field, and the optimiser avoids it. Taking
more derivatives buys smoothness here rather than costing it.

**Phase 2 status: valid at every resolution tested.** There is mild
quadrature optimism on C_up (1.7311 at nq=32 vs 1.7592 exact, ~1.6%),
exactly the size Rule 2 predicts at headroom 2.67 — but it never crosses
the true value. The lower bound is essentially exact (1.7250 vs 1.7252).
So the Phase 2 headline numbers stand; they should still be RE-CERTIFIED
through the exact path before publication, per Block 0.5.

---

# UNIVERSALITY: collocation overfitting in a VANILLA PINN

`examples/run_pinn_universality.py`. Textbook problem, nothing to do with
homogenization: -u'' = f on (0,1), u(0)=u(1)=0, u* = sin(pi x), exact BCs
via u = x(1-x)N(x), plain tanh MLP (width 64, depth 4), residual-only loss
on N FIXED collocation points, 4000 Adam steps.

| N_coll | res @ train points | res @ 20x finer grid | ratio | bandwidth | Nyquist | headroom | true L2 err |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2.84e-05 | 4.51e-01 | **15,891** | 6.0 | 2.0 | 0.33 | 1.06% |
| 6 | 1.55e-06 | 2.83e-01 | **182,755** | 6.0 | 3.0 | 0.50 | 0.17% |
| 8 | 1.14e-06 | 1.66e-02 | **14,550** | 6.0 | 4.0 | 0.67 | 0.00% |
| 16 | 2.10e-01 | 2.10e-01 | 1.0 | 6.0 | 8.0 | 1.33 | 0.77% |
| 32 | 2.25e-03 | 2.27e-03 | 1.0 | 6.0 | 16.0 | 2.67 | 0.00% |

**A reported residual of 1.55e-06 corresponds to a true residual of 0.283 —
five orders of magnitude of self-deception**, and it looks exactly like
machine-precision convergence in a loss curve.

**The headroom criterion transfers exactly.** Every configuration with
headroom < 1 overfits (ratios 1e4-1e5); every configuration with headroom
> 1 has ratio 1.000. The threshold found in homogenization (headroom = 1)
reproduces on a completely different problem, architecture and loss.

**An important nuance, stated honestly.** In this easy problem the SOLUTION
stays accurate (L2 error 0.0-1.1%) even when the reported residual is
meaningless. So the failure is epistemic rather than always catastrophic:
the training residual cannot be used as evidence of convergence, and in a
problem where you have no analytic solution you would have no way to know.
That is precisely the situation of most published PINN results.

**Practical recommendations (no FFT needed for the first one):**
1. Always report the residual on an INDEPENDENT, finer point set. The ratio
   between it and the training residual is a one-line honesty check.
2. Use the bandwidth-vs-Nyquist headroom to choose the collocation count a
   priori: N > 2B.
3. Where the problem has variational structure, certify by exact
   integration and the issue disappears entirely.

**Consequence for publication strategy.** This establishes the universality
needed for the standalone methods paper (see ROADMAP.md): the criterion is
not a homogenization result, it is a collocation-training result.

---

# Closing the loop: are bandwidth blow-up and seed volatility ONE mechanism?

`examples/run_harmonics_mechanism.py`. Laminate contrast 6, nq=32, width 48,
depth 3, 250 steps, 3 seeds per setting. Gaps from EXACT integration.
Mediator tested: weight inflation, measured as trunk gain (product of layer
spectral norms) and the derivative-amplification proxy (2 pi M) x gain.

| n_harm | gap median | gap spread | bandwidth | trunk gain | (2piM)*gain | valid |
|---:|---:|---:|---:|---:|---:|:--:|
| 2 | 64.8% | 133.0% | 23.0 | 3.7 | 46.2 | yes |
| 3 | 50.8% | 147.6% | 26.0 | 3.6 | 67.0 | yes |
| **4** | **0.4%** | **5.2%** | **7.0** | **0.6** | **14.2** | yes |
| 6 | 17.0% | 137.3% | 17.0 | 0.8 | 31.3 | yes |
| 10 | 155.4% | 54.4% | 32.0 | 0.9 | 59.5 | yes |

Rank correlations across the sweep (n = 5 settings, so weak evidence):
amplification vs bandwidth **+0.90**, amplification vs spread **+0.60**,
bandwidth vs spread **+0.30**.

**Verdict: PARTIALLY unified — do not overclaim.**
* All four quantities share a sharp common optimum at n_harmonics = 4.
  A single hyperparameter moves the certified gap from 155% to 0.4%.
* The proposed mediator (weight inflation) predicts BANDWIDTH strongly
  (+0.90) and seed spread moderately (+0.60), consistent with the
  mechanism: an under-parameterised embedding forces large weights, and
  tanh then manufactures high harmonics.
* But bandwidth and seed spread are NOT interchangeable (+0.30). At
  n_harmonics = 10 the bandwidth is highest (32) while the spread is
  comparatively low (54%) — the runs are consistently BAD (median gap
  155%) rather than erratic. Bandwidth measures the magnitude of the
  pathology; spread measures its instability. Related through the same
  knob, not the same quantity.
* Caveat that must survive into the paper: 3 seeds and 5 settings are thin.
  Rerun with >=8 seeds and a finer harmonic grid before claiming a
  correlation.

**Certificates never failed**, at any setting, including the 155%-gap runs.
Validity was 100% throughout — the pathology widens intervals, exactly as
the architecture guarantees, and never invalidates them.

**A useful by-product: hyperparameter selection WITHOUT ground truth.**
Because a tighter certificate is verifiably better, the certified gap can
be used to select n_harmonics with no validation set and no reference
solution — the guarantee makes model selection self-supervised. That is
worth a short subsection; ordinary surrogates cannot do this.

---

# T3.3 — FFT baseline, and an honest reckoning

`certihom/fft_reference.py` implements the Moulinec-Suquet basic scheme,
verified against exact solutions (laminate to 1.45e-11; checkerboard 0.062%
at contrast 5 after ~4900 iterations).

| case | FFT estimate | FFT error | FFT time | certified interval | gap | time | FFT inside? |
|---|---:|---:|---:|---|---:|---:|:--:|
| checkerboard c=5 | 2.2320 | 0.184% | 1.0 s | [2.1800, 2.2958] | 4.9% | 52 s | yes |
| checkerboard c=50 | 6.7976 | 3.867% | 1.2 s | [4.5586, 10.5049] | 78.5% | 50 s | yes |

(exact values 2.2361 and 7.0711; the certified interval contained the exact
tensor in both cases.)

## What this honestly shows

**1. FFT is faster per query, by roughly 50x. Do not pretend otherwise.**
Any framing of this work as "faster than FFT" for a single microstructure
is false and a referee will catch it immediately.

**2. Many-query crossover is around N ~ 300 designs** (amortized: 123 s
training + 19 ms per certificate; FFT ~0.5 s per solve):

| designs | FFT (no guarantee) | amortized certified | winner |
|---:|---:|---:|---|
| 10 | 5 s | 123 s | FFT |
| 100 | 50 s | 125 s | FFT |
| 1 000 | 500 s | 142 s | **amortized** |
| 10 000 | 5 000 s | 313 s | **amortized** |

**3. The real differentiator is not speed, it is the guarantee.** At
contrast 50 the FFT estimate was 3.87% wrong — and nothing in the FFT
output reveals that. Without the analytic answer the practitioner has a
number and no error bar. The certified method returns an interval that
provably contains the truth.

**4. A genuine weakness, stated plainly: the interval is too wide at high
contrast on hard morphologies.** 78.5% on the contrast-50 checkerboard is
valid but not useful as an engineering number. Contrast 10 disks gave
12.6%, contrast 50 disks 27.8% — so width depends on both contrast AND
morphology (corner singularities hurt). The paper must publish this map
rather than the favourable cases, and it makes the Hashin-Shtrikman
polarisation variant the clear next contribution rather than an optional
extra.

**5. No solver-audit catch this time.** Unlike the contrast-200 FV case,
the FFT estimate fell inside the certified interval in both runs. The audit
capability is real but only bites when the interval is tighter than the
competing solver's error.

## Correct positioning for the paper

Claim: *certified intervals at amortized cost*, differentiability for
design, and the ability to audit other solvers — NOT raw speed. The
comparison table above should appear in the paper as-is, including the
rows where FFT wins.

---

# THE DECISIVE TEST: are amortized certificates tighter than Voigt-Reuss?

Prior art forced this experiment. Keshav, Herb & Fritzen's "Voigt-Reuss Net"
(GAMM-Mitteilungen 2025; IJNME 2026) is an amortized neural stiffness
surrogate that guarantees, by spectral normalisation, that predictions lie
inside the analytic Voigt-Reuss envelope in the Loewner sense — during
training, inference AND gradient-based design. That overlaps this project on
four axes simultaneously. If our amortized certificate is not strictly
tighter than Voigt-Reuss, this project is a subset of theirs.

`examples/run_vs_voigt_reuss.py`, contrast 10, operator trained on 8 disk
microstructures, all certificates by EXACT integration:

| microstructure | Voigt-Reuss width | certified width | tightening | valid |
|---|---:|---:|---:|:--:|
| disks (held out) | 83.0% | 20.8% | **4.0x** | yes |
| disks (held out) | 89.6% | 26.7% | **3.4x** | yes |
| disks (held out) | 94.0% | 34.6% | **2.7x** | yes |
| laminate f=.25 (OOD) | 86.3% | 23.8% | **3.6x** | yes |
| checkerboard (OOD) | 100.6% | 42.4% | **2.4x** | yes |

**Median tightening factor 3.4x on unseen microstructures.**

## The argument that actually settles it

Voigt-Reuss depends ONLY on volume fraction, so it carries no geometric
information whatsoever:

| microstructure | volume fraction | Voigt-Reuss width | certified width |
|---|---:|---:|---:|
| laminate f=0.50 | 0.500 | 100.6% | **18.5%** |
| checkerboard | 0.500 | 100.6% | **42.4%** |

Identical volume fraction gives an IDENTICAL Voigt-Reuss envelope, yet the
certified intervals differ by more than a factor of two because the
geometries differ. A guarantee derived from volume fraction alone cannot
distinguish a laminate from a checkerboard; a variational certificate
computed from admissible trial fields for THIS microstructure can. That is
the difference between a physical-admissibility guarantee and a certificate.

## An honest subtlety

The held-out laminate reports "not strictly inside the Voigt-Reuss
envelope", and that is CORRECT rather than a failure: a laminate attains
the Voigt bound along the layers and the Reuss bound across them, so its
effective tensor lies ON the envelope boundary in the principal directions.
No method can be strictly tighter there. Report this rather than hiding it —
it demonstrates the certificate is honest about extremal microstructures.

---

# EXPERIMENTS 1 + 2 — the central claim, now supported in ELASTICITY

Closing the asymmetry identified in RESULTS_GAP.md: until now the tensor
case rested on a smoke test. `examples/run_amortized_elasticity.py`,
E-contrast 10 (matrix E=1 nu=0.3; inclusions E=10 nu=0.2), plane strain,
ONE conditioned operator trained on 40 microstructures (450 steps, 117 s,
1 CPU core), all certificates by EXACT integration (Q1 primal + BFS C1
Airy dual).

## Experiment 1 — amortized certified elasticity, held-out instances

| held-out | C_low[0,0] | C_up[0,0] | gap | Loewner valid | rigorous FEM audit | inference |
|---|---:|---:|---:|:--:|:--:|---:|
| #1 | 1.6530 | 1.9679 | 21.1% | yes | yes | 0.05 s |
| #2 | 1.6770 | 2.2301 | 26.0% | yes | yes | 0.04 s |
| #3 | 1.8768 | 2.5314 | 30.3% | yes | yes | 0.04 s |
| #4 | 2.1251 | 3.3146 | 41.0% | yes | yes | 0.04 s |
| #5 | 2.0885 | 3.0940 | 38.4% | yes | yes | 0.04 s |
| #6 | 1.6029 | 1.8953 | 24.0% | yes | yes | 0.04 s |

**Median certified gap 28.1% (range 21.1-41.0%); validity 6/6 including
the tolerance-free FEM audit (C_low <= C_FEM must hold since conforming Q1
FEM is itself an upper bound); ~40 ms per certificate on one CPU core.**

The central claim of the paper now has real support rather than a smoke
test: certified two-sided bounds on effective STIFFNESS TENSORS for unseen
microstructures at inference cost.

## Experiment 2 — tightening vs Voigt-Reuss, in the headline physics

**Median tightening factor 2.6x (range 2.1-3.1x)** against the analytic
Voigt-Reuss envelope — the guarantee provided by the nearest competitor,
the Voigt-Reuss Net. This is the answer to that work ON ITS OWN GROUND
(elasticity), not by analogy from conductivity.

| microstructure | volume fraction | Voigt-Reuss width | certified width |
|---|---:|---:|---:|
| laminate f=0.50 | 0.500 | 90.1% | 58.6% |
| checkerboard | 0.500 | 90.1% | 52.8% |

## Honest assessment — where elasticity is WEAKER than conductivity

1. **Tightening is 2.6x here vs 3.4x in conductivity**, and the certified
   gaps are wider (28.1% vs ~12.6% median). Elasticity is genuinely harder:
   six primal heads and three Airy potentials versus two and two.
2. **The identical-volume-fraction demonstration is much less striking in
   elasticity** (58.6% vs 52.8%) than in conductivity (18.5% vs 42.4%).
   Both test cases are far out of the training distribution (disks) and the
   operator is under-trained. Do NOT present the elasticity version of this
   demonstration as the headline; use the conductivity one, and report the
   elasticity numbers honestly alongside.
3. **The results had NOT converged.** Going from 120 to 450 steps moved the
   median gap 37.5% -> 28.1% and the tightening 2.1x -> 2.6x, with no sign
   of a plateau. These numbers are a LOWER BOUND on achievable quality on
   one CPU core; proper GPU training with a larger family and wider
   networks is the obvious next step (Block 4). State them as such rather
   than as the method's ceiling.

---

# CONFIRMED ON REAL HARDWARE: headroom predicts a 127x swing

A 5-seed multi-seed run (Kaggle, CPU) returned a median certified gap of
**162.3%** on the laminate case — against 1.1% from an earlier short
sandbox run. Diagnosis and fix, reproduced exactly (the same seed-by-seed
numbers appeared on both machines):

| nq | headroom | gaps across 3 seeds | median |
|---:|---:|---|---:|
| 32 | **0.64** | 101.5%, 172.8%, 89.3% | 101.5% |
| 64 | **4.57** | 0.8%, 1.7%, 0.4% | **0.8%** |

Raising only the quadrature rule improved the certified gap by **127x**.
The mechanism is the one documented above: at headroom < 1 the optimiser
hides energy between collocation points, and the honest exact-integration
certificate then reports the resulting field truthfully — wide, but valid
(5/5 in the Kaggle run). **The bounds never lied; the field was bad.**

This is the strongest evidence yet that the headroom criterion is
practically load-bearing rather than a curiosity, and it was found because
a collaborator ran the harness at a setting I had not checked.
`examples/run_multiseed.py` and `kaggle/01_multiseed.py` now default to
nq=64.

# NEGATIVE RESULT: GPU scale-up did NOT improve amortized elasticity

Kaggle T4, float32 training + float64 CPU certification (the precision
split, working as designed on real hardware):

| | CPU run | **GPU run** |
|---|---|---|
| training microstructures | 40 | **200** |
| steps | 450 | **4000** |
| batch | 3 | **16** |
| training wall clock | 82 s | 878 s |
| held-out median gap | 28.2% | **28.3%** |
| held-out gap range | [21.5%, 41.3%] | **[11.2%, 45.5%]** |
| median tightening vs Voigt-Reuss | 2.6x | **2.7x** |
| validity | 6/6 | **12/12** |
| inference | 40 ms | 62 ms |

The monitor gap fell 75.4% -> 25.3% by step 1200 and then **plateaued
flat** through step 4000. Nine times the data and nine times the steps
bought essentially nothing.

**This falsifies the claim (made in this project's own notes) that the CPU
numbers were a lower bound that GPU training would improve.** The
bottleneck is architecture, capacity or conditioning — not compute. Report
it as such.

Hypothesis tested and NOT confirmed: that the plateau came from
`AmortizedElasticity` defaulting to n_harmonics=2, the setting the
harmonics study flagged as pathological for the scalar case. A short
comparison gave 40.5% (nh=2) vs 44.4% (nh=4), i.e. no improvement. The
budget was small (90 steps), so treat this as inconclusive rather than
settled, but do not assume the harmonics fix will work.

**Open diagnostic questions for whoever picks this up:**
1. Decompose the gap — is C_up or C_low the limiting side? That says where
   to spend capacity.
2. Sweep width/depth properly (the GPU run used width 64, depth 3 for
   6 primal + 3 dual heads — plausibly too small).
3. Enrich the conditioner (kmax=3 with two smoothing scales may not carry
   enough microstructure information to separate instances).
4. Confirm elasticity headroom is adequate at nq=32 (the Phase 2 audit
   suggested yes: primal 2.67, dual 5.33 — so quadrature is probably NOT
   the elasticity bottleneck, unlike the scalar laminate case).

---

# The criterion becomes a LABEL-FREE QUALITY GATE

A 5-seed run at the corrected nq=64 still produced one catastrophic outlier
(seed 3: 98.1% against a 1.4% median). Per-seed diagnostics, reproduced
in-sandbox to within noise of the Kaggle numbers:

| seed | certified gap | bandwidth | headroom | refinement ratio | verdict |
|---:|---:|---:|---:|---:|---|
| 0 | 0.8% | 7.0 | 4.57 | 1.030 | ok |
| 1 | 1.7% | 6.0 | 5.33 | 1.016 | ok |
| 2 | 0.4% | 7.0 | 4.57 | 1.005 | ok |
| **3** | **97.8%** | **41.0** | **0.78** | **7.076** | **SUSPECT** |
| 4 | 0.9% | 6.4 | 5.00 | 1.016 | ok |

**Two conclusions.**

1. **Headroom is a property of the TRAINED FIELD, not of the
   configuration.** Identical settings and identical nq produced headroom
   4.87 on four seeds and 0.78 on one. Checking headroom per configuration
   is not enough; it must be checked per run.
2. **The diagnostics flagged the bad run WITHOUT the true answer.** Both
   signals (headroom < 2, refinement ratio > 1.05) fired on exactly the
   failing seed and on none of the healthy ones. That makes this a usable
   accept/reject gate in the situation where it is actually needed — when
   no reference solution exists.

Implemented as `certihom.diagnostics.quality_gate`, and wired into
`examples/run_multiseed.py`, which now prints headroom and ACCEPT/REJECT
per seed and reports the median over ACCEPTED runs.

**A rejected run is not an invalid one.** Its exact-integration certificate
still provably contains the truth (validity was 5/5 even including seed 3)
— it is merely uselessly wide. The gate separates trustworthy TIGHTNESS
from guaranteed VALIDITY, which is exactly the distinction this project
exists to make.

**Overfitting accumulates with training.** At 250 steps all five laminate
seeds were clean (gaps 1.0-1.1%, headroom 4.57 uniformly); the seed-3
collapse appeared only at 400 steps. So longer training is not
monotonically better here, and the gate gives a principled early-stopping
signal that the loss curve cannot: stop when headroom degrades, not when
the loss stops falling.

---

# THE LIMITS TABLE — certified gap vs contrast AND morphology

The prior-art assessment demanded this: the nearest competing per-instance
work tops out at 10:1 contrast and the classical bounds work avoids high
contrast, so the honest map occupies open ground.
`examples/run_contrast_morphology.py`, nq=64, 200 steps, exact integration.
**Every entry passed the label-free quality gate and every certificate was
valid** — so these numbers reflect genuine difficulty, not bad runs.

## Certified gap

| contrast | disks | laminate | checkerboard |
|---:|---:|---:|---:|
| 10 | 12.4% | **2.1%** | 19.3% |
| 50 | 31.8% | **5.2%** | 88.7% |
| 100 | 44.1% | **7.4%** | 125.3% |
| 200 | 61.1% | **9.9%** | 155.3% |

## Tightening vs Voigt-Reuss

| contrast | disks | laminate | checkerboard |
|---:|---:|---:|---:|
| 10 | 7.5x | **40.7x** | 5.2x |
| 50 | 5.3x | **31.2x** | 1.9x |
| 100 | 4.1x | **24.5x** | 1.5x |
| 200 | 3.1x | **19.2x** | **1.2x** |

## What this settles

**Morphology matters far more than contrast.** At 200:1 the laminate is
still excellent (9.9%, 19x tighter than Voigt-Reuss) while the checkerboard
is useless (155.3%, only 1.2x tighter — i.e. barely better than the trivial
analytic envelope). Contrast alone is the wrong axis to report; the paper
must publish the two-dimensional map.

**The mechanism is singularity approximation, not optimisation.** The
ordering laminate << disks << checkerboard tracks the regularity of the
exact corrector: piecewise-linear for the laminate, smooth-but-curved for
disks, and CORNER-SINGULAR for the checkerboard, where four phases meet at
a point. Smooth networks approximate singular fields badly, and no amount
of training fixes it — the quality gate accepted every one of these runs.

**Consequence for the Hashin-Shtrikman decision.** HS polarisation was
being considered as the fix for high contrast. This map argues it is the
WRONG fix, or at least not the whole one: the failure is concentrated in
corner-singular morphologies, and an HS polarisation field would have to
represent the same singular behaviour. The better-targeted remedies are
(a) enriching the trial space with singular functions at the corners (an
XFEM-style fix), or (b) scoping the paper honestly to morphologies without
四-phase junctions and declaring the rest open.

**Recommended scoping sentence for the paper:** the method delivers
certificates 3x-40x tighter than the analytic envelope for layered and
inclusion morphologies across 10:1-200:1 contrast, and degrades to
near-trivial tightness for corner-singular morphologies above roughly 50:1,
where the limitation is the approximation of singular corrector fields
rather than the certification machinery.

---

# THE LIMITS TABLE — ELASTICITY (the headline physics)

`examples/run_contrast_morphology_elastic.py`, nq=32, 120 steps, exact
integration (Q1 primal + BFS C1 Airy dual), matrix E=1 nu=0.3, inclusions
E=contrast nu=0.2, plane strain.

## Certified gap, elasticity

| contrast | disks | laminate | checkerboard |
|---:|---:|---:|---:|
| 10 | 16.3% | **5.4%** | 28.1% |
| 50 | 40.3% | **18.9%** | 101.1% |
| 200 | 68.8% * | 63.5% | 161.8% |

`*` = failed the quality gate (refinement ratio 1.075 > 1.05). **The gate
fired unprompted inside a production sweep** — exactly the intended use.

Tightening vs Voigt-Reuss: disks 5.0x -> 4.0x -> 2.8x; laminate
14.2x -> 8.4x -> 3.0x; checkerboard 3.2x -> 1.6x -> 1.2x.

## Both physics side by side

| contrast | | disks | laminate | checkerboard |
|---:|---|---:|---:|---:|
| 10 | conductivity | 12.4% | 2.1% | 19.3% |
| 10 | **elasticity** | 16.3% | 5.4% | 28.1% |
| 50 | conductivity | 31.8% | 5.2% | 88.7% |
| 50 | **elasticity** | 40.3% | 18.9% | 101.1% |
| 200 | conductivity | 61.1% | 9.9% | 155.3% |
| 200 | **elasticity** | 68.8%* | 63.5% | 161.8% |

## Findings

1. **The morphology ordering transfers exactly**: laminate << disks <<
   checkerboard in both physics, at every contrast. The mechanism
   (regularity of the exact corrector; corner singularities where four
   phases meet) is physics-independent.
2. ~~**Elasticity is uniformly wider than conductivity.**~~ **SUPERSEDED —
   see "MATCHED-BUDGET COMPARISON" below.** This comparison used 120-step
   elasticity against 200-step conductivity. At matched budget the claim is
   false: elasticity is BETTER for disks at 50:1 and 200:1, essentially
   identical for the checkerboard, and worse only for the laminate.
3. **At 200:1 elasticity degrades across ALL morphologies**, including the
   laminate that stayed excellent in conductivity. This is the first regime
   where the method is not useful regardless of geometry.
4. **The quality gate caught a bad run in a live sweep** (disks at 200:1),
   which is the first time it has fired outside a deliberately constructed
   test.

**Caveat, stated plainly:** the elasticity sweep used 120 steps against the
conductivity sweep's 200, so the two tables are NOT strictly like-for-like.
The morphology ordering is robust, but the elasticity-vs-conductivity
magnitude comparison should be re-run at matched budget before publication.

## Recommended scoping for the paper

Certificates are 3x-40x tighter than the analytic envelope for LAYERED and
INCLUSION morphologies up to roughly 100:1 in conductivity and 50:1 in
elasticity. They degrade to near-trivial tightness for CORNER-SINGULAR
morphologies above ~50:1, and for all morphologies in elasticity above
~100:1. The limitation is the approximation of singular or
strongly-heterogeneous corrector fields, not the certification machinery —
validity held in every single run reported here.

---

# CERTIFIED DESIGN IN BOTH PHYSICS, at EXACTLY fixed material budget

The earlier conductivity design result carried a documented blemish: the
material budget was enforced by a soft penalty and drifted ~6%, so part of
the reported gain could in principle have been bought with extra material.
Both design scripts now enforce the budget by PROJECTION — after each
design step a global radius scale is bisected until the rendered volume
fraction returns exactly to target (projected gradient ascent).

| | conductivity | **elasticity** |
|---|---|---|
| objective | guaranteed K_low[0,0] | guaranteed E_x floor |
| initial | 1.9165 | 2.3701 |
| optimised | **2.9769** | **3.2346** |
| **gain in the GUARANTEED value** | **+55.3%** | **+36.5%** |
| certified interval (final) | [2.9769, 3.0011] | [3.2346, 3.3886] |
| final gap | 0.6% | 3.7% |
| budget drift | **0.000** | **+0.00%** |
| validity | Loewner + FV inside | Loewner + rigorous FEM audit |

`examples/run_certified_design.py`, `run_certified_design_elastic.py`.

**Notes.**

1. **The blemish is gone.** Budget drift is now exactly zero in both
   physics, so neither gain can be attributed to extra material. The
   conductivity number moved only slightly (+57.1% -> +55.3%) once the
   budget was held exactly, which retrospectively confirms the earlier
   result was dominated by geometry rather than material.
2. **The elasticity objective has a clean form.** Since
   E(d) >= 1/(a . C_low^{-1} a) and C_low = S_up^{-1}, maximising the
   guaranteed floor is exactly MINIMISING the quadratic form a . S_up a in
   the dual Gram matrix — no inversion inside the optimisation loop.
3. **The certificate tightened during design** (conductivity 12.6% -> 0.6%;
   elasticity 9.0% -> 3.7%), because the trial fields keep training while
   the geometry moves. Optimising the floor and tightening the certificate
   are complementary, not competing.
4. **Self-penalising by construction.** A looser dual field yields a larger
   a . S_up a and hence a LOWER guaranteed floor, so the optimiser cannot
   profit from fooling the surrogate. This is the failure mode that makes
   ordinary surrogate-driven design untrustworthy, and it is structurally
   absent here.

---

# MATCHED-BUDGET COMPARISON — and a correction

The elasticity limits table above was run at 120 steps against
conductivity's 200, and I flagged that the cross-physics comparison was not
like-for-like. Re-running conductivity at 120 steps gives the strict
comparison, and it **falsifies the earlier claim that "elasticity is
uniformly wider than conductivity."**

## Both physics at 120 steps, nq as before, exact integration

| contrast | morphology | conductivity | elasticity | ratio |
|---:|---|---:|---:|---:|
| 10 | disks | 14.9% | 16.3% | 1.09 |
| 10 | laminate | 3.0% | 5.4% | 1.80 |
| 10 | checkerboard | 22.9% | 28.1% | 1.23 |
| 50 | disks | 45.7% | **40.3%** | **0.88** |
| 50 | laminate | 7.1% | 18.9% | 2.66 |
| 50 | checkerboard | 98.2% | 101.1% | 1.03 |
| 200 | disks | 86.0% | **68.8%*** | **0.80** |
| 200 | laminate | 15.0% | 63.5% | 4.23 |
| 200 | checkerboard | **161.2%** | **161.8%** | **1.00** |

`*` = failed the quality gate.

## Corrected findings

1. **"Elasticity is uniformly wider" was an artefact of the unmatched
   budget.** At matched steps, elasticity is *better* than conductivity for
   the disks morphology at 50:1 and 200:1 (ratios 0.88 and 0.80). The
   earlier statement has been superseded; do not cite it.
2. **The laminate is the one morphology where elasticity is genuinely and
   substantially worse** — by 4.2x at 200:1. Plausible cause: the scalar
   laminate corrector is piecewise linear and easy, whereas the elastic
   laminate involves three loading modes with Poisson coupling and an Airy
   dual carrying second derivatives, so the "easy" case stops being easy.
3. **The checkerboard is essentially PHYSICS-INDEPENDENT: 161.2% vs 161.8%
   at 200:1 (ratio 1.00), and 98.2% vs 101.1% at 50:1 (ratio 1.03).** This
   is strong independent support for the proposed mechanism: if the
   bottleneck were the physics (more heads, second derivatives, Poisson
   coupling) the two numbers would differ. They do not. The bottleneck is
   the approximation of corner-singular corrector fields, which is a
   geometric and approximation-theoretic limitation shared by both
   problems. **This is the cleanest evidence in the project for the
   singularity explanation, and it exists only because the budgets were
   matched.**
4. **Training budget matters more than expected at high contrast.**
   Conductivity disks at 200:1 improved from 86.0% (120 steps) to 61.1%
   (200 steps). Any tightness number must therefore be reported WITH its
   budget, and comparisons across methods or physics are meaningless
   without it.

## Methodological note for the paper

This correction is worth reporting rather than quietly fixing: a plausible
and intuitive claim ("the harder physics gives wider bounds") survived
until the budgets were matched, and then failed in two of three
morphologies. It is a concrete argument for the reporting discipline the
paper advocates — matched budgets, quality gates, and medians over
accepted runs.

---

# MULTI-SEED: the final configurations

## Conductivity limits map, 3 seeds (120 steps, exact integration)

| contrast | morphology | median (accepted) | full range | accepted | valid |
|---:|---|---:|---|:--:|:--:|
| 10 | disks | 14.2% | [13.7%, 14.9%] | 3/3 | yes |
| 10 | laminate | 3.0% | [2.9%, 3.2%] | 3/3 | yes |
| 10 | checkerboard | 23.0% | [22.9%, 24.0%] | 3/3 | yes |
| 200 | disks | 86.0% | [79.3%, 92.8%] | 3/3 | yes |
| 200 | laminate | 15.0% | [13.6%, 15.8%] | 3/3 | yes |
| 200 | checkerboard | 161.2% | [161.2%, 161.9%] | 3/3 | yes |

Every cell gate-accepted on every seed. The widest relative spread is disks
at 200:1 (~15% of the median); everything else is within a few percent.
**The limits map is seed-robust and can be reported as medians with ranges.**

## Amortized elasticity — the paper's central claim, 3 independent operators

(20-microstructure family, 300 steps, nq=24, 5 held-out instances each)

| seed | median gap | range | tightening | validity | training |
|---:|---:|---|---:|:--:|---:|
| 0 | 36.5% | [34.4%, 38.2%] | 2.2x | 5/5 | 77 s |
| 1 | 36.7% | [34.1%, 38.4%] | 2.2x | 5/5 | 70 s |
| 2 | 36.4% | [31.7%, 38.5%] | 2.1x | 5/5 | 72 s |

Pooled over 3 seeds x 5 instances: **median gap 36.5%, range [31.7%,
38.5%], tightening 2.2x, validity 15/15** (Loewner + rigorous FEM audit).

(The 36.5% here is wider than the 28.2% reported earlier because this study
used a smaller budget — 20 microstructures and 300 steps versus 40 and 450
— in order to fit three independent trainings. Budgets must always be
quoted with tightness numbers; see the matched-budget section.)

## A finding hiding in the seed statistics: AMORTIZATION REGULARISES

Compare the two training regimes on seed sensitivity:

* **Per-instance** training occasionally collapses. On the laminate at
  nq=64, four seeds gave 0.4-1.7% and one gave 97.8% — a catastrophic
  outlier caught only by the quality gate.
* **Amortized** training did not collapse once. Three independent operators
  agreed to within **0.3 percentage points** of median gap, and no held-out
  certificate in 15 was rejected.

Plausible mechanism: amortized training averages the objective over many
microstructures, so no single instance can drive the network into the
degenerate high-bandwidth configuration that produces a collapse. The
per-instance objective has no such averaging and can be captured by one
pathological trajectory.

**This is a benefit of amortization beyond cost.** The framing so far has
been "certificates at inference cost"; the seed statistics say amortization
also buys training STABILITY. Worth a short subsection — it is measured,
mechanistically plausible, and not something the per-instance prior art can
claim.

---

# REALISTIC MORPHOLOGIES — and a refinement of the mechanism

Four standard microstructure models from the composites literature
(generators added to `certihom/microstructures.py`), nq=64, 120 steps,
exact integration, quality gate, FV cross-check.

| contrast | morphology | phase frac | V-R gap | certified | tightening | gate | FV inside |
|---:|---|---:|---:|---:|---:|:--:|:--:|
| 10 | disks (baseline) | 0.31 | 92.6% | 14.9% | 6.2x | ACCEPT | yes |
| 10 | **spinodal / GRF** | 0.40 | 98.6% | **50.4%** | **2.0x** | ACCEPT | yes |
| 10 | Voronoi polycrystal | 0.46 | 100.2% | 16.5% | 6.1x | ACCEPT | yes |
| 10 | fibre packing (RSA) | 0.25 | 86.3% | 16.7% | 5.2x | ACCEPT | yes |
| 50 | disks (baseline) | 0.31 | 167.3% | 48.8% | 3.4x | ACCEPT | yes |
| 50 | **spinodal / GRF** | 0.40 | 170.4% | **129.5%** | **1.3x** | ACCEPT | yes |
| 50 | Voronoi polycrystal | 0.46 | 171.2% | 56.7% | 3.0x | ACCEPT | yes |
| 50 | fibre packing (RSA) | 0.25 | 163.6% | 43.1% | 3.8x | ACCEPT | yes |

## Findings

1. **Realistic inclusion-type morphologies behave like the toy disks.**
   Non-overlapping fibre packings (16.7%) and Voronoi polycrystals (16.5%)
   are indistinguishable from the disk baseline (14.9%) at 10:1. So the
   synthetic results transfer — the method is not an artefact of
   overlapping-disk geometry.
2. **Triple junctions are NOT the problem.** The Voronoi polycrystal is
   full of three-phase junctions and performs like smooth inclusions. The
   checkerboard's four-phase corner is a genuinely harsher singularity, not
   a generic grain-boundary effect.
3. **NEW LIMITATION FOUND: bicontinuous/spinodal morphologies are hard.**
   50.4% at 10:1 and 129.5% at 50:1 (tightening 1.3x — barely better than
   the trivial analytic envelope). This matters commercially: it is the
   morphology of porous media, aerogels, polymer blends, and battery
   electrodes.
4. **The mechanism claim must be widened.** Earlier the difficulty was
   attributed to corner singularities. The spinodal case has no corners, so
   the correct statement is that difficulty tracks **the complexity of the
   corrector field** — driven both by singular junctions AND by
   percolation with tortuous, multi-scale interfaces. Bicontinuous
   morphologies have a nontrivial corrector everywhere rather than only
   near isolated inclusions.
5. **Everything remained valid and gate-accepted**, with FV inside the
   interval in all eight cases. The wide numbers are honest difficulty, not
   broken runs.

**Revised scoping sentence.** The method delivers 3x-6x tightening for
particulate, fibrous and polycrystalline morphologies at moderate contrast,
and degrades toward the trivial envelope for bicontinuous and
corner-singular morphologies, or at contrasts above ~50:1. The limitation
is approximation of complex corrector fields by smooth networks, not the
certification machinery — validity held in every run reported here.

---

# THE SPINODAL LIMITATION IS PARTLY FIXABLE — and the criterion PRESCRIBES the fix

The bicontinuous/spinodal morphology was the worst case found (50.4% at
10:1). Diagnosis: the GRF has a dominant wavenumber of 6 cycles, but the
network embedding reached only n_harmonics=4 — the trial space could not
represent the microstructure's own length scale.

## Raising trial-space richness alone

| contrast | n_harm | nq | certified | tightening | headroom | gate |
|---:|---:|---:|---:|---:|---:|:--:|
| 10 | 4 | 64 | 50.4% | 2.0x | 2.86 | ACCEPT |
| 10 | 8 | 64 | 36.4% | 2.7x | 2.00 | ACCEPT |
| 10 | 12 | 64 | **29.9%** | 3.3x | **1.48** | **REJECT** |
| 50 | 4 | 64 | 129.5% | 1.3x | 2.86 | ACCEPT |
| 50 | 8 | 64 | 107.4% | 1.6x | 2.13 | REJECT |
| 50 | 12 | 64 | 96.4% | 1.8x | 1.66 | REJECT |

More harmonics tighten the bound monotonically — and simultaneously drive
headroom below the threshold, because the richer trial space outruns the
fixed quadrature rule.

## Raising BOTH together, as headroom = (nq/2)/B demands

| contrast | n_harm | nq | certified | tightening | headroom | gate |
|---:|---:|---:|---:|---:|---:|:--:|
| 10 | 4 | 64 | 50.4% | 2.0x | 2.86 | ACCEPT |
| 10 | 12 | 64 | 29.9% | 3.3x | 1.48 | REJECT |
| 10 | 12 | **128** | **33.2%** | **3.0x** | **3.16** | **ACCEPT** |
| 50 | 4 | 64 | 129.5% | 1.3x | 2.86 | ACCEPT |
| 50 | 12 | 64 | 96.4% | 1.8x | 1.66 | REJECT |
| 50 | 12 | **128** | **100.8%** | **1.7x** | **3.47** | **ACCEPT** |

## Findings

1. **The worst-case morphology improves substantially and honestly**:
   50.4% -> 33.2% at 10:1 (tightening 2.0x -> 3.0x) and 129.5% -> 100.8%
   at 50:1, with the quality gate satisfied in both cases.
2. **The gate separates a GENUINELY tighter bound from an ARTIFICIALLY
   tight one.** At 10:1 the (12, 64) configuration reported 29.9% but was
   rejected; the honest (12, 128) configuration reports 33.2%. The 3.3
   percentage points between them are pure quadrature illusion — and the
   gate identified this with no reference solution. This is the clearest
   demonstration in the project of why the criterion is needed: without it,
   29.9% would have been published.
3. **The criterion is PRESCRIPTIVE, not merely diagnostic.** It does not
   just flag bad runs; it tells you how to configure: choose n_harmonics
   from the microstructure's dominant wavenumber, then choose nq to keep
   headroom >= 2. Trial-space richness and quadrature resolution are
   COUPLED requirements, and tuning either alone is a mistake.
4. **The limitation is reduced, not eliminated.** Spinodal at 33.2% is
   still worse than disks at 14.9%. Bicontinuous morphologies remain the
   hardest realistic class, and should be reported as such.

**Practical recipe for the paper.**
(i) estimate the microstructure's dominant wavenumber k;
(ii) set n_harmonics >~ 2k;
(iii) measure the trained field's bandwidth B and set nq >= 4B;
(iv) re-check the gate and iterate if it fails.


---

# INDEPENDENT REPRODUCTION (Kaggle CPU, collaborator)

The Voigt-Reuss tightening study was re-run independently on different
hardware. Cross-machine agreement, same seeds:

| microstructure | V-R width | certified | tightening |
|---|---:|---:|---:|
| disks (held out) | 83.0% | 20.8% | 4.0x |
| disks (held out) | 89.6% | 25.8% | 3.5x |
| disks (held out) | 94.0% | 33.7% | 2.8x |
| laminate f=.25 (OOD) | 86.3% | 21.1% | 4.1x |
| checkerboard (OOD) | 100.6% | 43.2% | 2.3x |

**Median tightening 3.5x** (sandbox run gave 3.4x; the difference is BLAS
threading noise). Identical-volume-fraction pair reproduced: laminate
f=0.50 at 18.3% and checkerboard at 43.2%, both against a 100.6% envelope
(sandbox: 18.5% and 42.4%).

The manuscript quotes the Kaggle reproduction, since it is the later and
independently executed run. Both are regenerable from
`examples/run_vs_voigt_reuss.py`.


## Collaborator's GPU run — full per-instance record

Kaggle T4, float32 training + float64 CPU certification, 200 training
microstructures, 4000 steps, batch 16, 12 held out:

| instance | gap | Loewner | FEM audit | V-R width | tightening | t |
|---|---:|:--:|:--:|---:|---:|---:|
| 1 | 29.0% | yes | yes | 80.9% | 2.8x | 0.11 s |
| 2 | 29.0% | yes | yes | 77.6% | 2.7x | 0.06 s |
| 3 | 21.3% | yes | yes | 71.4% | 3.3x | 0.06 s |
| 4 | 45.5% | yes | yes | 88.7% | 2.0x | 0.06 s |
| 5 | 26.9% | yes | yes | 84.8% | 3.2x | 0.06 s |
| 6 | 25.8% | yes | yes | 77.6% | 3.0x | 0.06 s |
| 7 | 16.4% | yes | yes | 63.1% | 3.8x | 0.06 s |
| 8 | 27.7% | yes | yes | 73.7% | 2.7x | 0.06 s |
| 9 | 31.8% | yes | yes | 83.6% | 2.6x | 0.06 s |
| 10 | 31.8% | yes | yes | 85.8% | 2.7x | 0.05 s |
| 11 | 39.1% | yes | yes | 89.2% | 2.3x | 0.07 s |
| 12 | 11.2% | yes | yes | 56.2% | 5.0x | 0.06 s |

**median gap 28.3%, range [11.2%, 45.5%]; median tightening 2.7x, range
[2.0x, 5.0x]; validity 12/12; 62 ms per certificate.** Training 878 s.

Identical-volume-fraction pair, elasticity (collaborator's 450-step CPU
run): laminate f=0.50 at **60.1%** and checkerboard at 52.8%, both against
a 90.1% envelope. The sandbox run of the same script gave 58.6% and 52.8%;
the manuscript quotes the collaborator's values, being the independently
executed run.

---

# ARCHITECTURE ABLATION (closes referee objection R2.2)

The amortized plateau was previously reported without identifying its
cause. Sweeps at fixed budget (14 microstructures, 150 steps, nq=24,
elasticity, 4 held out):

| sweep | setting | parameters | median gap |
|---|---|---:|---:|
| width (depth 3) | 32 | 5,609 | 35.7% |
| | 64 | 19,401 | 34.5% |
| | **96** | **41,385** | **31.8%** |
| depth (width 64) | 2 | 11,081 | 35.2% |
| | 3 | 19,401 | 34.5% |
| | **4** | **27,721** | **32.6%** |

Conditioner richness, tested separately on 16-pixel microstructures where
higher modes are representable (width 64, depth 3, 120 steps):

| conditioner kmax | modes | median gap |
|---:|---:|---:|
| 3 | 27 | **25.1%** |
| 6 | 90 | 26.5% |

**The bottleneck is network capacity, not the geometry encoding.** Width
and depth both improve the certified width monotonically, while a
threefold richer conditioner does not help at all (marginally worse, within
noise). This supersedes the earlier statement that the limiting factor was
unidentified: the GPU run used width 64 and depth 3, in the middle of the
range where capacity still helps, which is why nine times the optimisation
budget changed nothing.

A practical constraint surfaced during the sweep and is worth recording:
the conditioner requires kmax < n/2 for an n-pixel microstructure, so
8-pixel geometries admit at most kmax = 3. Coarse microstructure grids
therefore cap the conditioning richness independently of the network.

# SCREENING DEMONSTRATION (closes referee objection R2.1)

The objection: what decision does a 28% interval support? Answer: a
provable one. Given a requirement, a certified interval decides a candidate
whenever the requirement falls outside it. Pool of 40 unseen
microstructures, screened by the amortized operator in 0.4 s total
(10 ms each):

| requirement | PASS | FAIL | undecided | decided | wrong decisions |
|---:|---:|---:|---:|---:|---:|
| 1.6 | 25 | 2 | 13 | 68% | **0** |
| 1.8 | 17 | 8 | 15 | 62% | **0** |
| 2.0 | 11 | 14 | 15 | 62% | **0** |
| 2.2 | 10 | 16 | 14 | 65% | **0** |
| 2.4 | 6 | 19 | 15 | 62% | **0** |

PASS means the certified lower bound already meets the requirement; FAIL
means the certified upper bound cannot. **Between 62% and 68% of the pool
is decided provably, and not one decision is contradicted by the
finite-volume reference** — as must be the case, since the decisions follow
from bounds that provably contain the truth.

The remaining third is not a failure but an honest referral: those
candidates require per-instance certification, which reaches widths below
1%. A point-estimate surrogate, however accurate, cannot make either
statement---it can rank candidates but never prove one acceptable.

---

# REAL SEGMENTED MICROGRAPH (closes referee objection R3.2)

A published segmented micrograph (three-panel figure; the right panel is
the binary segmentation) was cropped square, downsampled to 32x32, and its
geometry used directly as the microstructure. Phase fraction 0.802,
nominal contrast 10. `examples/run_micrograph.py`.

**What this claims:** the method operates on genuine micrograph
geometry---rough interfaces, clustering, percolating veins---rather than on
procedurally generated shapes. **What it does not claim:** anything about a
physical specimen. No measured property is used; the certificate bounds the
idealised periodic two-phase problem posed on this geometry.

## Applying the paper's own prescription

References: FV 6.0477, FFT (Moulinec--Suquet) 6.8094; Voigt--Reuss width
78.3%.

| nq | certified gap | tightening | headroom | gate | K_low | K_up | brackets FV / FFT |
|---:|---:|---:|---:|:--:|---:|---:|:--:|
| 32 | 58.6% | 1.3x | 0.87 | **REJECT** | 5.5457 | 7.6734 | yes / yes |
| 64 | 34.6% | 2.3x | 1.83 | **REJECT** | 5.4652 | 7.7238 | yes / yes |
| 128 | 36.8% | 2.1x | **3.72** | **ACCEPT** | 5.3732 | 7.7692 | yes / yes |

Three observations.

1. **The gate fired on real data, twice, before the certificate could be
   trusted.** At nq=32 the headroom is 0.87---below one---and the reported
   gap of 58.6% is not to be believed; at nq=64 it is still below the
   threshold. Only at nq=128 is the run acceptable, and the accepted
   certificate (36.8%) is slightly *wider* than the rejected nq=64 value
   (34.6%), exactly the pattern seen in the synthetic prescription study:
   the rejected number was flattered by under-resolved quadrature.
2. **Validity never failed**, at any resolution, including both rejected
   runs: all six certificates bracket both independent references.
3. **A periodicity convention was tested, not assumed.** Taking the crop
   directly as a unit cell gives 36.8%; reflecting it into a genuinely
   periodic cell of twice the side gives 60.4% at headroom 3.56. The
   mirrored cell is harder because reflection doubles the domain and the
   corrector must resolve twice as much structure. Both are valid; the
   direct convention is reported as primary and the mirrored figure as a
   sensitivity check, since neither is a property of the specimen.

The certified width of roughly 37% places this micrograph between the
inclusion morphologies and the bicontinuous class of
Section~\ref{sec:limits}, which is consistent with its appearance: a
percolating network of veins through a majority phase.

## Micrograph, second instance (collaborator's image, higher resolution)

The image is 504x504 after border removal, white-phase fraction 0.794,
with an autocorrelation half-width of **4 pixels** — roughly 126 features
across the cell. A 32x32 downsample therefore does NOT resolve the
microstructure, and the numbers obtained at that resolution describe a
different (aliased) geometry.

At resolutions that resolve the features (contrast 10, 250 steps,
n_harmonics 8, exact integration):

| grid | nq | gap | tightening | headroom | gate | K_low | K_up | FV | FFT | brackets |
|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|:--:|
| 32 | 32 | 19.6% | 3.7x | 1.67 | **REJECT** | 6.2901 | 7.6965 | 6.5962 | 7.0421 | yes |
| 64 | 64 | 47.5% | 1.7x | 1.35 | **REJECT** | 4.8202 | 7.7589 | 6.0720 | 6.6492 | yes |
| **64** | **128** | **49.7%** | **1.6x** | **2.78** | **ACCEPT** | 4.7368 | 7.7833 | 6.0720 | 6.6492 | yes |

**Three lessons, all of which belong in the paper.**

1. **The gate rejected the flattering number twice.** The 19.6% obtained
   on the 32-grid is the tightest figure in the table and the least
   trustworthy: it comes from an aliased geometry AND fails the headroom
   check. The honest, gate-accepted certificate is **49.7%**.
2. **Resolution is a modelling choice, not a numerical detail.** The 32-
   and 64-grids are different microstructures, not two discretisations of
   one: at 32 the 4-pixel features are lost entirely. Certified widths must
   be quoted with the resolution at which the geometry was represented, and
   a certificate on an under-resolved image bounds the wrong problem.
3. **Validity held everywhere**, including both rejected runs: all three
   intervals contain both reference solvers.

This morphology — a fine, percolating, bicontinuous network — is the
hardest realistic class identified in Section~\ref{sec:limits:realistic},
and 49.7% is consistent with the synthetic bicontinuous result (50.4%).
The method's own limits map predicted the outcome on an image it had never
seen, which is a stronger validation of the map than any single tight
number would have been.

**A note on the references.** FV gives 6.0720 and FFT 6.6492, a 9.5%
disagreement between two independent solvers on this geometry. The
certified interval contains both, as it must. Neither reference is
converged here, which is worth stating plainly: on real micrographs the
"reference" is itself an estimate, and only the certified interval carries
a guarantee.

## Micrograph, done correctly: NATIVE-RESOLUTION CROPS

The two attempts above both downsampled the full 504-pixel frame, and the
requirements check (`examples/check_micrograph_requirements.py`) shows that
BOTH were aliased: with a 4-pixel correlation half-width, features span
0.5 px on a 64-grid. The certified numbers there (19.6%, 49.7%) therefore
bound geometries that are not the one in the image.

The correct procedure preserves resolution and reduces the field of view.
Native-resolution crops, no downsampling, contrast 10, 220 steps:

| crop | feature px | corr. lengths | fraction | nq | gap | tightening | headroom | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 48x48 | 3 | 16 | 0.758 | 96 | **9.6%** | **8.9x** | 3.97 | **ACCEPT** |
| 64x64 | 3 | 21 | 0.810 | 128 | **10.5%** | **7.3x** | 4.76 | **ACCEPT** |

**The honest certified width on this micrograph is about 10%, not 50%.**
Preserving the geometry improves the certificate fivefold, and both runs
pass the quality gate.

Reference audit for the 48x48 crop, as full matrices:

```
K_low = [[6.1748, 0.1029], [0.1029, 5.0405]]
K_up  = [[6.6921, 0.1297], [0.1297, 5.6491]]
FV(h)    [[6.3801, 0.0992], [0.0992, 5.2144]]   INSIDE
FV(h/2)  [[6.4379, 0.1002], [0.1002, 5.2865]]   INSIDE
FFT      [[6.4757, 0.1032], [0.1032, 5.3395]]   INSIDE
```

All three references lie strictly inside, and the two solvers now agree to
1.5% (against 9.5% on the aliased grid) — under-resolved geometry was
degrading the references too.

**A test error caught here and worth recording.** The first version of the
bracket check compared the certificate against a fabricated isotropic
matrix diag(K11, K11). Real micrographs are anisotropic (here K11 = 6.38
against K22 = 5.21, with nonzero off-diagonals), so that test reported
spurious failures. The Loewner comparison must use the FULL reference
matrix. The certificates were correct throughout; the test was not.

**Requirements for certifying a microstructure image** are now checked by
`examples/check_micrograph_requirements.py`: segmentation, feature
resolution (>= 3 px per feature ON THE CERTIFICATION GRID), square
periodic-compatible cell, representativeness (>= 10 correlation lengths),
assigned phase contrast, and post-hoc quadrature headroom.

## Two micrographs, four windows, one protocol

A second segmented micrograph was supplied independently. Applying the
requirements check first: correlation half-width 4 px in a 504-px frame, so
a 32-grid gives 0.3 px per feature — aliased, exactly as before. Certified
on native-resolution windows instead (48x48, nq=96, 200 steps, contrast 10):

| image | window | fraction | gap | tightening | headroom | gate | K_low[0,0] | K_up[0,0] | refs inside |
|---|---:|---:|---:|---:|---:|:--:|---:|---:|:--:|
| binary | (0,0) | 0.758 | 9.9% | 8.6x | 4.00 | ACCEPT | 6.1570 | 6.7013 | yes |
| binary | (240,240) | 0.824 | 9.9% | 7.5x | 3.95 | ACCEPT | 6.9482 | 7.5014 | yes |
| binary_2 | (0,0) | 0.792 | 9.7% | 8.3x | 4.10 | ACCEPT | 6.7146 | 7.2813 | yes |
| binary_2 | (240,240) | 0.821 | 9.3% | 8.1x | 3.88 | ACCEPT | 6.8206 | 7.3621 | yes |

**Certified widths of 9.3--9.9% and tightening of 7.5--8.6x across two
independent images and four windows**, every run gate-accepted with
headroom near four, every certificate containing the finite-volume
reference. The aliased 32-grid runs on the same images reported 21.7% and
19.6% — tighter-looking numbers for geometries that were not photographed.

Two observations worth carrying into the paper.

1. **The protocol is reproducible across images and windows.** The spread
   in certified width is under a percentage point, while the certified
   VALUES differ substantially between windows (K_low[0,0] from 6.16 to
   6.95, a 13% spread). The method's precision is stable; the material's
   local property is not. That is the expected signature of a statistical
   volume element smaller than a true RVE, and it is visible only because
   each window carries its own certificate.
2. **Real micrographs certify better than the synthetic bicontinuous
   class** (9--10% against 50.4%), once resolved. The synthetic generator
   produces a finer, more tortuous structure than these images; the
   micrographs are closer to a dense particulate morphology with occasional
   percolating veins, which the limits map places in the favourable regime.

## Berea sandstone (digital rock physics benchmark)

A segmented slice of Berea sandstone, the standard benchmark of the digital
rock physics literature. Native resolution 994x994, grain fraction 0.787,
correlation half-width 6 px, so roughly 166 correlation lengths across the
full slice — genuinely representative, unlike the crops used earlier.

Phases: grain 10, pore 1. The pore phase is REGULARISED rather than void;
a true void breaks the dual bound (Section 11), so this certificate is for
a two-phase conductive problem, not for permeability.

Native-resolution windows, exact integration:

| window | corr. lengths | fraction | nq | gap | tightening | headroom | gate | K_low | K_up | FV inside |
|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|:--:|
| 48 | 10 | 0.444 | 96 | 11.2% | 8.9x | 3.99 | ACCEPT | 3.1766 | 3.5205 | yes |
| 72 | 14 | 0.608 | 144 | 13.3% | 7.4x | 5.98 | ACCEPT | 4.5243 | 5.0895 | yes |

### RVE study — the result the four-window observation was pointing at

Three disjoint windows per size, native resolution, all gate-accepted:

| window | corr. lengths | certified width (median, range) | certified K_low[0,0] (median, range) | value spread |
|---:|---:|---|---|---:|
| 32 | 5 | 12.1% [8.8%, 12.4%] | 2.710 [2.011, 3.033] | 1.51x |
| 48 | 8 | 9.5% [7.9%, 14.2%] | 4.929 [3.115, 5.317] | 1.71x |
| 64 | 11 | 9.8% [8.8%, 16.4%] | 6.069 [4.334, 6.317] | 1.46x |

**The two quantities behave completely differently, and separating them is
the point.**

* **Certified width is roughly constant at 9--12%** across a fourfold
  change in window area. It is a property of the METHOD on this
  morphology, and it does not improve with more material.
* **Certified value rises steeply and stays highly variable**: the median
  K_low[0,0] more than doubles from 2.71 to 6.07 between windows of 5 and
  11 correlation lengths, and windows of the same size still differ by
  factors of 1.5--1.7. At 11 correlation lengths this slice is nowhere near
  a representative volume for conductivity, despite the full image
  containing 166.

This is a statement a point-estimate surrogate cannot make. Given four
different numbers from four windows, an uncertified method cannot say
whether the variation is the material or the model. Here each window
carries its own guarantee, so the ~10% width is attributable to the method
and the 1.5x spread is attributable to the material — a decomposition the
certificate provides for free.

**Practical consequence.** Reporting a single effective property from one
window of a micrograph is unsafe at these window sizes regardless of
solver accuracy, and the certificate makes the unsafety visible rather than
silent.
