# TASKS — work plan for the researcher

Ordered by dependency. Each task has an **acceptance criterion**: something
that either passes or doesn't, so you know when you're done. Time estimates
assume you are new to the codebase.

---

## Block 0 — Own the machinery (do not skip) · ~1 week

You cannot defend this work at review unless you can break it on purpose.

### T0.1 Reproduce everything (2 h)
Run `pytest -q` and all five example scripts. Record wall-clock times on
your hardware in a new file `REPRO.md`.
**Accept:** 24/24 tests pass; your numbers for the checkerboard and
elasticity demos match RESULTS.md to within ~1% (small drift is fine —
different BLAS, different thread counts).

### T0.2 Falsification drills (1 day)
For each drill: **write your prediction first**, then run it, then revert.
A drill you predicted correctly proves you understand that mechanism; a
drill you got wrong is the thing to go study.

| # | Break this | Predict which tests fail and WHY |
|---|---|---|
| a | `energy.py: dual_fields` — remove the rotation (use the raw gradient instead of the curl) | ? |
| b | `fields.py: FieldPair` — delete the zero-init of `head` | ? |
| c | `certify.py: certified_bounds_scalar` — use 1×1 Gauss instead of 2×2 | ? |
| d | `bfs.py` — drop the `phi_,xy` DOF (set it to zero) | ? |
| e | `quadrature.py` — use `nq` NOT a multiple of `n` (e.g. 30 with n=16) | ? |

**Accept:** a table in `REPRO.md` with your prediction, the actual failure,
and one sentence on the mechanism for each. Drill (a) should destroy
validity, not just tightness — make sure you can say why.

### T0.2b Multi-seed the headline numbers (half a day) — START HERE
Every number in RESULTS.md was single-seed. Run the harness:

```
python examples/run_multiseed.py --seeds 5 --steps 400
```

**Status: a 3-seed / 120-step smoke run at standard settings came back
stable** — checkerboard 7.3% (spread 0.0%), laminate 1.1% (0.1%), disks
12.6% (0.9%), validity 3/3 on all cases. So this is a confirmation task,
not a rescue.

One caveat worth chasing: an earlier probe using `n_harmonics=3` on a
coarser n=8 laminate produced gaps of 50.8% / 1.1% / 148.8% across seeds.
Under-parameterized trial spaces optimize unstably even though they stay
valid. Worth one paragraph in the paper and one figure in the ablations
(T3.2): tightness volatility vs. `n_harmonics`.

**Accept:** RESULTS.md tables carry medians and ranges over >=5 seeds;
validity is 100% everywhere; you can name the volatility threshold.

### T0.3 Re-derive the mathematics by hand (2 days)
On paper, without looking at MATH.md, derive:
1. the primal Gram bound `K_up ⪰ K_eff` (why does linearity of the trial
   map in the load matter?);
2. the dual bound and why `A ⪰ B ≻ 0 ⟹ B⁻¹ ⪰ A⁻¹`;
3. why `curl ψ` has exactly zero mean for periodic ψ;
4. the BFS traction-continuity argument (§11) — this is the one a reviewer
   will probe hardest;
5. the unit-mean-stress normalization lemma (§9b) from scratch.

**Accept:** your derivations agree with MATH.md, and you found at least one
place where MATH.md is terse enough that you'd expand it for the paper.

---

## Block 0.5 — RE-CERTIFY EVERYTHING (new, urgent) · ~2 days

A session experiment found that networks trained against a fixed midpoint
rule can hide energy BETWEEN quadrature points, producing upper bounds
BELOW the exact tensor. Measured at ordinary settings (laminate c6,
nq=32): K_up[0,0] = 2.5207 vs true 2.6667. See RESULTS.md, "CRITICAL
FINDING".

### T0.5.1 Re-certify every reported number with exact integration
Any bound you cite must come from `certify.py` / `bfs.py`, never from the
midpoint rule at training resolution.
**Accept:** every table in RESULTS.md carries exact-integration values;
"quad" columns are relabelled as training diagnostics.

### T0.5.2 Add the refinement diagnostic to every study
Evaluate each trained field at nq, 2nq, 4nq. A large jump in K_up means the
training resolution was hiding energy.
**Accept:** a diagnostic column in every results table, and a stated rule
for when nq is high enough relative to network capacity.

### T0.5.3 Understand WHEN it happens — FIRST PASS DONE
`certihom/diagnostics.py` + `examples/run_quadrature_rule.py` now measure
spectral bandwidth B, Nyquist nq/2, headroom, and the refinement ratio.
Measured: headroom < 1 is catastrophic (ratios 10x-3500x); headroom ~10 was
needed for a strictly valid midpoint bound; exact integration removes the
requirement. Bandwidth is NON-MONOTONIC in n_harmonics (minimum at 4).

**Your remaining job here — and this may be the most citable thing in the
project:**
* extend the sweep to width, depth, steps, lr. (The elasticity check is
  DONE and the guess was wrong: the Airy dual is SMOOTHER than the primal,
  bandwidth 3.0 vs 6.0, because complementary energy penalises a mode like
  |xi|^4 instead of |xi|^2. Phase 2 was valid at every resolution. Use this
  as a worked example of a falsified prediction in the paper — it is
  evidence the diagnostics are informative rather than decorative.)
* test whether the bandwidth minimum at n_harmonics=4 is problem-specific
  or general;
* connect it to seed volatility — FIRST PASS DONE
  (`examples/run_harmonics_mechanism.py`): weight inflation predicts
  bandwidth (+0.90) and spread (+0.60), with a sharp common optimum at
  n_harmonics=4, but bandwidth and spread are not interchangeable (+0.30).
  Rerun with >=8 seeds and a finer harmonic grid before claiming
  correlations; n=5 settings is thin evidence;
* state the guideline as a recommendation other PINN practitioners can
  apply without reading the rest of the paper.
**Accept:** a figure (headroom vs refinement ratio, all configurations,
log scale) and a one-sentence rule.

## Block 1 — The conformity-trap figure · ~3 days
**Status: partially done, and the result SOFTENED the claim.** Lookup
conditioning was ~2x looser but produced no observed violations
(`examples/run_conformity_trap.py`). Your job is to either (a) construct a
sharper adversarial case that does violate, or (b) establish that
non-conformity reliably costs tightness without breaking validity in
practice, and phrase the paper accordingly. Either outcome is publishable;
overclaiming is not.


Right now MATH.md §10 *asserts* that pixel-lookup conditioning destroys
validity. Assertions get challenged; measurements don't.

### T1.1 Build the broken conditioner
Add `LookupConditioner` to `amortized.py`: features = k at the query point
and at 4 neighbour offsets (±h, 0), (0, ±h), via nearest-pixel lookup —
i.e. piecewise constant in y, discontinuous across pixel edges.

### T1.2 Demonstrate the failure
Train both conditioners identically on the same disk family. For held-out
instances compute bounds with **exact integration** (`certify.py`), and
audit against exact laminates and refined FV.

**Accept:** a table/figure showing the smooth conditioner is valid on 100%
of instances while the lookup conditioner produces measurable *violations*
(reference outside the interval). If the lookup version does NOT violate,
that is a real finding too — report it honestly and investigate why (hint:
check whether the Q1 projection is silently repairing the discontinuity,
and if so, whether the un-projected midpoint bound violates instead).

**Why it matters:** this converts a theoretical caveat into the paper's
sharpest methodological warning, and no prior work has had to face it.

---

## Block 2 — The next real contribution: HS polarization · ~3–4 weeks

The contrast table gives you a target to beat: **27.8% at contrast 50 and
58.6% at contrast 200.**

### T2.1 Derive the Hashin–Shtrikman variational principle with a neural
polarization field (comparison medium k₀, polarization τ(y) = (k − k₀)e).
Work out which sign of (k − k₀) gives an upper vs. lower bound.

### T2.2 Implement `hs.py` with the same architectural-admissibility
discipline as the rest of the codebase (no soft penalties).

### T2.3 Reproduce the contrast sweep with the HS variant.
**Accept:** at contrast ≥50 the HS bounds are *strictly tighter* than the
current primal–dual pair on the same geometry and compute budget, with
machine-precision anchors (laminate, and the HS bounds must respect the
classical analytic HS bounds for two-phase isotropic composites).
**If it does not beat the current pair, that is publishable too** — report
the crossover contrast and explain it.

---

## Block 3 — Studies that fill obvious reviewer gaps · ~2 weeks

### T3.1 RVE / window-size study
Fixed inclusion size, growing cell (n = 16, 32, 48, 64), ≥10 realizations
each. Plot certificate width and bound spread vs. window size.
**Accept:** you can state at which window size the certified interval
stops shrinking, and connect it to Kanit-style RVE determination.
**Note:** use the amortized operator for throughput — but first verify it
stays valid at resolutions it was not trained on (it should; the
conditioner is resolution-agnostic — confirm, don't assume).

### T3.2 Ablations a reviewer will demand
Dual-network capacity ratio; `n_harmonics`; conditioner `kmax` and `sigmas`;
Adam vs. L-BFGS polish on the final iterate.
**Accept:** one figure per knob, each with a one-line takeaway.

### T3.3 Cost-to-accuracy curves vs. FFT/FEM
The "why not just use FFT?" question needs a plot, not a paragraph.
**Accept:** certified width vs. wall-clock for (i) per-instance training,
(ii) amortized inference, (iii) FFT/FEM at matched cost, in the many-query
regime.

---

## Block 4 — Scale and extend · ~4–6 weeks

### T4.1 Get GPU access. Re-run Phase 3 with ≥1000 training microstructures,
larger grids, and a CNN/FNO encoder ablated against the current Fourier
features. **Accept:** amortized gap at contrast 10 below the current 28%.

### T4.2 3D elasticity: Beltrami potentials + a C¹ (or Hermite-type) 3D
projection. Start with the *primal* 3D bound only — it needs no new theory
and gives you a result while the dual side is in progress.

### T4.3 Amortized elasticity at production size (currently only a smoke
test exists).

---

## Block 5 — Write it · ~4 weeks, overlapping

### T5.1 Related-work delta (do this EARLY — it may change your framing)
Read and take notes on: Gaynutdinova, Doškář, Rokoš & Pultarová
(arXiv:2509.07579 — closest prior art), Milton *The Theory of Composites*
ch. 9 & 23, Hashin & Shtrikman (1962, 1963), Prager & Synge (1947),
Dvořák & Haslinger, Vondřejc et al. (Fourier–Galerkin guaranteed bounds),
Moulinec & Suquet (1994/98). **Verify every citation yourself** — do not
trust any bibliography you did not check, including mine.
**Accept:** a 2-page related-work section with an explicit delta table
(scalar→tensor, per-instance→amortized, quadrature-caveated→exact).

### T5.2 Draft around the four tables already in RESULTS.md
(contrast/certification, amortization economics, OOD validity, design).

### T5.3 Reviewer-proofing pass
Write the three harshest referee reports you can against your own paper,
then fix what you can and pre-empt the rest in the limitations section.

---

## Standing rules

1. **Never report a number you did not personally regenerate.**
2. Every figure regenerates from a seeded script, one command, no manual steps.
3. If a result surprises you, assume a bug until you have falsified that
   assumption. The FV-outside-the-interval finding at contrast 200 looked
   like a bug and turned out to be a result — but only checking told us.
4. Keep the machine-precision anchors green at all times. If a refactor
   breaks them, the refactor is wrong.
5. Authorship follows contribution. Blocks 1–5 are the science; make sure
   the work in them is genuinely yours.
