# PAPER OUTLINE — every section mapped to a verified result

Positioning CONFIRMED against the full text (all five checks passed).
Target: CMAME primary; IJNME / Computational Mechanics as Q1 fallbacks.

---

## The locked positioning

> Gaynutdinova et al. (IJNME 2026) established guaranteed primal–dual
> bounds for per-instance PINN and VPINN homogenization of 2D steady-state
> scalar conductivity, with certification obtained by projecting neural
> solutions onto conforming P1 finite elements and performing exact
> variational integration. Their framework cannot certify unseen
> microstructures without retraining, because every network is optimized
> for a single representative volume element. We contribute an amortized
> framework producing inference-time variational certificates; demonstrate
> certificates substantially tighter than volume-fraction-only Voigt–Reuss
> bounds while distinguishing morphologies sharing identical volume
> fractions; extend the framework **to** tensor-valued elasticity in 2D by
> carrying the classical Airy/BFS dual into neural trial spaces; and
> introduce a bandwidth-versus-Nyquist criterion that predicts
> quadrature-induced overfitting before certification deteriorates.

Three-way map to state in the introduction:

| | certificate | scope |
|---|---|---|
| Gaynutdinova et al. (IJNME 2026) | variational, tight | **per-instance**, scalar |
| Voigt–Reuss Net (GAMM-M 2025 / IJNME 2026) | **amortized**, but volume-fraction-only | tensor elasticity |
| **this work** | **amortized AND per-instance-tight** | scalar + tensor, 2D |

---

## Section plan

### 1. Introduction
Claim the three-way gap above explicitly, in the second paragraph, naming
both competitors. Do NOT bury either in related work.

### 2. Primal–dual certificates (background, cite generously)
Prager–Synge hypercircle; Dvořák–Haslinger; Vondřejc/Zeman Fourier–Galerkin
exact integration; Gaynutdinova et al. for the neural per-instance case.
**Explicitly credit their P1-projection certification.**
→ MATH.md §§2–3.

### 3. Architectural admissibility
Periodic primal fluctuations; scalar dual by stream function (credit their
construction); **tensor dual by Airy potential**, crediting PAMM 2023
(Gaynutdinova, Ladecký, Pultarová & Zeman) for the classical C¹ BFS dual.
→ MATH.md §§2–3, 9, 9b.
**Table 1 — verification ladder:** laminate & checkerboard to ~1e-11 in
both physics; Q1 and BFS projections machine-precision anchored.

### 4. Exact-integration certification
Q1 for primal and scalar dual; **BFS bicubic Hermite C¹ for the Airy dual**
— the C¹ continuity argument is the technical contribution here.
→ MATH.md §11.
**Table 2 — certification price:** 0.1–3.0 points of width across contrasts
2–200 (RESULTS.md contrast/certification study).

### 5. Amortization  ← THE CORE CONTRIBUTION
Conformity-preserving conditioning (MATH.md §10 — the smooth in-graph
Fourier features, and why pixel lookups would void the theorem).
**Table 3 — conductivity:** 8/8 held-out valid, 19 ms/certificate, ~1340×
faster than per-instance at ~2.7× the width.
**Table 4 — elasticity:** 12/12 valid on GPU, ~62 ms, median gap 28.3%;
3 independent operators agree within 0.3 points, 15/15 valid.
**Figure — OOD:** validity against exact tensors under family shift and a
10× contrast shift.

### 6. Tightness vs the Voigt–Reuss envelope  ← THE COMPETITIVE DEFENCE
**Table 5:** median 3.5× (conductivity), 2.2–2.7× (elasticity) on unseen
microstructures.
**Table 6 — the argument that settles it:** identical volume fraction
(0.500) gives an identical Voigt–Reuss envelope, but certified widths of
18.3% (laminate) vs 43.2% (checkerboard). A volume-fraction-only guarantee
provably cannot distinguish these.

### 7. Quadrature safety  ← THE TRANSFERABLE CONTRIBUTION
Credit Berrone et al. (2022) for proving collocation losses can be false,
and Gaynutdinova et al. for observing integration overfitting empirically.
**Our delta: prediction and prescription, not discovery.**
**Table 7:** headroom < 1 → refinement ratios 10×–3500×.
**Table 8 — the money table:** (n_harm 12, nq 64) reports 29.9% and is
REJECTED; (12, 128) reports 33.2% and is ACCEPTED. The 3.3 points between
them are quadrature illusion, identified with no reference solution.
**Quality gate:** per-run, label-free; on the laminate, four seeds at
headroom ~4.9 and one at 0.78 with a 97.8% gap — flagged without knowing
the answer.
**Prescriptive recipe:** n_harmonics ≳ 2×(dominant wavenumber), then
nq ≥ 4B; the two are coupled.

### 8. Limits — publish the bad cases
**Table 9 — contrast × morphology, both physics, seeded and gated.**
Key numbers: laminate 3.0% → 15.0% (10:1 → 200:1); checkerboard 23.0% →
161.2%; spinodal the hardest realistic class (50.4% → 33.2% once
n_harmonics and nq are coupled correctly).
**Mechanism:** difficulty tracks corrector-field complexity — singular
junctions *and* percolation with multi-scale interfaces. Evidence: the
checkerboard is **physics-independent** (161.2% vs 161.8%), which rules out
the extra elastic modes as the cause.
**Realistic morphologies:** fibre packing and Voronoi behave like disks;
triple junctions are NOT the problem.

### 9. Applications
**9.1 Certified design.** +55.3% (conductivity) and +36.5% (elasticity)
guaranteed floors at *exactly* fixed material budget (projection, 0.00%
drift). Self-penalising: a looser dual gives a LOWER floor.
**9.2 Solver audit.** At contrast 200 the FV reference sat below the
certified lower bound; refinement confirmed the reference was
under-converged. A rigorous bound as a convergence diagnostic for
conventional solvers.

### 10. Cost, honestly
**Table 10:** FFT is ~50× faster per query; crossover at ~300 designs.
Include the rows where FFT wins. Claim certificates + differentiability +
audit capability, never raw speed.

### 11. Limitations (write this before the conclusion, and mean it)
2D only; corner-singular and bicontinuous morphologies; contrasts above
~50:1 in elasticity; GPU scale-up gave no improvement (plateau at step
1200) so the bottleneck is architecture, not compute; off-diagonal entries
not guaranteed by the diagonal Loewner ordering (as PAMM 2023 also notes).

---

## Figure list (all regenerate from seeded scripts)

1. Method schematic: primal/dual architecture → Gram → certificate
2. Verification ladder (machine-precision anchors)
3. Amortization: gap vs inference cost, with the per-instance comparison
4. Voigt–Reuss tightening + the identical-volume-fraction pair
5. Headroom vs refinement ratio, log scale, all configurations
6. Contrast × morphology heatmap, both physics
7. Certified design: before/after microstructure and guaranteed floor

## Writing rules
1. Never a number you did not regenerate.
2. Medians over ACCEPTED runs, with ranges and seed counts.
3. Every tightness number quoted WITH its budget.
4. Exact-integration certificates only; midpoint values are diagnostics.
5. Credit the Prague group in the introduction, three times over: P1
   certification, the stream-function dual, and the C¹ Airy/BFS dual.
