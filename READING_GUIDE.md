# READING GUIDE — the prior art, in the order you should read it

Everything below is a digest from a full-text read plus publisher metadata.
**Verify it yourself.** If any item contradicts what you read, that changes
the paper's framing and you should say so immediately.

---

## 1. THE paper — read this first, in full, today

**Gaynutdinova, Doškář, Rokoš & Pultarová, "Homogenization with Guaranteed
Bounds via Primal-Dual Physically Informed Neural Networks."**

* **Free full text:** https://arxiv.org/abs/2509.07579 (PDF via the
  "Download" link; v1 Sep 2025, v2 Jan 2026)
* **Published version:** *International Journal for Numerical Methods in
  Engineering* 2026, DOI **10.1002/nme.70302** (paywalled; the arXiv v2 is
  the same content)
* Affiliations: Czech Technical University Prague; Rokoš at TU Eindhoven

### What the full text contains (digest — verify each line)

| aspect | what they do |
|---|---|
| physics | **2D steady-state heat equation only.** Scalar conductivity. No elasticity |
| regime | **strictly per-instance** — networks trained for one fixed microstructure; no conditioning, no operator learning, no unseen-microstructure evaluation |
| dual construction | **stream function**: the divergence-free flux fluctuation is written as a 90-degree rotation applied to the gradient of a scalar network output, so admissibility is EXACT by architecture (not a penalty) |
| formulations | strong-form PINN on smoothed material, and VPINN/VSPINN weak forms with spectral and NN test functions |
| **integration** | **TWO TIERS.** (a) quick estimates by trapezoidal rule on the training grid — they state explicitly these do NOT preserve the Loewner ordering, because of integration imprecision. (b) **guaranteed** bounds by projecting the network onto piecewise-linear (P1) triangular finite elements aligned with the phases, where integration is exact |
| benchmark | single square inclusion in a 2 pi x 2 pi cell, conductivity 1 vs 0.1 (**10:1 contrast**), Obnosov analytical effective value approx 0.6476. v2 adds a multi-inclusion case |
| accuracy | best primal bound approx 0.24%, dual approx -1.2% to -1.6% at smoothing eps=1/30. At a sharper interface (eps=1/40) it degrades badly: primal 11-19%, dual -33% to -45% |
| networks | 65 / 391 / 1801 / 15601 parameters; 40,000 epochs; 128x128 collocation grid |
| cost | **no wall-clock times, no hardware reported** |
| headline framing | the primal-dual **gap as a convergence-quality indicator** — a diagnostic for PINN failures that training curves and residual fields hide |
| baselines | fine FEM primal/dual. **No Voigt-Reuss or Hashin-Shtrikman comparison** |
| future work | 3D and linear elasticity; they note the 2D stream-function trick does not extend to 3D |

### The five things to check while reading (30 minutes)

1. Confirm the P1-projection exact-integration scheme exists and is what
   makes their bounds guaranteed. **This is the single most important
   check** — it removes "we invented exact-integration certification" from
   your contribution list.
2. Confirm they observe overfitting to integration imprecision (they report
   larger VSPINN bases with higher-frequency test functions doing WORSE,
   and note lower training loss not translating into better estimates).
   Your bandwidth-vs-Nyquist criterion must be positioned as sharpening
   this observation, not discovering it.
3. Confirm there is no amortization anywhere. **This is your contribution's
   load-bearing gap.**
4. Note their contrast (10:1) and count their microstructures. Your limits
   map to 200:1 across four morphology classes is genuinely new territory.
5. Note the absence of wall-clock/hardware. Your amortization crossover
   argument cannot cite their timings because they do not report any.

---

## 2. The group's classical elasticity work — read next (1 hour)

This is what makes your elasticity claim narrower than it looks.

* **Gaynutdinova, Ladecký, Pultarová & Zeman, "Guaranteed lower bounds to
  effective stiffness," PAMM 23 (2023) e202300098**, DOI
  10.1002/pamm.202300098 — open access via Wiley.
  **Digest:** guaranteed two-sided Loewner bounds on the effective
  STIFFNESS tensor, 2D and 3D. Primal by Lagrange finite elements with
  FFT preconditioning; **lower bound by the dual formulation discretised
  with Bogner-Fox-Schmit bicubic Hermite C1 elements** — i.e. a
  self-equilibrated stress field from a stress potential, the classical
  Airy construction in 2D. They state they must use exact quadrature.
  Caveats they themselves note: high contrast not treated; off-diagonal
  entries not guaranteed by the diagonal ordering.
  **Consequence: your C1 Airy/BFS dual is THEIR construction. Cite it as
  the origin and position yours as the neural + amortized extension.**

* **Gaynutdinova, Ladecký, Nekvinda, Pultarová & Zeman, arXiv:2208.09940**,
  "Efficient numerical method for reliable upper and lower bounds on
  homogenized parameters" — the bounds machinery the neural paper builds
  on; primal FEM upper bounds, Helmholtz-decomposition dual for lower.

---

## 3. Your actual competitor — read this third (1 hour)

**Keshav, Herb & Fritzen, "Voigt-Reuss net."**
* arXiv:2504.00712; *GAMM-Mitteilungen* 48 (2025) e70005, DOI
  10.1002/gamm.70005
* Inverse-design sequel: arXiv:2511.11388, accepted IJNME 2026

**Digest:** an amortized neural surrogate for effective stiffness (2D and
3D), trained on order 10^6 FFT-generated labels, which guarantees by a
Cholesky-like factorisation and spectral normalisation that every predicted
tensor lies inside the Voigt-Reuss envelope in the Loewner sense — during
training, inference AND gradient-based design.

**Why this matters more than the Prague paper for your positioning:** it
overlaps you on amortization, tensor elasticity, Loewner bracketing and
differentiable certified design simultaneously. Your distinction is that
their guarantee is the **volume-fraction-only** envelope, while yours is a
per-microstructure variational certificate. You have measured this:
**3.5x tighter (conductivity), 2.2-2.7x (elasticity)**, and certificates
that distinguish a laminate from a checkerboard at identical volume
fraction — which their envelope provably cannot.

**Check:** whether the IJNME 2026 version adds any per-instance tightness
claim. If it does, tell me and we re-plan.

---

## 4. The quadrature prior art — read fourth (30 minutes)

**Berrone, Canuto & Pintore, "Variational PINNs: the role of quadratures
and test functions," J. Sci. Comput. 92:100 (2022)**, DOI
10.1007/s10915-022-01950-4; arXiv:2109.02035.

**Go straight to Appendix B.2.** They construct a nonzero network whose
VPINN loss is exactly zero at all quadrature points while the error is
O(1). So "collocation-trained losses can be arbitrarily false" is PROVEN
prior art, not your observation. Their criterion is **polynomial
precision** (quadrature precision >= twice the test-function degree).

**Your remaining claim** is the spectral bandwidth-vs-Nyquist criterion
tied to the trained network's MEASURED spectrum, its use as a per-run
label-free quality gate, and its prescriptive form (couple n_harmonics to
nq). Position explicitly against Berrone's polynomial rule.

---

## 5. Classical background — skim as needed

* Milton, *The Theory of Composites*, ch. 9 (laminates) and ch. 23
  (bounds) — your ground truth chapters
* Hashin & Shtrikman 1962, 1963 — needed only if you revisit polarisation
* Prager & Synge 1947 — the hypercircle method, ancestor of the
  primal-dual pair; reviewers of this vintage expect the citation
* Moulinec & Suquet 1994, 1998 — your FFT baseline
* Vondřejc, Zeman & Marek 2015 (DOI 10.1016/j.cma.2015.09.003) and
  Vondřejc 2016 (DOI 10.1002/nme.5199) — Fourier-Galerkin guaranteed bounds
  with exact integration; the closest CLASSICAL analogue of your Phase 1

---

## 6. The one-paragraph output you must produce

After reading, write this and send it to me. If you cannot write it without
hedging, the framing is wrong and we replan before any drafting.

> Gaynutdinova et al. (IJNME 2026) established ________. Their approach
> cannot ________ because ________. We contribute ________, which we
> demonstrate by ________.

My current draft, to be confirmed or destroyed by what you read:

> Gaynutdinova et al. (IJNME 2026) established per-instance,
> scalar-conductivity primal-dual PINN bounds, certified by projection onto
> conforming P1 finite elements. Their formulation is per-instance, so it
> cannot certify an unseen microstructure without retraining. We contribute
> (i) amortized certificates at inference cost, measured 3.5x tighter than
> the volume-fraction-only Voigt-Reuss envelope guaranteed by amortized
> surrogates, and able to distinguish microstructures that envelope cannot;
> (ii) tensor-valued elasticity, carrying the group's own classical C1
> Airy/BFS stress-potential dual onto amortized neural trial fields; and
> (iii) a prescriptive bandwidth-vs-Nyquist criterion with a label-free
> per-run quality gate, sharpening the polynomial-precision rule of Berrone
> et al. (2022).

---

## 7. Claims already DELETED from all drafts

Keep these deleted:
* that we invented exact-integration certification (their P1 projection);
* that duality as a failure diagnostic is ours (their headline);
* that collocation-trained losses being false is a new observation
  (Berrone et al. 2022, proven);
* that the Airy/BFS C1 dual is a new construction (their PAMM 2023).
