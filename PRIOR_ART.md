# PRIOR ART BRIEFING — read this, then read the papers yourself

This is a starting point and a set of questions to answer, **not a
substitute for reading the PDFs**. Everything below was gathered from
abstracts, citation lists and publisher pages. I have not read the full
texts, and neither have you yet. Verify every claim.

---

## 1. THE closest prior work — and it is now PUBLISHED

**Gaynutdinova, Doškář, Rokoš & Pultarová, "Homogenization with Guaranteed
Bounds via Primal-Dual Physically Informed Neural Networks."**
arXiv:2509.07579 — **published in International Journal for Numerical
Methods in Engineering (2026), DOI 10.1002/nme.70302.**

That matters: IJNME was on your fallback venue list. This is no longer a
preprint you might outrun; it is a peer-reviewed Q1 paper you must
position against explicitly, in your introduction, not your related work.

**What they do** (from the abstract): a dual formulation for the PINN
framework applied to homogenization of periodic thermo-conductive
composites, in both strong and variational (weak) formulations. The dual
approach yields guaranteed upper and lower error bounds and, in their
framing, acts as a reliability indicator — a way to detect PINN failure
rather than only to bound the answer. They compare strong-form PINNs on
smoothed material approximations against VPINNs using spectral and
neural-network test functions, and report that strong-form PINNs can win in
controlled settings but are sensitive to material discontinuities and can
fail without clear diagnostics, while VPINNs handle piecewise-constant
coefficients directly but need careful test-function choice.

**Their motivation overlaps yours almost exactly**: PINNs fail on
piecewise-constant coefficients, and duality is the fix.

---

## 2. The part that should worry you most

The same group has prior work on guaranteed bounds for **ELASTICITY**, by
classical numerical methods:

* Gaynutdinova, Ladecký, Nekvinda, Pultarová & Zeman, "Efficient Numerical
  Method for Reliable Upper and Lower Bounds on Homogenized Parameters"
  (arXiv:2208.09940, 2022).
* Gaynutdinova, Ladecký, Pultarová & Zeman, **"Guaranteed Lower Bounds to
  Effective Stiffness,"** PAMM 23 (2023) e202300098.

So they already have (a) guaranteed bounds machinery for **stiffness**, and
(b) a neural primal-dual formulation for **conductivity**. The obvious next
paper for them is the combination — which is precisely your Phase 2. The
group also spans CTU Prague, TU Eindhoven and Zeman's FFT-homogenization
lineage, so they have the numerics depth to do it quickly.

**Implication for you.** Your tensorial (elasticity) result is a much
narrower moat than it looked a week ago. Your durable differentiators are
the ones a numerics-first group is least likely to build first:

1. **Amortization** — certificates for unseen microstructures at inference
   cost. Their work is per-instance.
2. **Exact-integration certification** including the BFS C¹ Airy element.
3. **The quadrature-safety criterion** — and this one is not even
   homogenization-specific.
4. **Certified design** — optimizing a guaranteed floor.

Lead with those. Do NOT lead with "we extend bounds to elasticity."

---

## 3. Questions you must answer FROM THE FULL TEXT

Write the answers down; they determine your framing.

1. **Are their bounds exactly integrated, or quadrature-based?** If
   quadrature-based, does the paper address the failure mode you measured
   (energy hidden between collocation points)? If it does not, your Phase 1
   + quadrature criterion is a genuine and citable contribution *against a
   published baseline*.
2. **Do they use Airy/stream-function potentials, or Lagrange multipliers /
   a separate dual network?** Your dual construction is
   admissible-by-architecture; if theirs enforces admissibility softly,
   that is a real distinction — verify it rather than assuming.
3. **How wide are their bounds at high contrast?** You measured 78.5% on a
   contrast-50 checkerboard. If theirs are comparable, your weakness is a
   field-wide open problem (much better framing). If theirs are tighter,
   find out why before writing anything.
4. **Do they report per-instance training cost?** Needed for your
   amortization crossover argument to be credible.
5. **Is the "failure detection" framing theirs?** They emphasise duality as
   a convergence diagnostic. Your solver-audit result (bounds detecting an
   under-converged FV solve) is adjacent — check whether they got there
   first, and cite them if so.
6. **What exactly is in the PAMM stiffness paper?** If the elasticity dual
   is already worked out classically, your novelty is the neural +
   amortized version, not the formulation.

---

## 4. Also read (shorter list, in priority order)

* **Vondřejc, Zeman et al.** — Fourier–Galerkin guaranteed bounds with
  exact integration. Likely the closest CLASSICAL analogue of your Phase 1;
  find out what their exact-integration argument is and how yours differs.
* **Milton, *The Theory of Composites*, ch. 9 and 23** — laminates,
  variational bounds, Hashin–Shtrikman. Your ground-truth chapter.
* **Hashin & Shtrikman (1962, 1963)** — needed before you attempt the
  polarization variant.
* **Prager & Synge (1947)** — the hypercircle method; the ancestor of the
  primal–dual bound pair. Cite it; reviewers of this vintage of work
  expect it.
* **Moulinec & Suquet (1994, 1998)** — the FFT baseline you now implement.
* **Dvořák & Haslinger** — computable bounds via conforming approximations.
* **Ma, Pitt, Azizzadenesheli & Anandkumar (TMLR 2024)** — conformal UQ for
  operator learning; the contrast between probabilistic coverage and your
  deterministic guarantee is worth one paragraph.

---

## 5. Decision this reading should produce

By the end of week 1 you should be able to complete this sentence in one
line, without hedging:

> "Gaynutdinova et al. (IJNME 2026) established X. We contribute Y, which
> their approach cannot do because Z."

If you cannot, do not start writing. Reframe first.

## VERDICT AFTER READING THE FULL TEXT (see the prior-art report)

Corrections that MUST be made to the project's claims:

* **Exact-integration certification is NOT novel.** Gaynutdinova et al.
  already project the PINN onto piecewise-linear (P1) triangular elements
  aligned with the phases and integrate exactly; they explicitly state the
  quick trapezoidal estimates do NOT preserve the Loewner ordering while
  the projected ones do. Reframe Phase 1 as carrying their certificate onto
  amortized neural fields, not as inventing it.
* **Dual-as-failure-detector is THEIR headline framing**, not ours. Our
  distinct version is cross-solver detection (our bounds caught an
  under-converged FV solve), which they do not do.
* **The C1 Airy / BFS bicubic Hermite dual is the group's own classical
  construction** (Gaynutdinova, Ladecky, Pultarova & Zeman, PAMM 23 (2023)
  e202300098, "Guaranteed lower bounds to effective stiffness"), with exact
  quadrature. Cite it as the origin of our elasticity dual.
* **"Collocation-trained bounds can be false" was proven in 2022** by
  Berrone, Canuto & Pintore (J. Sci. Comput. 92:100, Appendix B.2): a
  nonzero network can have exactly zero VPINN loss at all quadrature points
  while being O(1) wrong. Their criterion is polynomial-precision
  (q >= 2 k_test). Only a spectral bandwidth-vs-Nyquist criterion tied to
  the trained network's measured spectrum can still be claimed as new.
* **The main competitor is the Voigt-Reuss Net** (Keshav, Herb & Fritzen,
  GAMM-Mitteilungen 2025 / IJNME 2026): amortized, tensor elasticity,
  Loewner guarantees, differentiable design. Our defence is MEASURED (see
  RESULTS.md): median 3.4x tighter than Voigt-Reuss on unseen
  microstructures, and certificates that distinguish geometries with
  identical volume fraction, which their envelope provably cannot.

Superseded draft sentence (kept for the record):

> "Gaynutdinova et al. established per-instance primal–dual PINN bounds for
> scalar conductivity. We contribute certificates for unseen microstructures
> at inference cost, exact-integration certification for tensor-valued
> elasticity, and a quadrature-safety criterion showing that
> collocation-trained bounds can be false — none of which follows from a
> per-instance formulation."

Test that sentence against what you actually read. If the full text already
covers exact integration or amortization, tell me and we replan.


---

## CORRECTED POSITIONING SENTENCE (use this)

> Gaynutdinova et al. (IJNME 2026) established per-instance,
> scalar-conductivity primal-dual PINN bounds, certified by projection onto
> conforming finite elements. We extend this in three ways that do not
> follow from a per-instance formulation: (i) AMORTIZATION — guaranteed
> two-sided Loewner certificates for unseen microstructures at inference
> cost, measured 3.4x tighter than the volume-fraction-only Voigt-Reuss
> envelope guaranteed by amortized surrogates such as the Voigt-Reuss Net,
> and able to distinguish microstructures that envelope cannot; (ii) neural
> TENSOR-VALUED elasticity, carrying the group's classical C1 Airy/BFS
> stress-potential dual onto amortized neural trial fields with
> exact-integration certification; and (iii) a quantitative
> bandwidth-vs-Nyquist collocation-density criterion, sharpening the
> polynomial-precision rule of Berrone et al. (2022).

Three claims to DELETE from all drafts: that we invented exact-integration
certification; that duality as a failure diagnostic is ours; that
collocation-trained losses being false is a new observation.


---

# CONFIRMED BY FULL-TEXT READING (collaborator, verified independently)

All five checks passed. Recorded verbatim in substance:

1. **P1 projection + exact integration IS their certification mechanism.**
   The paper explicitly separates quick trapezoidal estimates on collocation
   points (which do NOT satisfy the guaranteed ordering, because the
   integration is inexact) from the certified bounds obtained by projecting
   onto P1 elements on an interface-aligned mesh, where gradients are
   constant per triangle and exact integration is trivial. **Our
   contribution must begin AFTER this projection step.**
2. **They observe integration overfitting empirically**, for BOTH spectral
   VPINNs (more test functions, dramatically lower loss, poorer estimates,
   attributed to overfitting the imprecision of numerical integration,
   worse for higher-frequency bases) AND neural test functions (loss falls,
   primal-dual gap converges early, further optimisation only fits the test
   functions). **Our delta is therefore prediction and prescription, not
   discovery** — we say WHEN it happens, flag it BEFORE certification
   fails, and prescribe the quadrature.
3. **No amortization of any kind.** One network per microstructure
   throughout; no conditioning, no latent variable, no operator learning,
   no unseen-microstructure inference, no RVE dataset, no generalisation
   study. **This is the largest open gap and our core contribution.**
4. **Contrast is exactly 10:1** (matrix 1, inclusion 0.1), with one square
   inclusion plus one four-inclusion example. Our 200:1 sweep across
   laminate / checkerboard / disks / random media is outside their
   demonstrated regime.
5. **No wall-clock, hardware, FLOPs or latency reported.** Do not compare
   amortization speed against timings that do not exist.

**Positioning locked** (see PAPER_OUTLINE.md). One strengthening applied to
the collaborator's draft: "extend the framework TOWARD tensor-valued
elasticity" was changed to "extend the framework TO tensor-valued
elasticity in 2D", because the elasticity case is fully demonstrated —
amortized certificates 15/15 valid across three independent operators,
2.2-2.7x tightening, a limits map, certified design at +36.5%, and
machine-precision BFS C1 anchors. Hedging invites a referee to ask what is
missing.
