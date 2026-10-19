# VALIDATION — how do we know the research succeeded?

## The distinction you must not blur

This project produces a **mathematical certificate**: for a given geometry
with given phase properties, under periodic linear conductivity/elasticity,
the effective tensor provably lies in [C_low, C_up].

That is a statement **about a model**, not a prediction about a specimen.
Two very different questions:

| question | how it is tested | status |
|---|---|---|
| **Is the certificate valid for the model?** | mathematics + exact solutions | **PROVEN, exhaustively** |
| **Does the model describe a real specimen?** | laboratory experiment | a separate question, not this paper's claim |

If a lab measurement fell outside our interval it would NOT mean the bounds
failed. It would mean a model assumption failed: imperfect interfaces,
non-periodicity, 3D effects out of plane, nonlinearity, porosity, or
measurement error. Conflating the two is a category error, and a referee
will punish it. **Do not present experimental agreement as validation of
the bounds.**

## What already constitutes success (and is complete)

1. **Exact analytic solutions reproduced to machine precision.** Rank-1
   laminate and Dykhne checkerboard, both physics, ~1e-11 through the
   production code path (`tests/`). For a bounds paper this is the gold
   standard: the certificate provably contains a value known in closed form.
2. **Tolerance-free one-sided audits.** Conforming Q1 FEM is itself an
   upper bound, so C_low <= C_FEM(h) must hold at every mesh level with no
   tolerance. It did, in every run.
3. **Independent solver cross-checks.** FV and Moulinec-Suquet FFT
   references land inside the intervals.
4. **The certificate caught a WRONG solver.** At contrast 200 the FV
   reference sat below the certified lower bound; mesh refinement confirmed
   the reference — not the bound — was under-converged. A bound that
   detects an error in an independent solver is strong evidence it means
   what it claims.
5. **Every reported run remained valid** across contrasts 2-200, four
   morphology classes, both physics, and all seeds — including runs the
   quality gate rejected as too wide.

Nothing an experiment can add strengthens (1)-(5). The mathematical claim
is already as validated as it can be.

## What real data DOES add: realistic geometry

The honest gap was never validity — it was that the microstructures were
toys. Since this project's own finding is that **morphology matters more
than contrast**, testing realistic morphology is the right use of real
data. Now done for four standard literature models (RESULTS.md):

| morphology | models | certified gap @10:1 | verdict |
|---|---|---:|---|
| overlapping disks | particulate composite | 14.9% | good |
| fibre packing (RSA) | UD composite cross-section | 16.7% | good |
| Voronoi polycrystal | sintered ceramic, metal | 16.5% | good |
| **spinodal / bicontinuous** | **porous media, aerogel, polymer blend, battery electrode** | **50.4% -> 33.2%** | **hardest class; partly fixable by coupling n_harmonics and nq (RESULTS.md)** |

## The validation ladder (in order of what a referee weighs)

**Rung 1 — analytic (DONE).** Laminate, checkerboard, Hashin coated-cylinder
assemblage. Machine precision. Non-negotiable, and worth more than
anything below it.

**Rung 2 — independent numerics (DONE).** FV, Q1 FEM, Moulinec-Suquet FFT.
Include mesh-refinement studies so a disagreement can be attributed.

**Rung 3 — realistic synthetic morphology (DONE).** GRF/spinodal, Voronoi,
RSA fibre packings. This is where the honest limitations appeared.

**Rung 4 — real segmented microstructure IMAGES (recommended next).** Take
a published micrograph or micro-CT slice, segment it, and certify it. This
tests the method on genuine geometry — rough interfaces, polydispersity,
clustering, percolation — without making any claim about a specimen's
measured properties. Sources worth checking:
* Digital Rocks Portal (segmented rock micro-CT with computed transport
  properties)
* NIST micro-CT and materials data repositories
* the DREAM.3D synthetic/experimental microstructure ecosystem
* open micro-CT of fibre composites from composites journals
Practical note: you only need the SEGMENTED BINARY IMAGE, not measured
properties. Crop to a periodic-ish window, or apply a periodic embedding,
and state which you did.

**Rung 5 — experimental effective properties (optional, frame carefully).**
Only meaningful as a MODEL-ADEQUACY study, and only if you also report the
model assumptions being tested. Phrase it as: "the certified interval for
the idealised model contains / does not contain the measured value, which
bounds how much of the discrepancy is attributable to model idealisation
rather than numerical error." That framing is defensible and actually
interesting; "our bounds agree with experiment" is not.

## What would count as FAILURE

Be explicit about this in the paper — a claim that cannot fail is not a
claim:

* any certified interval failing to contain an exact analytic tensor;
* C_low exceeding a converged conforming-FEM upper bound;
* a certificate changing when the quadrature is refined (would indicate the
  exact-integration path is not exact);
* certified intervals no tighter than the Voigt-Reuss envelope on unseen
  microstructures (would collapse the contribution into existing amortized
  surrogates).

None of these has occurred. The first three are actively tested in `tests/`
and would turn the suite red.
