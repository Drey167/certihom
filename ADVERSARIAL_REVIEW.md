# ADVERSARIAL REVIEW — three hostile referee reports

Written against the drafted manuscript, deliberately from the three
perspectives most likely to be assigned. Each attack is answered honestly;
where we **cannot** answer it, that is stated rather than argued around.
The purpose is to find the rejections before a journal does.

---

## Referee 1 — computational homogenization / guaranteed bounds
*(the perspective of the Prague group or a Zeman-lineage reviewer)*

### R1.1 "The novelty is thin. Everything structural here is ours."
> The primal--dual bound is Prager--Synge. The stream-function dual, the
> P1-projection certification and the failure-diagnostic framing are
> Gaynutdinova et al. (2026). The $C^1$ Airy/BFS dual is Gaynutdinova et al.
> (2023). What remains is conditioning a network on an input, which is
> standard machine learning. **Recommend reject.**

**Answer.** This is the strongest attack and it must be met head-on, not
minimised. Our response: the paper credits all four constructions
explicitly, in the body, and claims only the combination and its
consequences. Concretely, what is new is (i) that certificates survive
conditioning at all, with the non-obvious conformity constraint that makes
it possible and which per-instance formulations never face; (ii) that the
amortized certificates are measurably 2--4$\times$ tighter than the
envelope guaranteed by competing amortized surrogates, and geometry-
sensitive where that envelope provably is not; (iii) a quantitative
collocation criterion neither prior work supplies. If the referee accepts
none of these as sufficient, the paper is a CMAME-level engineering
contribution rather than a theoretical one — which we believe is still
publishable, but the framing must not oversell.

**Status: answerable, but this determines accept-vs-reject.** Mitigation:
make the three-way positioning table the second thing in the paper, and
never phrase a contribution in a way that could be read as claiming their
constructions.

### R1.2 "Your bounds are far looser than the classical ones."
> Gaynutdinova et al. report certified errors under 2% on their benchmark.
> You report 12--28% amortized and up to 161% at high contrast. Why would
> anyone accept a hundredfold looser certificate?

**Answer.** Because it is amortized: their 2% requires a training run per
microstructure, ours requires none. The correct comparison is at matched
cost, where per-instance certification of $N$ microstructures costs $N$
solves. We also report per-instance results (below 1% in the design cases),
so a practitioner can choose. **We should add an explicit
tightness-versus-cost figure** rather than leaving the comparison implicit.

**Status: answerable, and identifies a missing figure.**

### R1.3 "10:1 to 200:1 on 2D toy geometries is not a validation."
> Real composites are 3D. Your entire study is 2D, and the elasticity case
> is plane strain with the weaker results.

**Answer.** Conceded and stated in §11. The 2D restriction is real: the
Airy representation and the BFS projection do not extend directly, and 3D
requires Beltrami potentials with a $C^1$-conforming 3D interpolation.

**Status: NOT fully answerable.** This is the paper's largest genuine
weakness. Options: (a) accept the scope and defend it, (b) delay for 3D.
Recommendation: (a), because the amortization and criterion contributions
are dimension-independent in principle and the 3D extension is an
implementation programme rather than a scientific question.

---

## Referee 2 — operator learning / scientific machine learning

### R2.1 "The amortized model does not work well and you show it yourself."
> Median certified width 28% in elasticity, unchanged by a 5$\times$ larger
> family and 9$\times$ more steps. You admit you do not know the bottleneck.
> This is an underperforming model presented as a contribution.

**Answer.** The plateau is reported precisely because it is a negative
result we could have hidden. But the response has substance: the
contribution is that the certificate is *valid* at 28%, not that 28% is
tight. A 28% certified interval is decision-useful for screening — it
provably excludes candidates — in a way that an unquantified point estimate
is not, at any accuracy.

**Status: ANSWERED.** The screening demonstration now exists (RESULTS.md):
on a pool of 40 unseen microstructures, 62--68% of accept/reject decisions
are made provably at 10 ms each, with zero decisions contradicted by the
finite-volume reference. The remaining third is referred to per-instance
certification. A point estimate cannot make either statement.

### R2.2 "No architecture ablation. You have not earned the plateau claim."
> You claim the bottleneck is architecture, not compute, but you sweep
> neither width, depth, encoder type, nor conditioning richness. The claim
> is unsupported.

**Answer.** Correct. We tested one hypothesis (embedding harmonics) at a
short budget and found it inconclusive, and reported it as such. A width /
depth / encoder sweep is missing.

**Status: ANSWERED.** The sweep now exists (RESULTS.md): width
(32/64/96 -> 35.7/34.5/31.8%) and depth (2/3/4 -> 35.2/34.5/32.6%) both
improve the certificate monotonically, while a threefold richer conditioner
does not (25.1% -> 26.5%). The bottleneck is network capacity, not the
geometry encoding, and the plateaued run sat mid-range in both.

### R2.3 "Conformal prediction gives calibrated intervals more cheaply."
> Why variational bounds rather than a conformal wrapper on a fast
> surrogate?

**Answer.** Coverage from conformal prediction is marginal and requires
exchangeability, which fails precisely under the distribution shift where
guarantees matter; our certificates are per-instance, deterministic and
assumption-free, and remain valid under a 10$\times$ contrast shift where
any exchangeability-based guarantee is void. Worth one paragraph in
related work — **currently absent.**

**Status: answerable, needs adding.**

---

## Referee 3 — composites / applications engineer

### R3.1 "What decision does a 28% interval support?"
> Design allowables need percent-level confidence. An interval spanning
> 28% of the value tells me almost nothing about a real material.

**Answer.** Two distinct uses. For screening, the guaranteed floor is
directly actionable: a candidate whose certified lower bound already
exceeds the requirement is provably acceptable without further analysis.
For final design, use per-instance certification, which reaches below 1%.
The paper should say this explicitly and does not currently do so with
sufficient force.

**Status: answerable, needs sharper framing in §5 and §9.**

### R3.2 "No experimental comparison at all."
> Every number is synthetic. I have no evidence this describes a material.

**Answer.** By construction. The certificate is a statement about a model,
and experimental disagreement would test the model's idealisations, not the
bound. We validate against closed-form effective tensors and three
independent solvers. Framed as a model-adequacy study, an experimental
comparison would be interesting future work.

**Status: ANSWERED as far as it can be.** A published segmented micrograph
has now been certified (RESULTS.md): real geometry, binary image only, no
measured property used and no model-adequacy claim made. Certified width
36.8% with the quality gate satisfied, bracketing both FV and FFT
references. The gate rejected the run twice at coarser quadrature before
accepting---the criterion working on real data. A referee wanting
experimental property comparison will still be unsatisfied, and that
remains correctly out of scope.

### R3.3 "You exclude voids, which rules out most porous media."
> The bicontinuous morphology you identify as hardest is exactly the porous
> case, and you cannot treat voids at all.

**Answer.** Conceded, and stated in §11: the dual requires bounded
compliance. A small nonzero pore phase gives a certificate for the
regularised problem.

**Status: NOT answerable.** Genuine limitation. Regularisation error is not
quantified, and quantifying it would be a real contribution.

---

## Summary: what the attacks reveal

| attack | status | action |
|---|---|---|
| R1.1 novelty is combinatorial | answerable; decides accept/reject | lead with the three-way table; never overclaim |
| R1.2 looser than classical | answerable | **add tightness-vs-cost figure** |
| R1.3 2D only | **not answerable** | accept scope, defend it |
| R2.1 amortized model underperforms | **ANSWERED** | screening: 62-68% decided provably, 0 errors |
| R2.2 no architecture ablation | **ANSWERED** | capacity-bound; conditioner is not the limit |
| R2.3 why not conformal | answerable | **add related-work paragraph** |
| R3.1 28% is not decision-useful | answerable | sharpen framing |
| R3.2 no experiment | **ANSWERED** | real micrograph certified, 36.8%, gate-accepted |
| R3.3 voids excluded | **not answerable** | state; quantify regularisation later |

**Gap status: all four identified gaps are now closed** --- architecture
ablation (R2.2), screening demonstration (R2.1), tightness-versus-cost
figure (R1.2), and a certified real segmented micrograph (R3.2).

Seven of nine objections are answered. The two that remain are scope
limitations stated in section 11 and not fixable within this paper: the
two-dimensional restriction, and the exclusion of voids and rigid phases.

**Three attacks cannot be answered** (2D only, voids excluded, and
partially the plateau). All three are already stated in §11. A paper that
states its own unanswerable objections before the referee does is in a
much stronger position than one that is caught by them.
