# Mathematical notes — certihom Phase 0 (scalar conductivity)

Notation: unit cell Y = [0,1)^2, ⟨·⟩ = ∫_Y (unit volume), conductivity
k(y) > 0 piecewise constant on pixels, unit loads e_1 = (1,0), e_2 = (0,1).

## 1. Cell problem and effective tensor

For a prescribed mean temperature gradient ḡ, the corrector θ is the
Y-periodic solution of div( k (ḡ + ∇θ) ) = 0, and the effective tensor is
characterized variationally:

    ḡ · K_eff ḡ  =  min over Y-periodic θ  ⟨ (ḡ+∇θ) · k (ḡ+∇θ) ⟩.      (P)

## 2. Primal (upper) Gram bound

Take ANY periodic trial pair (χ_1, χ_2) and define the linear-in-load trial
family θ(ḡ) = ḡ_1 χ_1 + ḡ_2 χ_2 with total gradients G_i = e_i + ∇χ_i.
Substituting into (P):

    ḡ · K_eff ḡ  ≤  ⟨ (Σ_i ḡ_i G_i) · k (Σ_j ḡ_j G_j) ⟩
                 =  ḡᵀ K_up ḡ,     K_up[i,j] := ⟨ k G_i · G_j ⟩,

for EVERY ḡ, i.e. K_up ⪰ K_eff in the Loewner order. Linearity of the trial
map in ḡ is the only structural requirement — that is why the network has
one head per unit load and loads are combined by superposition, never by
feeding ḡ as an input.

## 3. Dual (lower) Gram bound

The complementary variational principle: for a prescribed mean flux q̄,

    q̄ · K_eff⁻¹ q̄  =  min over q with div q = 0, ⟨q⟩ = q̄  ⟨ q · k⁻¹ q ⟩.  (D)

Admissible fluxes by construction: with a Y-periodic stream function ψ set

    curl ψ := ( ∂ψ/∂y₂ , −∂ψ/∂y₁ ),      q = q̄ + curl ψ.

Then div q = ψ,₂₁ − ψ,₁₂ = 0 identically, and ⟨curl ψ⟩ = 0 exactly because
the mean of any derivative of a periodic function vanishes — so the mean
constraint ⟨q⟩ = q̄ holds EXACTLY, not approximately. With heads ψ_1, ψ_2 and
Q_i = e_i + curl ψ_i, the same superposition argument gives

    R_up[i,j] := ⟨ k⁻¹ Q_i · Q_j ⟩  ⪰  K_eff⁻¹,
    K_low := R_up⁻¹  ⪯  K_eff,

using that A ⪰ B ≻ 0 implies B⁻¹ ⪰ A⁻¹ (Loewner order reverses under SPD
inversion). Together:  K_low ⪯ K_eff ⪯ K_up, with the width of the interval
a computable, per-instance error certificate.

## 4. Why zero-initialization gives Voigt–Reuss

With χ_i ≡ 0: K_up[i,j] = ⟨k⟩ δ_ij (Voigt). With ψ_i ≡ 0:
R_up = ⟨k⁻¹⟩ I ⇒ K_low = ⟨k⁻¹⟩⁻¹ I (Reuss). The last linear layer of each
FieldPair is initialized to zero, so the certificate is valid at step 0 and
training is monotone tightening in expectation; a bad run yields a wide
interval, never an invalid one.

## 5. Decoupled label-free loss

L = tr K_up + tr R_up = Σ_i [⟨k|e_i+∇χ_i|²⟩ + ⟨k⁻¹|e_i+curl ψ_i|²⟩].
Each summand depends on exactly one head and is minimized by the true
corrector of that load; off-diagonal Gram entries are then determined by
polarization. No FEM/FFT labels enter training; references are used only to
AUDIT the implementation.

## 6. Laminate worked example (unit-test ground truth)

Layers stacked along y₁, phases (k₁, f₁), (k₂, f₂), harmonic mean
h = (f₁/k₁ + f₂/k₂)⁻¹, arithmetic mean a = f₁k₁ + f₂k₂. Exact
K_eff = diag(h, a). Optimal admissible fields, all with pixel-wise constant
gradients (hence midpoint-exact quadrature):

* primal, load e₁: uniform flux ⇒ total gradient (h/k(y)) e₁; the
  fluctuation is the piecewise-linear periodic antiderivative of h/k − 1.
* primal, load e₂: χ₂ ≡ 0 (uniform tangential gradient), energy = a.
* dual, mean flux e₁: q ≡ e₁ (uniform), complementary energy ⟨1/k⟩ = 1/h.
* dual, mean flux e₂: q = (k(y)/a) e₂; the fluctuation (k/a − 1) e₂ depends
  on y₁ only and equals curl of the periodic ψ(y₁) = −∫(k/a − 1) dy₁.

Feeding these through the production `gram` routine must reproduce
K_low = K_up = diag(h, a) to machine precision — the strongest single test
of the certificate pipeline.

## 7. Checkerboard (validation target)

Dykhne/Keller duality gives K_eff = √(k₁k₂) I for the two-phase 2D
checkerboard at any contrast. Local fields are corner-singular; smooth
networks approach the optimum from outside, and the residual certificate
gap measures exactly how much the singularity costs. This is the honest
stress test, not a showcase.

## 8. Certificate caveat: quadrature, and the Phase-1 fix

Phase 0 evaluates ⟨·⟩ with a fixed midpoint rule aligned with pixels; the
bounds are guaranteed *up to quadrature error* (integrands are smooth per
pixel; error O(nq⁻²), zero for pixel-wise-constant-gradient fields). Phase 1
removes the caveat: interpolate χ onto a conforming periodic P1/Q1 FE space
and ψ onto a C⁰ space whose curl lies in a piecewise-polynomial space, then
evaluate the energies with EXACT per-element integration against the
piecewise-constant k. Interpolation preserves admissibility (periodicity is
nodal; curl remains exact), so "network proposes, projection certifies".

## 9. Roadmap to elasticity (Phase 2) — the paper's core

Primal: fluctuation displacement heads χ_k (one per unit macroscopic strain,
3 in 2D), Gram matrix over the strain basis, C_up ⪰ C_eff by the identical
superposition argument. Dual: statically admissible periodic stresses via
the Airy potential, σ′₁₁ = φ,₂₂, σ′₂₂ = φ,₁₁, σ′₁₂ = −φ,₁₂ (equilibrium
identically, mean-zero exactly), complementary Gram S_up ⪰ C_eff⁻¹, hence
C_low = S_up⁻¹. 3D uses the Beltrami potential σ′ = inc Φ. Amortization
(Phase 3): condition both field networks on a microstructure encoding
(CNN/FNO features modulating the trunk, FiLM-style); every statement above
is per-instance and survives conditioning untouched, which is what makes
certificates at inference cost possible.

## 10. References to carry into the paper

Voigt (1889); Reuss (1929); Hashin & Shtrikman (1962/63); Prager & Synge
(1947); Dykhne (1971) / Keller (1964); Milton, *The Theory of Composites*;
Dvořák & Haslinger computable bounds; Vondřejc et al., Fourier–Galerkin
guaranteed bounds; Moulinec & Suquet (1994/98); Gaynutdinova, Doškář,
Rokoš & Pultarová, arXiv:2509.07579 (closest prior art — scalar,
per-instance primal–dual PINN bounds); Wang et al., DCEM (IJNME 2024,
stress-function networks). Verify exact bibliographic details before
submission.

## 9b. Phase 2 delivered: elasticity details and one subtlety

Implemented as specified in §9 (Mandel notation throughout; primal = three
displacement heads, one per unit Mandel strain; dual = three Airy potential
heads with sigma' = (phi,22, phi,11, -sqrt2 phi,12), equilibrated and
exactly mean-zero; C_low = S_up^{-1}). Two additions worth recording:

**Laminate ground truth.** For layers stacked along y1, fields vary in y1
only; the exact per-phase constant states solve, per unit load: e22 common
to both phases (kinematics), sigma11 and sigma12 common (traction
continuity), phase-average = load — a 6x6 linear system per basis load
(`laminate_exact_elasticity`). The resulting piecewise-constant fields make
midpoint quadrature exact, giving the machine-precision anchor.

**Unit-mean-stress normalization (a bug the tests caught).** The stress
fields of the *strain*-basis solutions have mean stress C_exact[:, k], not
the unit basis; their S-weighted Gram is therefore C_exact itself, and
inverting it yields C_exact^{-1}, not C_exact. The dual Gram presumes unit
mean stress: recombine sigma(T_j) = sum_k [C_exact^{-1}]_{kj} sigma^{(k)}
(the stress solution map is linear in the mean), after which
C_low = C_exact to ~1e-11. The production network path is immune — its
trial means are the unit T_k by construction — but any future analytic or
imported trial fields must respect this normalization.

**FEM as a one-sided auditor.** Conforming Q1 displacements are a subspace
of the admissible fluctuations, so C_FEM(h) >= C_eff for every h; hence
C_low <= C_FEM(h) is a rigorous, tolerance-free check (and C_FEM equals the
exact laminate tensor because the exact solution lies in the Q1 space).

## 10. Phase 3: amortization without losing the certificate

**Statement.** Let the trial fields be conditioned on the microstructure m,
chi(y; m) and psi/phi(y; m), through any construction that keeps them
Y-periodic and H^1-conforming (dual side: C^1) IN y for every fixed m. Then
all bound theorems of sections 2-3 and 9 hold per instance, for every m,
trained or not, in or out of distribution. Amortization changes the
statistics of the certificate WIDTH, never its validity.

**The conformity trap.** The natural conditioning — feeding pixel lookups
k(y + delta_j) as network inputs — violates the hypothesis: lookups are
piecewise constant in y, so the trial field inherits jump discontinuities
across pixel boundaries, leaves H^1, and its quadrature "energy" bounds
nothing. The failure is silent: the code runs, the numbers look plausible,
and the guarantee is gone. Any conditioning channel must therefore be
smooth and periodic in y *inside the autograd graph*.

**Chosen construction.** Per instance, project the log conductivity (or a
log stiffness proxy) onto low half-plane Fourier modes, Gaussian-blur the
coefficients at two length scales, and evaluate analytically at query time:
features = [k~_s(y), grad k~_s(y)]_s plus (mean, std) globals, with cos/sin
computed from y in-graph. Fields remain C^infinity and periodic; spatial
autograd derivatives are exact; the quadrature caveat is unchanged from
Phases 0-2; zero-initialized heads give Voigt-Reuss for EVERY instance at
step 0. The construction is resolution-agnostic (projection is defined for
any n > 2 kmax) and its locality (grad k~ encodes interface orientation) is
what the measured laminate transfer results attribute their tightness to.

**Status of the claim (measured).** A direct test (examples/
run_conformity_trap.py) compared lookup-conditioned against smooth-
conditioned fields, judged against exact laminate tensors: the
non-conforming version was roughly twice as loose but produced NO observed
validity violations. So the defensible statement is that non-conformity
voids the theorem and costs tightness, not that it reliably produces wrong
numbers. A sharper adversarial construction may still exhibit a violation;
until one is exhibited, do not overclaim.

**Related, and more dangerous (see RESULTS.md).** Quadrature overfitting
DOES produce false certificates: a field trained against a fixed midpoint
rule can hide energy between sample points, and midpoint "upper bounds"
below the exact tensor were measured both adversarially (0.0014 vs 3.0769)
and at ordinary settings (2.5207 vs 2.6667). Exact integration (section 11)
is therefore mandatory rather than merely tidy.

**Measured consequences (RESULTS.md, Phase 3).** Validity 100% everywhere
including a 10x contrast extrapolation audited against exact tensors;
certificate width orders itself by difficulty and shift severity; ~1,300x
inference speedup over per-instance training at ~2.7x width — the first
point on the paper's amortization trade-off curve.

## 11. Phase 1 delivered: exact-integration certificates

**Primal (Q1).** Nodal interpolation of a continuous periodic field is
continuous, piecewise bilinear and periodic, hence in the conforming
subspace of H^1_per; the primal principle applies verbatim to chi_h.

**Scalar dual (Q1, the elegant case).** For continuous piecewise-bilinear
psi_h, q = curl psi_h satisfies div q = 0 pointwise inside elements, and
across an edge the continuity of psi_h forces continuity of its TANGENTIAL
derivative, which curl rotates into continuity of the NORMAL component of
q — exactly the H(div) condition. Mean-zero is exact (integral of a
derivative of a periodic piecewise polynomial). So interpolation certifies
the dual side, and no Raviart-Thomas machinery is needed.

**Elasticity dual (BFS, C^1).** Q1 cannot carry an Airy stress: bilinears
have phi,11 = phi,22 = 0 identically. Use Bogner-Fox-Schmit bicubic Hermite
with nodal DOFs (phi, phi_,x, phi_,y, phi_,xy) — all four supplied exactly
by autodiff of the network. Global C^1 continuity makes the Airy traction
across an edge with normal e1, namely (phi,22, -phi,12), continuous:
phi,22 is the second tangential derivative of a continuous function, and
phi,12 is the tangential derivative of the C^1-continuous phi,1. Hence
sigma in H(div), div sigma = 0, mean exact.

**Exactness of quadrature.** With a pixel-aligned mesh the coefficient is
constant per element. Q1 gradients are bilinear -> integrands of degree <=2
per coordinate -> 2x2 Gauss exact. BFS second derivatives are degree <=3
per coordinate -> integrands of degree <=6 -> 4x4 Gauss (exact to degree 7).

**Consequence.** The certificates hold as mathematical statements up to
floating-point arithmetic only; the "up to quadrature error" caveat of
Phases 0/2/3 is retired. Measured cost of the retirement: 0.1-3.0
percentage points of certificate width over contrasts 2-200 (RESULTS.md).

## 12. Two application-level results

**Certificates audit conventional solvers.** Because the interval is
rigorous, any reference value falling outside it proves the REFERENCE
wrong. Measured at contrast 200: the FV solution on the native mesh sat
below K_low and entered the interval only under refinement. Rigorous
two-sided bounds are therefore a convergence diagnostic for classical
solvers, not merely a surrogate-quality measure.

**Certified design is self-penalizing.** Maximizing K_low (the guaranteed
floor) rather than a point estimate removes the standard failure mode of
surrogate-driven design: a loose model cannot inflate the objective,
because looseness lowers K_low. Measured: +57% guaranteed floor at fixed
material budget, the gain attributable to arrangement (percolation) since
volume fraction alone moves Voigt/Reuss by <5%.

## 13. Remaining theory gaps (honest list)

* 3D elasticity dual needs the Beltrami potential and a C^1 (or
  Hermite-type) 3D projection; the 2D BFS argument generalizes but is not
  implemented.
* No inf-sup/approximation THEORY for the neural trial spaces: tightness is
  characterized empirically, validity is unconditional.
* Hashin-Shtrikman polarization variant (comparison medium) not yet
  implemented; the measured dual-side lag at high contrast motivates it.
* Certificate width vs. RVE size (statistical volume elements) not yet
  studied.

## 14. Quadrature safety: a measurable admissibility condition

Sections 2-3 assume the bound integrals are evaluated exactly. Phases 0/2/3
approximate them by a midpoint rule, and that approximation is not benign:
an optimizer minimizing a rule-based energy will, if the trial space
permits, place energy BETWEEN the sample points. The rule then reports an
energy below the field's true energy and the "bound" can fall below
K_eff — a false certificate, measured both adversarially and at ordinary
settings (RESULTS.md).

**Diagnostic.** Let B be the field's spectral bandwidth (radius containing
99.9% of its spectral energy) and nq/2 the Nyquist limit of the rule.
Define headroom h = (nq/2)/B. Measured behaviour:

    h < 1     refinement ratio 10x - 3500x; reported bound is fiction
    h ~ 2-5   ratio 1.03 - 1.09; bound marginally invalid
    h ~ 10    ratio 1.00; bound valid

**Resolution.** Exact-integration certification (section 11) removes the
condition entirely: interpolation onto a conforming FE space and
element-exact Gauss integration cannot hide energy between sample points,
because the integral has no sample points to hide between. This upgrades
Phase 1 from a tidiness measure to a correctness requirement.

**Bandwidth is non-monotonic in embedding richness.** B was minimised at
n_harmonics = 4 and much larger at 2 and 12. An under-parameterised
embedding cannot represent the target smoothly, so optimisation inflates
the weights and the tanh nonlinearity generates high harmonics; an
over-parameterised one is simply free to oscillate. The same mechanism
plausibly explains the seed volatility observed at n_harmonics = 3.

**Scope.** Nothing here is specific to bounds, homogenization, or elliptic
problems: any physics-informed network trained on fixed collocation points
has this exposure, and the check costs one FFT.

## 15. The quality gate: per-run, label-free acceptance

Section 14 gave headroom h = (nq/2)/B as a configuration-level criterion.
Measurement forces a refinement: at FIXED configuration and fixed nq,
individual training runs differ. On the laminate at nq=64, four seeds
trained to bandwidth ~6.6 (h ~ 4.9, refinement ratio ~1.02) and one to
bandwidth 41.0 (h = 0.78, ratio 7.08), with certified gaps of 0.4-1.7%
versus 97.8%.

Hence **h is a property of the trained field, not of the configuration**,
and must be evaluated per run. Accept a run when

    h >= 2      AND      refinement ratio <= 1.05,

both computable from the trained network alone — no reference solution, no
exact answer, no held-out labels. This is the practically important
property: the check is available precisely when nothing else is.

Two corollaries worth stating in the paper.

* A rejected run remains VALID if certified by exact integration
  (section 11); it is simply wide. The gate separates trustworthy tightness
  from guaranteed validity — a distinction only a certified method can
  even express.
* Because overfitting accumulates with training (the seed-3 collapse
  appeared between 250 and 400 steps), the gate supplies an early-stopping
  signal that the training loss cannot: stop when headroom degrades, not
  when the loss plateaus.
