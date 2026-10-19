# Application — Intelligent Matter Lab (Prof. Wenjie Zhou, UIUC MatSE)

**Verify everything here yourself before sending.** Lab details were gathered
from UIUC MatSE and MRL pages this session; the LinkedIn post itself could
not be fetched (LinkedIn blocks automated access), so check the post for the
actual application instructions, deadlines, and what he asks for. If he
specifies a format, his format wins over anything below.

---

## Why you are a plausible fit (and where you are not)

**The strong alignment.** His stated organising principle is that
architecture and connectivity are primary design variables alongside
composition. Your certified-homogenization result is a guaranteed,
quantitative statement of precisely that claim: two microstructures at
identical volume fraction receive an identical composition-based envelope
but certified intervals differing by more than a factor of two. You can
prove that architecture matters independently of composition, per instance.
Very few applicants will be able to say something that specific about his
central thesis.

**The second alignment.** The lab does AI-guided design. You have done
gradient-based design *on a guarantee* — raising a provable performance
floor by 55.3% (conductivity) and 36.5% (stiffness) at exactly fixed
material budget, with the property that a worse model yields a lower
guaranteed value rather than a falsely attractive one. For a lab that will
eventually have to trust an optimiser's output enough to fabricate it, that
is a meaningful capability.

**The gap, which you should raise before he does.** His polycatenated and
architected materials work lives in contact, friction, large deformation
and granular-to-solid transitions. Your framework is linear, small-strain,
two-dimensional, and has no contact model. The lab is also substantially
experimental. You are a computational candidate applying to a group that
synthesises, prints and measures.

**How to convert the gap into the proposal.** Certified bounds for
*frictional contact* is a real open problem — contact is a variational
inequality, and the primal--dual machinery you have built extends to it
through complementarity formulations. That is a natural PhD project that
starts from what you have and ends inside his research programme. Propose
it. It shows you understand his field well enough to see the next problem,
not just to describe your last one.

---

## Email to Prof. Zhou

Keep it short. He will read the first three lines and decide.

> **Subject:** PhD application — certified bounds for architected materials
> (guaranteed property intervals from geometry)
>
> Dear Professor Zhou,
>
> I am writing about PhD positions in the Intelligent Matter Lab. Your
> framing of architecture and connectivity as primary design variables
> alongside composition is close to a result I have been working on, and I
> think it points at a project.
>
> I have developed a method that computes *guaranteed* two-sided bounds on
> the effective properties of a heterogeneous material — not an estimate,
> but an interval that provably contains the true value. Two results seem
> relevant to your programme. First, the bounds are geometry-sensitive in a
> way composition-based bounds provably are not: a laminate and a
> checkerboard at identical volume fraction receive an identical classical
> envelope, while my certified intervals for them differ by more than a
> factor of two. Second, because the certificate is differentiable, the
> design objective can be the guarantee itself — I raised a provable
> stiffness floor by 36% at exactly fixed material budget, with the
> property that a worse-trained model returns a lower guaranteed value
> rather than an overconfident one.
>
> The method currently handles linear, small-strain problems in two
> dimensions, so it does not yet reach the contact, friction and large
> deformation that govern polycatenated and interlocked architectures.
> That is exactly why the lab interests me: certified bounds for frictional
> contact — a variational inequality — is an open problem, and the
> primal–dual construction I have built is the natural starting point for
> it. I would like to work on it with you.
>
> A manuscript is in preparation for CMAME. The code, a full run record,
> and an independent verifier are at [link]; every number in the paper is
> regenerable from a seeded script.
>
> My CV is attached. I would welcome a short conversation.
>
> With best regards,
> [Name]
> [Affiliation, if any]

**Notes on the email.** No adjectives about how excited you are. One
specific result, one specific gap, one specific proposed project. The
sentence about a worse model giving a *lower* guaranteed value is worth
keeping — it is the part a mechanician will find interesting, because it
inverts the usual failure mode of surrogate-driven design.

---

## One-page research statement (attach if he asks, or if the post requests one)

**Structure that works for this lab:**

1. **The problem, in his language** (3 sentences). Effective properties of
   architected materials are computed constantly and certified rarely.
   Designers get a number with no error bar, which is tolerable for
   screening and not for anything that must be trusted.

2. **What you built** (1 short paragraph + one figure). The primal–dual
   certificate; admissibility enforced by architecture so a badly trained
   model widens the interval rather than falsifying it; amortization so an
   unseen microstructure is certified in milliseconds. Use the
   corrector-field figure — it makes the mechanism visible in one glance
   and it is the most striking image you have.

3. **The evidence** (a compact table, not prose). Machine-precision
   agreement with closed-form solutions; three independent reference
   solvers; a certificate that caught an under-converged solver; certified
   design gains at fixed budget; and the real-micrograph result on Berea
   sandstone where certified width (method) and certified value (material)
   separated cleanly across windows.

4. **What it cannot do yet, and the project you propose** (1 paragraph).
   2D, linear, small strain, no contact, no voids. Then the proposal:
   certified bounds for frictional contact and interlocked architectures.
   Name the mathematical obstacle — contact is a variational inequality, so
   the dual side needs a complementarity formulation rather than a plain
   equilibrated field — because naming the obstacle is what distinguishes a
   proposal from a wish.

5. **Why this lab specifically** (3 sentences). Not flattery. The concrete
   reason: his materials are the case where composition tells you almost
   nothing and architecture tells you everything, which is the regime where
   a geometry-sensitive certificate is worth having.

---

## Things to check before sending

- Read the actual LinkedIn post. It may name required documents, a
  deadline, a subject-line format, or specific topics. Follow it exactly.
- Confirm the UIUC MatSE PhD application deadline and requirements
  (GRE/TOEFL/IELTS policy, transcripts, letters). Departmental admission
  and PI interest are separate processes — you generally need both.
- Look at his recent papers before writing, especially the polycatenated
  architected materials work with Daraio. Reference one specifically and
  accurately; a wrong reference is worse than none.
- Have the repository link live and the README readable before you send it.
  He may look, and the reproducibility package is genuinely a differentiator.

## Things NOT to claim

- Do not say the paper is published or accepted. "In preparation for CMAME"
  is accurate; anything stronger is checkable and will be checked.
- Do not describe the method as validated against experiment. It is
  validated against closed-form solutions and independent solvers, which is
  the correct claim and a strong one.
- Do not overstate the amortized elasticity results (28% widths are
  screening-grade). If asked, say so plainly — being calibrated about your
  own work is a signal, and this lab is computational enough to notice.
- Do not claim contact mechanics experience you do not have. Propose it as
  the direction, not as something you have done.
