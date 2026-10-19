# SUBMISSION CHECKLIST — what is actually left

State: 28-page manuscript, 8 figures, 12 tables, 26/26 tests passing,
every numeral traceable to `RESULTS.md`, seven of nine adversarial
objections answered.

**Nothing scientific is blocking.** What remains is authorship, verification
you must do personally, and journal formatting. Estimated 1--2 weeks of
your time, none of it research.

---

## BLOCKING — cannot submit without these

### 1. Verify the bibliography yourself · half a day
`paper/refs.bib` has 11 entries assembled from search results. Volume,
issue and page fields are unconfirmed in several. Open each publisher page
and check every field.

This is the single highest-risk item in the package: it is the cheapest
possible referee catch, it is entirely mechanical, and I cannot do it for
you reliably.

Also decide `Milton2002` — currently present but uncited. Either cite it in
Section 2 for the laminate closed form, or remove it.

### 2. Confirm the two prior-art readings you have not yet done · half a day
You have read Gaynutdinova et al. (2026) and confirmed all five findings.
Two remain, and both are load-bearing:

* **PAMM 2023** — confirm the $C^1$ Airy/BFS dual is theirs. Sections 3
  and 4 credit it to them; if that is wrong, the credit is misplaced.
* **Voigt--Reuss net** (GAMM-M 2025 / IJNME 2026) — confirm their guarantee
  is the volume-fraction envelope and nothing tighter. **If they claim
  per-instance tightness, Section 6 must be rewritten.** This is the only
  remaining reading that could still change a claim.

### 3. Authorship and administration · a day
Author names, affiliations, corresponding author, acknowledgements, funding
statement, data availability statement, competing interests. Decide the
author list before anything else — it is easier now than after submission.

### 4. Reformat to the journal template · half a day
Currently `article` class, for drafting. CMAME uses `elsarticle`. Convert,
recompile, check that all 8 figures and 12 tables still place sensibly, and
verify the page count against the journal's limits.

### 5. Write your own answer to objection R1.1 · half a day
`ADVERSARIAL_REVIEW.md` R1.1 — that the novelty is merely combinatorial —
is the objection that decides acceptance. My answer is in that file. Write
yours without reading mine, then compare. **You will be defending this in
the response letter, so the argument must be yours.**

---

## STRONGLY RECOMMENDED — not blocking, materially strengthening

### 6. RVE convergence study · 1 day of compute
Your four-window micrograph result showed certified widths stable to under
a percentage point while certified values spanned 13%. Growing the window
(32, 48, 64, 96) and plotting both would turn that observation into a
result, and would answer the representativeness question a composites
referee will certainly ask.

### 7. Related-work paragraph on conformal prediction · an hour
Referee objection R2.3, currently unanswered in the text. One paragraph
distinguishing distribution-free coverage under exchangeability from
per-instance deterministic bounds.

### 8. Figures reading from data files · half a day
`paper/figures/make_figures.py` currently uses transcribed literals with
provenance comments. Having each figure read a CSV written by its example
script would make a rerun that changes a number change the figure
automatically. Good practice; not required.

### 9. arXiv preprint · an hour
Post before or at submission. Establishes priority, and the closest
competing group is actively working in this area.

---

## EXPLICITLY OUT OF SCOPE — do not let these delay submission

* 3D (Beltrami potentials, $C^1$ 3D projection) — stated in Section 11
* voids and rigid phases — stated in Section 11
* Hashin--Shtrikman polarisation — the limits map indicates it is not the
  right remedy for the observed failure mode
* experimental property comparison — a model-adequacy study, not this paper
* the quadrature criterion's standalone methods paper — a separate,
  probably higher-cited publication; write it after this one

---

## Realistic timeline

| | |
|---|---|
| blocking items (1--5) | ~3 days of your time |
| recommended items (6--9) | ~3 days |
| internal review by a colleague | 1--2 weeks elapsed |
| **submission-ready** | **2--3 weeks** |

---

## What I would do in your position

Day 1: bibliography and the two prior-art readings. Both are mechanical and
both could change the paper, so do them before anything else.

Day 2: write your R1.1 answer. If you cannot make it convincingly, that is
the signal to strengthen the framing before submitting rather than after
rejection.

Day 3: authorship, `elsarticle`, arXiv.

Then the RVE study while a colleague reads the draft cold. A reader who has
not seen this conversation will find things neither of us can.
