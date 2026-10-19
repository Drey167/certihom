# HOW TO CHECK THIS YOURSELF

Written so that you can verify every claim in this project without
trusting anything I produced. Work down the list; each step is
independently meaningful, and the early ones matter most.

---

## Step 0 — the five-minute check that matters most

```bash
python verify_independently.py
```

This file imports **only numpy**. It does not use the project's code. It
derives the laminate's optimal fields by hand, computes the bounds with
plain array arithmetic, and checks them against the closed-form answer.
Then it feeds deliberately *bad* random fields through the same arithmetic
and checks the sandwich still holds.

Expected: agreement to ~1e-16, bad fields still bracketing, and the
project's own implementation agreeing with it to ~1e-15.

If you run it inside a notebook cell rather than as a script, section 3
needs to know where the package is:

```python
import os; os.chdir('/kaggle/working/certihom')   # or your repo root
```

Sections 1 and 2 need nothing --- they import only numpy.

**Read the whole file before running it.** It is about a hundred lines and
contains no cleverness. If you believe that file, you believe the central
mathematical claim, independently of every other line of code here.

If it ever disagrees with `certihom`, **trust this file** and treat the
disagreement as a bug in the package.

---

## Step 1 — does the code do what it says?

```bash
python -m pytest -q            # expect 26 passed
```

Then break it on purpose:

```bash
python kaggle/02_falsification_drills.py     # edit INDEX between runs
```

Write your prediction down before each drill. A drill you predict
correctly proves you understand that mechanism. Note that drill (c) does
NOT break anything, for a real mathematical reason --- work out why before
reading the hint.

**Independent cross-checks already in the suite:** the certificates are
compared against three separately written solvers (finite volume, conforming
Q1 finite elements, Moulinec--Suquet FFT). They were written at different
times and share no code path with the bound assembly.

---

## Step 2 — is every number in the paper real?

Every numeral in the manuscript appears in `RESULTS.md`, and every table in
`RESULTS.md` names the script that produced it. Pick any number in the
paper at random and trace it:

```bash
grep -n "36.8" RESULTS.md            # find the claim
python examples/run_micrograph.py --image <your image>   # regenerate it
```

Do this for five numbers you choose yourself, not five I suggest. If any
cannot be traced or regenerated, that is a real problem and you should tell
me.

The automated versions of this check are in `paper/README.md`; the point of
doing it manually is that you choose the samples.

---

## Step 3 — do the claims survive the literature?

This is the part no script can do for you, and the part most likely to
change the paper.

1. Read arXiv:2509.07579 in full. Confirm for yourself the five findings
   recorded in `PRIOR_ART.md` --- especially that their P1 projection is
   what makes their bounds guaranteed. If that is wrong, our positioning is
   wrong.
2. Read the PAMM 2023 paper. Confirm the C1 Airy/BFS dual is theirs.
3. Read the Voigt--Reuss net papers. Confirm their guarantee is the
   volume-fraction envelope and not something tighter. **If they claim
   per-instance tightness, section 6 of our paper needs rewriting.**
4. Check every entry in `paper/refs.bib` against the publisher page. I
   assembled these from search results and some fields are unconfirmed.

---

## Step 4 — can you defend it under attack?

Read `ADVERSARIAL_REVIEW.md`. Then, without looking at my answers, write
your own response to R1.1 --- the objection that the novelty is merely
combinatorial. That is the objection that decides acceptance, and the
answer has to be yours, because you will be the one defending it.

Then write a fourth referee report I did not write. The perspective I have
not taken is the one most likely to catch what I missed.

---

## Step 5 — reproduce the headline result from scratch

The strongest check available to you: on a fresh machine, from the zip
alone,

```bash
python -m pytest -q
python examples/run_multiseed.py --seeds 5 --steps 400
python examples/run_vs_voigt_reuss.py --steps 300
```

and confirm the numbers match `RESULTS.md` to within a percent or so. You
have already done a version of this on Kaggle and the agreement was close
(3.5x vs 3.4x tightening; 18.3% vs 18.5%), which is itself evidence: two
different machines, same result.

---

## What each level of checking actually buys you

| check | what it establishes |
|---|---|
| `verify_independently.py` | the mathematics is right, without trusting the package |
| test suite + drills | the implementation matches the mathematics |
| three independent solvers | no shared-bug explanation for the agreement |
| tracing numbers to `RESULTS.md` | the paper reports what was measured |
| reading the prior art | the contribution is what we say it is |
| adversarial review | the claims survive attack |
| fresh-machine reproduction | none of it depends on one environment |

The first row is the one to do today. The fifth is the one most likely to
change the paper.

---

## A standing caution

I generated most of this code and all of these documents. The protections
against that are structural rather than a matter of trust: exact solutions
that would expose an error, three independently written reference solvers,
a test suite designed to fail loudly, every number traceable to a named
script, and a hundred-line verifier that shares no code with the thing it
verifies.

Use them. Where a claim rests on my reading rather than on a computation
--- the literature review above all --- treat it as a hypothesis to
confirm, not a result.
