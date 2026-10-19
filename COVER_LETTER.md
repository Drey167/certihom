# Cover letter (adapt, do not send verbatim)

Keep it under a page. Editors read the first paragraph and skim the rest.

---

Dear Editors,

We submit for your consideration "Certified amortized homogenization:
guaranteed two-sided bounds on effective tensors at inference cost" as a
Full Length Article.

Computational homogenization returns effective properties without error
bars. Variational bounds return certified intervals but require a separate
solve for every microstructure, and amortized neural surrogates remove
that cost while giving up the guarantee. This paper combines the two: a
single trained operator emits certified two-sided Löwner bounds for
microstructures it has never seen, in tens of milliseconds, with the
guarantee following from the architecture rather than from the training
data. On unseen microstructures the intervals are two to four times
narrower than the Voigt–Reuss envelope, and they distinguish
microstructures that a composition-based guarantee provably cannot: a
laminate and a checkerboard of identical phase fraction receive certified
widths of 18.3% and 43.2%.

We are explicit about what is and is not new. The primal–dual construction
is classical, and its neural realisation with exact-integration
certification is due to Gaynutdinova et al. (IJNME 2026); amortized
homogenization is established, including in this journal (Hendriks et al.
2024; Jiang et al. 2023). The contribution is their combination, together
with the conformity constraint that makes it possible and a computable
criterion for collocation adequacy that we show applies to
physics-informed networks beyond homogenization.

The paper reports the regimes in which the method degrades, including
morphologies where the certified interval approaches the trivial envelope,
a plateau under increased compute, and a false-certificate mode we found
in our own method. All code, a complete run record, and an independent
verifier are openly available at [DOI]; every reported number regenerates
from a seeded script.

The manuscript is original, is not under consideration elsewhere, and all
authors have approved the submission. We declare no competing interests.

Yours sincerely,
[Corresponding author, on behalf of the authors]
