# Manuscript drafts

Each section is a standalone `.tex` fragment for inclusion in the CMAME
manuscript. Draft status and provenance:

| file | section | status | numbers audited |
|---|---|---|---|
| `sec7_quadrature.tex` | 7. Quadrature safety | **draft 1 complete** (~1600 words) | yes — all traceable to RESULTS.md |

## Rules for these drafts
1. Every numeral must be traceable to RESULTS.md and regenerable from a
   seeded script in `examples/`. Run the audit before editing:
   grep each number in both files.
2. Credit prior art in the section that uses it, not only in the
   introduction. Section 7 credits Berrone et al. (2022) for proving
   collocation losses can be false, and Gaynutdinova et al. (2026) for the
   empirical observation in this setting.
3. State scope and calibration limits inside the section, not only in a
   global limitations section.

## Citation keys used so far
* `Berrone2022` — Berrone, Canuto, Pintore, *J. Sci. Comput.* 92:100 (2022),
  DOI 10.1007/s10915-022-01950-4
* `Gaynutdinova2026` — Gaynutdinova, Doskar, Rokos, Pultarova, *IJNME*
  (2026), DOI 10.1002/nme.70302

| `sec5_amortization.tex` | 5. Amortization (core) | **draft 1** (~1160 words) | yes |
| `sec6_tightness.tex` | 6. Tightness vs Voigt-Reuss | **draft 1** (~730 words) | yes |
| `sec8_limits.tex` | 8. Limits: contrast, morphology, mechanism | **draft 1** (~1050 words) | yes (51 numbers) |
| `sec9_10_applications_cost.tex` | 9. Applications, 10. Cost | **draft 1** (~690 words) | yes (29 numbers) |

| `sec2_4_method.tex` | 2. Certificates, 3. Admissibility, 4. Exact integration | **draft 1** (~1300 words) | yes (anchors + MATH.md cross-check) |

| `sec11_limitations.tex` | 11. Limitations | **draft 1** (~1200 words) | yes; 15/15 caveats covered |
| `sec1_introduction.tex` | 1. Introduction | **draft 1** (~1180 words) | yes; discipline test passed |
| `main.tex` | assembly + abstract | **draft 1** | — |

## Status: ALL 11 SECTIONS DRAFTED, ~8900 words + abstract

Remaining: a short conclusion (no new claims), the bibliography file,
and the seven figures listed in PAPER_OUTLINE.md.

### The discipline test (run it after every edit to section 1 or 11)
Every claim in the introduction must be permitted by section 11, and every
numeral must appear in RESULTS.md. The test currently passes on 9 claim
pairs and 9 numerals. It caught one real gap during drafting: the
introduction claimed the solver-audit capability while section 11 did not
carry its limitation (the audit only detects an error larger than the
certified width). Section 11 now does.

Drafted: 2-4 (method), 5 (amortization, core), 6 (tightness),
7 (quadrature safety), 8 (limits), 9-10 (applications, cost).

Caveats now consolidated in section 11 (audited, 14/14 present): 2D only
and no Beltrami; linear/periodic/small-strain; voids and rigid phases
excluded; phase-aligned meshes; Loewner ordering is not componentwise;
corner-singular and bicontinuous morphologies; the amortized plateau being
architecture- not compute-limited; the weaker elastic
geometry-discrimination result; NOT benchmarked against the competitor's
point predictions, only against the guarantee they enforce; empirical gate
thresholds; bandwidth requires periodicity; the criterion is sufficient but
not necessary; model-vs-specimen; and no speed claim against FFT.

Additional citation keys:
* `PragerSynge1947` — Prager & Synge, Quart. Appl. Math. 5 (1947)
* `Vondrejc2015` — Vondrejc, Zeman, Marek, CMAME (2015),
  DOI 10.1016/j.cma.2015.09.003
* `Vondrejc2016` — Vondrejc, IJNME (2016), DOI 10.1002/nme.5199

Additional citation keys:
* `Gaynutdinova2022` — arXiv:2208.09940
* `Gaynutdinova2023` — PAMM 23 (2023) e202300098, DOI 10.1002/pamm.202300098
* `Keshav2025` — Keshav, Herb, Fritzen, *GAMM-Mitteilungen* 48 (2025)
  e70005, DOI 10.1002/gamm.70005
* `Keshav2026` — arXiv:2511.11388, accepted *IJNME* 2026
