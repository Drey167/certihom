# CMAME submission — step by step

Journal: Computer Methods in Applied Mechanics and Engineering (Elsevier)
System: Editorial Manager, reached from the journal's ScienceDirect page
Review: single anonymized (reviewers know you; you do not know them)
Typical timeline: ~3 days to first editorial decision, ~36 days after review

---

## BEFORE you open Editorial Manager

These five must be done first. None is optional.

### 1. Fill every placeholder in `main_cmame.tex`
Search the file for `[` and replace:
- `[FIRST AUTHOR]`, `[SECOND AUTHOR]` and the author list order
- `[Department, Institution, City, Country]`
- `[email@institution]` for the corresponding author
- CRediT roles for each author (Elsevier taxonomy: Conceptualization,
  Methodology, Software, Validation, Formal analysis, Investigation,
  Resources, Data curation, Writing, Visualization, Supervision, Funding)
- `[REPOSITORY URL / DOI]` in the data availability statement
- `[FUNDING SOURCES AND ACKNOWLEDGEMENTS]`
- the generative-AI declaration — state your actual usage per Elsevier
  policy; AI tools must not be listed as authors

Decide author order and the corresponding author before anything else.
It is far easier now than after submission.

### 2. Verify all 18 bibliography entries
Eight carry `note = {Verify...}` in `refs.bib`. Open each publisher page,
confirm authors, year, volume, pages and DOI, then delete the note field.
Author lists for Hendriks, Jiang, Xing, Ernst, Eiras, Liu and Caradot were
reconstructed from database records and are the likeliest errors in the
package. Also settle the Gaynutdinova year: one database indexes it as
IJNME 2025, the manuscript says 2026.

### 3. Publish the code and mint a DOI
Push the repository to GitHub, then connect it to Zenodo and create a
release. Zenodo issues a DOI. Put that DOI in the data availability
statement. A live repository with 26 passing tests is a genuine advantage
for a computational methods paper; make sure the README is readable.

### 4. Read the manuscript end to end, once, on paper
It was assembled section by section and has never been read continuously.
Look for repetition across sections and for seams where two passages say
the same thing differently. Mark with a pen; do not edit while reading.

### 5. Compile the final PDF
`latexmk -pdf main_cmame.tex`. Confirm zero undefined references and
citations. Check that every figure and table sits where it should.

---

## IN Editorial Manager

### Step 1 — Account and start
Go to the CMAME page on ScienceDirect and follow "Submit your paper".
Register or log in to Editorial Manager. Choose **Submit New Manuscript**.

### Step 2 — Article type
Select **Full Length Article** (or Short Communication if you have cut it
down, which you have not). CMAME does not publish review or survey papers,
so do not pick anything resembling one.

### Step 3 — Upload files, in this order
| item | file | notes |
|---|---|---|
| Manuscript source | `main_cmame.tex` + all `sec*.tex` + `refs.bib` | Elsevier wants editable source, not only PDF |
| Figures | `figures/*.pdf` (8 files) | vector; upload individually, in order |
| Highlights | `highlights.txt` | **separate file**, "highlights" in the filename |
| Graphical abstract | `figures/graphical_abstract.png` | separate item, not a figure |
| Manuscript PDF | `main_cmame.pdf` | if the system asks for a compiled version |

The system builds a merged PDF for review. Check it before approving.

### Step 4 — Metadata
- Title, abstract (239 words), keywords (6) — paste from the manuscript
- Authors, affiliations, corresponding author, ORCID
- Funding information

### Step 5 — Suggested reviewers
Most Elsevier journals ask for 3–5. Suggest people who work in
computational homogenization or certified numerics but who are **not**
co-authors of your closest prior work and not at your institution.
You may also request exclusions, with brief reasons, if there is a genuine
conflict. Do not exclude the Prague group simply because they are
competitors; that reads badly and they are legitimate referees.

### Step 6 — Cover letter
Short. Three paragraphs: what the paper does, why CMAME specifically, and
a statement that the work is original, not under consideration elsewhere,
and that all authors approve. Name the closest prior work and your delta
in one sentence — editors screen on novelty and you want that visible
before they open the PDF.

### Step 7 — Declarations
- Competing interests: as stated in the manuscript
- Data availability: point to the Zenodo DOI
- Generative AI use: complete honestly per Elsevier policy

### Step 8 — Open access choice
You will be offered subscription (no charge) or open access (an Article
Publishing Charge). Elsevier states the choice does not affect peer review
or acceptance. Choose **subscription** unless your institution has an
agreement that covers the charge; the personalised price is shown during
submission. You get open access for free anyway by posting to arXiv.

### Step 9 — Approve the built PDF, then submit

---

## THE SAME DAY

**Post to arXiv.** Category `math.NA` (cross-list `cs.LG` and
`cond-mat.mtrl-sci`). This establishes priority, which matters because the
closest competing group is actively publishing in this area. Elsevier
permits preprint posting. Add the arXiv ID to your repository README.

---

## AFTER submission

- **Status "With Editor"** — desk screening. CMAME reports a ~3 day median
  first decision, so silence beyond a week usually means it went out to
  review, which is good news.
- **Desk reject** — the most common causes are scope and missing required
  elements. You have addressed both.
- **Reviews arrive** — read once, wait a day, then respond. Your R1.1
  answer is already written; keep it with the manuscript.
- **Respond to every comment**, including ones you disagree with. State
  the disagreement politely and give the reason. Reviewers accept "we
  disagree, and here is why" far better than silence.

---

## Files, all in `certihom_CMAME_submission.zip`
```
main_cmame.tex          manuscript
sec*.tex                8 section files
refs.bib                18 references
highlights.txt          5 bullets, 68-73 chars
figures/*.pdf           8 figures
figures/graphical_abstract.png   3984 x 1593 px
SUBMISSION_CHECKLIST.md what is verified, what is left
PROSE_EDIT_LOG.md       AI-marker scan and edits
```
