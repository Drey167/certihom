# Publishing the code and minting a DOI

Roughly one hour. Do this before you submit, because the data availability
statement in the manuscript needs the DOI.

---

## Step 0 — fill in your details first

Four files now carry placeholders. Replace them before the first commit,
because they end up baked into the Zenodo record:

| file | replace |
|---|---|
| `LICENSE` | `[AUTHOR NAME(S)]` and the year if not 2026 |
| `CITATION.cff` | names, affiliation, ORCID, release date |
| `.zenodo.json` | creators block, and the arXiv ID once you have it |
| `README.md` | add a "How to cite" section pointing at the DOI |

MIT is suggested in `LICENSE` because it is permissive and imposes no
conditions on reuse of the method. If your institution requires something
else (BSD-3, Apache-2.0, or a specific policy), use that instead. Check
before you publish; changing a licence after release is awkward.

---

## Step 1 — create the GitHub repository

```bash
cd certihom
git init
git add .
git commit -m "certihom: certified amortized homogenization"
git branch -M main
git remote add origin https://github.com/<you>/certihom.git
git push -u origin main
```

Make it **public**. A private repository cannot be archived by Zenodo and
defeats the purpose of the data availability statement.

Before pushing, check that nothing unwanted is included:

```bash
git status --short          # should be clean
du -sh .                    # should be a few MB, not hundreds
```

`.gitignore` already excludes `__pycache__`, LaTeX build artefacts,
generated CSVs and checkpoints.

---

## Step 2 — connect Zenodo

1. Go to zenodo.org and sign in **with GitHub** (this authorises the link).
2. Open your Zenodo account settings, then **GitHub**.
3. Find `certihom` in the repository list and switch the toggle **on**.

Zenodo now watches that repository for releases. Nothing is archived until
you make one.

---

## Step 3 — make a release

On GitHub: **Releases** → **Create a new release**.

- Tag: `v1.0.0`
- Title: `certihom v1.0.0 — CMAME submission`
- Description: one paragraph, and note which manuscript version it
  accompanies.

Publish the release. Zenodo picks it up within a few minutes and mints a
DOI.

---

## Step 4 — collect the DOI, and use the right one

Zenodo issues **two** DOIs:

- a **version DOI**, pointing at v1.0.0 specifically
- a **concept DOI**, pointing at "all versions" and always resolving to the
  latest

**Use the concept DOI in the manuscript.** If you later release v1.1.0 in
response to reviewers, the concept DOI still resolves; a version DOI would
freeze readers on the submitted code.

Then, in `main_cmame.tex`:

```latex
\section*{Data availability}
All code, data and scripts required to reproduce every result in this
paper are openly available at https://doi.org/10.5281/zenodo.XXXXXXX
(source: https://github.com/<you>/certihom). The repository contains the
complete implementation, a verification suite of 26 tests including
machine-precision anchors against closed-form effective tensors, 19 seeded
scripts that regenerate every reported number and figure, a full run
record, and a standalone verifier written independently of the main
package that reproduces the central certificate using only numpy.
```

Recompile and check the DOI resolves before submitting.

---

## Step 5 — after arXiv

Once the preprint has an ID:

- add it to `.zenodo.json` under `related_identifiers`
- add it to the repository README
- link the repository from the arXiv abstract page

---

## Sanity check before you release

```bash
python -m pytest -q            # expect 26 passed
python verify_independently.py # expect agreement to ~1e-16
```

A repository whose tests fail on a clean clone is worse than no repository.
Clone it fresh into a temporary directory and run both, exactly as a
referee would.
