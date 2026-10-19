# Kaggle drivers — run these in order

Each file is a plain `.py` split into `# %%` cells. Either run the whole
file, or paste each cell block into its own Kaggle notebook cell.

**Accelerator: CPU for drivers 1-4; GPU (T4 is fine) for drivers 5-6.**
Driver 5 uses a float32-training / float64-certification split; driver 6 is
point-prediction training and therefore runs entirely in float32.  A T4 is
not appropriate for all-float64 training (see `../KAGGLE.md`).

| order | file | what it gives you | time |
|---|---|---|---|
| 1 | `00_setup_and_verify.py` | env recorded, 24/24 tests green | ~2 min |
| 2 | `01_multiseed.py` | **your first real result**: medians + ranges, CSV, figure | ~15 min |
| 3 | `02_falsification_drills.py` | proof you understand the machinery | ~1 h (thinking, not compute) |
| 4 | `03_reproduce_headlines.py` | all headline numbers regenerated + logs | ~12 min |
| 5 | `04_gpu_elasticity.py` | **the load-bearing experiment** — GPU-trained amortized elasticity, certified on CPU | ~30-60 min (T4) |
| 6 | `05_vrnet_elasticity_baselines.py` | fixed 200/100 external-elasticity point-prediction baselines | GPU, roughly minutes |

Driver 6 needs the public `feature_engineering_mechanical_2D.h5` file from
DaRUS-5730 attached as an input, unless it was deliberately bundled under
`data/vrnet_2d/`.  It saves all fixed indices, device/package metadata,
per-cell predictions, and checkpoints.  It is a matched small-data
diagnostic, not a reproduction of the source paper's full-data sweep.

## Read-only datasets (this bit bit me)

`/kaggle/input/...` is **read-only**. The drivers therefore separate:

* `ROOT` — an importable, **writable** copy of the package. If the source
  is found under `/kaggle/input`, it is automatically staged into
  `/kaggle/working/certihom`. This is required, not cosmetic: drill 02
  edits source files, and nothing can be written beside a read-only dataset.
* `OUT` — where CSVs, figures, logs and `REPRO.md` go (`/kaggle/working`).

Verified against a forced read-only source: staged copy created, 24 tests
passed from it, writes to `OUT` succeeded.

## Getting the code onto Kaggle

Upload `certihom_full.zip` as a private Kaggle **Dataset** named
`certihom-src`, attach it to the notebook, then:

```python
!cp -r /kaggle/input/certihom-src/certihom /kaggle/working/
%cd /kaggle/working/certihom
```

The drivers search `/kaggle/working`, `/kaggle/input/certihom-src`, and the
current directory automatically, so after that they just work.

Better once you are iterating: put the repo on GitHub and
`!pip install git+https://github.com/<you>/certihom.git` — re-uploading a
Dataset for every edit gets old fast.

## Notes

* Long runs: use **Save & Run All (commit)** so the job survives your
  browser tab closing. Write outputs to `/kaggle/working` (persisted);
  everything else is discarded.
* `02_falsification_drills.py` edits source files and restores them in a
  `finally` block. Verified to restore byte-identical files — but commit
  to git first anyway, because that is the habit that saves you eventually.
* Do not switch to float32 to chase speed. Training may be float32;
  certification may not. `KAGGLE.md` explains the split and what was
  actually measured.
