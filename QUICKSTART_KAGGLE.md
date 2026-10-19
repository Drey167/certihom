# KAGGLE QUICKSTART — copy, paste, run

Five steps. Total hands-on time ~10 minutes; the long job runs unattended.

---

## STEP 1 — Upload the code as a Dataset (once)

1. Kaggle → **Datasets** → **New Dataset**
2. Drag in `certihom_full.zip`. **Kaggle auto-extracts zip uploads**, so you
   end up with a `certihom/` folder inside the dataset.
3. Name it `certihom-src`, set it Private, **Create**.

You only redo this when the code changes. (Once your repo is on GitHub,
`!pip install git+https://github.com/<you>/certihom.git` is less friction.)

---

## STEP 2 — New notebook, attach the dataset

1. Kaggle → **Code** → **New Notebook**
2. Right panel → **Input** → **Add Input** → your `certihom-src` dataset
3. Right panel → **Settings**:
   * **Accelerator: None (CPU)** for drivers 00–03
   * **Accelerator: GPU T4 x2** only for driver 04
   * Internet: off is fine (nothing needs it except `pip install pytest`,
     which Kaggle usually has preinstalled — turn Internet on if that fails)

---

## STEP 3 — Cell 1: setup (ALWAYS the first cell, in EVERY session)

### Read this first: switching the accelerator wipes /kaggle/working

Changing Accelerator (e.g. None -> GPU T4) **restarts the container and
clears `/kaggle/working`**. So does a session timeout, and so does
"Save & Run All", which reruns every cell from a clean container. Any staged
copy disappears. Symptom:

    python3: can't open file '/kaggle/working/kaggle/04_gpu_elasticity.py'

**Fix: the setup cell below is idempotent — just run it again.** Set your
accelerator BEFORE staging, and keep this cell as cell 1 of the notebook so
committed runs always stage before they compute.

```python
import os, shutil, glob, sys

DEST = "/kaggle/working/certihom"
if not os.path.isdir(os.path.join(DEST, "certihom")):          # re-stage if wiped
    cands = [p for d in ("*", "*/*", "*/*/*", "*/*/*/*")
               for p in sorted(glob.glob(f"/kaggle/input/{d}/certihom"))
               if os.path.isdir(os.path.join(p, "certihom"))]
    assert cands, "dataset not attached — run: !ls /kaggle/input/*/"
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)
    shutil.copytree(cands[0], DEST)
    print("staged from:", cands[0])
os.chdir(DEST)
if DEST not in sys.path:
    sys.path.insert(0, DEST)
print("cwd:", os.getcwd())
print("drivers present:", os.path.isfile("kaggle/04_gpu_elasticity.py"))
```

Then verify:

```python
!pip -q install pytest
!python -m pytest -q
```

**Expect `26 passed`.** If you do not get that, stop and investigate — do
not run any science on a red test suite.

Staging to `/kaggle/working` matters twice over: `/kaggle/input` is
**read-only**, and driver 02 edits source files.

### Use absolute paths in run commands

`!` shell commands are less forgiving about the working directory than
Python is. Prefer absolute paths so a stale cwd cannot bite:

```python
!python /kaggle/working/certihom/examples/run_multiseed.py --seeds 5 --steps 400
```

---

## STEP 4 — Run the drivers

Each is a plain script. One cell each, in this order.

```python
# 4a. multi-seed study — your first real result (~15 min, CPU)
!python examples/run_multiseed.py --seeds 5 --steps 400
```

```python
# 4b. reproduce the headline numbers (~12 min, CPU)
!python examples/run_checkerboard.py --contrast 5 --steps 900
!python examples/run_elasticity_disks.py --steps 400
!python examples/run_amortized.py --steps 400
!python examples/run_contrast_certification.py --steps 200
!python examples/run_certified_design.py --steps 240
```

```python
# 4c. the results that answer the competitor (~5 min, CPU)
!python examples/run_vs_voigt_reuss.py --steps 300
!python examples/run_amortized_elasticity.py --steps 450 --n_train 40
```

```python
# 4d. the methodological contribution (~10 min, CPU)
!python examples/run_quadrature_rule.py --steps 400
!python examples/run_pinn_universality.py --steps 4000
!python examples/run_harmonics_mechanism.py --seeds 3 --steps 250
```

For the **falsification drills** (driver 02), paste the file's cells into
separate notebook cells instead of running it as a script — you need to
change `INDEX` between runs and write down your prediction first.

---

## STEP 5 — The GPU run (driver 04)

1. **Settings → Accelerator → GPU T4 x2.** This RESTARTS the container and
   wipes `/kaggle/working`.
2. **Re-run the Step 3 setup cell** (it is idempotent and will re-stage).
3. Then:

```python
!python /kaggle/working/certihom/kaggle/04_gpu_elasticity.py
```

**Use "Save & Run All" (Commit)** for this one — it takes 30–60 minutes and
a committed run continues after you close the browser. Interactive sessions
die with the tab.

To change the budget, edit the constants at the top of cell 2 of that file
(`N_TRAIN`, `STEPS`, `BATCH`). Start smaller the first time:

```python
!python -c "
src = open('kaggle/04_gpu_elasticity.py').read()
src = src.replace('N_TRAIN = 200','N_TRAIN = 50').replace('STEPS = 4000','STEPS = 800')
exec(compile(src,'drv','exec'))"
```

---

## Getting your results out

Everything is written to `/kaggle/working`: `multiseed.csv`,
`multiseed.png`, `amortized_elasticity.pt`, `REPRO.md`, `logs/`.

* In a committed run: notebook → **Output** tab → download.
* In an interactive session: the file browser on the right, or
  `from IPython.display import FileLink; FileLink('multiseed.csv')`.

---

## Things that will bite you

| symptom | cause | fix |
|---|---|---|
| `OSError: Read-only file system` | writing into `/kaggle/input` | you skipped the staging in Step 3 |
| `certihom not found` | dataset not attached, or a different folder depth | check `!ls /kaggle/input/*/` |
| job dies at ~9–12 h | session limit | use Save & Run All; checkpoint |
| results vanish | wrote outside `/kaggle/working` | write to `/kaggle/working` |
| GPU slower than CPU | you ran float64 on the T4 | driver 04 handles this; don't bypass `to_certification` |
| `24 passed` not `26` | old zip | re-upload the current `certihom_full.zip` |
| `can't open file '/kaggle/working/kaggle/...'` | accelerator switch / restart wiped `/kaggle/working` | re-run the Step 3 setup cell, then use the absolute path |

---

## Suggested first session (about 40 minutes)

1. Step 3 → confirm `26 passed`
2. `run_multiseed.py --seeds 5 --steps 400` → your first real result
3. `run_vs_voigt_reuss.py` → the competitive defence
4. Switch to GPU, **re-run the setup cell**, commit
   `04_gpu_elasticity.py`, close the laptop

Then read the numbers against `RESULTS.md` and tell me what differs.
