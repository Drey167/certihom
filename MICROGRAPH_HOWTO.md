# Certifying a microstructure image — practical guide

## The one-line answer to "how do I raise nq"

```bash
python examples/run_micrograph.py --image mine.png --crop --size 48 --nq 96
```

`--nq` sets the quadrature resolution. It must be a **multiple of `--size`**.
Or let it escalate on its own until the gate accepts:

```bash
python examples/run_micrograph.py --image mine.png --crop --size 48 --auto-nq
```

## Recommended workflow

**Step 1 — check the requirements first.** This costs seconds and tells you
what the image needs before you spend a training run:

```bash
python examples/check_micrograph_requirements.py --image mine.png --size 48
```

Read the `R2 feature scale` line. If it says ALIASED, do not downsample —
use `--crop`.

**Step 2 — certify a native-resolution crop.**

```bash
python examples/run_micrograph.py --image mine.png --crop --size 48 --auto-nq
```

`--crop` takes a `size x size` window at the image's own pixel resolution.
No downsampling, so the geometry is preserved exactly. This is the
important flag: downsampling a fine microstructure aliases it, and the
certificate then bounds a geometry that is not the one you photographed.

**Step 3 — read the gate column.**

* `ACCEPT` — the width is trustworthy. Quote this number.
* `REJECT` — the certificate is still VALID (it contains the truth) but its
  WIDTH is not trustworthy. Raise `--nq` and rerun. Expect the accepted
  number to be slightly **wider** than the rejected one; the rejected value
  was flattered by under-resolved quadrature.

**Step 4 — sample more than one window.** Certified values vary between
windows far more than certified widths do. Use `--offset` to move the crop:

```bash
for off in 0 200 400 600; do
  python examples/run_micrograph.py --image mine.png --crop --size 48 \
      --offset $off --auto-nq --steps 200
done
```

Spread across windows tells you how far the window is from a representative
volume — a statement you can only make because each window carries its own
certificate.

## All the flags

| flag | meaning |
|---|---|
| `--image` | path to the segmented image |
| `--crop` | native-resolution window (recommended) instead of downsampling |
| `--size` | window/grid size in pixels |
| `--offset` | top-left corner of the crop, native pixels |
| `--nq` | quadrature resolution; must be a multiple of `--size` |
| `--auto-nq` | double `nq` until the gate accepts |
| `--contrast` | phase property ratio (default 10) |
| `--steps` | training steps (default 300) |
| `--panel` | which third of a 3-panel figure holds the segmentation |
| `--invert` | swap which phase is which |

## Cost

`nq` is a grid, so doubling it **quadruples** the work. Start at
`nq = 2 x size` and double only if the gate rejects. On a Kaggle CPU
session, `--size 48 --nq 96` takes a couple of minutes at 200 steps.

## Worked example (Berea sandstone)

```
--crop --size 48 --nq 48   ->  4.6% width, 11.4x tightening, headroom 2.26, ACCEPT
--crop --size 64 --nq 64   ->  6.5% width, 10.2x tightening, headroom 3.20, ACCEPT
```

Both accepted first time. By contrast, downsampling the whole 994 px frame
to 32 px gave a *tighter-looking* 21.7% for a geometry that was aliased
beyond recognition, and the gate rejected it.
