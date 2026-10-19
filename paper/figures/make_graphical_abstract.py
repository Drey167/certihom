"""Graphical abstract for Elsevier submission — gridspec layout.

Elsevier requires at least 531 x 1328 px (h x w), readable at 5 x 13 cm.
Rendered at 3x for print crispness.

Layout uses GridSpec with constrained_layout rather than hand-placed figure
coordinates, so panels cannot collide regardless of font metrics. All text
lives inside its own axes with explicit alignment.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", ".."))
from certihom.microstructures import checkerboard, laminate  # noqa: E402

INK, ACC, BAD, GREY = "#1a1a1a", "#1f5c99", "#a01d2d", "#d9d9d9"
HERE = os.path.dirname(os.path.abspath(__file__))

W_IN, H_IN = 13.0 / 2.54, 5.2 / 2.54
fig = plt.figure(figsize=(W_IN, H_IN), constrained_layout=True)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.06)

gs = GridSpec(2, 4, figure=fig, width_ratios=[0.80, 0.40, 1.20, 2.55],
              height_ratios=[1, 1])

# microstructures -----------------------------------------------------------
for row, (name, K) in enumerate([("laminate", laminate(24, 1., 10., .5, 0)),
                                 ("checkerboard", checkerboard(24, 1., 10.))]):
    ax = fig.add_subplot(gs[row, 0])
    ax.imshow(K.T, origin="lower", cmap="binary", interpolation="nearest",
              vmin=0, vmax=11)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color(INK)
    ax.set_ylabel(name, fontsize=6.2, color=INK, labelpad=2)
    if row == 0:
        ax.set_title("same 50/50\ncomposition", fontsize=6.2, color=INK,
                     style="italic", pad=3, linespacing=1.2)

# arrow ---------------------------------------------------------------------
axa = fig.add_subplot(gs[:, 1]); axa.axis("off")
axa.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5), xycoords="axes fraction",
             arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1,
                             shrinkA=0, shrinkB=0))
axa.text(0.5, 0.60, "admissible\nneural fields", ha="center", va="bottom",
         fontsize=5.8, color=ACC, linespacing=1.25, transform=axa.transAxes)
axa.text(0.5, 0.40, "60 ms", ha="center", va="top", fontsize=5.8,
         color=INK, transform=axa.transAxes)

# interval bars -------------------------------------------------------------
axb = fig.add_subplot(gs[:, 2])
ypos, widths, colors = [1.0, 0.0], [18.3, 43.2], [ACC, BAD]
axb.barh(ypos, [100.6, 100.6], height=0.34, color=GREY, zorder=1)
for y, w, c in zip(ypos, widths, colors):
    left = (100.6 - w) / 2
    axb.barh([y], [w], left=left, height=0.34, color=c, zorder=2)
    axb.text(left + w / 2, y, f"{w}%", ha="center", va="center",
             color="white", fontsize=6.6, zorder=3)
axb.set_yticks(ypos)
axb.set_yticklabels(["laminate", "checkerboard"], fontsize=6.2, color=INK)
axb.set_xlim(-2, 103); axb.set_xticks([]); axb.set_ylim(-0.6, 1.62)
axb.set_title("certified width\nvs identical envelope", fontsize=6.2,
              color=INK, pad=3, linespacing=1.2)
axb.tick_params(axis="y", length=0, pad=1)
for s in axb.spines.values():
    s.set_visible(False)
axb.text(100.6, 1.40, "envelope", ha="right", va="bottom", fontsize=5.5,
         color="#777777")

# message -------------------------------------------------------------------
axt = fig.add_subplot(gs[:, 3]); axt.axis("off")
axt.text(0, 1.02, "Guaranteed bounds, not estimates", fontsize=8.6,
         weight="bold", color=INK, va="top", ha="left", transform=axt.transAxes)
body = ("One trained operator certifies microstructures it has\n"
        "never seen, in tens of milliseconds.\n\n"
        "The guarantee is architectural: an unfamiliar input\n"
        "widens the interval — it cannot falsify it.\n\n"
        "Composition-based bounds give these two structures\n"
        "the same answer. Certified bounds tell them apart.")
axt.text(0, 0.82, body, fontsize=6.4, color=INK, va="top", ha="left",
         linespacing=1.5, transform=axt.transAxes)
axt.text(0, 0.03,
         r"$K_{\mathrm{low}} \preceq K_{\mathrm{eff}} \preceq K_{\mathrm{up}}$"
         "    provably, for every input",
         fontsize=7.0, color=ACC, va="bottom", ha="left", transform=axt.transAxes)

fig.savefig(os.path.join(HERE, "graphical_abstract.pdf"))
fig.savefig(os.path.join(HERE, "graphical_abstract.png"), dpi=3 * 1328 / W_IN)
plt.close(fig)

from PIL import Image  # noqa: E402
im = Image.open(os.path.join(HERE, "graphical_abstract.png"))
print(f"PNG {im.size[0]} x {im.size[1]} px  (Elsevier minimum 1328 x 531)")
print("meets requirement:", im.size[0] >= 1328 and im.size[1] >= 531)
