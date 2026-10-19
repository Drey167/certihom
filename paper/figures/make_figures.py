"""Manuscript figures.

Every data table below is annotated with the script and settings that
produced it, so any figure can be regenerated from scratch by re-running
the named example. Values are transcribed from RESULTS.md, which is itself
the record of those runs.

    python paper/figures/make_figures.py

Outputs vector PDFs into this directory for \\includegraphics.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # notebook cell: __file__ undefined
    HERE = os.path.join(os.getcwd(), "paper", "figures")
    if not os.path.isdir(HERE):
        HERE = os.getcwd()
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
})
INK, ACC, BAD = "#1a1a1a", "#2166ac", "#b2182b"


# ---------------------------------------------------------------- fig 1
def fig_headroom():
    """Headroom vs refinement ratio.
    Source: examples/run_quadrature_rule.py (laminate c6, width 48, depth 3,
    400 steps) and examples/run_pinn_universality.py (1D Poisson PINN).
    """
    # (headroom, refinement ratio, valid?) — homogenization
    hom = [(0.50, 3498.46, False), (1.78, 1.09, False), (4.57, 1.03, False),
           (10.67, 1.00, True), (0.44, 21.60, False), (0.53, 9.84, False),
           (0.47, 1093.23, False)]
    # (headroom, ratio) — vanilla PINN, 1D Poisson
    pinn = [(0.33, 15891), (0.50, 182755), (0.67, 14550),
            (1.33, 1.00), (2.67, 1.00)]

    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(top=0.82, left=0.19, right=0.97, bottom=0.17)
    ax.axvspan(1e-2, 1.0, color=BAD, alpha=0.07, lw=0)
    ax.axvline(1.0, color=BAD, lw=1, ls="--")
    ax.axhline(1.05, color=INK, lw=0.8, ls=":")

    h, r, v = zip(*hom)
    ax.scatter([x for x, k in zip(h, v) if not k],
               [y for y, k in zip(r, v) if not k],
               s=34, color=ACC, marker="o", label="homogenization", zorder=3)
    ax.scatter([x for x, k in zip(h, v) if k], [y for y, k in zip(r, v) if k],
               s=34, facecolors="none", edgecolors=ACC, marker="o", zorder=3)
    ax.scatter(*zip(*pinn), s=34, color=INK, marker="^",
               label="1D Poisson PINN", zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.28, 26); ax.set_ylim(0.35, 3e6)
    ax.set_xlabel(r"headroom  $h = (n_q/2)\,/\,B$")
    ax.set_ylabel(r"refinement ratio  $\rho$")
    # annotation in the empty upper-right quadrant, not over the data
    ax.text(3.2, 4e4, "energy hidden\nbetween points", color=BAD,
            fontsize=7, ha="left", va="center", linespacing=1.25)
    ax.annotate("", xy=(0.75, 6e3), xytext=(3.0, 3e4),
                arrowprops=dict(arrowstyle="->", color=BAD, lw=0.8))
    ax.text(20, 1.6, r"$\rho \approx 1$", color=INK, fontsize=7,
            ha="right", va="bottom")
    # legend outside the data area entirely
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, handletextpad=0.4, columnspacing=1.2, fontsize=7.5)
    fig.savefig(os.path.join(HERE, "fig_headroom.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig_map():
    """Certified width vs contrast and morphology, both physics.
    Source: examples/run_contrast_morphology.py (conductivity, 120 steps,
    3 seeds) and run_contrast_morphology_elastic.py (elasticity, 120 steps).
    Matched budget, so the two panels are directly comparable.
    """
    contrasts = [10, 50, 200]
    cond = {"laminate": [3.0, 7.1, 15.0], "disks": [14.9, 45.7, 86.0],
            "checkerboard": [22.9, 98.2, 161.2]}
    elas = {"laminate": [5.4, 18.9, 63.5], "disks": [16.3, 40.3, 68.8],
            "checkerboard": [28.1, 101.1, 161.8]}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, data, title in zip(axes, (cond, elas),
                               ("conductivity", "elasticity")):
        for m, mk, c in (("laminate", "o", ACC), ("disks", "s", INK),
                         ("checkerboard", "^", BAD)):
            ax.plot(contrasts, data[m], marker=mk, color=c, lw=1.3, ms=5,
                    label=m)
        ax.axhline(100, color="gray", lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_xticks(contrasts)
        ax.set_xticklabels([str(c) for c in contrasts])
        ax.set_xlabel("phase contrast"); ax.set_title(title)
    axes[0].set_ylabel("certified width [%]")
    axes[0].text(11, 108, "envelope-level", color="gray", fontsize=7)
    axes[0].legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(HERE, "fig_map.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_tightening():
    """Tightening vs the Voigt-Reuss envelope, and the identical-fraction
    pair. Source: examples/run_vs_voigt_reuss.py (contrast 10).
    """
    names = ["disks\n(held out)", "disks\n(held out)", "disks\n(held out)",
             "laminate\n(OOD)", "checker\n(OOD)"]
    vr = [83.0, 89.6, 94.0, 86.3, 100.6]
    ce = [20.8, 25.8, 33.7, 21.1, 43.2]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6),
                                 gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(len(names))
    a1.bar(x - 0.19, vr, 0.36, color="lightgray", label="Voigt\u2013Reuss")
    a1.bar(x + 0.19, ce, 0.36, color=ACC, label="certified")
    for i, (u, c) in enumerate(zip(vr, ce)):
        a1.text(i + 0.19, c + 3, f"{u/c:.1f}\u00d7", ha="center",
                fontsize=7.5, color=ACC)
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=7)
    a1.set_ylabel("relative width [%]"); a1.set_ylim(0, 118)
    a1.legend(frameon=False, loc="upper left")
    a1.set_title("unseen microstructures")

    # identical volume fraction, different geometry
    a2.bar([0, 1], [100.6, 100.6], 0.5, color="lightgray")
    a2.bar([0, 1], [18.3, 43.2], 0.5, color=ACC)
    a2.set_xticks([0, 1])
    a2.set_xticklabels(["laminate\n$f=0.50$", "checkerboard\n$f=0.50$"],
                       fontsize=7.5)
    a2.set_ylim(0, 118); a2.set_title("identical volume fraction")
    a2.text(0, 22, "18.3%", ha="center", fontsize=8, color="white")
    a2.text(1, 47, "43.2%", ha="center", fontsize=8, color=ACC)
    a2.text(0.5, 104, "same envelope", ha="center", fontsize=7.5, color="gray")
    fig.savefig(os.path.join(HERE, "fig_tightening.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_amortization():
    """Cost to a certified answer, many-query regime.
    Source: RESULTS.md amortization economics (123 s training, 19 ms per
    certificate; per-instance 25 s each; FFT ~0.5 s, no guarantee).
    """
    N = np.logspace(0, 4, 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(top=0.78, left=0.17, right=0.97, bottom=0.17)
    ax.plot(N, 25.0 * N, color=INK, lw=1.4, label="per-instance certified")
    ax.plot(N, 123 + 0.019 * N, color=ACC, lw=1.6, label="amortized certified")
    ax.plot(N, 0.5 * N, color="gray", lw=1.2, ls="--",
            label="FFT estimate (no guarantee)")
    xo = 123 / (0.5 - 0.019)
    ax.axvline(xo, color=BAD, lw=0.9, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1e4); ax.set_ylim(0.3, 1e6)
    # crossover label low and to the right of the line, clear of every curve
    ax.text(xo * 1.4, 0.9, f"crossover\n$\\approx{int(round(xo/10)*10)}$",
            fontsize=7, color=BAD, ha="left", va="bottom", linespacing=1.2)
    ax.set_xlabel("number of microstructures certified")
    ax.set_ylabel("wall clock [s]")
    # legend above the axes: the upper-left region is occupied by the curves
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=1, fontsize=7.2, handlelength=1.8, handletextpad=0.5,
              labelspacing=0.3)
    fig.savefig(os.path.join(HERE, "fig_amortization.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_tightness_cost():
    """Certified width against cost per certificate. Answers the referee
    question 'why accept a looser certificate than the classical schemes?'
    Source: RESULTS.md — per-instance fits at several budgets, and the
    amortized operator; conductivity, contrast 10 disks.
    """
    # (wall clock per certificate [s], certified width [%], label)
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    fig.subplots_adjust(top=0.94, left=0.16, right=0.97, bottom=0.17)

    xs = [0.019, 25.0, 52.0]
    ys = [26.6, 9.9, 4.9]
    ax.plot(xs, ys, color=INK, lw=1.0, ls=":", zorder=2)
    # markers distinguish the three routes; a legend replaces point labels,
    # which cannot be placed without collision at this point spacing
    ax.scatter([xs[0]], [ys[0]], s=75, color=ACC, marker="D", zorder=3,
               label="amortized (this work)")
    ax.scatter([xs[1]], [ys[1]], s=52, color=INK, marker="o", zorder=3,
               label="per-instance, 400 steps")
    ax.scatter([xs[2]], [ys[2]], s=52, color=INK, marker="s", zorder=3,
               label="per-instance, 900 steps")

    ax.axhline(90.1, color="gray", lw=1.0, ls="--")
    ax.text(0.006, 92, "Voigt\u2013Reuss envelope", fontsize=7, color="gray",
            ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 200); ax.set_ylim(0, 108)
    ax.set_xlabel("cost per certificate [s]")
    ax.set_ylabel("certified width [%]")
    ax.legend(frameon=False, loc="center right", fontsize=7,
              handletextpad=0.5, labelspacing=0.5, borderaxespad=0.6)
    fig.savefig(os.path.join(HERE, "fig_tightness_cost.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 6
def fig_microstructures():
    """Gallery of the morphology classes studied. No training required."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(HERE, "..", ".."))
    from certihom.microstructures import (boolean_disks, checkerboard,
                                          fiber_packing,
                                          gaussian_random_field, laminate,
                                          voronoi_polycrystal)
    cases = [("laminate", laminate(64, 1., 10., .25, 0)),
             ("disks", boolean_disks(64, 1., 10., 6, .14, 1)),
             ("fibre packing", fiber_packing(64, 1., 10., 10, .09, 0)),
             ("Voronoi", voronoi_polycrystal(64, 10., 1., 12, .4, 0)),
             ("checkerboard", checkerboard(64, 1., 10.)),
             ("bicontinuous", gaussian_random_field(64, 10., 1., .4, 6., 0))]
    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.45))
    for ax, (name, K) in zip(axes, cases):
        ax.imshow(K.T, origin="lower", cmap="bone_r", interpolation="nearest")
        ax.set_title(name, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(_os.path.join(HERE, "fig_microstructures.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 7
def fig_correctors():
    """The mechanism, made visible: corrector-gradient magnitude for three
    morphologies. Difficulty tracks the regularity of these fields."""
    import sys, os as _os, torch
    sys.path.insert(0, _os.path.join(HERE, "..", ".."))
    from certihom.fields import heads_and_grads
    from certihom.microstructures import (boolean_disks, checkerboard,
                                          gaussian_random_field, laminate)
    from certihom.quadrature import midpoint_grid
    from certihom.train import fit_bounds

    cases = [("laminate\n2.1% width", laminate(32, 1., 10., .25, 0)),
             ("disks\n12.4% width", boolean_disks(32, 1., 10., 6, .14, 1)),
             ("checkerboard\n19.3% width", checkerboard(32, 1., 10.)),
             ("bicontinuous\n50.4% width",
              gaussian_random_field(32, 10., 1., .4, 6., 0))]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.9))
    n = 96
    pts = midpoint_grid(n)
    for ax, (name, K) in zip(axes, cases):
        _, _, (chi, _) = fit_bounds(K, nq=64, n_harmonics=4, width=48,
                                    depth=3, steps=250, lr=2e-3, seed=0,
                                    verbose=False)
        _, g = heads_and_grads(chi, pts, False)
        m = g[0].detach().norm(dim=1).reshape(n, n).numpy()
        ax.imshow(m.T, origin="lower", cmap="magma", interpolation="bilinear")
        ax.contour(K.T, levels=[5.0], colors="w", linewidths=0.6,
                   extent=(0, n, 0, n))
        ax.set_title(name, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(_os.path.join(HERE, "fig_correctors.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 8
def fig_convergence():
    """Bounds closing from the Voigt-Reuss envelope onto the exact value.
    Validity holds at every iterate, including the first."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(HERE, "..", ".."))
    from certihom.exact import checkerboard_exact
    from certihom.microstructures import checkerboard
    from certihom.train import fit_bounds
    K = checkerboard(16, 1., 5.)
    _, hist, _ = fit_bounds(K, nq=64, n_harmonics=4, width=64, depth=3,
                            steps=900, lr=1e-3, seed=0, log_every=25,
                            verbose=False)
    s = [h["step"] for h in hist]
    up = [h["tr_K_up"] / 2 for h in hist]
    lo = [h["tr_K_low"] / 2 for h in hist]
    ex = checkerboard_exact(1., 5.)[0, 0]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.fill_between(s, lo, up, color=ACC, alpha=0.18, lw=0,
                    label="certified interval")
    ax.plot(s, up, color=ACC, lw=1.3); ax.plot(s, lo, color=ACC, lw=1.3)
    ax.axhline(ex, color=INK, lw=1.1, ls="--", label="exact $\\sqrt{k_1k_2}$")
    ax.set_xlabel("training step"); ax.set_ylabel("effective conductivity")
    ax.set_xlim(0, max(s)); ax.legend(frameon=False, loc="upper right")
    ax.text(20, up[0] + 0.06, "Voigt", fontsize=7.5, color=ACC)
    ax.text(20, lo[0] - 0.12, "Reuss", fontsize=7.5, color=ACC)
    fig.savefig(_os.path.join(HERE, "fig_convergence.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_headroom, fig_map, fig_tightening, fig_amortization,
              fig_tightness_cost, fig_microstructures, fig_correctors,
              fig_convergence):
        f()
        print("wrote", f.__name__.replace("fig_", "fig_") + ".pdf")
