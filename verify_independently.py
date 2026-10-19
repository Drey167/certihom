"""INDEPENDENT VERIFICATION — reads in one sitting, imports nothing.

This file deliberately uses only numpy. It does NOT import certihom. Its
purpose is to let you verify the central mathematical claim of the project
WITHOUT trusting any of the project's code, and then to check that the
project agrees with it.

If you can read this file and believe it, and it agrees with certihom, then
the certificates are sound at their root. If it disagrees, something in
certihom is wrong and you should trust this file instead --- it is short
enough to check by hand.

    python verify_independently.py

Written to be read top to bottom. Roughly 100 lines of substance.
"""

import numpy as np


# =====================================================================
# 1. THE CLAIM, restated
# =====================================================================
# For a periodic two-phase cell with conductivity k(y), the effective
# tensor K_eff satisfies, for ANY periodic scalar fields chi_i and psi_i:
#
#   K_up[i,j] = < k (e_i + grad chi_i) . (e_j + grad chi_j) >   >=  K_eff
#   R_up[i,j] = < (1/k) (e_i + curl psi_i) . (e_j + curl psi_j) > >= K_eff^-1
#   K_low     = R_up^-1                                          <=  K_eff
#
# curl psi := (d psi/dy2, -d psi/dy1), which is divergence-free and has
# exactly zero mean for periodic psi. That is the whole construction.


# =====================================================================
# 2. A CASE WITH A KNOWN ANSWER: the rank-one laminate
# =====================================================================
# Layers stacked along y1. Transport across the layers sees the phases in
# series (harmonic mean); along them, in parallel (arithmetic mean).
# K_eff = diag(harmonic, arithmetic) exactly. This is textbook.

K1, K2, F1, N = 1.0, 6.0, 0.25, 400          # phases, fraction, resolution
h = 1.0 / N
y1 = (np.arange(N) + 0.5) * h                 # cell midpoints along y1
k = np.where(y1 < F1, K1, K2)                 # piecewise-constant, y1 only

harm = 1.0 / (F1 / K1 + (1 - F1) / K2)
arith = F1 * K1 + (1 - F1) * K2
K_exact = np.diag([harm, arith])


# =====================================================================
# 3. THE OPTIMAL FIELDS, derived by hand
# =====================================================================
# Load e1 (across the layers): the flux must be uniform, so the total
# gradient is (harm/k) e1. Load e2 (along): the gradient is uniform, e2.
# Dual, mean flux e1: the flux is uniform, e1. Dual, mean flux e2: the
# gradient is uniform so the flux is (k/arith) e2.
#
# We only need the TOTAL fields, not the potentials, to evaluate the Gram
# matrices --- which is exactly why the construction is easy to check.

G1 = np.stack([harm / k, np.zeros(N)], axis=1)     # e1 + grad chi_1
G2 = np.stack([np.zeros(N), np.ones(N)], axis=1)   # e2 + grad chi_2
Q1 = np.stack([np.ones(N), np.zeros(N)], axis=1)   # e1 + curl psi_1
Q2 = np.stack([np.zeros(N), k / arith], axis=1)    # e2 + curl psi_2


def gram(F, G, w):
    """< w F.G > over the cell; fields vary in y1 only, so a mean over y1
    IS the cell average."""
    return float(np.mean(w * np.sum(F * G, axis=1)))


K_up = np.array([[gram(G1, G1, k), gram(G1, G2, k)],
                 [gram(G2, G1, k), gram(G2, G2, k)]])
R_up = np.array([[gram(Q1, Q1, 1 / k), gram(Q1, Q2, 1 / k)],
                 [gram(Q2, Q1, 1 / k), gram(Q2, Q2, 1 / k)]])
K_low = np.linalg.inv(R_up)

print("=" * 66)
print("1. OPTIMAL FIELDS REPRODUCE THE EXACT TENSOR")
print("=" * 66)
print("exact  diag :", np.round(np.diag(K_exact), 12))
print("K_up   diag :", np.round(np.diag(K_up), 12))
print("K_low  diag :", np.round(np.diag(K_low), 12))
print("max |K_up  - K_exact| =", f"{np.abs(K_up - K_exact).max():.2e}")
print("max |K_low - K_exact| =", f"{np.abs(K_low - K_exact).max():.2e}")


# =====================================================================
# 4. THE PROPERTY THAT MATTERS: ARBITRARY fields still bracket
# =====================================================================
# The bound must hold for any admissible field, not only the optimum. We
# build deliberately BAD ones from random periodic Fourier series and
# check the sandwich still holds. This is the claim the whole project
# rests on, and it needs no neural network to test.

rng = np.random.default_rng(0)
print()
print("=" * 66)
print("2. ARBITRARY (NON-OPTIMAL) FIELDS MUST STILL BRACKET")
print("=" * 66)
print(f"{'trial':>6} | {'K_up[0,0]':>10} {'K_eff[0,0]':>10} {'K_low[0,0]':>10}"
      f" | {'valid?':>7}")
print("-" * 56)

ok_all = True
for t in range(6):
    # d/dy1 of a random periodic function: sum of a_m 2 pi m cos(2 pi m y1)
    def dfield():
        d = np.zeros(N)
        for m in range(1, 5):
            a = 0.4 * rng.standard_normal()
            d += a * 2 * np.pi * m * np.cos(2 * np.pi * m * y1)
        return d

    g1 = np.stack([1 + dfield(), np.zeros(N)], axis=1)   # e1 + grad chi
    g2 = np.stack([np.zeros(N), np.ones(N)], axis=1)
    # curl of a periodic psi(y1) is (0, -dpsi/dy1): affects component 2
    q1 = np.stack([np.ones(N), np.zeros(N)], axis=1)
    q2 = np.stack([np.zeros(N), 1 + dfield()], axis=1)

    Ku = np.array([[gram(g1, g1, k), gram(g1, g2, k)],
                   [gram(g2, g1, k), gram(g2, g2, k)]])
    Ru = np.array([[gram(q1, q1, 1 / k), gram(q1, q2, 1 / k)],
                   [gram(q2, q1, 1 / k), gram(q2, q2, 1 / k)]])
    Kl = np.linalg.inv(Ru)

    up_ok = np.linalg.eigvalsh(Ku - K_exact).min() >= -1e-12
    lo_ok = np.linalg.eigvalsh(K_exact - Kl).min() >= -1e-12
    ok_all &= up_ok and lo_ok
    print(f"{t:>6} | {Ku[0,0]:10.5f} {K_exact[0,0]:10.5f} {Kl[0,0]:10.5f}"
          f" | {str(up_ok and lo_ok):>7}")

print(f"\nall trials bracketed the exact tensor: {ok_all}")
print("These admissible non-optimal fields still bracket the exact tensor.")
print("Their widths depend on the chosen fields; width alone is not an OOD detector.")


# =====================================================================
# 5. NOW compare against the project's own implementation
# =====================================================================
print()
print("=" * 66)
print("3. DOES certihom AGREE WITH THIS FILE?")
print("=" * 66)
try:
    import sys, os, glob

    def _roots():
        """Locate the package whether run as a script, in a notebook cell
        (no __file__), or from a staged Kaggle copy."""
        cands = [os.getcwd(), "/kaggle/working/certihom"]
        try:
            cands.insert(0, os.path.dirname(os.path.abspath(__file__)))
        except NameError:
            pass                      # notebook cell: __file__ undefined
        for d in ("*", "*/*", "*/*/*", "*/*/*/*"):
            cands += sorted(glob.glob(f"/kaggle/input/{d}/certihom"))
        return [p for p in cands if os.path.isdir(os.path.join(p, "certihom"))]

    roots = _roots()
    if not roots:
        raise ImportError("certihom package directory not found near "
                          + os.getcwd())
    sys.path.insert(0, roots[0])
    from certihom.exact import laminate_exact
    from certihom.energy import as_numpy, gram as cgram
    from certihom.microstructures import laminate, sample_k
    from certihom.quadrature import midpoint_grid
    import torch

    Kf = laminate(8, K1, K2, F1, axis=0)
    pts = midpoint_grid(32)
    kv = sample_k(Kf, pts)
    E = torch.eye(2, dtype=torch.float64)
    prim = [(harm / kv).unsqueeze(1) * E[0], E[1].expand_as(pts).clone()]
    dual = [E[0].expand_as(pts).clone(), (kv / arith).unsqueeze(1) * E[1]]
    cu = as_numpy(cgram(prim, kv))
    cl = np.linalg.inv(as_numpy(cgram(dual, 1.0 / kv)))

    print("certihom exact tensor  :", np.round(np.diag(laminate_exact(K1, K2, F1, 0)), 12))
    print("this file's tensor     :", np.round(np.diag(K_exact), 12))
    print("certihom K_up diag     :", np.round(np.diag(cu), 12))
    print("this file's K_up diag  :", np.round(np.diag(K_up), 12))
    d = max(np.abs(cu - K_up).max(), np.abs(cl - K_low).max())
    print(f"\nmax disagreement between the two implementations: {d:.2e}")
    print("INDEPENDENT AGREEMENT:", d < 1e-10)
except Exception as exc:                                   # noqa: BLE001
    print("certihom not importable here:", exc)
    print("\nSections 1 and 2 above are the mathematical claim and stand on")
    print("their own: they import only numpy. Section 3 is a cross-check")
    print("against the package, not part of the argument.")
    print("\nTo run it, either execute this file from the repository root")
    print("    python verify_independently.py")
    print("or, in a notebook, chdir there first:")
    print("    import os; os.chdir('/kaggle/working/certihom')")
