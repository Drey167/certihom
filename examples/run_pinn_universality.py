"""Is the collocation-overfitting failure UNIVERSAL? A vanilla PINN test.

The homogenization experiments showed that a network trained against fixed
collocation points can drive the reported objective to ~0 while the true
objective is large, because it hides error BETWEEN the points. If that is a
property of collocation training rather than of homogenization, it must
reproduce in the simplest possible PINN.

Setup: the textbook 1D Poisson problem
        -u''(x) = f(x) on (0,1),  u(0) = u(1) = 0,
with the analytic solution u*(x) = sin(4 pi x), so f = (4 pi)^2 sin(4 pi x)
and the TRUE error is computable exactly.

Boundary conditions are imposed exactly by the ansatz
        u_theta(x) = x (1 - x) N_theta(x),
so the only training signal is the PDE residual on N fixed collocation
points -- isolating the effect under study.

Reported per configuration:
  * residual RMS at the TRAINING points        (what a paper would report)
  * residual RMS on a 20x finer grid            (the honest value)
  * relative L2 error against u*                (ground truth)
  * spectral bandwidth of u_theta vs Nyquist N/2, and the headroom

If the criterion is universal, low headroom must coincide with a large
gap between reported and honest residuals AND a large true error.

Run:  python examples/run_pinn_universality.py [--steps 4000]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

PI = np.pi
K = float(os.environ.get("PINN_K", "1.0"))  # u* = sin(K pi x)


def u_exact(x):
    return torch.sin(K * PI * x)


def f_rhs(x):
    return (K * PI) ** 2 * torch.sin(K * PI * x)


class PINN(nn.Module):
    """u_theta(x) = x(1-x) N(x) — exact Dirichlet BCs, plain tanh MLP."""

    def __init__(self, width=64, depth=4):
        super().__init__()
        L, d = [], 1
        for _ in range(depth):
            L += [nn.Linear(d, width, dtype=torch.float64), nn.Tanh()]
            d = width
        L += [nn.Linear(d, 1, dtype=torch.float64)]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        # x: (N,) -> add the feature axis for the MLP, then drop it again
        return x * (1.0 - x) * self.net(x.unsqueeze(-1)).squeeze(-1)


def residual(model, x):
    """r = -u'' - f, computed by autograd."""
    x = x.detach().clone().requires_grad_(True)
    u = model(x)
    du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
    return -d2u - f_rhs(x)


def bandwidth_1d(model, n_probe=2048, quantile=0.999):
    """Frequency (cycles on [0,1]) containing `quantile` of the spectral
    energy of u_theta. The ansatz vanishes at both ends, so the periodic
    extension is continuous and leakage is mild."""
    x = torch.linspace(0, 1, n_probe + 1, dtype=torch.float64)[:-1]
    with torch.no_grad():
        u = model(x)
    P = (torch.fft.rfft(u).abs() ** 2).numpy()
    P[0] = 0.0
    c = np.cumsum(P)
    if c[-1] <= 0:
        return 0.0
    return float(np.searchsorted(c, quantile * c[-1]))


def run(n_coll, steps, width=64, depth=4, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    model = PINN(width, depth)
    # interior collocation points, fixed for the whole run
    xc = (torch.arange(n_coll, dtype=torch.float64) + 0.5) / n_coll
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        (residual(model, xc) ** 2).mean().backward()
        opt.step()

    xf = (torch.arange(20 * n_coll, dtype=torch.float64) + 0.5) / (20 * n_coll)
    r_train = float(residual(model, xc).pow(2).mean().sqrt().detach())
    r_fine = float(residual(model, xf).pow(2).mean().sqrt().detach())
    with torch.no_grad():
        ue, up = u_exact(xf), model(xf)
        err = float((up - ue).norm() / ue.norm())
    B = bandwidth_1d(model)
    return dict(n=n_coll, r_train=r_train, r_fine=r_fine,
                ratio=r_fine / max(r_train, 1e-30), err=err,
                B=B, nyq=n_coll / 2, head=(n_coll / 2) / max(B, 1e-9))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--collocation", type=int, nargs="+",
                   default=[3, 4, 6, 8, 16, 32])
    args = p.parse_args()

    print(f"# Vanilla PINN, -u'' = f, u* = sin({K:.0f} pi x), "
          f"exact BCs, {args.steps} Adam steps")
    print(f"# solution's own frequency is {K/2:.0f} cycles on [0,1]\n")
    print(f"{'N_coll':>7} | {'res@train':>10} {'res@fine':>10} {'ratio':>9} | "
          f"{'bandwidth':>9} {'Nyquist':>8} {'headroom':>8} | {'L2 err':>8}")
    print("-" * 88)
    rows = []
    for n in args.collocation:
        r = run(n, args.steps)
        rows.append(r)
        print(f"{r['n']:>7} | {r['r_train']:10.2e} {r['r_fine']:10.2e} "
              f"{r['ratio']:9.1f} | {r['B']:9.1f} {r['nyq']:8.1f} "
              f"{r['head']:8.2f} | {r['err']:8.2%}")

    print("\nInterpretation")
    print(" res@train is what a paper reports; res@fine is the honest value.")
    print(" A large ratio means the network hid error between its own")
    print(" collocation points. Check whether large ratio coincides with")
    print(" low headroom and large true L2 error.")
    bad = [r for r in rows if r["ratio"] > 5]
    if bad:
        print(f"\n COLLOCATION OVERFITTING OBSERVED at N = "
              f"{[r['n'] for r in bad]}")
        print(" -> the failure is NOT specific to homogenization.")
    else:
        print("\n No overfitting observed in this sweep; try fewer points, "
              "more capacity, or more steps.")


if __name__ == "__main__":
    main()
