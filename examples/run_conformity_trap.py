"""Does non-conforming conditioning actually BREAK validity? (MATH.md §10)

The claim under test: conditioning the trial fields on per-point pixel
LOOKUPS makes them discontinuous across pixel boundaries, hence not
H^1-conforming, hence the bound theorems no longer apply.

Mechanism predicted before running:
  lookup features are constant inside each pixel, so autograd sees no
  contribution from them and MISSES the jump at pixel edges. The computed
  energy therefore UNDER-estimates the true energy of the field it claims
  to bound. Under-estimating the primal energy pulls K_up DOWN (can fall
  below K_eff); under-estimating the complementary energy pulls R_up down,
  so K_low = R_up^{-1} goes UP (can rise above K_eff). Both bounds move
  toward the truth and can cross it => measurable VIOLATIONS.

Second, subtler question: does the Phase-1 Q1 projection REPAIR this?
Nodal interpolation of a discontinuous field produces a continuous Q1
field, which IS admissible. If so, the projected bound should stay valid
(though looser) even for the broken conditioner — i.e. projection does not
merely remove quadrature error, it restores conformity.

Verdict is decided against EXACT laminate tensors, so there is no
reference-solver slack anywhere in this experiment.

Run:  python examples/run_conformity_trap.py [--steps 150]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

from certihom.bounds import CertifiedBounds
from certihom.certify import certified_bounds_scalar, node_grid
from certihom.energy import as_numpy, gram
from certihom.exact import laminate_exact
from certihom.fields import PeriodicEmbedding
from certihom.microstructures import boolean_disks, laminate, sample_k
from certihom.quadrature import midpoint_grid

_E = torch.eye(2, dtype=torch.float64)


# --------------------------------------------------------------- conditioners
class SmoothCond(nn.Module):
    """Phase-3 conditioning: smooth, periodic, in-graph (CONFORMING)."""

    def __init__(self, kmax=4):
        super().__init__()
        m = [(a, b) for a in range(kmax + 1) for b in range(-kmax, kmax + 1)
             if not (a == 0 and b <= 0)]
        self.register_buffer("modes", torch.tensor(m, dtype=torch.float64))
        self.n_feats = 4

    def forward(self, y, K):
        f = torch.as_tensor(np.log(K), dtype=torch.float64)
        n = K.shape[0]
        c = (torch.arange(n, dtype=torch.float64) + 0.5) / n
        Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
        pix = torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1)
        ph0 = 2 * torch.pi * pix @ self.modes.T
        fv = f.reshape(-1)
        a = (2.0 / n**2) * (fv @ torch.cos(ph0))
        b = (2.0 / n**2) * (fv @ torch.sin(ph0))
        blur = torch.exp(-2 * torch.pi**2 * (1 / 16.0)**2 *
                         (self.modes**2).sum(1))
        ph = 2 * torch.pi * y @ self.modes.T          # in-graph -> smooth
        cph, sph = torch.cos(ph), torch.sin(ph)
        kt = fv.mean() + ((blur * a) * cph + (blur * b) * sph).sum(1)
        t = -(blur * a) * sph + (blur * b) * cph
        g1 = 2 * torch.pi * (t * self.modes[:, 0]).sum(1)
        g2 = 2 * torch.pi * (t * self.modes[:, 1]).sum(1)
        s = 1.0 / (2 * torch.pi * 4)
        return torch.stack([kt, s * g1, s * g2, fv.std().expand_as(kt)], dim=1)


class LookupCond(nn.Module):
    """The trap: per-point pixel lookups — piecewise constant in y, hence
    DISCONTINUOUS across pixel boundaries (NON-conforming)."""

    def __init__(self):
        super().__init__()
        self.n_feats = 5

    def forward(self, y, K):
        n = K.shape[0]
        h = 1.0 / n
        Kt = torch.as_tensor(np.log(K), dtype=torch.float64)

        def look(off):
            p = (y + torch.tensor(off, dtype=torch.float64)) % 1.0
            idx = torch.clamp((p * n).long(), 0, n - 1)
            return Kt[idx[:, 0], idx[:, 1]]

        return torch.stack([look((0, 0)), look((h, 0)), look((-h, 0)),
                            look((0, h)), look((0, -h))], dim=1)


class Net(nn.Module):
    def __init__(self, cond, width=48, depth=3, n_harm=3):
        super().__init__()
        self.cond = cond
        self.embed = PeriodicEmbedding(n_harm)
        d = self.embed.out_dim + cond.n_feats
        L = []
        for _ in range(depth):
            L += [nn.Linear(d, width, dtype=torch.float64), nn.Tanh()]
            d = width
        self.trunk = nn.Sequential(*L)
        self.head = nn.Linear(d, 2, dtype=torch.float64)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, y, K):
        return self.head(self.trunk(
            torch.cat([self.embed(y), self.cond(y, K)], dim=1)))


# ------------------------------------------------------------------- bounds
def grads(net, y, K, create_graph):
    yy = y.detach().clone().requires_grad_(True)
    out = net(yy, K)
    return [torch.autograd.grad(out[:, i].sum(), yy, create_graph=create_graph,
                                retain_graph=True)[0] for i in range(2)]


def midpoint_bounds(chi, psi, K, nq):
    pts = midpoint_grid(nq)
    kv = sample_k(K, pts)
    gc, gp = grads(chi, pts, K, False), grads(psi, pts, K, False)
    prim = [_E[i] + gc[i] for i in range(2)]
    dual = [_E[i] + torch.stack([gp[i][:, 1], -gp[i][:, 0]], dim=1)
            for i in range(2)]
    K_up = gram(prim, kv)
    R_up = gram(dual, 1.0 / kv)
    return CertifiedBounds(as_numpy(torch.linalg.inv(R_up)), as_numpy(K_up))


def projected_bounds(chi, psi, K, ne):
    def nod(net):
        with torch.no_grad():
            return net(node_grid(ne), K).reshape(ne, ne, -1).numpy()
    return certified_bounds_scalar(nod(chi), nod(psi), K)


def train(cond_factory, fam, nq, steps, seed=0, lr=2e-3):
    torch.manual_seed(seed)
    chi, psi = Net(cond_factory()), Net(cond_factory())
    opt = torch.optim.Adam(list(chi.parameters()) + list(psi.parameters()), lr=lr)
    pts = midpoint_grid(nq)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        K = fam[rng.integers(len(fam))]
        kv = sample_k(K, pts)
        opt.zero_grad(set_to_none=True)
        gc, gp = grads(chi, pts, K, True), grads(psi, pts, K, True)
        loss = 0.0
        for i in range(2):
            Fp = _E[i] + gc[i]
            Fd = _E[i] + torch.stack([gp[i][:, 1], -gp[i][:, 0]], dim=1)
            loss = loss + (kv * (Fp * Fp).sum(1)).mean() \
                        + ((Fd * Fd).sum(1) / kv).mean()
        loss.backward(); opt.step()
    return chi, psi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--nq", type=int, default=32)
    args = p.parse_args()
    NQ = args.nq

    fam = [boolean_disks(16, 1.0, 10.0, n_disks=int(d), radius=float(r), seed=s)
           for d, r, s in [(5, .12, 0), (6, .13, 1), (7, .11, 2), (6, .15, 3),
                           (5, .14, 4), (7, .12, 5)]]
    # Verdict cases have EXACT effective tensors -> no reference slack.
    tests = [("laminate f=.25", laminate(16, 1., 10., .25, 0),
              laminate_exact(1., 10., .25, 0)),
             ("laminate f=.50", laminate(16, 1., 10., .50, 0),
              laminate_exact(1., 10., .50, 0)),
             ("laminate axis1", laminate(16, 1., 10., .25, 1),
              laminate_exact(1., 10., .25, 1))]

    print("# Conformity trap: smooth (conforming) vs lookup (non-conforming)")
    print(f"# {args.steps} steps, verdict against EXACT tensors, tol 1e-9\n")

    for label, factory in (("SMOOTH (conforming)", lambda: SmoothCond()),
                           ("LOOKUP (non-conforming)", lambda: LookupCond())):
        chi, psi = train(factory, fam, NQ, args.steps)
        print(f"## {label}")
        print(f"{'case':>16} | {'midpoint bound':>28} | {'Q1-projected bound':>28}")
        print(f"{'':>16} | {'valid  contains  gap':>28} | "
              f"{'valid  contains  gap':>28}")
        print("-" * 80)
        for name, K, K_ex in tests:
            bm = midpoint_bounds(chi, psi, K, NQ)
            bp = projected_bounds(chi, psi, K, NQ)
            def fmt(b):
                return (f"{str(b.loewner_valid(1e-9)):>5}  "
                        f"{str(b.brackets(K_ex, tol=1e-9)):>8}  "
                        f"{b.gap_rel():7.1%}")
            print(f"{name:>16} | {fmt(bm):>28} | {fmt(bp):>28}")
        print()

    print("Read the 'contains' columns. A False there is a VIOLATION: the\n"
          "certified interval failed to contain the exact answer, which for\n"
          "a conforming field is mathematically impossible.")


if __name__ == "__main__":
    main()
