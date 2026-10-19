"""Phase 3 — AMORTIZED certified homogenization.

One trained operator produces certified bounds for UNSEEN microstructures at
inference cost (a few forward/input-gradient passes), with no per-instance
training. The scientific basis is a single observation:

    Every certificate in Phases 0-2 is a per-instance statement that
    depends ONLY on admissibility of the trial fields, and admissibility is
    architectural. Conditioning the networks on the microstructure does not
    touch admissibility — so certificates for out-of-distribution inputs
    remain VALID; poor generalization can only WIDEN them. The certificate
    width is therefore a built-in, assumption-free trust signal.

THE CONFORMITY TRAP (why the conditioning below is designed as it is).
The obvious conditioning — per-point lookups of the pixel field k(y+delta)
as input features — silently destroys the guarantee: lookup features are
piecewise constant in y, so the trial field JUMPS across pixel boundaries,
is not H^1-conforming, and its "energy" bounds nothing. Conditioning
features must be smooth and periodic IN THE AUTOGRAD GRAPH of y. Here we
use truncated-Fourier reconstructions of the (Gaussian-blurred) log
conductivity, evaluated analytically from per-instance coefficients:

    feats(y; m) = [ k~_s(y), grad k~_s(y) ]_{s in sigmas}  +  global stats,
    k~_s(y) = mean + sum_{xi in H} e^{-2 pi^2 s^2 |xi|^2}
              ( a_xi cos(2 pi xi.y) + b_xi sin(2 pi xi.y) ).

cos/sin are computed from y inside forward, so fields are C^inf and
Y-periodic in y, spatial autograd derivatives are exact, and the quadrature
caveat is unchanged from Phases 0-2. Zero-initialized heads reproduce
Voigt-Reuss for EVERY instance at step 0 — validity-from-step-0 survives
amortization.

The Mandel/scalar field formulas are re-expressed here for conditioned
grads; ``tests/test_amortized.py::test_formula_consistency_with_phase_0_2``
pins them bit-for-bit against energy.py / elasticity.py on a shared
unconditioned network, eliminating transcription risk.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .bounds import CertifiedBounds
from .elasticity import sample_mandel
from .energy import as_numpy, gram
from .elasticity import gram_mandel
from .fields import PeriodicEmbedding
from .microstructures import sample_k
from .quadrature import midpoint_grid

SQ2 = np.sqrt(2.0)
def _eye(n: int, t: torch.Tensor) -> torch.Tensor:
    return torch.eye(n, dtype=t.dtype, device=t.device)
TWO_PI = 2.0 * torch.pi


# ----------------------------------------------------------------------------
# Microstructure -> smooth conditioning coefficients
# ----------------------------------------------------------------------------

def build_modes(kmax: int) -> torch.Tensor:
    """Half-plane integer modes H = {xi1>0} U {xi1=0, xi2>0}, |xi|_inf<=kmax.
    Resolution-agnostic: any pixel grid with n > 2*kmax projects onto them."""
    modes = []
    for x1 in range(0, kmax + 1):
        for x2 in range(-kmax, kmax + 1):
            if x1 == 0 and x2 <= 0:
                continue
            modes.append((x1, x2))
    return torch.tensor(modes, dtype=torch.float64)


def project_coeffs(scalar_field: np.ndarray, modes: torch.Tensor,
                   sigmas: tuple[float, ...]):
    """Real Fourier coefficients of a pixel scalar field (evaluated at pixel
    centers), Gaussian-blurred per sigma.

    Returns (coeffs, glob): coeffs (S, P, 2) float64 with [...,0]=a, [...,1]=b
    for k~_s(y) = mean + sum_xi blur * (a cos + b sin); glob = (mean, std)
    of the field. Projection is the discrete real DFT restricted to the mode
    set (exact orthogonality for kmax < n/2).
    """
    f = torch.as_tensor(scalar_field, dtype=modes.dtype, device=modes.device)
    n = f.shape[0]
    assert 2 * int(modes.abs().max().item()) < n, "kmax must satisfy kmax < n/2"
    c = (torch.arange(n, dtype=modes.dtype, device=modes.device) + 0.5) / n
    Y1, Y2 = torch.meshgrid(c, c, indexing="ij")
    pts = torch.stack([Y1.reshape(-1), Y2.reshape(-1)], dim=1)  # (n^2, 2)
    ph = TWO_PI * pts @ modes.T                                  # (n^2, P)
    fv = f.reshape(-1)
    a = (2.0 / n**2) * (fv @ torch.cos(ph))
    b = (2.0 / n**2) * (fv @ torch.sin(ph))
    mean = fv.mean()
    xi2 = (modes**2).sum(dim=1)
    coeffs = []
    for s in sigmas:
        blur = torch.exp(-2.0 * torch.pi**2 * (s**2) * xi2)
        coeffs.append(torch.stack([blur * a, blur * b], dim=1))
    glob = torch.stack([mean, fv.std()])
    return torch.stack(coeffs, dim=0), glob  # (S,P,2), (2,)


class Conditioner(nn.Module):
    """Builds smooth in-graph features [k~_s, grad k~_s / (2 pi kmax)]_s and
    appends the per-instance globals. Gradients are analytic and flow
    through y (module docstring: the conformity requirement)."""

    def __init__(self, kmax: int = 4, sigmas: tuple[float, ...] = (1 / 16, 3 / 16),
                 dtype=torch.float64):
        super().__init__()
        self.register_buffer("modes", build_modes(kmax).to(dtype))
        self.sigmas = tuple(sigmas)
        self.grad_scale = 1.0 / (TWO_PI * kmax)

    @property
    def n_feats(self) -> int:
        return 3 * len(self.sigmas) + 2

    def project(self, scalar_field: np.ndarray):
        return project_coeffs(scalar_field, self.modes, self.sigmas)

    def forward(self, y: torch.Tensor, coeffs: torch.Tensor,
                glob: torch.Tensor) -> torch.Tensor:
        """y: (N,2) requires-grad; coeffs: (N,S,P,2) per-point (gathered per
        instance); glob: (N,2). Returns (N, n_feats)."""
        ph = TWO_PI * y @ self.modes.T          # (N, P) — in-graph
        cph, sph = torch.cos(ph), torch.sin(ph)
        feats = []
        mean = glob[:, 0]
        for s_idx in range(len(self.sigmas)):
            a = coeffs[:, s_idx, :, 0]
            b = coeffs[:, s_idx, :, 1]
            kt = mean + (a * cph + b * sph).sum(dim=1)
            t = -a * sph + b * cph               # d/d(phase)
            g1 = TWO_PI * (t * self.modes[:, 0]).sum(dim=1)
            g2 = TWO_PI * (t * self.modes[:, 1]).sum(dim=1)
            feats += [kt, self.grad_scale * g1, self.grad_scale * g2]
        feats += [glob[:, 0], glob[:, 1]]
        return torch.stack(feats, dim=1)


# ----------------------------------------------------------------------------
# Conditioned field network + conditioned derivatives
# ----------------------------------------------------------------------------

class ConditionedFieldPair(nn.Module):
    """FieldPair with extra conditioning inputs; zero-init head keeps the
    family-wide Voigt-Reuss-at-step-0 property."""

    def __init__(self, n_feats: int, n_harmonics: int = 3, width: int = 64,
                 depth: int = 3, n_out: int = 2, dtype=torch.float64):
        super().__init__()
        self.embed = PeriodicEmbedding(n_harmonics, dtype=dtype)
        layers: list[nn.Module] = []
        d_in = self.embed.out_dim + n_feats
        for _ in range(depth):
            layers += [nn.Linear(d_in, width, dtype=dtype), nn.Tanh()]
            d_in = width
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(d_in, n_out, dtype=dtype)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, y, feats):
        return self.head(self.trunk(torch.cat([self.embed(y), feats], dim=1)))


def cond_heads_and_grads(net, conditioner, pts, coeffs, glob, create_graph):
    y = pts.detach().clone().requires_grad_(True)
    out = net(y, conditioner(y, coeffs, glob))
    grads = [torch.autograd.grad(out[:, i].sum(), y, create_graph=create_graph,
                                 retain_graph=True)[0]
             for i in range(out.shape[1])]
    return out, grads


def cond_hessians(net, conditioner, pts, coeffs, glob, create_graph):
    y = pts.detach().clone().requires_grad_(True)
    out = net(y, conditioner(y, coeffs, glob))
    hessians = []
    for i in range(out.shape[1]):
        gi = torch.autograd.grad(out[:, i].sum(), y, create_graph=True,
                                 retain_graph=True)[0]
        rows = [torch.autograd.grad(gi[:, c].sum(), y,
                                    create_graph=create_graph,
                                    retain_graph=True)[0] for c in range(2)]
        hessians.append(torch.stack(rows, dim=1))
    return hessians


# --- field formulas (pinned against Phase 0/2 by the consistency test) ------

def fields_scalar_primal(grads):
    E = _eye(2, grads[0])
    return [E[i] + grads[i] for i in range(2)]


def fields_scalar_dual(grads):
    E = _eye(2, grads[0])
    return [E[i] + torch.stack([g[:, 1], -g[:, 0]], dim=1)
            for i, g in enumerate(grads)]


def fields_mandel_primal(grads6):
    I3 = _eye(3, grads6[0])
    out = []
    for k in range(3):
        gu, gv = grads6[2 * k], grads6[2 * k + 1]
        eps = torch.stack([gu[:, 0], gv[:, 1], (gu[:, 1] + gv[:, 0]) / SQ2], dim=1)
        out.append(I3[k] + eps)
    return out


def fields_mandel_dual(hessians3):
    I3 = _eye(3, hessians3[0])
    out = []
    for k in range(3):
        H = hessians3[k]
        sig = torch.stack([H[:, 1, 1], H[:, 0, 0], -SQ2 * H[:, 0, 1]], dim=1)
        out.append(I3[k] + sig)
    return out


# ----------------------------------------------------------------------------
# Amortized models
# ----------------------------------------------------------------------------

class _AmortizedBase(nn.Module):
    def __init__(self, kmax, sigmas, n_harmonics, width, depth, n_out_primal,
                 n_out_dual, dtype=torch.float64):
        super().__init__()
        self.cond = Conditioner(kmax, sigmas, dtype=dtype)
        F = self.cond.n_feats
        self.chi = ConditionedFieldPair(F, n_harmonics, width, depth,
                                        n_out_primal, dtype=dtype)
        self.dual_net = ConditionedFieldPair(F, n_harmonics, width, depth,
                                             n_out_dual, dtype=dtype)

    @staticmethod
    def _gather(pts, B, coeffs_all, glob_all):
        """Repeat quadrature points per instance and gather per-point
        conditioning tensors. Returns (y_rep, coeffs_rep, glob_rep, N)."""
        N = pts.shape[0]
        y = pts.repeat(B, 1)
        idx = torch.arange(B, device=pts.device).repeat_interleave(N)
        return y, coeffs_all[idx], glob_all[idx], N


class AmortizedConductivity(_AmortizedBase):
    """Operator: pixel conductivity field -> certified [K_low, K_up]."""

    def __init__(self, kmax=4, sigmas=(1 / 16, 3 / 16), n_harmonics=3,
                 width=64, depth=3, dtype=torch.float64):
        super().__init__(kmax, sigmas, n_harmonics, width, depth, 2, 2,
                         dtype=dtype)

    def condition(self, K: np.ndarray):
        return self.cond.project(np.log(K))

    def grams_batch(self, Ks: list[np.ndarray], pts, create_graph):
        B = len(Ks)
        proj = [self.condition(K) for K in Ks]
        coeffs = torch.stack([p[0] for p in proj])
        glob = torch.stack([p[1] for p in proj])
        y, c_rep, g_rep, N = self._gather(pts, B, coeffs, glob)
        _, gp = cond_heads_and_grads(self.chi, self.cond, y, c_rep, g_rep, create_graph)
        _, gd = cond_heads_and_grads(self.dual_net, self.cond, y, c_rep, g_rep, create_graph)
        out = []
        for b, K in enumerate(Ks):
            sl = slice(b * N, (b + 1) * N)
            kv = sample_k(K, pts)
            prim = fields_scalar_primal([g[sl] for g in gp])
            dual = fields_scalar_dual([g[sl] for g in gd])
            out.append((gram(prim, kv), gram(dual, 1.0 / kv)))
        return out

    @torch.enable_grad()
    def bounds_for(self, K: np.ndarray, nq: int) -> CertifiedBounds:
        """Certified interval for an (unseen) microstructure at inference
        cost — no training, no labels, validity by admissibility."""
        d = self.cond.modes
        (K_up, R_up), = self.grams_batch(
            [K], midpoint_grid(nq, dtype=d.dtype, device=d.device), False)
        return CertifiedBounds(as_numpy(torch.linalg.inv(R_up)), as_numpy(K_up))


class AmortizedElasticity(_AmortizedBase):
    """Operator: (n,n,3,3) Mandel field -> certified [C_low, C_up].
    Conditioning scalar proxy: log trace of the pixel stiffness."""

    def __init__(self, kmax=3, sigmas=(1 / 8, 3 / 8), n_harmonics=2,
                 width=40, depth=3, dtype=torch.float64):
        super().__init__(kmax, sigmas, n_harmonics, width, depth, 6, 3,
                         dtype=dtype)

    def condition(self, Cf: np.ndarray):
        return self.cond.project(np.log(np.trace(Cf, axis1=2, axis2=3)))

    def grams_batch(self, Cfs: list[np.ndarray], pts, create_graph):
        B = len(Cfs)
        proj = [self.condition(Cf) for Cf in Cfs]
        coeffs = torch.stack([p[0] for p in proj])
        glob = torch.stack([p[1] for p in proj])
        y, c_rep, g_rep, N = self._gather(pts, B, coeffs, glob)
        _, gp = cond_heads_and_grads(self.chi, self.cond, y, c_rep, g_rep, create_graph)
        hd = cond_hessians(self.dual_net, self.cond, y, c_rep, g_rep, create_graph)
        out = []
        for b, Cf in enumerate(Cfs):
            sl = slice(b * N, (b + 1) * N)
            Cv = sample_mandel(Cf, pts)
            Sv = torch.linalg.inv(Cv)
            prim = fields_mandel_primal([g[sl] for g in gp])
            dual = fields_mandel_dual([H[sl] for H in hd])
            out.append((gram_mandel(prim, Cv), gram_mandel(dual, Sv)))
        return out

    @torch.enable_grad()
    def bounds_for(self, Cf: np.ndarray, nq: int) -> CertifiedBounds:
        d = self.cond.modes
        (C_up, S_up), = self.grams_batch(
            [Cf], midpoint_grid(nq, dtype=d.dtype, device=d.device), False)
        return CertifiedBounds(as_numpy(torch.linalg.inv(S_up)), as_numpy(C_up))


# ----------------------------------------------------------------------------
# Family training (label-free, mirrors Phase 0/2 contracts)
# ----------------------------------------------------------------------------

def fit_amortized(model, train_set: list, nq: int, steps: int, batch: int,
                  lr: float = 1e-3, seed: int = 0, log_every: int = 100,
                  monitor: int = 4, verbose: bool = True):
    """Minimize E_m [ tr G_up(m) + tr G_dual(m) ] over minibatches of the
    family. Returns (model, history); history logs the mean relative gap on
    a fixed monitor subset — every monitored iterate is a valid certificate.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _d = model.cond.modes
    pts = midpoint_grid(nq, dtype=_d.dtype, device=_d.device)
    mon = train_set[:monitor]
    import time
    t0 = time.time()
    history = []

    def log(step):
        gaps = []
        for m in mon:
            b = model.bounds_for(m, nq)
            gaps.append(b.gap_rel())
        rec = {"step": step, "mean_gap": float(np.mean(gaps)),
               "wall_s": time.time() - t0}
        history.append(rec)
        if verbose:
            print(f"step {step:5d} | monitor mean gap {rec['mean_gap']:.3%} "
                  f"| {rec['wall_s']:.1f}s")

    log(0)  # family-wide Voigt-Reuss by zero-initialization
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        idx = rng.choice(len(train_set), size=min(batch, len(train_set)),
                         replace=False)
        opt.zero_grad(set_to_none=True)
        grams = model.grams_batch([train_set[i] for i in idx], pts, True)
        loss = sum(torch.trace(Gu) + torch.trace(Gd) for Gu, Gd in grams)
        (loss / len(grams)).backward()
        opt.step()
        if step % log_every == 0 or step == steps:
            log(step)
    return model, history
