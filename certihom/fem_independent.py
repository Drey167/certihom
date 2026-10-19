"""Matrix-free periodic Q1 elasticity reference solve.

This module is deliberately independent of :mod:`certihom.fem_reference`.
It is intended as an audit tool for a reported primal certificate, not as a
new certification route.  It minimizes the periodic Q1 energy with a
matrix-free, Jacobi-preconditioned conjugate-gradient solve.  Consequently,
for a pixel-aligned Q1 trial displacement ``u_trial`` and the discrete
minimizer ``C_q1`` it must satisfy

    energy(u_trial) >= C_q1

in the Loewner order.  A violation points to an implementation problem in
the trial-energy evaluation rather than to any external reference label.

The implementation intentionally does not import ``fem_reference`` or the
certificate assembly code.  Its only inputs are the per-pixel Mandel
stiffnesses and the usual 2D periodic grid convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SQ2 = np.sqrt(2.0)
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _b_matrix(xi: float, eta: float, h: float) -> np.ndarray:
    """Mandel strain-displacement matrix for a counter-clockwise Q1 cell."""
    dN = np.array([
        [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
    ]) / (2.0 * h)
    B = np.zeros((3, 8))
    for a in range(4):
        B[0, 2 * a] = dN[0, a]
        B[1, 2 * a + 1] = dN[1, a]
        B[2, 2 * a] = dN[1, a] / SQ2
        B[2, 2 * a + 1] = dN[0, a] / SQ2
    return B


def _zero_mean(v: np.ndarray) -> np.ndarray:
    """Project a periodic displacement field onto the translation-free space."""
    return v - v.mean(axis=(0, 1), keepdims=True)


@dataclass(frozen=True)
class Q1ReferenceReport:
    """Residual diagnostics from a matrix-free three-load Q1 solve."""

    iterations: tuple[int, int, int]
    relative_residuals: tuple[float, float, float]


class PeriodicQ1Elasticity:
    """Independent matrix-free periodic Q1 operator for a pixel stiffness map."""

    def __init__(self, Cfield: np.ndarray):
        C = np.asarray(Cfield, dtype=np.float64)
        if C.ndim != 4 or C.shape[0] != C.shape[1] or C.shape[2:] != (3, 3):
            raise ValueError("Cfield must have shape (n, n, 3, 3)")
        if not np.allclose(C, np.swapaxes(C, -1, -2), atol=1e-12):
            raise ValueError("Cfield must be symmetric")
        if np.linalg.eigvalsh(C.reshape(-1, 3, 3)).min() <= 0.0:
            raise ValueError("Cfield must be positive definite per pixel")
        self.C = C
        self.n = C.shape[0]
        self.h = 1.0 / self.n
        self.weight = self.h * self.h / 4.0
        self.Bs = tuple(_b_matrix(xi, eta, self.h) for xi in _GP for eta in _GP)
        self.diag = self._jacobi_diagonal()

    @staticmethod
    def _element_dofs(u: np.ndarray) -> np.ndarray:
        """Gather four periodic Q1 corners into shape ``(n,n,8)``."""
        corners = (
            u,
            np.roll(u, -1, axis=0),
            np.roll(np.roll(u, -1, axis=0), -1, axis=1),
            np.roll(u, -1, axis=1),
        )
        return np.concatenate(corners, axis=2)

    @staticmethod
    def _scatter_element_vectors(local: np.ndarray) -> np.ndarray:
        """Assemble local ``(n,n,8)`` vectors without a sparse matrix."""
        out = np.zeros((*local.shape[:2], 2), dtype=local.dtype)
        out += local[..., 0:2]
        out += np.roll(local[..., 2:4], 1, axis=0)
        out += np.roll(np.roll(local[..., 4:6], 1, axis=0), 1, axis=1)
        out += np.roll(local[..., 6:8], 1, axis=1)
        return out

    def _jacobi_diagonal(self) -> np.ndarray:
        """Diagonal of the assembled Q1 stiffness used by PCG."""
        local_diag = np.zeros((self.n, self.n, 8), dtype=np.float64)
        for B in self.Bs:
            local_diag += self.weight * np.einsum("aq,ijab,bq->ijq", B, self.C, B)
        diag = self._scatter_element_vectors(local_diag)
        if np.any(diag <= 0.0):  # pragma: no cover - material sanity guard
            raise RuntimeError("nonpositive Q1 Jacobi diagonal")
        return diag

    def apply(self, u: np.ndarray) -> np.ndarray:
        """Apply the periodic Q1 stiffness operator to nodal displacement ``u``."""
        u = np.asarray(u, dtype=np.float64)
        if u.shape != (self.n, self.n, 2):
            raise ValueError(f"u must have shape ({self.n}, {self.n}, 2)")
        ue = self._element_dofs(u)
        return _zero_mean(self._apply_robust(ue))

    def _apply_robust(self, ue: np.ndarray) -> np.ndarray:
        local = np.zeros((self.n, self.n, 8), dtype=np.float64)
        for B in self.Bs:
            strain = np.einsum("ab,ijb->ija", B, ue)
            stress = np.einsum("ijab,ijb->ija", self.C, strain)
            local += self.weight * np.einsum("qa,ija->ijq", B.T, stress)
        return self._scatter_element_vectors(local)

    def load(self, macro_strain: np.ndarray) -> np.ndarray:
        """Right hand side ``-int B^T C E`` for one prescribed Mandel strain."""
        E = np.asarray(macro_strain, dtype=np.float64)
        if E.shape != (3,):
            raise ValueError("macro_strain must be a length-3 Mandel vector")
        stress = np.einsum("ijab,b->ija", self.C, E)
        local = np.zeros((self.n, self.n, 8), dtype=np.float64)
        for B in self.Bs:
            local -= self.weight * np.einsum("qa,ija->ijq", B.T, stress)
        return _zero_mean(self._scatter_element_vectors(local))

    def solve(self, rhs: np.ndarray, *, rtol: float = 1e-10,
              maxiter: int | None = None) -> tuple[np.ndarray, int, float]:
        """Solve ``K u = rhs`` in the translation-free periodic subspace."""
        b = _zero_mean(np.asarray(rhs, dtype=np.float64))
        if maxiter is None:
            maxiter = max(1000, 40 * self.n)
        bnorm = float(np.linalg.norm(b))
        if bnorm == 0.0:
            return np.zeros_like(b), 0, 0.0
        x = np.zeros_like(b)
        r = b.copy()
        z = _zero_mean(r / self.diag)
        p = z.copy()
        rz = float(np.vdot(r, z))
        for iteration in range(1, maxiter + 1):
            Ap = self.apply(p)
            pAp = float(np.vdot(p, Ap))
            if not np.isfinite(pAp) or pAp <= 0.0:
                raise RuntimeError("PCG lost positive definiteness")
            alpha = rz / pAp
            x = _zero_mean(x + alpha * p)
            r = _zero_mean(r - alpha * Ap)
            rel = float(np.linalg.norm(r) / bnorm)
            if rel <= rtol:
                return x, iteration, rel
            z = _zero_mean(r / self.diag)
            rz_new = float(np.vdot(r, z))
            if not np.isfinite(rz_new) or rz_new <= 0.0:
                raise RuntimeError("PCG lost positive definiteness")
            p = z + (rz_new / rz) * p
            rz = rz_new
        raise RuntimeError(
            f"PCG did not reach relative residual {rtol:g} in {maxiter} iterations; "
            f"last residual {float(np.linalg.norm(r) / bnorm):.3e}")

    def total_strain(self, u: np.ndarray, macro_strain: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return Q1 total Mandel strains at the four Gauss points."""
        ue = self._element_dofs(u)
        E = np.asarray(macro_strain, dtype=np.float64)
        return tuple(E + np.einsum("ab,ijb->ija", B, ue) for B in self.Bs)

    def effective(self, *, rtol: float = 1e-10,
                  maxiter: int | None = None) -> tuple[np.ndarray, Q1ReferenceReport]:
        """Compute the periodic Q1 effective stiffness independently by PCG."""
        E = np.eye(3)
        displacements: list[np.ndarray] = []
        iterations: list[int] = []
        residuals: list[float] = []
        for k in range(3):
            u, it, rel = self.solve(self.load(E[k]), rtol=rtol, maxiter=maxiter)
            displacements.append(u)
            iterations.append(it)
            residuals.append(rel)
        Ceff = np.zeros((3, 3), dtype=np.float64)
        for k, u in enumerate(displacements):
            for strain in self.total_strain(u, E[k]):
                stress = np.einsum("ijab,ijb->ija", self.C, strain)
                Ceff[:, k] += self.weight * stress.sum(axis=(0, 1))
        Ceff = 0.5 * (Ceff + Ceff.T)
        return Ceff, Q1ReferenceReport(tuple(iterations), tuple(residuals))


def periodic_q1_effective(Cfield: np.ndarray, *, rtol: float = 1e-10,
                          maxiter: int | None = None) -> tuple[np.ndarray, Q1ReferenceReport]:
    """Convenience wrapper for the independent periodic Q1 reference solve."""
    return PeriodicQ1Elasticity(Cfield).effective(rtol=rtol, maxiter=maxiter)
